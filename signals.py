from __future__ import annotations

import logging
from collections import Counter
from datetime import timezone
from typing import List, Optional

import joblib

from utils import (
    engineer_features,
    fetch_recent_data,
    model_path_for_interval,
    resolve_interval,
)


def _resolve_take_profit_pcts(risk_cfg: dict) -> List[float]:
    configured = risk_cfg.get("take_profit_pcts")
    if configured:
        return [abs(float(pct)) for pct in configured if pct]

    base = abs(float(risk_cfg.get("take_profit_pct", 0.0)))
    if base <= 0:
        return []
    multipliers = (0.5, 1.0, 1.5)
    return [base * mult for mult in multipliers]


def compute_stop_take_prices(signal: str, price: float, risk_cfg: dict) -> tuple[float, List[float]]:
    stop_pct = risk_cfg.get("stop_loss_pct", 0.0)
    stop_price = price * (1 - stop_pct) if signal == "BUY" else price * (1 + stop_pct)

    take_pcts = _resolve_take_profit_pcts(risk_cfg)
    if signal == "BUY":
        take_prices = [price * (1 + pct) for pct in take_pcts]
        take_prices.sort()
    else:
        take_prices = [price * (1 - pct) for pct in take_pcts]
        take_prices.sort(reverse=True)

    return stop_price, take_prices


def load_interval_signal(
    config: dict,
    client,
    interval: str,
    history_window: int,
    threshold_override: Optional[float],
) -> Optional[dict]:
    logger = logging.getLogger("signal")
    path = model_path_for_interval(config, interval)
    try:
        artifact = joblib.load(path)
    except FileNotFoundError:
        logger.warning("Model artifact missing for interval %s at %s", interval, path)
        return None

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    threshold = threshold_override
    if threshold is None:
        threshold = artifact.get("best_threshold")
    if threshold is None:
        threshold = config.get("monitor", {}).get("signal_threshold", 0.5)

    df = fetch_recent_data(client, config, limit=history_window, interval=interval)
    data = engineer_features(df, config)
    if data.features.empty:
        logging.getLogger("signal").warning(
            "Insufficient feature rows for interval %s (history_window=%d)", interval, history_window
        )
        return None

    X = data.features.reindex(columns=feature_columns, fill_value=0.0)
    latest_features = X.iloc[[-1]]
    proba = model.predict_proba(latest_features)[0, 1]
    signal = "BUY" if proba > threshold else "SELL"
    latest_row = data.raw.iloc[-1]
    price = latest_row["close"]
    ts = latest_row.name.to_pydatetime().astimezone(timezone.utc)

    return {
        "interval": interval,
        "probability": float(proba),
        "threshold": float(threshold),
        "signal": signal,
        "price": float(price),
        "time": ts,
        "model_path": path,
    }


def collect_signals(
    config: dict,
    client,
    intervals: List[str],
    history_window: int,
    threshold_override: Optional[float],
) -> List[dict]:
    unique_intervals = list(dict.fromkeys(intervals))
    results: List[dict] = []
    for interval in unique_intervals:
        res = load_interval_signal(
            config=config,
            client=client,
            interval=interval,
            history_window=history_window,
            threshold_override=threshold_override,
        )
        if res:
            results.append(res)
    return results


def log_interval_details(results: List[dict], risk_cfg: dict) -> None:
    logger = logging.getLogger("signal")
    for res in results:
        stop_price, take_prices = compute_stop_take_prices(res["signal"], res["price"], risk_cfg)
        take_repr = ", ".join(f"{tp:.2f}" for tp in take_prices) if take_prices else "n/a"
        logger.info(
            "interval=%s signal=%s prob=%.3f thresh=%.3f entry=%.2f stop=%.2f takes=[%s] time=%s",
            res["interval"],
            res["signal"],
            res["probability"],
            res["threshold"],
            res["price"],
            stop_price,
            take_repr,
            res["time"].isoformat(),
        )


def aggregate_vote(results: List[dict], base_interval: str) -> dict:
    votes = Counter(res["signal"] for res in results)
    final_signal, vote_count = votes.most_common(1)[0]
    confidence = vote_count / len(results)
    avg_probability = sum(res["probability"] for res in results if res["signal"] == final_signal) / vote_count
    base_result = next((res for res in results if res["interval"] == base_interval), results[0])
    return {
        "signal": final_signal,
        "confidence": confidence,
        "votes_for": vote_count,
        "votes_total": len(results),
        "avg_probability": avg_probability,
        "entry_price": base_result["price"],
        "time": base_result["time"],
        "base_interval": base_result["interval"],
    }


def build_single_summary(result: dict) -> dict:
    return {
        "signal": result["signal"],
        "confidence": 1.0,
        "votes_for": 1,
        "votes_total": 1,
        "avg_probability": result["probability"],
        "entry_price": result["price"],
        "time": result["time"],
        "base_interval": result["interval"],
    }


def ensemble_signal(
    config: dict,
    client,
    intervals: Optional[List[str]],
    history_window: int,
    threshold_override: Optional[float],
) -> dict:
    trading_cfg = config.get("trading", {})
    if not intervals:
        intervals = trading_cfg.get("intervals") or [resolve_interval(config)]

    results = collect_signals(
        config=config,
        client=client,
        intervals=intervals,
        history_window=history_window,
        threshold_override=threshold_override,
    )

    if not results:
        raise RuntimeError("No interval signals could be generated; check model artifacts and data availability.")

    base_interval = resolve_interval(config)
    if len(results) == 1:
        final_summary = build_single_summary(results[0])
    else:
        final_summary = aggregate_vote(results, base_interval=base_interval)

    return {
        "interval_results": results,
        "final": final_summary,
        "base_interval": base_interval,
    }
