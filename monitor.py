import argparse
import logging
import time
from datetime import datetime, timezone

import joblib

from utils import (
    engineer_features,
    fetch_recent_data,
    get_binance_client,
    load_config,
    model_path_for_interval,
    resolve_interval,
    setup_logging,
)


def _run_monitor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("monitor")

    interval = resolve_interval(config, getattr(args, "timeframe", None))
    config.setdefault("trading", {})["interval"] = interval

    model_path = model_path_for_interval(config, interval)
    logger.info("Loading model for interval %s from %s", interval, model_path)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    client = get_binance_client(config)
    monitor_cfg = config.get("monitor", {})
    risk_cfg = config.get("backtest", {})
    history_window = args.window or monitor_cfg.get("history_window", 200)
    poll_interval = args.poll_interval or monitor_cfg.get("poll_interval_seconds", 60)
    threshold = args.threshold
    if threshold is None:
        threshold = artifact.get("best_threshold")
    if threshold is None:
        threshold = monitor_cfg.get("signal_threshold", 0.5)

    logger.info("Starting monitor for %s", config.get("trading", {}).get("symbol"))
    logger.info(
        "Using history window=%d, poll interval=%ds, probability threshold=%.2f",
        history_window,
        poll_interval,
        threshold,
    )
    if risk_cfg:
        logger.info(
            "Risk settings -> stop_loss=%.2f%% take_profit=%.2f%% max_hold=%d bars cooldown=%d bars",
            risk_cfg.get("stop_loss_pct", 0.0) * 100,
            risk_cfg.get("take_profit_pct", 0.0) * 100,
            risk_cfg.get("max_hold_bars", 0),
            risk_cfg.get("cooldown_bars", 0),
        )

    try:
        while True:
            df = fetch_recent_data(client, config, limit=history_window, interval=interval)
            data = engineer_features(df, config)
            if data.features.empty:
                logger.warning("Not enough data to compute features yet.")
                time.sleep(poll_interval)
                continue

            X = data.features.reindex(columns=feature_columns, fill_value=0.0)
            latest_features = X.iloc[[-1]]
            proba = model.predict_proba(latest_features)[0, 1]
            signal = "BUY" if proba > threshold else "SELL"
            latest_row = data.raw.iloc[-1]
            price = latest_row["close"]
            ts = latest_row.name.to_pydatetime().astimezone(timezone.utc)

            stop_pct = risk_cfg.get("stop_loss_pct", 0.0)
            take_pct = risk_cfg.get("take_profit_pct", 0.0)
            stop_price = price * (1 - stop_pct) if signal == "BUY" else price * (1 + stop_pct)
            take_price = price * (1 + take_pct) if signal == "BUY" else price * (1 - take_pct)

            logger.info(
                "signal=%s prob=%.3f entry=%.2f stop=%.2f take=%.2f time=%s",
                signal,
                proba,
                price,
                stop_price,
                take_price,
                ts.isoformat(),
            )

            if args.once:
                break

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user.")


def run_monitor(
    config_path: str = "config.yaml",
    window: int = None,
    poll_interval: int = None,
    threshold: float = None,
    once: bool = False,
    timeframe: str | None = None,
) -> None:
    args = argparse.Namespace(
        config=config_path,
        window=window,
        poll_interval=poll_interval,
        threshold=threshold,
        once=once,
        timeframe=timeframe,
    )
    _run_monitor(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor live market data to generate trading signals.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--window", type=int, help="Number of candles to fetch for each evaluation.")
    parser.add_argument("--poll-interval", type=int, help="Polling interval in seconds.")
    parser.add_argument("--timeframe", help="Override trading interval (e.g., 15m, 1h).")
    parser.add_argument("--threshold", type=float, help="Probability threshold for buy signal.")
    parser.add_argument("--once", action="store_true", help="Fetch and print a single signal then exit.")
    return parser.parse_args()


if __name__ == "__main__":
    _run_monitor(parse_args())
