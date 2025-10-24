import argparse
import logging
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from utils import (
    engineer_features,
    fetch_historical_data,
    get_binance_client,
    load_config,
    model_path_for_interval,
    resolve_interval,
    setup_logging,
)


def parse_dates(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        datetime.fromisoformat(date_str)
        return date_str
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{date_str}'. Use ISO format e.g. 2023-01-01.") from exc


def interval_to_hours(interval: str) -> float:
    mapping = {
        "1m": 1 / 60,
        "3m": 3 / 60,
        "5m": 5 / 60,
        "15m": 15 / 60,
        "30m": 0.5,
        "1h": 1,
        "2h": 2,
        "4h": 4,
        "6h": 6,
        "8h": 8,
        "12h": 12,
        "1d": 24,
        "3d": 72,
        "1w": 168,
        "1M": 24 * 30,
    }
    return mapping.get(interval, 1)


def simulate_trades(
    probabilities: pd.Series,
    raw: pd.DataFrame,
    threshold: float,
    transaction_cost: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_bars: int,
    cooldown_bars: int,
    probability_exit_buffer: Optional[float],
) -> List[dict]:
    index = probabilities.index
    closes = raw.loc[index, "close"]
    highs = raw.loc[index, "high"]
    lows = raw.loc[index, "low"]

    trades: List[dict] = []
    cooldown = 0
    i = 0
    timestamps = list(index)

    while i < len(index) - 1:
        if cooldown > 0:
            cooldown -= 1
        prob = probabilities.iloc[i]
        if prob <= threshold or cooldown > 0:
            i += 1
            continue

        entry_idx = i
        entry_time = timestamps[entry_idx]
        entry_price = closes.iloc[entry_idx]

        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "end_of_data"

        current_pos = entry_idx + 1
        while current_pos < len(index):
            hold_bars = current_pos - entry_idx
            high = highs.iloc[current_pos]
            low = lows.iloc[current_pos]
            close_price = closes.iloc[current_pos]

            runup = high / entry_price - 1
            drawdown = low / entry_price - 1

            if take_profit_pct and runup >= take_profit_pct:
                exit_idx = current_pos
                exit_price = entry_price * (1 + take_profit_pct)
                exit_reason = "take_profit"
                break

            if stop_loss_pct and drawdown <= -stop_loss_pct:
                exit_idx = current_pos
                exit_price = entry_price * (1 - stop_loss_pct)
                exit_reason = "stop_loss"
                break

            if probability_exit_buffer is not None:
                proba_current = probabilities.iloc[current_pos]
                if proba_current < max(threshold - probability_exit_buffer, 0.0):
                    exit_idx = current_pos
                    exit_price = close_price
                    exit_reason = "probability_drop"
                    break

            if max_hold_bars and hold_bars >= max_hold_bars:
                exit_idx = current_pos
                exit_price = close_price
                exit_reason = "time_exit"
                break

            current_pos += 1

        if exit_idx == entry_idx:
            exit_idx = len(index) - 1
            exit_price = closes.iloc[exit_idx]
            exit_reason = "end_of_data"

        gross_return = exit_price / entry_price - 1
        net_return = gross_return - 2 * transaction_cost

        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": timestamps[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "holding_bars": exit_idx - entry_idx,
                "probability": prob,
                "exit_reason": exit_reason,
            }
        )

        cooldown = cooldown_bars
        i = exit_idx + 1

    return trades


def _run_backtest(
    args: argparse.Namespace,
) -> None:
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("backtest")

    interval = resolve_interval(config, getattr(args, "interval", None))
    config.setdefault("trading", {})["interval"] = interval
    model_path = model_path_for_interval(config, interval)
    logger.info("Loading model artifact for interval %s from %s", interval, model_path)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    client = get_binance_client(config)
    cfg_backtest = config.get("backtest", {})

    start = parse_dates(args.start or cfg_backtest.get("start"))
    end = parse_dates(args.end or cfg_backtest.get("end"))

    df = fetch_historical_data(client, config, start_str=start, end_str=end, interval=interval)
    data = engineer_features(df, config)
    X = data.features.reindex(columns=feature_columns, fill_value=0.0)

    probabilities = model.predict_proba(X)[:, 1]
    threshold = args.threshold
    if threshold is None:
        threshold = artifact.get("best_threshold")
    if threshold is None:
        threshold = config.get("monitor", {}).get("signal_threshold", 0.5)

    transaction_cost = args.cost if args.cost is not None else cfg_backtest.get("transaction_cost", 0.0007)
    risk_cfg = cfg_backtest
    stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.0)
    take_profit_pct = risk_cfg.get("take_profit_pct", 0.0)
    max_hold_bars = risk_cfg.get("max_hold_bars", 12)
    cooldown_bars = risk_cfg.get("cooldown_bars", 0)
    probability_exit_buffer = risk_cfg.get("probability_exit_buffer")

    trades = simulate_trades(
        probabilities=pd.Series(probabilities, index=X.index),
        raw=data.raw,
        threshold=threshold,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_bars=max_hold_bars,
        cooldown_bars=cooldown_bars,
        probability_exit_buffer=probability_exit_buffer,
    )

    initial_capital = args.capital if args.capital is not None else cfg_backtest.get("initial_capital", 10000)
    capital = initial_capital
    equity = [capital]

    for trade in trades:
        capital *= 1 + trade["net_return"]
        equity.append(capital)

    total_return = capital / initial_capital - 1

    if len(trades) > 0:
        start_time = trades[0]["entry_time"]
        end_time = trades[-1]["exit_time"]
    else:
        start_time = X.index[0]
        end_time = X.index[-1]

    years = max((end_time - start_time).days / 365.25, 1e-6)
    annualized_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0

    trade_returns = np.array([t["net_return"] for t in trades])
    wins = (trade_returns > 0).sum()
    win_rate = wins / len(trades) if trades else 0.0

    avg_trade_return = float(trade_returns.mean()) if len(trade_returns) else 0.0
    median_trade_return = float(np.median(trade_returns)) if len(trade_returns) else 0.0
    std_trade_return = float(trade_returns.std(ddof=1)) if len(trade_returns) > 1 else 0.0
    trades_per_year = len(trades) / years if years > 0 else 0.0
    sharpe = (
        (avg_trade_return / (std_trade_return + 1e-9)) * np.sqrt(trades_per_year)
        if len(trades) > 1 and std_trade_return > 0
        else 0.0
    )

    equity_array = np.array(equity)
    rolling_max = np.maximum.accumulate(equity_array)
    drawdowns = equity_array / rolling_max - 1
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

    logger.info("Backtest completed with %d trades.", len(trades))
    logger.info("Total Return: %.2f%%", total_return * 100)
    logger.info("Annualized Return: %.2f%%", annualized_return * 100)
    logger.info("Sharpe Ratio: %.2f", sharpe)
    logger.info("Max Drawdown: %.2f%%", max_drawdown * 100)
    logger.info("Win Rate: %.2f%%", win_rate * 100)
    logger.info("Average Trade Return: %.2f%%", avg_trade_return * 100)

    if args.export:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(args.export, index=False)
        logger.info("Exported trade log to %s", args.export)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest with risk management over historical data.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--start", help="Override backtest start date (ISO format).")
    parser.add_argument("--end", help="Override backtest end date (ISO format).")
    parser.add_argument("--capital", type=float, help="Override initial capital.")
    parser.add_argument("--cost", type=float, help="Override transaction cost per trade.")
    parser.add_argument("--threshold", type=float, help="Probability threshold for taking trades.")
    parser.add_argument("--export", help="CSV path to export trade log.")
    parser.add_argument("--interval", help="Interval to backtest (e.g., 15m, 1h, 4h).")
    return parser.parse_args()


def run_backtest(
    config_path: str = "config.yaml",
    start: Optional[str] = None,
    end: Optional[str] = None,
    capital: Optional[float] = None,
    cost: Optional[float] = None,
    threshold: Optional[float] = None,
    export: Optional[str] = None,
    interval: Optional[str] = None,
) -> None:
    args = argparse.Namespace(
        config=config_path,
        start=start,
        end=end,
        capital=capital,
        cost=cost,
        threshold=threshold,
        export=export,
        interval=interval,
    )
    _run_backtest(args)


if __name__ == "__main__":
    _run_backtest(parse_args())
