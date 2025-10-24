import argparse
import logging
from typing import List, Optional

from signals import (
    compute_stop_take_prices,
    ensemble_signal,
    log_interval_details,
)
from utils import get_binance_client, load_config, resolve_interval, setup_logging


def _run_signal(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("signal")

    client = get_binance_client(config)
    monitor_cfg = config.get("monitor", {})
    risk_cfg = config.get("backtest", {})
    history_window = args.window or monitor_cfg.get("history_window", 200)

    intervals: Optional[List[str]]
    if args.intervals:
        intervals = args.intervals
    elif args.interval:
        intervals = [args.interval]
    else:
        trading_cfg = config.get("trading", {})
        intervals = trading_cfg.get("intervals") or [resolve_interval(config)]

    result_bundle = ensemble_signal(
        config=config,
        client=client,
        intervals=intervals,
        history_window=history_window,
        threshold_override=args.threshold,
    )

    interval_results = result_bundle["interval_results"]
    final = result_bundle["final"]

    log_interval_details(interval_results, risk_cfg)

    stop_price, take_prices = compute_stop_take_prices(final["signal"], final["entry_price"], risk_cfg)
    take_repr = ", ".join(f"{tp:.2f}" for tp in take_prices) if take_prices else "n/a"
    logger.info(
        "final=%s votes=%d/%d confidence=%.2f avg_prob=%.3f entry=%.2f stop=%.2f takes=[%s] time=%s",
        final["signal"],
        final["votes_for"],
        final["votes_total"],
        final["confidence"],
        final["avg_probability"],
        final["entry_price"],
        stop_price,
        take_repr,
        final["time"].isoformat(),
    )


def run_signal(
    config_path: str = "config.yaml",
    window: int = None,
    threshold: float = None,
    interval: Optional[str] = None,
    intervals: Optional[List[str]] = None,
) -> None:
    args = argparse.Namespace(
        config=config_path,
        window=window,
        threshold=threshold,
        interval=interval,
        intervals=intervals,
    )
    _run_signal(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print latest trading signal based on most recent data.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--window", type=int, help="Candles to fetch for feature computation.")
    parser.add_argument("--threshold", type=float, help="Override probability threshold.")
    parser.add_argument("--interval", help="Evaluate a single interval (e.g., 1h).")
    parser.add_argument("--intervals", nargs="+", help="List of intervals for ensemble voting.")
    return parser.parse_args()


if __name__ == "__main__":
    _run_signal(parse_args())
