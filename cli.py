import argparse

from backtest import run_backtest
from monitor import run_monitor
from signal_cli import run_signal
from train import run_training


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gold trading CLI powered by machine learning signals.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train the configured machine learning model.")
    parser_train.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_train.add_argument("--model-type", help="Override model_type defined in config (e.g., lightgbm).")
    parser_train.add_argument("--interval", help="Override interval defined in config (e.g., 15m).")
    parser_train.set_defaults(
        handler=lambda args: run_training(
            args.config,
            override_model_type=args.model_type,
            override_interval=args.interval,
        )
    )

    parser_backtest = subparsers.add_parser("backtest", help="Run backtest over historical data.")
    parser_backtest.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_backtest.add_argument("--start", help="Override backtest start date (ISO format).")
    parser_backtest.add_argument("--end", help="Override backtest end date (ISO format).")
    parser_backtest.add_argument("--capital", type=float, help="Override initial capital.")
    parser_backtest.add_argument("--cost", type=float, help="Override transaction cost per trade.")
    parser_backtest.add_argument("--threshold", type=float, help="Probability threshold for taking trades.")
    parser_backtest.add_argument("--export", help="CSV path to export detailed results.")
    parser_backtest.add_argument("--interval", help="Interval to backtest (e.g., 15m, 1h, 4h).")
    parser_backtest.set_defaults(
        handler=lambda args: run_backtest(
            config_path=args.config,
            start=args.start,
            end=args.end,
            capital=args.capital,
            cost=args.cost,
            threshold=args.threshold,
            export=args.export,
            interval=args.interval,
        )
    )

    parser_monitor = subparsers.add_parser("monitor", help="Monitor live market data for signals.")
    parser_monitor.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_monitor.add_argument("--window", type=int, help="Number of candles to fetch each cycle.")
    parser_monitor.add_argument("--poll-interval", type=int, help="Polling interval in seconds.")
    parser_monitor.add_argument("--timeframe", help="Override interval used for monitoring (e.g., 15m).")
    parser_monitor.add_argument("--threshold", type=float, help="Probability threshold for buy signal.")
    parser_monitor.add_argument("--once", action="store_true", help="Fetch a single signal then exit.")
    parser_monitor.set_defaults(
        handler=lambda args: run_monitor(
            config_path=args.config,
            window=args.window,
            poll_interval=args.poll_interval,
            threshold=args.threshold,
            once=args.once,
            timeframe=args.timeframe,
        )
    )

    parser_signal = subparsers.add_parser("signal", help="Print latest model signal.")
    parser_signal.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_signal.add_argument("--window", type=int, help="Candles to fetch for feature computation.")
    parser_signal.add_argument("--threshold", type=float, help="Probability threshold override.")
    parser_signal.add_argument("--interval", help="Evaluate a single interval (overrides config default).")
    parser_signal.add_argument("--intervals", nargs="+", help="List of intervals for ensemble voting.")
    parser_signal.set_defaults(
        handler=lambda args: run_signal(
            config_path=args.config,
            window=args.window,
            threshold=args.threshold,
            interval=args.interval,
            intervals=args.intervals,
        )
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()
