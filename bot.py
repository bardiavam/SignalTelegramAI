import argparse
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode

from signals import (
    compute_stop_take_prices,
    ensemble_signal,
    log_interval_details,
)
from utils import (
    compute_rsi,
    fetch_recent_data,
    get_binance_client,
    load_config,
    resolve_interval,
    setup_logging,
)


def load_trade_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                logging.getLogger("bot").warning("Trade state file %s is empty; ignoring.", path)
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        logging.getLogger("bot").warning("Trade state file %s is corrupted; ignoring.", path)
        return None


def save_trade_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def clear_trade_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def trade_status(state: Optional[dict]) -> str:
    if not state:
        return ""
    status = state.get("status")
    if isinstance(status, str):
        return status.strip().lower()
    return ""


def build_chart(
    config: Dict[str, Any],
    client,
    interval: str,
    candles: int,
    output_path: Path,
) -> None:
    df = fetch_recent_data(client, config, limit=candles, interval=interval)
    if df.empty:
        raise RuntimeError(f"No data returned for interval {interval}")

    ohlc = df[["open", "high", "low", "close", "volume"]].copy()
    ohlc.index = ohlc.index.tz_convert(None)
    ohlc.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    ohlc["EMA20"] = ohlc["Close"].ewm(span=20, adjust=False).mean()
    ohlc["EMA50"] = ohlc["Close"].ewm(span=50, adjust=False).mean()
    rsi = compute_rsi(ohlc["Close"], period=14)

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={
            "font.size": 9,
            "figure.facecolor": "#0f1117",
            "axes.facecolor": "#0f1117",
        },
    )

    add_plots = [
        mpf.make_addplot(ohlc["EMA20"], color="#1f77b4", width=1.2),
        mpf.make_addplot(ohlc["EMA50"], color="#ff7f0e", width=1.2),
        mpf.make_addplot(rsi, panel=1, color="#2ca02c", ylabel="RSI", width=1.0),
        mpf.make_addplot(pd.Series(70, index=ohlc.index), panel=1, color="#c70039", width=0.8, linestyle="--"),
        mpf.make_addplot(pd.Series(30, index=ohlc.index), panel=1, color="#17becf", width=0.8, linestyle="--"),
    ]

    mpf.plot(
        ohlc,
        type="candle",
        style=style,
        addplot=add_plots,
        volume=True,
        volume_panel=2,
        panel_ratios=(3, 1, 1),
        title=f"{config.get('trading', {}).get('symbol')} ({interval})",
        ylabel="Price",
        ylabel_lower="Volume",
        figratio=(16, 9),
        figscale=1.2,
        savefig=dict(fname=str(output_path), dpi=180, bbox_inches="tight"),
        datetime_format="%Y-%m-%d\n%H:%M",
    )

    plt.close("all")


def derive_trade_levels(
    final: dict,
    interval_results: List[dict],
    risk_cfg: dict,
) -> tuple[float, float, List[float]]:
    entries = [res["price"] for res in interval_results if res["signal"] == final["signal"]]
    entry_price = sum(entries) / len(entries) if entries else final["entry_price"]
    stop_price, take_prices = compute_stop_take_prices(final["signal"], entry_price, risk_cfg)
    return entry_price, stop_price, take_prices


def compose_message(final: dict, entry_price: float, stop_price: float, take_prices: List[float]) -> str:
    header_emoji = "🟢" if final["signal"] == "BUY" else "🔴"
    lines = [
        f"{header_emoji} *{final['signal']}*  · votes {final['votes_for']}/{final['votes_total']} "
        f"· confidence {final['confidence']*100:.1f}%",
        "━━━━━━━━━━━━━━━━━━━━",
        "🪙 *Levels*",
        f"• Entry: `{entry_price:.2f}`",
        f"• Stop : `{stop_price:.2f}`",
    ]
    lines.append("🎯 *Targets*")
    for idx, target in enumerate(take_prices, start=1):
        lines.append(f"• TP{idx}: `{target:.2f}`")

    lines.extend(
        [
            "",
            f"🕒 Time: `{final['time'].isoformat()}`",
        ]
    )
    return "\n".join(lines)


async def generate_signal_bundle(
    config: Dict[str, Any],
    client,
    intervals: Optional[List[str]],
    history_window: int,
    threshold_override: Optional[float],
) -> dict:
    return await asyncio.to_thread(
        ensemble_signal,
        config,
        client,
        intervals,
        history_window,
        threshold_override,
    )


async def create_chart_file(
    config: Dict[str, Any],
    client,
    interval: str,
    candles: int,
) -> Path:
    def _build() -> Path:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            chart_path = Path(tmp.name)
        build_chart(config, client, interval, candles, chart_path)
        return chart_path

    return await asyncio.to_thread(_build)


async def send_photo(
    bot: Bot,
    chat_id: str,
    chart_path: Path,
    caption: str,
) -> None:
    photo_bytes = await asyncio.to_thread(chart_path.read_bytes)
    await bot.send_photo(
        chat_id=chat_id,
        photo=photo_bytes,
        caption=caption,
        parse_mode=ParseMode.MARKDOWN,
    )


def compose_close_message(
    state: dict,
    exit_price: float,
    exit_reason: str,
    pnl_pct: float,
    exit_time: datetime,
) -> str:
    if exit_reason.startswith("take"):
        emoji = "🎯"
        suffix = exit_reason.split("_", 1)[-1].upper() if "_" in exit_reason else "TP"
        reason_text = f"Take Profit ({suffix})"
    else:
        emoji = "🛑"
        reason_text = "Stop Loss"

    take_levels = state.get("take_levels") or ([state["take"]] if state.get("take") is not None else [])
    lines = [
        f"{emoji} *{reason_text}* ({state['signal']})",
        "━━━━━━━━━━━━━━━━━━━━",
        "🪙 *Trade*",
        f"• Entry: `{state['entry']:.2f}`",
        f"• Stop : `{state['stop']:.2f}`",
    ]
    lines.append("🎯 *Targets*")
    for idx, target in enumerate(take_levels, start=1):
        lines.append(f"• TP{idx}: `{target:.2f}`")

    lines.extend(
        [
            f"• Exit : `{exit_price:.2f}`",
            f"• PnL : `{pnl_pct*100:.2f}%`",
            "",
            f"🕒 Opened: `{state['entry_time']}`",
            f"🕒 Closed: `{exit_time.isoformat()}`",
        ]
    )
    return "\n".join(lines)


async def run_once(
    bot: Bot,
    config: Dict[str, Any],
    client,
    intervals: Optional[List[str]],
    history_window: int,
    threshold_override: Optional[float],
    risk_cfg: dict,
    chart_interval: Optional[str],
    chart_candles: int,
    chat_id: str,
    logger: logging.Logger,
    state_path: Path,
) -> None:
    state = await asyncio.to_thread(load_trade_state, state_path)
    if trade_status(state) == "open":
        take_levels = state.get("take_levels") or ([state["take"]] if state.get("take") is not None else [])

        df = await asyncio.to_thread(fetch_recent_data, client, config, 2, state.get("base_interval"))
        if df.empty:
            logger.warning("No market data available to evaluate existing position.")
            return

        latest = df.iloc[-1]
        high = float(latest["high"])
        low = float(latest["low"])
        exit_reason = None
        exit_price = None
        take_hit_index: Optional[int] = None

        if state["signal"] == "BUY":
            for idx, target in enumerate(take_levels or []):
                if high >= target:
                    exit_reason = f"take_tp{idx + 1}"
                    exit_price = target
                    take_hit_index = idx
                    break
            if exit_reason is None and low <= state["stop"]:
                exit_reason = "stop"
                exit_price = state["stop"]
        else:
            for idx, target in enumerate(take_levels or []):
                if low <= target:
                    exit_reason = f"take_tp{idx + 1}"
                    exit_price = target
                    take_hit_index = idx
                    break
            if exit_reason is None and high >= state["stop"]:
                exit_reason = "stop"
                exit_price = state["stop"]

        if exit_reason is None:
            logger.info(
                "Active %s trade still open (entry %.2f | stop %.2f | targets=%s | last %.2f)",
                state["signal"],
                state["entry"],
                state["stop"],
                ", ".join(f"{tp:.2f}" for tp in take_levels) if take_levels else "n/a",
                float(latest["close"]),
            )
            return

        exit_time = datetime.now(timezone.utc)
        if state["signal"] == "BUY":
            pnl_pct = (exit_price - state["entry"]) / state["entry"]
        else:
            pnl_pct = (state["entry"] - exit_price) / state["entry"]

        message = compose_close_message(state, exit_price, exit_reason, pnl_pct, exit_time)
        await bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.MARKDOWN)

        closed_state = dict(state)
        closed_state.update(
            {
                "status": "closed",
                "exit_price": exit_price,
                "exit_time": exit_time.isoformat(),
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
            }
        )
        if take_hit_index is not None:
            closed_state["take_hit_index"] = take_hit_index
        await asyncio.to_thread(save_trade_state, state_path, closed_state)
        logger.info("Trade closed via %s at %.2f (PnL %.4f)", exit_reason, exit_price, pnl_pct)
        return

    bundle = await generate_signal_bundle(
        config=config,
        client=client,
        intervals=intervals,
        history_window=history_window,
        threshold_override=threshold_override,
    )

    interval_results = bundle["interval_results"]
    final = bundle["final"]
    log_interval_details(interval_results, risk_cfg)

    entry_price, stop_price, take_prices = derive_trade_levels(final, interval_results, risk_cfg)
    trade_state = {
        "status": "open",
        "signal": final["signal"],
        "entry": entry_price,
        "stop": stop_price,
        "take": take_prices[-1] if take_prices else None,
        "take_levels": take_prices,
        "entry_time": final["time"].isoformat(),
        "base_interval": final["base_interval"],
        "intervals": intervals,
        "confidence": final["confidence"],
        "votes_for": final["votes_for"],
        "votes_total": final["votes_total"],
    }

    chart_interval_resolved = chart_interval or final["base_interval"] or resolve_interval(config)
    chart_path = await create_chart_file(
        config=config,
        client=client,
        interval=chart_interval_resolved,
        candles=chart_candles,
    )

    try:
        caption = compose_message(final, entry_price, stop_price, take_prices)
        latest_state = await asyncio.to_thread(load_trade_state, state_path)
        if trade_status(latest_state) == "open":
            logger.info(
                "Existing %s trade opened at %s still active; skipping new %s signal.",
                latest_state.get("signal"),
                latest_state.get("entry_time"),
                final["signal"],
            )
            return

        await send_photo(bot, chat_id, chart_path, caption)
        await asyncio.to_thread(save_trade_state, state_path, trade_state)
        logger.info(
            "Signal %s sent to %s (votes %d/%d, confidence %.2f, avg prob %.3f)",
            final["signal"],
            chat_id,
            final["votes_for"],
            final["votes_total"],
            final["confidence"],
            final["avg_probability"],
        )
    finally:
        await asyncio.to_thread(chart_path.unlink, missing_ok=True)


async def run_loop(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("bot")

    telegram_cfg = config.get("telegram", {})
    token = telegram_cfg.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Telegram bot token missing. Set config.telegram.bot_token or TELEGRAM_BOT_TOKEN env.")

    chat_id = (
        args.chat_id
        or telegram_cfg.get("channel_id")
        or telegram_cfg.get("chat_id")
    )
    if not chat_id:
        raise RuntimeError("Telegram chat/channel ID missing. Provide --chat-id or config.telegram.channel_id/chat_id.")

    client = get_binance_client(config)
    monitor_cfg = config.get("monitor", {})
    risk_cfg = config.get("backtest", {})
    history_window = args.window or monitor_cfg.get("history_window", 200)
    chart_interval = args.chart_interval
    intervals = args.intervals or config.get("trading", {}).get("intervals") or [resolve_interval(config)]
    state_cfg = config.get("bot", {})
    state_path = Path(state_cfg.get("state_path", "state/trade_state.json"))

    bot = Bot(token=token)

    logger.info(
        "Starting Telegram signal loop for %s | intervals=%s | poll_interval=%ss",
        chat_id,
        ", ".join(intervals),
        args.poll_interval if not args.once else "N/A",
    )

    try:
        while True:
            try:
                await run_once(
                    bot=bot,
                    config=config,
                    client=client,
                    intervals=intervals,
                    history_window=history_window,
                    threshold_override=args.threshold,
                    risk_cfg=risk_cfg,
                    chart_interval=chart_interval,
                    chart_candles=args.chart_candles,
                    chat_id=str(chat_id),
                    logger=logger,
                    state_path=state_path,
                )
            except Exception as exc:
                logger.exception("Failed to generate/send signal: %s", exc)

            if args.once:
                break

            await asyncio.sleep(args.poll_interval)
    finally:
        await asyncio.to_thread(client.close_session, True) if hasattr(client, "close_session") else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram monitoring bot for trading signals.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--window", type=int, help="Candles to fetch per interval.")
    parser.add_argument("--threshold", type=float, help="Override probability threshold.")
    parser.add_argument("--intervals", nargs="+", help="Intervals to include in voting.")
    parser.add_argument("--chart-interval", help="Interval to use for chart (defaults to base interval).")
    parser.add_argument("--chart-candles", type=int, default=180, help="Candles to display on the chart.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between signal updates.")
    parser.add_argument("--chat-id", help="Override Telegram chat/channel ID.")
    parser.add_argument("--once", action="store_true", help="Send a single update then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    main()
