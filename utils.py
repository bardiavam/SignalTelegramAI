import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from binance.client import Client
from sklearn.model_selection import train_test_split


@dataclass
class DataBundle:
    features: pd.DataFrame
    target: pd.Series
    raw: pd.DataFrame


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    if path.suffix.lower() in {".yml", ".yaml"}:
        with path.open("r") as f:
            return yaml.safe_load(f)

    if path.suffix.lower() == ".json":
        with path.open("r") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config file format: {path.suffix}")


def setup_logging(config: Dict[str, Any]) -> None:
    logging_cfg = config.get("logging", {})
    level_name = logging_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    handlers: List[logging.Handler] = [logging.StreamHandler()]

    log_file = logging_cfg.get("file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def get_binance_client(config: Dict[str, Any]) -> Client:
    api_cfg = config.get("api") or {}
    api_key = api_cfg.get("api_key")
    api_secret = api_cfg.get("api_secret")
    testnet = api_cfg.get("testnet", False)

    client = Client(api_key, api_secret)
    if testnet:
        client.API_URL = "https://api.binance.com/"
    return client


def ensure_symbol(config: Dict[str, Any], symbol: str) -> None:
    expected = config.get("trading", {}).get("symbol")
    if expected and symbol.upper() != expected.upper():
        raise ValueError(f"Symbol mismatch. Expected {expected}, received {symbol}")


def resolve_interval(config: Dict[str, Any], interval: str | None = None) -> str:
    if interval:
        return interval
    trading_cfg = config.get("trading", {})
    if "default_interval" in trading_cfg:
        return trading_cfg["default_interval"]
    if "interval" in trading_cfg:
        return trading_cfg["interval"]
    intervals = trading_cfg.get("intervals") or []
    if intervals:
        return intervals[0]
    return "1h"


def model_path_for_interval(config: Dict[str, Any], interval: str | None = None) -> str:
    trading_cfg = config.get("trading", {})
    interval_resolved = resolve_interval(config, interval)
    template = trading_cfg.get("model_path_template")
    if template:
        return template.format(interval=interval_resolved)
    path = trading_cfg.get("model_path")
    if path:
        if "{interval}" in path:
            return path.format(interval=interval_resolved)
        stem = Path(path)
        return str(stem.with_name(f"{stem.stem}_{interval_resolved}{stem.suffix}"))
    return f"models/model_{interval_resolved}.pkl"


def fetch_historical_data(
    client: Client,
    config: Dict[str, Any],
    start_str: str,
    end_str: str = None,
    interval: str | None = None,
) -> pd.DataFrame:
    trading_cfg = config.get("trading", {})
    symbol = trading_cfg["symbol"]
    resolved_interval = resolve_interval(config, interval)

    logging.info(
        "Downloading historical data for %s interval %s from %s to %s",
        symbol,
        resolved_interval,
        start_str,
        end_str or "now",
    )

    klines = client.get_historical_klines(
        symbol, resolved_interval, start_str=start_str, end_str=end_str
    )
    return klines_to_dataframe(klines)


def klines_to_dataframe(klines: List[List[Any]]) -> pd.DataFrame:
    if not klines:
        raise RuntimeError("No market data returned from Binance.")
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("close_time", inplace=True)
    return df


def fetch_recent_data(
    client: Client,
    config: Dict[str, Any],
    limit: int,
    interval: str | None = None,
) -> pd.DataFrame:
    trading_cfg = config.get("trading", {})
    klines = client.get_klines(
        symbol=trading_cfg["symbol"],
        interval=resolve_interval(config, interval),
        limit=limit,
    )
    return klines_to_dataframe(klines)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def engineer_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> DataBundle:
    df = df.copy()

    feature_cfg = config.get("feature_engineering", {})
    trading_cfg = config.get("trading", {})

    df["return_1"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    feature_columns = ["return_1", "log_return"]

    # Moving averages and ratios
    df["ma_fast"] = df["close"].rolling(window=5).mean()
    df["ma_slow"] = df["close"].rolling(window=20).mean()
    df["ma_ratio"] = df["ma_fast"] / df["ma_slow"]
    feature_columns.extend(["ma_ratio"])

    # Rolling volatility measures
    df["volatility_20"] = df["log_return"].rolling(window=20).std()
    feature_columns.append("volatility_20")
    for win in feature_cfg.get("rolling_vol_windows", []) or []:
        col = f"volatility_{win}"
        df[col] = df["log_return"].rolling(window=win).std()
        feature_columns.append(col)

    # RSI
    rsi_period = feature_cfg.get("rsi_period", 14)
    df[f"rsi_{rsi_period}"] = compute_rsi(df["close"], period=rsi_period)
    feature_columns.append(f"rsi_{rsi_period}")

    # Volume z-score
    df["volume_mean_20"] = df["volume"].rolling(window=20).mean()
    df["volume_std_20"] = df["volume"].rolling(window=20).std()
    df["volume_z"] = (df["volume"] - df["volume_mean_20"]) / df["volume_std_20"]
    feature_columns.append("volume_z")

    # MACD and EMA ratios
    macd_fast = feature_cfg.get("macd_fast", 12)
    macd_slow = feature_cfg.get("macd_slow", 26)
    macd_signal = feature_cfg.get("macd_signal", 9)
    df[f"ema_{macd_fast}"] = df["close"].ewm(span=macd_fast, adjust=False).mean()
    df[f"ema_{macd_slow}"] = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["ema_ratio"] = df[f"ema_{macd_fast}"] / df[f"ema_{macd_slow}"]
    df["macd"] = df[f"ema_{macd_fast}"] - df[f"ema_{macd_slow}"]
    df["macd_signal"] = df["macd"].ewm(span=macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    feature_columns.extend(["ema_ratio", "macd", "macd_signal", "macd_hist"])

    # Additional EMA spans
    for span in feature_cfg.get("ema_spans", []) or []:
        col = f"ema_span_{span}"
        df[col] = df["close"].ewm(span=span, adjust=False).mean()
        feature_columns.append(col)

    # Bollinger band derived features
    bollinger_window = feature_cfg.get("bollinger_window", 20)
    bollinger_std = feature_cfg.get("bollinger_std", 2.0)
    df["bollinger_mid"] = df["close"].rolling(window=bollinger_window).mean()
    df["bollinger_std"] = df["close"].rolling(window=bollinger_window).std()
    df["bollinger_upper"] = df["bollinger_mid"] + bollinger_std * df["bollinger_std"]
    df["bollinger_lower"] = df["bollinger_mid"] - bollinger_std * df["bollinger_std"]
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / df["bollinger_mid"]
    df["price_position"] = (df["close"] - df["bollinger_mid"]) / df["bollinger_mid"]
    feature_columns.extend(["bollinger_width", "price_position"])

    # ATR
    atr_window = feature_cfg.get("atr_window", 14)
    df[f"atr_{atr_window}"] = compute_atr(df, window=atr_window)
    feature_columns.append(f"atr_{atr_window}")

    # Momentum features
    for window in feature_cfg.get("momentum_windows", []) or []:
        col = f"momentum_{window}"
        df[col] = df["close"].pct_change(window)
        feature_columns.append(col)

    # Lagged returns
    for lag in feature_cfg.get("lag_returns", []) or []:
        col = f"lag_return_{lag}"
        df[col] = df["return_1"].shift(lag)
        feature_columns.append(col)

    threshold = trading_cfg.get("target_return_threshold", 0.001)
    horizon = max(int(trading_cfg.get("prediction_horizon", 1)), 1)
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (df["future_return"] > threshold).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=feature_columns + ["target", "future_return"])
    features = df[feature_columns].copy()

    constant_columns = [col for col in features.columns if features[col].nunique(dropna=False) <= 1]
    if constant_columns:
        logging.debug("Dropping constant feature columns: %s", constant_columns)
        features.drop(columns=constant_columns, inplace=True)

    target = df["target"].copy()

    return DataBundle(features=features, target=target, raw=df)


def train_test_splits(
    data: DataBundle,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        data.features,
        data.target,
        test_size=test_size,
        shuffle=False,
    )


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def walk_forward_splits(
    n_samples: int,
    train_window: int,
    val_window: int,
    max_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if train_window <= 0 or val_window <= 0:
        raise ValueError("train_window and val_window must be positive integers.")

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while True:
        train_end = start + train_window
        val_end = train_end + val_window
        if val_end > n_samples:
            break
        train_idx = np.arange(start, train_end)
        val_idx = np.arange(train_end, val_end)
        splits.append((train_idx, val_idx))
        start += val_window

    if not splits:
        raise ValueError("Not enough samples to create walk-forward splits with the provided windows.")

    return splits[-max_splits:] if max_splits > 0 else splits
