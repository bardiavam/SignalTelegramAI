import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

from utils import (
    engineer_features,
    ensure_parent_dir,
    fetch_historical_data,
    get_binance_client,
    load_config,
    model_path_for_interval,
    resolve_interval,
    setup_logging,
    walk_forward_splits,
)

METRIC_FUNCTIONS = {
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
}


def build_model(model_type: str, params: Dict) -> object:
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight"),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed. Install it or choose another model_type.")
        return lgb.LGBMClassifier(**params)
    if model_type == "logistic_regression":
        return LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
            max_iter=params.get("max_iter", 200),
            class_weight=params.get("class_weight"),
        )
    raise ValueError(f"Unsupported model_type '{model_type}'.")


def evaluate_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float],
    metric_name: str,
) -> Tuple[float, Dict[str, float]]:
    metric_func = METRIC_FUNCTIONS.get(metric_name.lower())
    if metric_func is None:
        raise ValueError(f"Unsupported metric '{metric_name}'. Choose from {list(METRIC_FUNCTIONS)}.")

    best_threshold = 0.5
    best_metric = -np.inf
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for thr in thresholds:
        preds = (proba > thr).astype(int)
        metric_value = metric_func(y_true, preds)
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = float(thr)
            best_precision = precision_score(y_true, preds, zero_division=0)
            best_recall = recall_score(y_true, preds, zero_division=0)
            best_f1 = f1_score(y_true, preds, zero_division=0)

    return best_threshold, {
        "metric": best_metric,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


def run_training(
    config_path: str,
    override_model_type: str | None = None,
    override_interval: str | None = None,
) -> None:
    config = load_config(config_path)
    setup_logging(config)
    logger = logging.getLogger("train")

    trading_cfg = config.get("trading", {})
    training_cfg = config.get("training", {})
    if override_model_type:
        training_cfg = dict(training_cfg)
        training_cfg["model_type"] = override_model_type
        config["training"] = training_cfg

    interval = resolve_interval(config, override_interval)
    config.setdefault("trading", {})["interval"] = interval
    logger.info("Using interval %s for training.", interval)

    model_type = training_cfg.get("model_type", "random_forest")
    model_params = (config.get("model_params") or {}).get(model_type, {})

    lookback_days = trading_cfg.get("lookback_days", 365)
    client = get_binance_client(config)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    logger.info(
        "Fetching historical data for %s from %s to %s",
        trading_cfg.get("symbol"),
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    df = fetch_historical_data(
        client=client,
        config=config,
        start_str=start.strftime("%Y-%m-%d %H:%M:%S"),
        interval=interval,
    )

    data = engineer_features(df, config)
    X = data.features
    y = data.target

    min_samples = training_cfg.get("min_training_samples", 500)
    if len(X) < min_samples:
        raise RuntimeError(
            f"Insufficient samples ({len(X)}) for training. "
            f"Increase lookback_days or adjust feature engineering."
        )

    train_window = training_cfg.get("walk_forward_train_candles")
    val_window = training_cfg.get("walk_forward_val_candles")
    n_splits = training_cfg.get("walk_forward_splits", 0)
    use_walk_forward = all(
        [
            train_window,
            val_window,
            n_splits,
            train_window > 0,
            val_window > 0,
            n_splits > 0,
        ]
    )

    all_val_true: List[np.ndarray] = []
    all_val_proba: List[np.ndarray] = []
    split_reports: List[str] = []
    auc_scores: List[float] = []

    if use_walk_forward:
        try:
            splits = walk_forward_splits(len(X), train_window, val_window, n_splits)
        except ValueError as exc:
            logger.warning("Walk-forward setup invalid (%s). Falling back to hold-out split.", exc)
            use_walk_forward = False
        else:
            logger.info("Performing walk-forward validation with %d splits.", len(splits))
            for idx, (train_idx, val_idx) in enumerate(splits, start=1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = build_model(model_type, model_params)
                model.fit(X_train, y_train)

                val_proba = model.predict_proba(X_val)[:, 1]
                val_pred = (val_proba > 0.5).astype(int)
                auc = roc_auc_score(y_val, val_proba)
                auc_scores.append(auc)

                report = classification_report(y_val, val_pred, zero_division=0)
                split_reports.append(f"Split {idx}:\n{report}")
                logger.info("Split %d ROC AUC: %.4f", idx, auc)

                all_val_true.append(y_val.to_numpy())
                all_val_proba.append(val_proba)

    if not use_walk_forward:
        logger.info("Using simple hold-out validation.")
        holdout_fraction = training_cfg.get("test_size", 0.2)
        split_point = int(len(X) * (1 - holdout_fraction))
        if split_point <= 0 or split_point >= len(X):
            raise ValueError("Invalid hold-out split. Adjust training.test_size.")
        splits = [(np.arange(0, split_point), np.arange(split_point, len(X)))]

        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
        model = build_model(model_type, model_params)
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba > 0.5).astype(int)
        auc = roc_auc_score(y_val, val_proba)
        auc_scores.append(auc)
        report = classification_report(y_val, val_pred, zero_division=0)
        split_reports.append(f"Hold-out:\n{report}")
        all_val_true.append(y_val.to_numpy())
        all_val_proba.append(val_proba)

    for rep in split_reports:
        logger.info("Validation report:\n%s", rep)

    val_true = np.concatenate(all_val_true)
    val_proba = np.concatenate(all_val_proba)
    mean_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")
    logger.info("Mean validation ROC AUC across splits: %.4f", mean_auc)

    thresholds = np.arange(
        training_cfg.get("threshold_min", 0.5),
        training_cfg.get("threshold_max", 0.5) + training_cfg.get("threshold_step", 0.01) / 2,
        training_cfg.get("threshold_step", 0.01),
    )
    best_threshold, metric_details = evaluate_thresholds(
        val_true,
        val_proba,
        thresholds,
        metric_name=training_cfg.get("metric", "f1"),
    )

    logger.info(
        "Selected probability threshold %.3f optimizing %s (value=%.3f)",
        best_threshold,
        training_cfg.get("metric", "f1"),
        metric_details["metric"],
    )
    logger.info(
        "Validation metrics at threshold %.3f -> Precision: %.3f Recall: %.3f F1: %.3f",
        best_threshold,
        metric_details["precision"],
        metric_details["recall"],
        metric_details["f1"],
    )

    # Train final model on full dataset
    final_model = build_model(model_type, model_params)
    final_model.fit(X, y)
    model_path = model_path_for_interval(config, interval)
    ensure_parent_dir(model_path)

    artifact = {
        "model": final_model,
        "model_type": model_type,
        "feature_columns": list(X.columns),
        "config": config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "interval": interval,
        "metrics": {
            "roc_auc": mean_auc,
            "precision": metric_details["precision"],
            "recall": metric_details["recall"],
            "f1": metric_details["f1"],
        },
        "best_threshold": best_threshold,
        "validation_reports": split_reports,
    }

    joblib.dump(artifact, model_path)
    logger.info("Model saved to %s", model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model for gold trading signals.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--model-type", help="Override model_type defined in config (e.g., lightgbm).")
    parser.add_argument("--interval", help="Override interval defined in config (e.g., 15m, 1h).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config, override_model_type=args.model_type, override_interval=args.interval)
