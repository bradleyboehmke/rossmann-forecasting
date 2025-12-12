"""Baseline models for the Rossmann forecasting project.

Implements simple baseline models to establish performance benchmarks.
"""

import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.cv import filter_open_stores, make_time_series_folds, remove_missing_features
from evaluation.metrics import rmspe
from evaluation.reporting import print_cv_summary, save_cv_results
from utils.log import get_logger

logger = get_logger(__name__)


def naive_last_week_model(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], lag_col: str = "Sales_Lag_7"
) -> dict[str, Any]:
    """
    Naive baseline: Predict sales using last week's sales (7-day lag).

    This is a simple benchmark that assumes sales patterns repeat weekly.
    Performance gives us a lower bound - any real model should beat this.

    Parameters
    ----------
    df : pd.DataFrame
        Featured dataset with lag columns
    folds : list of tuple
        CV folds from make_time_series_folds()
    lag_col : str, default='Sales_Lag_7'
        Column to use for predictions

    Returns
    -------
    dict
        Results dictionary with fold_scores, mean_score, std_score
    """
    logger.info("=" * 60)
    logger.info("Training Naive Last-Week Model")
    logger.info("=" * 60)
    logger.info(f"Using {lag_col} as prediction")

    fold_scores = []
    fold_results = []

    for fold_idx, (_train_idx, val_idx) in enumerate(folds):
        # Get validation data
        val_data = df.iloc[val_idx].copy()

        # Filter to open stores only
        val_data = val_data[val_data["Open"] == 1]

        # Use lag as prediction
        y_true = val_data["Sales"].values
        y_pred = val_data[lag_col].values

        # Remove any NaN predictions
        mask = ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Calculate RMSPE
        score = rmspe(y_true, y_pred)
        fold_scores.append(score)

        logger.info(f"Fold {fold_idx + 1}: RMSPE = {score:.6f}")

        fold_results.append({"fold": fold_idx + 1, "score": score, "val_size": len(y_true)})

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    logger.info(f"\nMean RMSPE: {mean_score:.6f} ± {std_score:.6f}")
    logger.info("=" * 60)

    results = {
        "model_name": "Naive_LastWeek",
        "metric": "RMSPE",
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_results": fold_results,
    }

    return results


def simple_lightgbm_baseline(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_cols: list[str],
    target_col: str = "Sales",
    params: dict[str, Any] = None,
) -> dict[str, Any]:
    """Simple LightGBM baseline with default parameters.

    Uses basic LightGBM with minimal tuning to establish a reasonable
    machine learning baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Featured dataset
    folds : list of tuple
        CV folds from make_time_series_folds()
    feature_cols : list of str
        Feature column names
    target_col : str, default='Sales'
        Target column name
    params : dict, optional
        LightGBM parameters (uses defaults if None)

    Returns
    -------
    dict
        Results dictionary with fold_scores, mean_score, std_score
    """
    logger.info("=" * 60)
    logger.info("Training Simple LightGBM Baseline")
    logger.info("=" * 60)
    logger.info(f"Number of features: {len(feature_cols)}")

    # Default parameters
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

    logger.info(f"Parameters: {params}")

    fold_scores = []
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        start_time = time.time()

        # Get train and validation data
        train_data = df.iloc[train_idx].copy()
        val_data = df.iloc[val_idx].copy()

        # Filter to open stores only
        train_data = train_data[train_data["Open"] == 1]
        val_data = val_data[val_data["Open"] == 1]

        # Remove rows with missing features
        train_data, valid_features = remove_missing_features(train_data, feature_cols)
        val_data, _ = remove_missing_features(val_data, valid_features)

        # Prepare data
        X_train = train_data[valid_features]
        y_train = train_data[target_col]
        X_val = val_data[valid_features]
        y_val = val_data[target_col]

        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  Train size: {len(X_train):,}")
        logger.info(f"  Val size: {len(X_val):,}")
        logger.info(f"  Features: {len(valid_features)}")

        # Create LightGBM datasets
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Train model
        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predict
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        # Calculate RMSPE
        score = rmspe(y_val.values, y_pred)
        fold_scores.append(score)

        train_time = time.time() - start_time

        logger.info(f"  Best iteration: {model.best_iteration}")
        logger.info(f"  RMSPE: {score:.6f}")
        logger.info(f"  Training time: {train_time:.2f}s")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "score": score,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "best_iteration": model.best_iteration,
                "train_time": train_time,
            }
        )

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    logger.info(f"\nMean RMSPE: {mean_score:.6f} ± {std_score:.6f}")
    logger.info("=" * 60)

    results = {
        "model_name": "LightGBM_Baseline",
        "metric": "RMSPE",
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_results": fold_results,
        "params": params,
        "features": valid_features,
    }

    return results


def get_feature_columns(df: pd.DataFrame, exclude_cols: list[str] = None) -> list[str]:
    """Get list of feature columns, excluding target and metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    exclude_cols : list of str, optional
        Additional columns to exclude

    Returns
    -------
    list of str
        Feature column names
    """
    # Default exclusions
    default_exclude = [
        "Sales",  # Target
        "Customers",  # Not available in test
        "Date",  # Date column
        "Store",  # Store ID (could be used as categorical, but we have store-level features)
    ]

    if exclude_cols:
        default_exclude.extend(exclude_cols)

    # Get all columns except excluded
    feature_cols = [col for col in df.columns if col not in default_exclude]

    # Filter to numeric and categorical columns only
    feature_cols = [
        col
        for col in feature_cols
        if df[col].dtype
        in [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "category",
            "object",
            "bool",
        ]
    ]

    return feature_cols


def main():
    """Main function to run baseline model training pipeline."""
    import yaml
    from utils.io import read_parquet

    logger.info("=" * 60)
    logger.info("Starting baseline model training pipeline")
    logger.info("=" * 60)

    # Load configuration
    config_path = Path("config/params.yaml")
    if config_path.exists():
        with open(config_path) as f:
            params = yaml.safe_load(f)
        cv_config = params.get("cv", {})
    else:
        logger.warning("Config file not found, using defaults")
        cv_config = {}

    # Load featured data
    logger.info("Loading featured data from data/processed/train_features.parquet")
    df = read_parquet("data/processed/train_features.parquet")
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter to open stores
    df = filter_open_stores(df)

    # Create CV folds
    folds = make_time_series_folds(
        df,
        n_folds=cv_config.get("n_folds", 5),
        fold_length_days=cv_config.get("fold_length_days", 42),
        min_train_days=cv_config.get("min_train_days", 365),
    )

    # Train naive baseline
    naive_results = naive_last_week_model(df, folds)
    print_cv_summary(naive_results)
    save_cv_results(naive_results, "naive_lastweek")

    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"\nIdentified {len(feature_cols)} feature columns")

    # Train LightGBM baseline
    lgb_results = simple_lightgbm_baseline(df, folds, feature_cols)
    print_cv_summary(lgb_results)
    save_cv_results(lgb_results, "lightgbm_baseline")

    logger.info("=" * 60)
    logger.info("Baseline model training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
