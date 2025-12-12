"""Advanced models for the Rossmann forecasting project.

Implements tuned LightGBM, XGBoost, and CatBoost models with hyperparameter optimization.
"""

import sys
import time
from pathlib import Path
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.cv import remove_missing_features
from evaluation.metrics import rmspe
from utils.log import get_logger

logger = get_logger(__name__)


def tuned_lightgbm_model(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_cols: list[str],
    target_col: str = "Sales",
    params: dict[str, Any] = None,
) -> dict[str, Any]:
    """Tuned LightGBM model with optimized hyperparameters.

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
        LightGBM parameters (uses tuned defaults if None)

    Returns
    -------
    dict
        Results dictionary with fold_scores, mean_score, std_score, models
    """
    logger.info("=" * 60)
    logger.info("Training Tuned LightGBM Model")
    logger.info("=" * 60)
    logger.info(f"Number of features: {len(feature_cols)}")

    # Tuned parameters (these would come from hyperparameter optimization)
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 50,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "max_depth": 8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": 42,
        }

    logger.info(f"Parameters: {params}")

    fold_scores = []
    fold_results = []
    models = []

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
            num_boost_round=2000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
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

        models.append(model)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    logger.info(f"\nMean RMSPE: {mean_score:.6f} ± {std_score:.6f}")
    logger.info("=" * 60)

    results = {
        "model_name": "LightGBM_Tuned",
        "metric": "RMSPE",
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_results": fold_results,
        "params": params,
        "features": valid_features,
        "models": models,
    }

    return results


def xgboost_model(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_cols: list[str],
    target_col: str = "Sales",
    params: dict[str, Any] = None,
) -> dict[str, Any]:
    """XGBoost model with optimized hyperparameters.

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
        XGBoost parameters (uses tuned defaults if None)

    Returns
    -------
    dict
        Results dictionary with fold_scores, mean_score, std_score, models
    """
    logger.info("=" * 60)
    logger.info("Training XGBoost Model")
    logger.info("=" * 60)
    logger.info(f"Number of features: {len(feature_cols)}")

    # Tuned parameters
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
            "verbosity": 0,
        }

    logger.info(f"Parameters: {params}")

    fold_scores = []
    fold_results = []
    models = []

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
        X_train = train_data[valid_features].copy()
        y_train = train_data[target_col]
        X_val = val_data[valid_features].copy()
        y_val = val_data[target_col]

        # XGBoost doesn't handle pandas categoricals - convert to codes
        for col in X_train.columns:
            if X_train[col].dtype.name == "category":
                X_train[col] = X_train[col].cat.codes
                X_val[col] = X_val[col].cat.codes

        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  Train size: {len(X_train):,}")
        logger.info(f"  Val size: {len(X_val):,}")
        logger.info(f"  Features: {len(valid_features)}")

        # Create XGBoost datasets
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Filter out training-specific params that should be separate arguments
        train_params = {
            k: v for k, v in params.items() if k not in ["num_boost_round", "early_stopping_rounds"]
        }

        # Train model
        evals = [(dtrain, "train"), (dval, "valid")]
        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        # Predict
        y_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))

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

        models.append(model)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    logger.info(f"\nMean RMSPE: {mean_score:.6f} ± {std_score:.6f}")
    logger.info("=" * 60)

    results = {
        "model_name": "XGBoost",
        "metric": "RMSPE",
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_results": fold_results,
        "params": params,
        "features": valid_features,
        "models": models,
    }

    return results


def catboost_model(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_cols: list[str],
    target_col: str = "Sales",
    params: dict[str, Any] = None,
    cat_features: list[str] = None,
) -> dict[str, Any]:
    """CatBoost model with optimized hyperparameters.

    CatBoost handles categorical features natively without encoding.

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
        CatBoost parameters (uses tuned defaults if None)
    cat_features : list of str, optional
        List of categorical feature names

    Returns
    -------
    dict
        Results dictionary with fold_scores, mean_score, std_score, models
    """
    logger.info("=" * 60)
    logger.info("Training CatBoost Model")
    logger.info("=" * 60)
    logger.info(f"Number of features: {len(feature_cols)}")

    # Identify categorical features
    if cat_features is None:
        cat_features = [col for col in feature_cols if df[col].dtype in ["object", "category"]]

    logger.info(f"Categorical features: {len(cat_features)}")
    if cat_features:
        logger.info(f"  {cat_features}")

    # Tuned parameters
    if params is None:
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3,
            "random_strength": 0.5,
            "bagging_temperature": 0.2,
            "border_count": 128,
            "verbose": False,
            "random_seed": 42,
        }

    logger.info(f"Parameters: {params}")

    fold_scores = []
    fold_results = []
    models = []

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

        # Update categorical features based on valid features
        valid_cat_features = [col for col in cat_features if col in valid_features]

        # Prepare data
        X_train = train_data[valid_features]
        y_train = train_data[target_col]
        X_val = val_data[valid_features]
        y_val = val_data[target_col]

        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  Train size: {len(X_train):,}")
        logger.info(f"  Val size: {len(X_val):,}")
        logger.info(f"  Features: {len(valid_features)}")

        # Create CatBoost datasets
        train_pool = cb.Pool(X_train, label=y_train, cat_features=valid_cat_features)
        val_pool = cb.Pool(X_val, label=y_val, cat_features=valid_cat_features)

        # Train model
        model = cb.CatBoost(params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

        # Predict
        y_pred = model.predict(val_pool)

        # Calculate RMSPE
        score = rmspe(y_val.values, y_pred)
        fold_scores.append(score)

        train_time = time.time() - start_time

        logger.info(f"  Best iteration: {model.best_iteration_}")
        logger.info(f"  RMSPE: {score:.6f}")
        logger.info(f"  Training time: {train_time:.2f}s")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "score": score,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "best_iteration": model.best_iteration_,
                "train_time": train_time,
            }
        )

        models.append(model)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    logger.info(f"\nMean RMSPE: {mean_score:.6f} ± {std_score:.6f}")
    logger.info("=" * 60)

    results = {
        "model_name": "CatBoost",
        "metric": "RMSPE",
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_results": fold_results,
        "params": params,
        "features": valid_features,
        "cat_features": valid_cat_features,
        "models": models,
    }

    return results


def get_feature_importance(
    models: list[Any], feature_names: list[str], model_type: str = "lightgbm"
) -> pd.DataFrame:
    """Extract and aggregate feature importance across CV folds.

    Parameters
    ----------
    models : list
        List of trained models from CV folds
    feature_names : list of str
        Feature column names
    model_type : str, default='lightgbm'
        Type of model: 'lightgbm', 'xgboost', or 'catboost'

    Returns
    -------
    pd.DataFrame
        Feature importance summary sorted by mean importance
    """
    importances = []

    for model in models:
        if model_type == "lightgbm":
            imp = model.feature_importance(importance_type="gain")
        elif model_type == "xgboost":
            imp = list(model.get_score(importance_type="gain").values())
        elif model_type == "catboost":
            imp = model.get_feature_importance()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        importances.append(imp)

    # Calculate statistics
    importances = np.array(importances)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances.mean(axis=0),
            "importance_std": importances.std(axis=0),
            "importance_min": importances.min(axis=0),
            "importance_max": importances.max(axis=0),
        }
    )

    # Sort by mean importance
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(
        drop=True
    )

    return importance_df


def main():
    """Main function to run advanced model training pipeline."""
    import yaml
    from evaluation.cv import filter_open_stores, make_time_series_folds
    from evaluation.reporting import print_cv_summary, save_cv_results
    from utils.io import read_parquet

    from models.train_baselines import get_feature_columns

    logger.info("=" * 60)
    logger.info("Starting advanced model training pipeline")
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

    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"\nIdentified {len(feature_cols)} feature columns")

    # Train tuned LightGBM
    lgb_results = tuned_lightgbm_model(df, folds, feature_cols)
    print_cv_summary(lgb_results)
    save_cv_results(lgb_results, "lightgbm_tuned", "outputs/metrics/advanced")

    # Train XGBoost
    xgb_results = xgboost_model(df, folds, feature_cols)
    print_cv_summary(xgb_results)
    save_cv_results(xgb_results, "xgboost", "outputs/metrics/advanced")

    # Train CatBoost
    cb_results = catboost_model(df, folds, feature_cols)
    print_cv_summary(cb_results)
    save_cv_results(cb_results, "catboost", "outputs/metrics/advanced")

    logger.info("=" * 60)
    logger.info("Advanced model training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
