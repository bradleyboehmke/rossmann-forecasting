"""Production ensemble model training pipeline for Rossmann forecasting.

This module provides the main training workflow for retraining the ensemble model with the latest
data and best hyperparameters from Optuna tuning.
"""

import json
import logging
import sys
import time
from typing import Optional

import catboost as cb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from utils.io import read_parquet
from utils.mlflow_utils import log_dvc_data_version, setup_mlflow

from models.ensemble import create_ensemble
from models.model_registry import register_ensemble_model

logger = logging.getLogger(__name__)


def load_training_data(
    data_path: str = "data/processed/train_features.parquet",
    holdout_days: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split training data into train and holdout sets.

    Parameters
    ----------
    data_path : str
        Path to the processed features parquet file
    holdout_days : int, default=42
        Number of days to use for holdout validation (6 weeks)

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, holdout_df)
    """
    logger.info(f"Loading data from: {data_path}")
    df = read_parquet(data_path)

    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Create train/holdout split
    max_date = df["Date"].max()
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)

    train_df = df[df["Date"] < holdout_start].copy()
    holdout_df = df[df["Date"] >= holdout_start].copy()

    logger.info(f"Train set: {len(train_df):,} rows")
    logger.info(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    logger.info(f"Holdout set: {len(holdout_df):,} rows")
    logger.info(f"  Date range: {holdout_df['Date'].min()} to {holdout_df['Date'].max()}")

    return train_df, holdout_df


def load_best_hyperparameters(
    config_path: str = "config/best_hyperparameters.json",
) -> dict:
    """Load best hyperparameters from Optuna tuning.

    Parameters
    ----------
    config_path : str
        Path to best hyperparameters JSON file

    Returns
    -------
    dict
        Hyperparameters for each model type
    """
    logger.info(f"Loading hyperparameters from: {config_path}")
    with open(config_path) as f:
        best_params = json.load(f)

    logger.info(f"Best model: {best_params['metadata']['best_model']}")
    logger.info(f"Best CV RMSPE: {best_params['metadata']['best_rmspe']:.6f}")

    return best_params


def prepare_training_data(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list, list]:
    """Prepare training data by filtering open stores and extracting features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training data

    Returns
    -------
    tuple
        (X_train, y_train, feature_cols, cat_features)
    """
    # Filter to open stores only
    train_data = train_df[train_df["Open"] == 1].copy()

    # Define feature columns
    exclude_cols = ["Sales", "Date", "Store", "Customers"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Remove rows with missing features
    train_data = train_data.dropna(subset=feature_cols)

    X_train = train_data[feature_cols]
    y_train = train_data["Sales"].values

    # Identify categorical features
    cat_features = [col for col in feature_cols if X_train[col].dtype.name == "category"]

    logger.info(f"Training data: {len(train_data):,} rows (open stores only)")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Categorical features: {len(cat_features)}")

    return X_train, y_train, feature_cols, cat_features


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    best_params: dict,
    cat_features: list,
) -> lgb.Booster:
    """Train LightGBM model with best hyperparameters.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training targets
    best_params : dict
        Best hyperparameters from Optuna
    cat_features : list
        Categorical feature names

    Returns
    -------
    lgb.Booster
        Trained LightGBM model
    """
    logger.info("Training LightGBM...")
    start_time = time.time()

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": int(best_params["lightgbm"]["hyperparameters"]["num_leaves"]),
        "learning_rate": best_params["lightgbm"]["hyperparameters"]["learning_rate"],
        "feature_fraction": best_params["lightgbm"]["hyperparameters"]["feature_fraction"],
        "bagging_fraction": best_params["lightgbm"]["hyperparameters"]["bagging_fraction"],
        "bagging_freq": int(best_params["lightgbm"]["hyperparameters"]["bagging_freq"]),
        "max_depth": int(best_params["lightgbm"]["hyperparameters"]["max_depth"]),
        "min_child_samples": int(best_params["lightgbm"]["hyperparameters"]["min_child_samples"]),
        "reg_alpha": best_params["lightgbm"]["hyperparameters"]["reg_alpha"],
        "reg_lambda": best_params["lightgbm"]["hyperparameters"]["reg_lambda"],
        "verbose": -1,
        "seed": 42,
    }

    # Log hyperparameters to MLflow
    for key, value in lgb_params.items():
        mlflow.log_param(f"lgb_{key}", value)

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=1600, callbacks=[lgb.log_evaluation(period=0)]
    )

    lgb_time = time.time() - start_time
    logger.info(f"LightGBM training complete in {lgb_time:.2f}s")
    mlflow.log_metric("lgb_train_time_seconds", lgb_time)

    return lgb_model


def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, best_params: dict) -> xgb.Booster:
    """Train XGBoost model with best hyperparameters.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training targets
    best_params : dict
        Best hyperparameters from Optuna

    Returns
    -------
    xgb.Booster
        Trained XGBoost model
    """
    logger.info("Training XGBoost...")
    start_time = time.time()

    # XGBoost needs categorical features as codes
    X_train_xgb = X_train.copy()
    for col in X_train_xgb.columns:
        if X_train_xgb[col].dtype.name == "category":
            X_train_xgb[col] = X_train_xgb[col].cat.codes

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(best_params["xgboost"]["hyperparameters"]["max_depth"]),
        "learning_rate": best_params["xgboost"]["hyperparameters"]["learning_rate"],
        "subsample": best_params["xgboost"]["hyperparameters"]["subsample"],
        "colsample_bytree": best_params["xgboost"]["hyperparameters"]["colsample_bytree"],
        "min_child_weight": int(best_params["xgboost"]["hyperparameters"]["min_child_weight"]),
        "reg_alpha": best_params["xgboost"]["hyperparameters"]["reg_alpha"],
        "reg_lambda": best_params["xgboost"]["hyperparameters"]["reg_lambda"],
        "gamma": best_params["xgboost"]["hyperparameters"]["gamma"],
        "seed": 42,
        "verbosity": 0,
    }

    # Log hyperparameters to MLflow
    for key, value in xgb_params.items():
        mlflow.log_param(f"xgb_{key}", value)

    dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1600, verbose_eval=False)

    xgb_time = time.time() - start_time
    logger.info(f"XGBoost training complete in {xgb_time:.2f}s")
    mlflow.log_metric("xgb_train_time_seconds", xgb_time)

    return xgb_model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    best_params: dict,
    cat_features: list,
) -> cb.CatBoost:
    """Train CatBoost model with best hyperparameters.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training targets
    best_params : dict
        Best hyperparameters from Optuna
    cat_features : list
        Categorical feature names

    Returns
    -------
    cb.CatBoost
        Trained CatBoost model
    """
    logger.info("Training CatBoost...")
    start_time = time.time()

    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "depth": int(best_params["catboost"]["hyperparameters"]["depth"]),
        "learning_rate": best_params["catboost"]["hyperparameters"]["learning_rate"],
        "l2_leaf_reg": best_params["catboost"]["hyperparameters"]["l2_leaf_reg"],
        "random_strength": best_params["catboost"]["hyperparameters"]["random_strength"],
        "bagging_temperature": best_params["catboost"]["hyperparameters"]["bagging_temperature"],
        "border_count": int(best_params["catboost"]["hyperparameters"]["border_count"]),
        "iterations": 1500,
        "verbose": False,
        "random_seed": 42,
    }

    # Log hyperparameters to MLflow
    for key, value in cb_params.items():
        mlflow.log_param(f"cb_{key}", value)

    train_pool = cb.Pool(X_train, label=y_train, cat_features=cat_features)
    cb_model = cb.CatBoost(cb_params)
    cb_model.fit(train_pool)

    cb_time = time.time() - start_time
    logger.info(f"CatBoost training complete in {cb_time:.2f}s")
    mlflow.log_metric("cb_train_time_seconds", cb_time)

    return cb_model


def main(
    data_path: str = "data/processed/train_features.parquet",
    config_path: str = "config/best_hyperparameters.json",
    model_name: str = "rossmann-ensemble",
    ensemble_weights: Optional[dict[str, float]] = None,
    run_name: str = "production_ensemble_training",
):
    """Main training pipeline for ensemble model.

    Parameters
    ----------
    data_path : str
        Path to processed features
    config_path : str
        Path to best hyperparameters JSON
    model_name : str
        Name for registered model in MLflow
    ensemble_weights : dict, optional
        Ensemble weights. Default: {'lightgbm': 0.3, 'xgboost': 0.6, 'catboost': 0.1}
    run_name : str
        Name for MLflow run

    Returns
    -------
    str
        Model version number
    """
    # Default ensemble weights
    if ensemble_weights is None:
        ensemble_weights = {"lightgbm": 0.30, "xgboost": 0.60, "catboost": 0.10}

    logger.info("=" * 70)
    logger.info("PRODUCTION ENSEMBLE MODEL TRAINING")
    logger.info("=" * 70)

    # Setup MLflow
    experiment_id = setup_mlflow()
    logger.info(f"MLflow experiment ID: {experiment_id}")

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log DVC data version
        log_dvc_data_version(data_path)

        # Load data
        train_df, holdout_df = load_training_data(data_path)

        # Load best hyperparameters
        best_params = load_best_hyperparameters(config_path)

        # Prepare training data
        X_train, y_train, feature_cols, cat_features = prepare_training_data(train_df)

        # Log dataset info
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("holdout_size", len(holdout_df))
        mlflow.log_param("n_features", len(feature_cols))

        # Log ensemble weights
        for model_type, weight in ensemble_weights.items():
            mlflow.log_param(f"{model_type}_weight", weight)

        # Train individual models
        logger.info("\nTraining individual models...")
        lgb_model = train_lightgbm(X_train, y_train, best_params, cat_features)
        xgb_model = train_xgboost(X_train, y_train, best_params)
        cb_model = train_catboost(X_train, y_train, best_params, cat_features)

        logger.info("✓ All models trained successfully!")

        # Create ensemble
        logger.info("\nCreating ensemble model...")
        ensemble = create_ensemble(
            lgb_model=lgb_model,
            xgb_model=xgb_model,
            cb_model=cb_model,
            weights=ensemble_weights,
            cat_features=cat_features,
        )

        # Prepare holdout data for input example
        from evaluation.cv import remove_missing_features

        holdout_data = holdout_df[holdout_df["Open"] == 1].copy()
        holdout_data, _ = remove_missing_features(holdout_data, feature_cols)
        X_holdout = holdout_data[feature_cols]

        # Note: We don't create a signature here due to categorical dtype issues
        # MLflow will infer it automatically when the model is saved

        # Define conda environment
        conda_env = {
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                f"python={sys.version_info.major}.{sys.version_info.minor}",
                "pip",
                {
                    "pip": [
                        f"lightgbm=={lgb.__version__}",
                        f"xgboost=={xgb.__version__}",
                        f"catboost=={cb.__version__}",
                        "pandas",
                        "numpy",
                        "scikit-learn",
                    ]
                },
            ],
            "name": "rossmann_ensemble_env",
        }

        # Register ensemble model
        logger.info(f"\nRegistering ensemble model: {model_name}")
        version = register_ensemble_model(
            ensemble_model=ensemble,
            model_name=model_name,
            conda_env=conda_env,
            signature=None,  # Let MLflow infer automatically
            input_example=X_holdout.head(5),
            description=f"Rossmann ensemble (weights: {ensemble_weights})",
        )

        logger.info("=" * 70)
        logger.info(f"✓ Training complete! Model version: {version}")
        logger.info("=" * 70)

        return version


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    version = main()
    print(f"\n✓ Successfully trained and registered model version: {version}")
    print("\nNext steps:")
    print("  1. Validate model: python src/models/validate_model.py")
    print("  2. Promote to Staging if passing validation")
    print("  3. Test inference: python src/models/predict.py")
