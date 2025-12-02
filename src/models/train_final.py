"""
Final model training for the Rossmann forecasting project.

Trains the final model on the full historical dataset and generates predictions
for a 6-week holdout period (simulating the test set).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import time
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import rmspe
from evaluation.cv import remove_missing_features
from utils.log import get_logger
from utils.io import ensure_dir

logger = get_logger(__name__)


def create_holdout_split(
    df: pd.DataFrame,
    holdout_days: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a train/holdout split based on the last N days.

    The holdout period simulates the Kaggle test set (6 weeks).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with Date column
    holdout_days : int, default=42
        Number of days for holdout (6 weeks)

    Returns
    -------
    train_df : pd.DataFrame
        Training data (all data before holdout period)
    holdout_df : pd.DataFrame
        Holdout data (last N days)
    """
    logger.info("="*60)
    logger.info("Creating Train/Holdout Split")
    logger.info("="*60)

    # Get max date
    max_date = df['Date'].max()

    # Calculate holdout start date
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)

    # Split data
    train_df = df[df['Date'] < holdout_start].copy()
    holdout_df = df[df['Date'] >= holdout_start].copy()

    logger.info(f"Full dataset: {len(df):,} rows")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"\nTrain set:")
    logger.info(f"  Rows: {len(train_df):,}")
    logger.info(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    logger.info(f"\nHoldout set:")
    logger.info(f"  Rows: {len(holdout_df):,}")
    logger.info(f"  Date range: {holdout_df['Date'].min()} to {holdout_df['Date'].max()}")
    logger.info(f"  Duration: {holdout_days} days ({holdout_days // 7} weeks)")
    logger.info("="*60)

    return train_df, holdout_df


def train_final_xgboost(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Sales',
    params: Dict[str, Any] = None
) -> xgb.Booster:
    """
    Train final XGBoost model on full training set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    feature_cols : list of str
        Feature column names
    target_col : str, default='Sales'
        Target column name
    params : dict, optional
        XGBoost parameters

    Returns
    -------
    model : xgb.Booster
        Trained XGBoost model
    """
    logger.info("="*60)
    logger.info("Training Final XGBoost Model")
    logger.info("="*60)

    # Filter to open stores
    train_df = train_df[train_df['Open'] == 1].copy()

    # Remove rows with missing features
    train_df, valid_features = remove_missing_features(train_df, feature_cols)

    logger.info(f"Training set size: {len(train_df):,}")
    logger.info(f"Features: {len(valid_features)}")

    # Prepare data
    X_train = train_df[valid_features].copy()
    y_train = train_df[target_col]

    # XGBoost doesn't handle pandas categoricals - convert to codes
    for col in X_train.columns:
        if X_train[col].dtype.name == 'category':
            X_train[col] = X_train[col].cat.codes

    # Default parameters from Phase 4 tuning
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
            'verbosity': 0
        }

    logger.info(f"Parameters: {params}")

    # Create XGBoost dataset
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Filter out training-specific params that should be separate arguments
    train_params = {k: v for k, v in params.items()
                   if k not in ['num_boost_round', 'early_stopping_rounds']}

    # Train model (use average best iteration from CV: ~1500)
    start_time = time.time()
    model = xgb.train(
        train_params,
        dtrain,
        num_boost_round=1600,  # Slightly more than CV average
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )

    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time:.2f}s")
    logger.info("="*60)

    return model, valid_features


def train_final_ensemble(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Sales',
    weights: Dict[str, float] = None,
    use_tuned_params: bool = True
) -> Dict[str, Any]:
    """
    Train final ensemble model (LightGBM + XGBoost + CatBoost weighted blend).

    Based on Optuna tuning results, the optimal ensemble uses:
    - ~50% XGBoost (best: 0.121780)
    - ~30% LightGBM (0.126174)
    - ~20% CatBoost (0.129670)

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    feature_cols : list of str
        Feature column names
    target_col : str, default='Sales'
        Target column name
    weights : dict, optional
        Model weights (defaults to optimized weights from tuning results)
    use_tuned_params : bool, default=True
        Use Optuna-tuned hyperparameters from outputs/tuning/best_hyperparameters.json

    Returns
    -------
    ensemble : dict
        Dictionary containing trained models and metadata
    """
    logger.info("="*60)
    logger.info("Training Final Ensemble Model (3-Model Blend)")
    logger.info("="*60)

    # Default weights optimized from Optuna tuning results
    # Based on inverse RMSPE weighting: XGBoost (best) gets highest weight
    if weights is None:
        weights = {
            'lightgbm': 0.30,
            'xgboost': 0.50,
            'catboost': 0.20
        }

    logger.info(f"Ensemble weights: {weights}")
    logger.info(f"Using tuned hyperparameters: {use_tuned_params}")

    # Load tuned hyperparameters if requested
    tuned_params = None
    if use_tuned_params:
        tuned_params_path = Path('outputs/tuning/best_hyperparameters.json')
        if tuned_params_path.exists():
            with open(tuned_params_path, 'r') as f:
                tuned_params = json.load(f)
            logger.info(f"Loaded tuned hyperparameters from {tuned_params_path}")
        else:
            logger.warning(f"Tuned parameters file not found: {tuned_params_path}")
            logger.warning("Using default hyperparameters instead")
            use_tuned_params = False

    # Filter to open stores
    train_df = train_df[train_df['Open'] == 1].copy()

    # Remove rows with missing features
    train_df, valid_features = remove_missing_features(train_df, feature_cols)

    # Identify categorical features
    cat_features = [
        col for col in valid_features
        if train_df[col].dtype in ['object', 'category']
    ]

    logger.info(f"Training set size: {len(train_df):,}")
    logger.info(f"Features: {len(valid_features)}")
    logger.info(f"Categorical features: {len(cat_features)}")

    # Prepare common training data
    y_train = train_df[target_col]

    # Train LightGBM
    logger.info("\nTraining LightGBM component...")
    X_train_lgb = train_df[valid_features]

    if use_tuned_params and tuned_params:
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': tuned_params['lightgbm']['num_leaves'],
            'learning_rate': tuned_params['lightgbm']['learning_rate'],
            'feature_fraction': tuned_params['lightgbm']['feature_fraction'],
            'bagging_fraction': tuned_params['lightgbm']['bagging_fraction'],
            'bagging_freq': tuned_params['lightgbm']['bagging_freq'],
            'max_depth': tuned_params['lightgbm']['max_depth'],
            'min_child_samples': tuned_params['lightgbm']['min_child_samples'],
            'reg_alpha': tuned_params['lightgbm']['reg_alpha'],
            'reg_lambda': tuned_params['lightgbm']['reg_lambda'],
            'verbose': -1,
            'seed': 42
        }
    else:
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'seed': 42
        }

    lgb_train = lgb.Dataset(X_train_lgb, label=y_train, categorical_feature=cat_features)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1600,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    logger.info("LightGBM training complete")

    # Train XGBoost
    logger.info("\nTraining XGBoost component...")
    X_train_xgb = train_df[valid_features].copy()

    for col in X_train_xgb.columns:
        if X_train_xgb[col].dtype.name == 'category':
            X_train_xgb[col] = X_train_xgb[col].cat.codes

    dtrain = xgb.DMatrix(X_train_xgb, label=y_train)

    if use_tuned_params and tuned_params:
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': tuned_params['xgboost']['max_depth'],
            'learning_rate': tuned_params['xgboost']['learning_rate'],
            'subsample': tuned_params['xgboost']['subsample'],
            'colsample_bytree': tuned_params['xgboost']['colsample_bytree'],
            'min_child_weight': tuned_params['xgboost']['min_child_weight'],
            'reg_alpha': tuned_params['xgboost']['reg_alpha'],
            'reg_lambda': tuned_params['xgboost']['reg_lambda'],
            'gamma': tuned_params['xgboost']['gamma'],
            'seed': 42,
            'verbosity': 0
        }
    else:
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
            'verbosity': 0
        }

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1600,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    logger.info("XGBoost training complete")

    # Train CatBoost
    logger.info("\nTraining CatBoost component...")
    X_train_cb = train_df[valid_features]

    train_pool = cb.Pool(X_train_cb, label=y_train, cat_features=cat_features)

    if use_tuned_params and tuned_params:
        cb_params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'depth': tuned_params['catboost']['depth'],
            'learning_rate': tuned_params['catboost']['learning_rate'],
            'l2_leaf_reg': tuned_params['catboost']['l2_leaf_reg'],
            'random_strength': tuned_params['catboost']['random_strength'],
            'bagging_temperature': tuned_params['catboost']['bagging_temperature'],
            'border_count': tuned_params['catboost']['border_count'],
            'iterations': 1500,
            'verbose': False,
            'random_seed': 42
        }
    else:
        cb_params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'depth': 8,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3,
            'random_strength': 0.5,
            'bagging_temperature': 0.2,
            'border_count': 128,
            'iterations': 1500,
            'verbose': False,
            'random_seed': 42
        }

    cb_model = cb.CatBoost(cb_params)
    cb_model.fit(train_pool)
    logger.info("CatBoost training complete")

    logger.info("="*60)

    ensemble = {
        'models': {
            'lightgbm': lgb_model,
            'xgboost': xgb_model,
            'catboost': cb_model
        },
        'weights': weights,
        'features': valid_features,
        'cat_features': cat_features
    }

    return ensemble


def predict_with_ensemble(
    ensemble: Dict[str, Any],
    test_df: pd.DataFrame
) -> np.ndarray:
    """
    Generate predictions using the ensemble model.

    Parameters
    ----------
    ensemble : dict
        Ensemble dictionary from train_final_ensemble()
    test_df : pd.DataFrame
        Test/holdout data

    Returns
    -------
    predictions : np.ndarray
        Blended predictions
    """
    # Filter to open stores
    test_df = test_df[test_df['Open'] == 1].copy()

    # Get valid features
    valid_features = ensemble['features']
    test_df, _ = remove_missing_features(test_df, valid_features)

    # LightGBM predictions
    X_test_lgb = test_df[valid_features]
    lgb_preds = ensemble['models']['lightgbm'].predict(X_test_lgb, num_iteration=ensemble['models']['lightgbm'].best_iteration)

    # XGBoost predictions
    X_test_xgb = test_df[valid_features].copy()
    for col in X_test_xgb.columns:
        if X_test_xgb[col].dtype.name == 'category':
            X_test_xgb[col] = X_test_xgb[col].cat.codes

    dtest = xgb.DMatrix(X_test_xgb)
    xgb_preds = ensemble['models']['xgboost'].predict(dtest)

    # CatBoost predictions
    X_test_cb = test_df[valid_features]
    test_pool = cb.Pool(X_test_cb, cat_features=ensemble['cat_features'])
    cb_preds = ensemble['models']['catboost'].predict(test_pool)

    # Weighted blend
    weights = ensemble['weights']
    blended_preds = (
        weights['lightgbm'] * lgb_preds +
        weights['xgboost'] * xgb_preds +
        weights['catboost'] * cb_preds
    )

    return blended_preds, test_df


def evaluate_final_model(
    predictions: np.ndarray,
    actuals: np.ndarray,
    holdout_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluate final model performance on holdout set.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    actuals : np.ndarray
        Actual sales values
    holdout_df : pd.DataFrame
        Holdout dataframe (for additional metrics)

    Returns
    -------
    metrics : dict
        Performance metrics
    """
    logger.info("="*60)
    logger.info("Evaluating Final Model Performance")
    logger.info("="*60)

    # Calculate RMSPE
    score = rmspe(actuals, predictions)

    # Calculate additional metrics
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    logger.info(f"Holdout Performance:")
    logger.info(f"  RMSPE: {score:.6f}")
    logger.info(f"  RMSE:  {rmse:.2f}")
    logger.info(f"  MAE:   {mae:.2f}")
    logger.info(f"  MAPE:  {mape:.2f}%")

    # Target comparison
    target_rmspe = 0.09856
    gap = score - target_rmspe
    gap_pct = (gap / target_rmspe) * 100

    logger.info(f"\nTarget Analysis:")
    logger.info(f"  Target RMSPE: {target_rmspe:.6f}")
    logger.info(f"  Current RMSPE: {score:.6f}")
    logger.info(f"  Gap: {gap:.6f} ({gap_pct:+.2f}%)")

    if score <= target_rmspe:
        logger.info(f"\n🎯 TARGET ACHIEVED!")
    else:
        logger.info(f"\n📊 Additional tuning needed to reach target")

    logger.info("="*60)

    metrics = {
        'rmspe': float(score),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'target_rmspe': target_rmspe,
        'gap': float(gap),
        'gap_pct': float(gap_pct),
        'target_achieved': bool(score <= target_rmspe),
        'n_predictions': len(predictions)
    }

    return metrics


def save_final_model(
    ensemble: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str = 'models/final'
):
    """
    Save final trained model and metadata.

    Parameters
    ----------
    ensemble : dict
        Ensemble dictionary with models
    metrics : dict
        Performance metrics
    output_dir : str
        Output directory for model artifacts
    """
    output_path = Path(output_dir)
    ensure_dir(output_path)

    logger.info("="*60)
    logger.info("Saving Final Model")
    logger.info("="*60)

    # Save LightGBM model
    lgb_path = output_path / 'lightgbm_final.txt'
    ensemble['models']['lightgbm'].save_model(str(lgb_path))
    logger.info(f"Saved LightGBM model to {lgb_path}")

    # Save XGBoost model
    xgb_path = output_path / 'xgboost_final.json'
    ensemble['models']['xgboost'].save_model(str(xgb_path))
    logger.info(f"Saved XGBoost model to {xgb_path}")

    # Save CatBoost model
    cb_path = output_path / 'catboost_final.cbm'
    ensemble['models']['catboost'].save_model(str(cb_path))
    logger.info(f"Saved CatBoost model to {cb_path}")

    # Save metadata
    metadata = {
        'weights': ensemble['weights'],
        'features': ensemble['features'],
        'cat_features': ensemble['cat_features'],
        'metrics': metrics,
        'model_type': 'ensemble',
        'components': ['lightgbm', 'xgboost', 'catboost']
    }

    metadata_path = output_path / 'ensemble_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info("="*60)


def save_predictions(
    predictions: np.ndarray,
    holdout_df: pd.DataFrame,
    output_path: str = 'outputs/predictions/final_holdout_predictions.csv'
):
    """
    Save predictions to CSV file.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    holdout_df : pd.DataFrame
        Holdout dataframe with Store, Date, Sales
    output_path : str
        Output file path
    """
    ensure_dir(Path(output_path).parent)

    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'Store': holdout_df['Store'].values,
        'Date': holdout_df['Date'].values,
        'Sales_Actual': holdout_df['Sales'].values,
        'Sales_Predicted': predictions
    })

    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def main():
    """
    Main function to train final model and evaluate on holdout set.
    """
    import yaml
    from utils.io import read_parquet
    from models.train_baselines import get_feature_columns

    logger.info("="*60)
    logger.info("PHASE 5: FINAL MODEL TRAINING & EVALUATION")
    logger.info("="*60)

    # Load configuration
    config_path = Path('config/params.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        logger.warning("Config file not found, using defaults")
        params = {}

    # Load featured data
    logger.info("Loading featured data...")
    df = read_parquet('data/processed/train_features.parquet')
    logger.info(f"Loaded {len(df):,} rows")

    # Create train/holdout split (6 weeks holdout)
    train_df, holdout_df = create_holdout_split(df, holdout_days=42)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"\nFeatures: {len(feature_cols)}")

    # Train final ensemble model
    ensemble = train_final_ensemble(train_df, feature_cols)

    # Generate predictions on holdout set
    logger.info("\nGenerating holdout predictions...")
    predictions, holdout_df_filtered = predict_with_ensemble(ensemble, holdout_df)
    actuals = holdout_df_filtered['Sales'].values

    # Evaluate performance
    metrics = evaluate_final_model(predictions, actuals, holdout_df_filtered)

    # Save model and predictions
    save_final_model(ensemble, metrics)
    save_predictions(predictions, holdout_df_filtered)

    logger.info("\n" + "="*60)
    logger.info("PHASE 5 COMPLETE!")
    logger.info("="*60)
    logger.info(f"Final RMSPE: {metrics['rmspe']:.6f}")
    logger.info(f"Gap to target: {metrics['gap_pct']:+.2f}%")
    logger.info("="*60)


if __name__ == "__main__":
    main()
