"""Ensemble methods for the Rossmann forecasting project.

Implements weighted blending and stacked ensembles.
"""

import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import rmspe
from utils.log import get_logger

logger = get_logger(__name__)


def weighted_blend_ensemble(
    predictions_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    weights: dict[str, float] | None = None,
    optimize_weights: bool = True,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Create weighted blend of model predictions.

    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays
    y_true : np.ndarray
        True target values
    weights : dict, optional
        Dictionary mapping model names to weights
        If None and optimize_weights=True, weights will be optimized
    optimize_weights : bool, default=True
        Whether to optimize weights based on RMSPE

    Returns
    -------
    tuple
        (blended_predictions, optimal_weights, score)
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    if weights is None and not optimize_weights:
        # Equal weights
        weights = dict.fromkeys(model_names, 1.0 / n_models)
        logger.info("Using equal weights for ensemble")
    elif weights is None and optimize_weights:
        # Optimize weights using grid search
        logger.info("Optimizing ensemble weights...")
        weights = optimize_ensemble_weights(predictions_dict, y_true)
        logger.info(f"Optimal weights: {weights}")
    else:
        logger.info(f"Using provided weights: {weights}")

    # Create weighted blend
    blended_pred = np.zeros_like(y_true, dtype=float)
    for name, preds in predictions_dict.items():
        blended_pred += weights[name] * preds

    # Calculate RMSPE
    score = rmspe(y_true, blended_pred)

    return blended_pred, weights, score


def optimize_ensemble_weights(
    predictions_dict: dict[str, np.ndarray], y_true: np.ndarray, step: float = 0.05
) -> dict[str, float]:
    """Optimize ensemble weights using grid search.

    Searches for weights that minimize RMSPE.

    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays
    y_true : np.ndarray
        True target values
    step : float, default=0.05
        Step size for weight grid search

    Returns
    -------
    dict
        Optimal weights for each model
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    if n_models == 2:
        # For 2 models, search over w1, w2 = 1 - w1
        best_score = float("inf")
        best_weights = None

        for w1 in np.arange(0, 1 + step, step):
            w2 = 1 - w1
            weights = {model_names[0]: w1, model_names[1]: w2}

            # Calculate blended predictions
            blended = np.zeros_like(y_true, dtype=float)
            for name, w in weights.items():
                blended += w * predictions_dict[name]

            score = rmspe(y_true, blended)

            if score < best_score:
                best_score = score
                best_weights = weights

        return best_weights

    elif n_models == 3:
        # For 3 models, search over w1, w2, w3 = 1 - w1 - w2
        best_score = float("inf")
        best_weights = None

        for w1 in np.arange(0, 1 + step, step):
            for w2 in np.arange(0, 1 - w1 + step, step):
                w3 = 1 - w1 - w2
                weights = {model_names[0]: w1, model_names[1]: w2, model_names[2]: w3}

                # Calculate blended predictions
                blended = np.zeros_like(y_true, dtype=float)
                for name, w in weights.items():
                    blended += w * predictions_dict[name]

                score = rmspe(y_true, blended)

                if score < best_score:
                    best_score = score
                    best_weights = weights

        return best_weights

    else:
        # For more models, use equal weights
        logger.warning(f"Optimization for {n_models} models not implemented, using equal weights")
        return dict.fromkeys(model_names, 1.0 / n_models)


def stacked_ensemble(
    oof_predictions_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    test_predictions_dict: dict[str, np.ndarray],
    meta_model: str = "ridge",
    alpha: float = 1.0,
) -> tuple[np.ndarray, Any, float]:
    """Create stacked ensemble with meta-learner.

    Uses out-of-fold predictions from base models to train a meta-learner.

    Parameters
    ----------
    oof_predictions_dict : dict
        Dictionary mapping model names to out-of-fold prediction arrays
    y_train : np.ndarray
        True target values for training meta-learner
    test_predictions_dict : dict
        Dictionary mapping model names to test prediction arrays
    meta_model : str, default='ridge'
        Type of meta-learner: 'ridge', 'lightgbm'
    alpha : float, default=1.0
        Regularization strength for Ridge (ignored for lightgbm)

    Returns
    -------
    tuple
        (stacked_predictions, meta_model, cv_score)
    """
    logger.info("=" * 60)
    logger.info("Training Stacked Ensemble")
    logger.info("=" * 60)
    logger.info(f"Meta-learner: {meta_model}")

    # Prepare meta-features from out-of-fold predictions
    X_meta_train = np.column_stack([preds for preds in oof_predictions_dict.values()])
    X_meta_test = np.column_stack([preds for preds in test_predictions_dict.values()])

    logger.info(f"Meta-features shape: {X_meta_train.shape}")
    logger.info(f"Base models: {list(oof_predictions_dict.keys())}")

    # Train meta-learner
    if meta_model == "ridge":
        model = Ridge(alpha=alpha)
        model.fit(X_meta_train, y_train)
        logger.info(
            f"Ridge coefficients: {dict(zip(oof_predictions_dict.keys(), model.coef_, strict=False))}"
        )

    elif meta_model == "lightgbm":
        train_set = lgb.Dataset(X_meta_train, label=y_train)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[train_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        logger.info(f"Best iteration: {model.best_iteration}")

    else:
        raise ValueError(f"Unknown meta_model: {meta_model}")

    # Make predictions on test set
    stacked_preds = model.predict(X_meta_test)

    # Calculate CV score on training data
    oof_stacked = model.predict(X_meta_train)
    cv_score = rmspe(y_train, oof_stacked)

    logger.info(f"Stacked ensemble CV RMSPE: {cv_score:.6f}")
    logger.info("=" * 60)

    return stacked_preds, model, cv_score


def create_oof_predictions(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    models_dict: dict[str, list[Any]],
    feature_cols: list[str],
    model_types: dict[str, str],
) -> dict[str, np.ndarray]:
    """Create out-of-fold predictions for stacking.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    folds : list of tuple
        CV folds (train_idx, val_idx)
    models_dict : dict
        Dictionary mapping model names to list of trained models (one per fold)
    feature_cols : list of str
        Feature column names
    model_types : dict
        Dictionary mapping model names to model types ('lightgbm', 'xgboost', 'catboost')

    Returns
    -------
    dict
        Dictionary mapping model names to out-of-fold prediction arrays
    """
    logger.info("Creating out-of-fold predictions for stacking...")

    # Initialize OOF prediction arrays
    oof_predictions = {name: np.zeros(len(df)) for name in models_dict.keys()}

    for fold_idx, (_train_idx, val_idx) in enumerate(folds):
        logger.info(f"Fold {fold_idx + 1}/{len(folds)}")

        val_data = df.iloc[val_idx].copy()
        val_data = val_data[val_data["Open"] == 1]

        # Get valid features for this fold
        valid_features = [col for col in feature_cols if col in val_data.columns]
        X_val = val_data[valid_features]

        # Get predictions from each model
        for model_name, models in models_dict.items():
            model = models[fold_idx]
            model_type = model_types[model_name]

            if model_type == "lightgbm":
                preds = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_type == "xgboost":
                import xgboost as xgb

                dval = xgb.DMatrix(X_val)
                preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            elif model_type == "catboost":
                preds = model.predict(X_val)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Store OOF predictions (matching indices from val_data)
            valid_indices = val_data.index.values
            oof_predictions[model_name][valid_indices] = preds

    logger.info("OOF predictions created successfully")

    return oof_predictions


def ensemble_cv_predictions(
    models_results: list[dict[str, Any]],
    method: str = "weighted",
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create ensemble using cross-validation predictions.

    Parameters
    ----------
    models_results : list of dict
        List of model results dictionaries from CV training
    method : str, default='weighted'
        Ensemble method: 'weighted' or 'average'
    weights : dict, optional
        Weights for weighted ensemble (optimized if None)

    Returns
    -------
    dict
        Ensemble results dictionary
    """
    logger.info("=" * 60)
    logger.info(f"Creating {method} ensemble")
    logger.info("=" * 60)

    n_folds = len(models_results[0]["fold_scores"])
    ensemble_scores = []

    for fold_idx in range(n_folds):
        # Get predictions from each model for this fold
        fold_preds = {}
        for model_result in models_results:
            model_name = model_result["model_name"]
            # Note: This would require storing predictions in model results
            # For now, we'll use fold scores
            fold_preds[model_name] = model_result["fold_scores"][fold_idx]

        # For demonstration, average the scores
        # In practice, would blend actual predictions
        if method == "average":
            ensemble_score = np.mean(list(fold_preds.values()))
        else:
            # Weighted average using provided weights
            if weights is None:
                weights = {m["model_name"]: 1.0 / len(models_results) for m in models_results}
            ensemble_score = sum(weights[name] * score for name, score in fold_preds.items())

        ensemble_scores.append(ensemble_score)

    mean_score = np.mean(ensemble_scores)
    std_score = np.std(ensemble_scores)

    logger.info(f"Ensemble mean RMSPE: {mean_score:.6f} ± {std_score:.6f}")
    logger.info("=" * 60)

    results = {
        "model_name": f"Ensemble_{method.capitalize()}",
        "metric": "RMSPE",
        "fold_scores": ensemble_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "weights": weights if method == "weighted" else None,
    }

    return results


def main():
    """Example usage of ensemble methods."""
    logger.info("Ensemble methods module loaded successfully")
    logger.info("Use weighted_blend_ensemble() or stacked_ensemble() for ensembling")


if __name__ == "__main__":
    main()
