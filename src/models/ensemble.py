"""Ensemble model for Rossmann sales forecasting.

This module provides a custom MLflow PyFunc wrapper that encapsulates LightGBM, XGBoost, and
CatBoost models into a weighted ensemble for production deployment.
"""

import mlflow.pyfunc
import numpy as np


class RossmannEnsemble(mlflow.pyfunc.PythonModel):
    """Custom MLflow PyFunc model that wraps LightGBM, XGBoost, and CatBoost models into a weighted
    ensemble for production deployment.

    This allows the entire ensemble to be registered as a single deployable
    artifact in MLflow Model Registry, encapsulating all ensemble logic.

    Parameters
    ----------
    lgb_model : lightgbm.Booster
        Trained LightGBM model
    xgb_model : xgboost.Booster
        Trained XGBoost model
    cb_model : catboost.CatBoost
        Trained CatBoost model
    weights : dict
        Ensemble weights for each model, e.g., {'lightgbm': 0.3, 'xgboost': 0.6, 'catboost': 0.1}
    cat_features : list
        List of categorical feature names

    Examples
    --------
    >>> ensemble = RossmannEnsemble(
    ...     lgb_model=lgb_model,
    ...     xgb_model=xgb_model,
    ...     cb_model=cb_model,
    ...     weights={'lightgbm': 0.3, 'xgboost': 0.6, 'catboost': 0.1},
    ...     cat_features=['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']
    ... )
    >>> predictions = ensemble.predict(None, X_test)
    """

    def __init__(self, lgb_model, xgb_model, cb_model, weights, cat_features):
        """Initialize the ensemble model with trained base models and weights.

        Parameters
        ----------
        lgb_model : lightgbm.Booster
            Trained LightGBM model
        xgb_model : xgboost.Booster
            Trained XGBoost model
        cb_model : catboost.CatBoost
            Trained CatBoost model
        weights : dict
            Ensemble weights, must sum to 1.0
        cat_features : list
            Categorical feature names
        """
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cb_model = cb_model
        self.weights = weights
        self.cat_features = cat_features

        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total_weight}. " f"Weights: {weights}"
            )

    def predict(self, context, model_input):
        """Generate ensemble predictions by combining predictions from all three models.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            MLflow model context (unused, required by interface)
        model_input : pd.DataFrame
            Input features for prediction

        Returns
        -------
        np.ndarray
            Weighted ensemble predictions
        """
        import catboost as cb
        import xgboost as xgb

        # LightGBM predictions
        lgb_preds = self.lgb_model.predict(model_input)

        # XGBoost predictions (needs categorical encoding)
        X_xgb = model_input.copy()
        for col in X_xgb.columns:
            if X_xgb[col].dtype.name == "category":
                X_xgb[col] = X_xgb[col].cat.codes
        dmatrix = xgb.DMatrix(X_xgb)
        xgb_preds = self.xgb_model.predict(dmatrix)

        # CatBoost predictions
        pool = cb.Pool(model_input, cat_features=self.cat_features)
        cb_preds = self.cb_model.predict(pool)

        # Weighted ensemble
        ensemble_preds = (
            self.weights["lightgbm"] * lgb_preds
            + self.weights["xgboost"] * xgb_preds
            + self.weights["catboost"] * cb_preds
        )

        return ensemble_preds


def create_ensemble(lgb_model, xgb_model, cb_model, weights=None, cat_features=None):
    """Factory function to create a RossmannEnsemble instance.

    Parameters
    ----------
    lgb_model : lightgbm.Booster
        Trained LightGBM model
    xgb_model : xgboost.Booster
        Trained XGBoost model
    cb_model : catboost.CatBoost
        Trained CatBoost model
    weights : dict, optional
        Ensemble weights. Default: {'lightgbm': 0.3, 'xgboost': 0.6, 'catboost': 0.1}
    cat_features : list, optional
        Categorical feature names. Default: ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']

    Returns
    -------
    RossmannEnsemble
        Initialized ensemble model

    Examples
    --------
    >>> ensemble = create_ensemble(lgb_model, xgb_model, cb_model)
    >>> predictions = ensemble.predict(None, X_test)
    """
    if weights is None:
        weights = {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}

    if cat_features is None:
        cat_features = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

    return RossmannEnsemble(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        cb_model=cb_model,
        weights=weights,
        cat_features=cat_features,
    )
