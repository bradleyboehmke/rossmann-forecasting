"""
Evaluation metrics for the Rossmann forecasting project.
"""

import numpy as np
from typing import Union


def rmspe(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    ignore_zero_sales: bool = True
) -> float:
    """
    Calculate Root Mean Square Percentage Error (RMSPE).

    This is the primary evaluation metric for the Rossmann competition.
    Observations where Sales = 0 are ignored by default.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    ignore_zero_sales : bool, default=True
        If True, exclude observations where y_true == 0 from calculation

    Returns
    -------
    float
        RMSPE score (lower is better)

    Notes
    -----
    Formula: RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Create mask to exclude zero sales if specified
    mask = y_true != 0 if ignore_zero_sales else np.ones_like(y_true, dtype=bool)

    # Calculate RMSPE only on masked values
    return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        RMSE score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def mae(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        MAE score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def mape(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    ignore_zero_sales: bool = True
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    ignore_zero_sales : bool, default=True
        If True, exclude observations where y_true == 0 from calculation

    Returns
    -------
    float
        MAPE score in percentage (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Create mask to exclude zero sales if specified
    mask = y_true != 0 if ignore_zero_sales else np.ones_like(y_true, dtype=bool)

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
