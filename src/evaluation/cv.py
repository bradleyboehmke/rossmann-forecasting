"""
Time-series cross-validation utilities for the Rossmann forecasting project.

CRITICAL: Uses expanding window splits to prevent data leakage.
Each fold trains on all historical data up to the validation period.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.log import get_logger

logger = get_logger(__name__)


def make_time_series_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    fold_length_days: int = 42,
    min_train_days: int = 365,
    date_col: str = 'Date'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create expanding window time-series cross-validation folds.

    Each fold:
    - Trains on all historical data up to the validation period
    - Validates on the next fold_length_days (typically 6 weeks = 42 days)
    - Expanding window: training set grows with each fold

    CRITICAL: This prevents data leakage by ensuring validation data is
    always in the future relative to training data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with date column (must be sorted by Store and Date)
    n_folds : int, default=5
        Number of cross-validation folds
    fold_length_days : int, default=42
        Length of validation period in days (6 weeks)
    min_train_days : int, default=365
        Minimum number of days in first training fold (1 year)
    date_col : str, default='Date'
        Name of date column

    Returns
    -------
    list of tuple
        List of (train_indices, val_indices) for each fold

    Example
    -------
    Fold 1: Train [2013-01-01 to 2014-06-30], Val [2014-07-01 to 2014-08-11]
    Fold 2: Train [2013-01-01 to 2014-08-11], Val [2014-08-12 to 2014-09-22]
    Fold 3: Train [2013-01-01 to 2014-09-22], Val [2014-09-23 to 2014-11-03]
    ...
    """
    logger.info("="*60)
    logger.info("Creating time-series cross-validation folds")
    logger.info("="*60)
    logger.info(f"Number of folds: {n_folds}")
    logger.info(f"Validation fold length: {fold_length_days} days ({fold_length_days//7} weeks)")
    logger.info(f"Minimum training days: {min_train_days} days ({min_train_days//365} year)")

    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        logger.info(f"Converted {date_col} to datetime type")

    # Ensure data is sorted by date
    df = df.sort_values(date_col).reset_index(drop=True)

    # Get unique dates and ensure they're datetime64 type
    unique_dates = pd.to_datetime(df[date_col].unique())
    unique_dates = pd.Series(unique_dates).sort_values().values
    # Convert to DatetimeIndex to maintain datetime type
    unique_dates = pd.DatetimeIndex(unique_dates)

    logger.info(f"Date range: {unique_dates.min()} to {unique_dates.max()}")
    logger.info(f"Total unique dates: {len(unique_dates)}")

    # Calculate fold boundaries
    # Start with min_train_days for first fold
    # Each subsequent fold adds fold_length_days to validation end
    folds = []

    # Find the date that is min_train_days from start
    start_date = unique_dates[0]
    first_val_start_date = start_date + pd.Timedelta(days=min_train_days)

    # Find closest actual date using searchsorted on DatetimeIndex
    first_val_start_idx = unique_dates.searchsorted(first_val_start_date)
    if first_val_start_idx >= len(unique_dates):
        raise ValueError(f"Not enough data for minimum training days ({min_train_days})")

    logger.info(f"\nFirst validation period starts at: {unique_dates[first_val_start_idx]}")

    for fold_idx in range(n_folds):
        # Calculate validation start date using date arithmetic (not index arithmetic)
        val_start_date = unique_dates[first_val_start_idx] + pd.Timedelta(days=fold_idx * fold_length_days)

        # Calculate validation end date
        val_end_date = val_start_date + pd.Timedelta(days=fold_length_days - 1)

        # Check if we have enough data
        if val_end_date > unique_dates[-1]:
            logger.warning(f"Not enough data for fold {fold_idx + 1}, stopping at {fold_idx} folds")
            break

        # Training period: everything before validation start
        train_mask = df[date_col] < val_start_date
        val_mask = (df[date_col] >= val_start_date) & (df[date_col] <= val_end_date)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        if len(train_indices) == 0 or len(val_indices) == 0:
            logger.warning(f"Fold {fold_idx + 1} has empty train or val set, skipping")
            continue

        folds.append((train_indices, val_indices))

        # Get actual date range for logging
        train_start = df.iloc[train_indices[0]][date_col]
        train_end = df.iloc[train_indices[-1]][date_col]
        val_start = df.iloc[val_indices[0]][date_col]
        val_end = df.iloc[val_indices[-1]][date_col]

        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  Train: {train_start} to {train_end} ({len(train_indices):,} samples)")
        logger.info(f"  Val:   {val_start} to {val_end} ({len(val_indices):,} samples)")

    logger.info("="*60)
    logger.info(f"Created {len(folds)} time-series CV folds")
    logger.info("="*60)

    return folds


def get_fold_summary(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    date_col: str = 'Date'
) -> pd.DataFrame:
    """
    Generate a summary dataframe of fold information.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    folds : list of tuple
        List of (train_indices, val_indices)
    date_col : str, default='Date'
        Name of date column

    Returns
    -------
    pd.DataFrame
        Summary with columns: fold, train_start, train_end, train_size,
        val_start, val_end, val_size
    """
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    summary = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_info = {
            'fold': fold_idx + 1,
            'train_start': df.iloc[train_idx[0]][date_col],
            'train_end': df.iloc[train_idx[-1]][date_col],
            'train_size': len(train_idx),
            'val_start': df.iloc[val_idx[0]][date_col],
            'val_end': df.iloc[val_idx[-1]][date_col],
            'val_size': len(val_idx)
        }

        # Calculate days
        fold_info['train_days'] = (fold_info['train_end'] - fold_info['train_start']).days + 1
        fold_info['val_days'] = (fold_info['val_end'] - fold_info['val_start']).days + 1

        summary.append(fold_info)

    return pd.DataFrame(summary)


def filter_open_stores(
    df: pd.DataFrame,
    open_col: str = 'Open',
    date_col: str = 'Date'
) -> pd.DataFrame:
    """
    Filter dataset to only include days when stores were open.

    This is important because:
    1. Closed stores have Sales=0 which is ignored in RMSPE
    2. Models shouldn't train on closed-store patterns

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Open column
    open_col : str, default='Open'
        Name of open/closed flag column
    date_col : str, default='Date'
        Name of date column for sorting

    Returns
    -------
    pd.DataFrame
        Filtered dataset with only open stores (sorted by date with reset index)
    """
    initial_size = len(df)
    df_filtered = df[df[open_col] == 1].copy()
    removed = initial_size - len(df_filtered)

    # Sort by date and reset index to ensure positional indices are sequential
    # This is critical for CV fold indices to work correctly
    df_filtered = df_filtered.sort_values(date_col).reset_index(drop=True)

    logger.info(f"Filtered out {removed:,} closed store-days ({removed/initial_size*100:.2f}%)")
    logger.info(f"Remaining: {len(df_filtered):,} open store-days")

    return df_filtered


def remove_missing_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove rows with missing values in feature columns.

    This is necessary because lag/rolling features have NaN values
    for early dates where there's insufficient history.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    feature_cols : list of str
        Feature column names to check

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        (Filtered dataset, list of removed feature columns if any had all NaN)
    """
    initial_size = len(df)

    # Check for completely empty feature columns
    empty_cols = []
    for col in feature_cols:
        if col in df.columns and df[col].isna().all():
            empty_cols.append(col)
            logger.warning(f"Feature '{col}' is entirely NaN, will be dropped")

    # Remove empty columns from feature list
    valid_features = [col for col in feature_cols if col not in empty_cols]

    # Remove rows with any missing values in valid features
    df_clean = df.dropna(subset=valid_features).copy()

    removed = initial_size - len(df_clean)
    logger.info(f"Removed {removed:,} rows with missing features ({removed/initial_size*100:.2f}%)")
    logger.info(f"Remaining: {len(df_clean):,} complete rows")

    return df_clean, valid_features
