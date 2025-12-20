"""Prepare new data for model predictions.

This module provides a unified interface for preparing raw prediction data through
the complete data processing pipeline:
1. Load and merge store metadata
2. Clean data
3. Engineer features
4. Extract model-ready features

This ensures consistency between training and inference data processing.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from features.build_features import build_all_features
from utils.io import read_csv
from utils.log import get_logger

from data.make_dataset import basic_cleaning, merge_store_info

logger = get_logger(__name__)

# Model feature columns (excluding metadata and target columns)
# These columns are excluded during training (see train_ensemble.py:111)
EXCLUDED_COLUMNS = ["Sales", "Date", "Store", "Customers"]


def prepare_prediction_data(
    raw_data: pd.DataFrame,
    store_metadata: Optional[pd.DataFrame] = None,
    store_path: str = "data/raw/store.csv",
) -> pd.DataFrame:
    """Prepare raw data for model predictions through complete pipeline.

    This function mimics the production batch prediction workflow:
    1. Merges with store metadata
    2. Cleans the data
    3. Engineers features
    4. Returns model-ready features

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw input data in train.csv format with columns:
        - Store: int
        - DayOfWeek: int
        - Date: str or datetime
        - Open: int
        - Promo: int
        - StateHoliday: str
        - SchoolHoliday: int
    store_metadata : pd.DataFrame, optional
        Store metadata (store.csv). If None, will be loaded from store_path
    store_path : str, default="data/raw/store.csv"
        Path to store metadata file (used if store_metadata is None)

    Returns
    -------
    pd.DataFrame
        Model-ready features (46 columns, excludes Sales/Date/Store/Customers)

    Examples
    --------
    >>> import pandas as pd
    >>> raw_data = pd.DataFrame({
    ...     "Store": [1],
    ...     "DayOfWeek": [5],
    ...     "Date": ["2015-08-01"],
    ...     "Open": [1],
    ...     "Promo": [1],
    ...     "StateHoliday": ["0"],
    ...     "SchoolHoliday": [0]
    ... })
    >>> features = prepare_prediction_data(raw_data)
    >>> features.shape
    (1, 46)
    """
    logger.info(f"Preparing {len(raw_data)} records for prediction")

    # Make a copy to avoid modifying input
    df = raw_data.copy()

    # Step 1: Add dummy Sales and Customers columns if not present
    # These are required by the cleaning pipeline but will be dropped later
    if "Sales" not in df.columns:
        df["Sales"] = 0
        logger.debug("Added dummy Sales column")
    if "Customers" not in df.columns:
        df["Customers"] = 0
        logger.debug("Added dummy Customers column")

    # Step 2: Load store metadata if not provided
    if store_metadata is None:
        logger.info(f"Loading store metadata from {store_path}")
        store_metadata = read_csv(store_path)
        logger.info(f"Loaded metadata for {len(store_metadata)} stores")

    # Step 3: Merge with store metadata
    logger.info("Merging with store metadata")
    merged_df = merge_store_info(df, store_metadata)

    # Step 4: Clean the data
    logger.info("Cleaning data")
    clean_df = basic_cleaning(merged_df)

    # Step 5: Engineer features
    logger.info("Engineering features")
    featured_df = build_all_features(clean_df)

    # Step 6: Extract model-ready features
    # CRITICAL: Must exclude same columns as training pipeline
    feature_cols = [c for c in featured_df.columns if c not in EXCLUDED_COLUMNS]
    model_features = featured_df[feature_cols]

    logger.info(f"Prepared {len(model_features)} records with {len(feature_cols)} features")

    return model_features


def prepare_prediction_data_from_csv(
    csv_path: str,
    store_metadata: Optional[pd.DataFrame] = None,
    store_path: str = "data/raw/store.csv",
) -> pd.DataFrame:
    """Load raw data from CSV and prepare for predictions.

    Convenience wrapper around prepare_prediction_data() that loads data from CSV.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with raw prediction data (train.csv format)
    store_metadata : pd.DataFrame, optional
        Store metadata. If None, will be loaded from store_path
    store_path : str, default="data/raw/store.csv"
        Path to store metadata file

    Returns
    -------
    pd.DataFrame
        Model-ready features

    Examples
    --------
    >>> features = prepare_prediction_data_from_csv("new_data.csv")
    >>> features.shape
    (100, 46)
    """
    logger.info(f"Loading raw data from {csv_path}")
    raw_data = read_csv(csv_path)
    logger.info(f"Loaded {len(raw_data)} records")

    return prepare_prediction_data(raw_data, store_metadata, store_path)


def get_feature_columns() -> list[str]:
    """Get the list of expected feature column names for the model.

    This is useful for validation and documentation purposes.

    Returns
    -------
    list of str
        Expected feature column names (sorted alphabetically)

    Notes
    -----
    This requires loading actual data to determine feature names.
    For a static list, see the training code or model artifacts.
    """
    # Load a small sample to get feature names
    logger.info("Determining feature columns from sample data")

    # Use a minimal sample
    sample_data = pd.DataFrame(
        {
            "Store": [1],
            "DayOfWeek": [1],
            "Date": ["2015-01-01"],
            "Open": [1],
            "Promo": [0],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
        }
    )

    features = prepare_prediction_data(sample_data)
    return sorted(features.columns.tolist())


def validate_input_data(raw_data: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that raw input data has required columns and types.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw input data to validate

    Returns
    -------
    tuple of (bool, list of str)
        (is_valid, list of error messages)

    Examples
    --------
    >>> data = pd.DataFrame({"Store": [1], "DayOfWeek": [5]})
    >>> is_valid, errors = validate_input_data(data)
    >>> is_valid
    False
    >>> "Date" in errors[0]
    True
    """
    errors = []

    # Required columns
    required_cols = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]

    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in raw_data.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check data types and ranges (basic validation)
    if "Store" in raw_data.columns:
        if not pd.api.types.is_integer_dtype(raw_data["Store"]):
            errors.append("Store must be integer type")
        elif (raw_data["Store"] < 1).any() or (raw_data["Store"] > 1115).any():
            errors.append("Store must be between 1 and 1115")

    if "DayOfWeek" in raw_data.columns:
        if not pd.api.types.is_integer_dtype(raw_data["DayOfWeek"]):
            errors.append("DayOfWeek must be integer type")
        elif (raw_data["DayOfWeek"] < 1).any() or (raw_data["DayOfWeek"] > 7).any():
            errors.append("DayOfWeek must be between 1 and 7")

    if "Open" in raw_data.columns:
        if not raw_data["Open"].isin([0, 1]).all():
            errors.append("Open must be 0 or 1")

    if "Promo" in raw_data.columns:
        if not raw_data["Promo"].isin([0, 1]).all():
            errors.append("Promo must be 0 or 1")

    if "SchoolHoliday" in raw_data.columns:
        if not raw_data["SchoolHoliday"].isin([0, 1]).all():
            errors.append("SchoolHoliday must be 0 or 1")

    if "StateHoliday" in raw_data.columns:
        valid_holidays = ["0", "a", "b", "c"]
        if not raw_data["StateHoliday"].isin(valid_holidays).all():
            errors.append(f"StateHoliday must be one of {valid_holidays}")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    """Example usage of prediction data preparation."""
    logger.info("=" * 70)
    logger.info("Prediction Data Preparation Example")
    logger.info("=" * 70)

    # Example 1: Prepare data from DataFrame
    sample_data = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "DayOfWeek": [5, 6, 5],
            "Date": ["2015-08-01", "2015-08-02", "2015-08-01"],
            "Open": [1, 1, 1],
            "Promo": [1, 0, 1],
            "StateHoliday": ["0", "0", "0"],
            "SchoolHoliday": [0, 0, 0],
        }
    )

    logger.info("\nValidating input data...")
    is_valid, errors = validate_input_data(sample_data)
    if is_valid:
        logger.info("✓ Input data is valid")
    else:
        logger.error(f"✗ Input data validation failed: {errors}")
        return

    logger.info("\nPreparing features...")
    features = prepare_prediction_data(sample_data)

    logger.info(f"\nPrepared features shape: {features.shape}")
    logger.info(f"Feature columns: {sorted(features.columns.tolist())}")

    logger.info("\n" + "=" * 70)
    logger.info("Example complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
