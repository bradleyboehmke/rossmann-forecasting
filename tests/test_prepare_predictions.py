"""Tests for prediction data preparation pipeline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.prepare_predictions import (
    prepare_prediction_data,
    validate_input_data,
)


@pytest.fixture
def sample_raw_data():
    """Sample raw prediction data in train.csv format."""
    return pd.DataFrame(
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


@pytest.fixture
def sample_store_metadata():
    """Sample store metadata."""
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "c"],
            "CompetitionDistance": [1000.0, 2000.0],
            "CompetitionOpenSinceMonth": [1, 2],
            "CompetitionOpenSinceYear": [2010, 2011],
            "Promo2": [1, 0],
            "Promo2SinceWeek": [1, 0],
            "Promo2SinceYear": [2012, 0],
            "PromoInterval": ["Jan,Apr,Jul,Oct", ""],
        }
    )


def test_prepare_prediction_data_shape(sample_raw_data, sample_store_metadata):
    """Test that prepare_prediction_data returns correct shape."""
    features = prepare_prediction_data(sample_raw_data, store_metadata=sample_store_metadata)

    # Should have 3 rows (same as input)
    assert len(features) == 3

    # Should have 46 features (excluding Sales, Date, Store, Customers)
    assert features.shape[1] == 46


def test_prepare_prediction_data_columns(sample_raw_data, sample_store_metadata):
    """Test that prepare_prediction_data returns expected columns."""
    features = prepare_prediction_data(sample_raw_data, store_metadata=sample_store_metadata)

    # Check that excluded columns are not present
    excluded_cols = ["Sales", "Date", "Store", "Customers"]
    for col in excluded_cols:
        assert col not in features.columns, f"Column '{col}' should be excluded"

    # Check that some expected features are present
    expected_features = [
        "DayOfWeek",
        "Open",
        "Promo",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Year",
        "Month",
        "Week",
        "IsWeekend",
        "Season",
    ]
    for col in expected_features:
        assert col in features.columns, f"Expected feature '{col}' is missing"


def test_prepare_prediction_data_no_nulls(sample_raw_data, sample_store_metadata):
    """Test that prepare_prediction_data handles missing values properly."""
    features = prepare_prediction_data(sample_raw_data, store_metadata=sample_store_metadata)

    # Check for unexpected NaN values in non-lag features
    # (Lag features may have NaN for first observations)
    non_lag_cols = [c for c in features.columns if "Lag" not in c and "Rolling" not in c]
    for col in non_lag_cols:
        null_count = features[col].isna().sum()
        assert (
            null_count == 0
        ), f"Column '{col}' has {null_count} null values (non-lag features should not have nulls)"


def test_validate_input_data_valid(sample_raw_data):
    """Test validation with valid data."""
    is_valid, errors = validate_input_data(sample_raw_data)

    assert is_valid is True
    assert len(errors) == 0


def test_validate_input_data_missing_columns():
    """Test validation with missing required columns."""
    invalid_data = pd.DataFrame({"Store": [1], "DayOfWeek": [5]})

    is_valid, errors = validate_input_data(invalid_data)

    assert is_valid is False
    assert len(errors) > 0
    assert "Missing required columns" in errors[0]


def test_validate_input_data_invalid_store_range():
    """Test validation with Store ID out of range."""
    invalid_data = pd.DataFrame(
        {
            "Store": [9999],  # Invalid: > 1115
            "DayOfWeek": [5],
            "Date": ["2015-08-01"],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
        }
    )

    is_valid, errors = validate_input_data(invalid_data)

    assert is_valid is False
    assert any("Store must be between 1 and 1115" in err for err in errors)


def test_validate_input_data_invalid_dayofweek():
    """Test validation with DayOfWeek out of range."""
    invalid_data = pd.DataFrame(
        {
            "Store": [1],
            "DayOfWeek": [8],  # Invalid: > 7
            "Date": ["2015-08-01"],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
        }
    )

    is_valid, errors = validate_input_data(invalid_data)

    assert is_valid is False
    assert any("DayOfWeek must be between 1 and 7" in err for err in errors)


def test_validate_input_data_invalid_open_value():
    """Test validation with invalid Open value."""
    invalid_data = pd.DataFrame(
        {
            "Store": [1],
            "DayOfWeek": [5],
            "Date": ["2015-08-01"],
            "Open": [2],  # Invalid: must be 0 or 1
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
        }
    )

    is_valid, errors = validate_input_data(invalid_data)

    assert is_valid is False
    assert any("Open must be 0 or 1" in err for err in errors)


def test_validate_input_data_invalid_stateholiday():
    """Test validation with invalid StateHoliday value."""
    invalid_data = pd.DataFrame(
        {
            "Store": [1],
            "DayOfWeek": [5],
            "Date": ["2015-08-01"],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": ["x"],  # Invalid: must be 0, a, b, or c
            "SchoolHoliday": [0],
        }
    )

    is_valid, errors = validate_input_data(invalid_data)

    assert is_valid is False
    assert any("StateHoliday must be one of" in err for err in errors)


def test_prepare_prediction_data_adds_dummy_columns(sample_store_metadata):
    """Test that prepare_prediction_data adds dummy Sales/Customers columns."""
    # Create data without Sales/Customers
    raw_data = pd.DataFrame(
        {
            "Store": [1],
            "DayOfWeek": [5],
            "Date": ["2015-08-01"],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
        }
    )

    # Should not raise error even without Sales/Customers
    features = prepare_prediction_data(raw_data, store_metadata=sample_store_metadata)

    # Should successfully create features
    assert len(features) == 1
    assert features.shape[1] == 46


def test_prepare_prediction_data_preserves_row_count(sample_raw_data, sample_store_metadata):
    """Test that prepare_prediction_data preserves input row count."""
    features = prepare_prediction_data(sample_raw_data, store_metadata=sample_store_metadata)

    assert len(features) == len(sample_raw_data)


def test_prepare_prediction_data_dtypes(sample_raw_data, sample_store_metadata):
    """Test that prepare_prediction_data returns appropriate dtypes."""
    features = prepare_prediction_data(sample_raw_data, store_metadata=sample_store_metadata)

    # Categorical features should be category dtype
    categorical_features = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
    for col in categorical_features:
        if col in features.columns:
            assert (
                features[col].dtype.name == "category"
            ), f"Column '{col}' should be category dtype"

    # Numeric features should be numeric
    numeric_features = ["Open", "Promo", "CompetitionDistance", "Year", "Month"]
    for col in numeric_features:
        if col in features.columns:
            assert pd.api.types.is_numeric_dtype(
                features[col]
            ), f"Column '{col}' should be numeric dtype"
