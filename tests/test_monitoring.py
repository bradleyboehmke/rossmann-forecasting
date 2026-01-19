"""Tests for monitoring module (prediction logging and drift detection)."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "deployment" / "api"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction_logger import PredictionLogger


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory

    Returns
    -------
    Path
        Path to temporary database
    """
    db_path = tmp_path / "test_predictions.db"
    return db_path


@pytest.fixture
def sample_raw_inputs():
    """Create sample raw inputs (train.csv format).

    Returns
    -------
    pd.DataFrame
        Sample raw input data
    """
    return pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "DayOfWeek": [1, 2, 1],
            "Date": ["2015-01-01", "2015-01-02", "2015-01-01"],
            "Open": [1, 1, 1],
            "Promo": [0, 1, 0],
            "StateHoliday": ["0", "0", "a"],
            "SchoolHoliday": [0, 0, 1],
        }
    )


@pytest.fixture
def sample_features():
    """Create sample feature data.

    Returns
    -------
    pd.DataFrame
        Sample feature data
    """
    return pd.DataFrame(
        {
            "month": [1, 1, 1],
            "year": [2015, 2015, 2015],
            "StoreType": ["c", "c", "a"],
            "Assortment": ["a", "a", "c"],
            "CompetitionDistance": [1270.0, 1270.0, 570.0],
            "Promo2": [1, 1, 0],
            "IsPromo2Active": [1, 1, 0],
            "sales_lag_7": [5530.0, 5550.0, 6100.0],
            "sales_rolling_mean_7": [5600.0, 5610.0, 6050.0],
        }
    )


@pytest.fixture
def sample_predictions():
    """Create sample predictions.

    Returns
    -------
    list[float]
        Sample predictions
    """
    return [5500.0, 5700.0, 6200.0]


# =============================================================================
# Tests for PredictionLogger
# =============================================================================


def test_prediction_logger_init(temp_db):
    """Test PredictionLogger initialization creates database and tables."""
    _ = PredictionLogger(temp_db)

    assert temp_db.exists()

    # Check tables were created
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        assert cursor.fetchone() is not None

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='actual_sales'"
        )
        assert cursor.fetchone() is not None


def test_log_predictions(temp_db, sample_raw_inputs, sample_features, sample_predictions):
    """Test logging predictions to database."""
    logger = PredictionLogger(temp_db)

    batch_id = logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="1",
        model_stage="Production",
        response_time_ms=150.5,
    )

    assert batch_id is not None

    # Verify data was logged
    with sqlite3.connect(temp_db) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)

    assert len(df) == 3
    assert df["batch_id"].iloc[0] == batch_id
    assert df["model_version"].iloc[0] == "1"
    assert df["model_stage"].iloc[0] == "Production"
    assert df["store_id"].tolist() == [1, 1, 2]
    assert df["prediction"].tolist() == sample_predictions


def test_log_predictions_extracts_features_correctly(
    temp_db, sample_raw_inputs, sample_features, sample_predictions
):
    """Test that log_predictions correctly extracts key features."""
    logger = PredictionLogger(temp_db)

    logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="1",
    )

    # Verify feature extraction
    with sqlite3.connect(temp_db) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)

    # Check raw input features
    assert df["promo"].tolist() == [0, 1, 0]
    assert df["day_of_week"].tolist() == [1, 2, 1]
    assert df["state_holiday"].tolist() == ["0", "0", "a"]

    # Check engineered features
    assert df["month"].tolist() == [1, 1, 1]
    assert df["store_type"].tolist() == ["c", "c", "a"]
    assert df["assortment"].tolist() == ["a", "a", "c"]


def test_get_predictions_all(temp_db, sample_raw_inputs, sample_features, sample_predictions):
    """Test retrieving all predictions."""
    logger = PredictionLogger(temp_db)

    logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="1",
    )

    # Retrieve predictions
    df = logger.get_predictions()

    assert len(df) == 3
    assert "prediction" in df.columns
    assert "model_version" in df.columns


def test_get_predictions_with_limit(
    temp_db, sample_raw_inputs, sample_features, sample_predictions
):
    """Test retrieving predictions with limit."""
    logger = PredictionLogger(temp_db)

    logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="1",
    )

    # Retrieve with limit
    df = logger.get_predictions(limit=2)

    assert len(df) == 2


def test_get_summary_stats(temp_db, sample_raw_inputs, sample_features, sample_predictions):
    """Test getting summary statistics."""
    logger = PredictionLogger(temp_db)

    logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="1",
        model_stage="Production",
    )

    stats = logger.get_summary_stats()

    assert stats["total_predictions"] == 3
    assert stats["first_prediction"] is not None
    assert stats["last_prediction"] is not None
    assert len(stats["model_versions"]) > 0
    assert stats["model_versions"][0]["model_version"] == "1"
    assert stats["model_versions"][0]["count"] == 3


def test_log_multiple_batches(temp_db, sample_raw_inputs, sample_features, sample_predictions):
    """Test logging multiple batches of predictions."""
    logger = PredictionLogger(temp_db)

    batch_id_1 = logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="1",
    )

    batch_id_2 = logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=sample_features,
        predictions=sample_predictions,
        model_version="2",
    )

    assert batch_id_1 != batch_id_2

    # Verify both batches were logged
    df = logger.get_predictions()
    assert len(df) == 6  # 3 predictions × 2 batches

    # Verify different model versions
    versions = df["model_version"].unique()
    assert "1" in versions
    assert "2" in versions


def test_handles_missing_lag_features(temp_db, sample_raw_inputs, sample_predictions):
    """Test that logger handles missing lag features gracefully."""
    logger = PredictionLogger(temp_db)

    # Features without lag columns
    features_no_lags = pd.DataFrame(
        {
            "month": [1, 1, 1],
            "year": [2015, 2015, 2015],
            "StoreType": ["c", "c", "a"],
            "Assortment": ["a", "a", "c"],
            "CompetitionDistance": [1270.0, 1270.0, 570.0],
            "Promo2": [1, 1, 0],
            "IsPromo2Active": [1, 1, 0],
        }
    )

    # Should not raise error
    logger.log_predictions(
        raw_inputs=sample_raw_inputs,
        features=features_no_lags,
        predictions=sample_predictions,
        model_version="1",
    )

    # Verify NULL values for lag features
    with sqlite3.connect(temp_db) as conn:
        df = pd.read_sql_query("SELECT sales_lag_7, sales_rolling_mean_7 FROM predictions", conn)

    assert df["sales_lag_7"].isna().all()
    assert df["sales_rolling_mean_7"].isna().all()


# =============================================================================
# Tests for Drift Detection (basic tests - full integration requires reference data)
# =============================================================================


def test_drift_detector_key_features():
    """Test that DriftDetector has correct key features defined.

    This is a simple smoke test - just checks the class can be imported
    and has the expected KEY_FEATURES attribute.
    """

    from monitoring.drift_detection import DriftDetector

    expected_features = [
        "promo",
        "day_of_week",
        "month",
        "state_holiday",
        "school_holiday",
        "store_type",
        "assortment",
        "competition_distance",
        "promo2",
        "is_promo2_active",
    ]

    assert DriftDetector.KEY_FEATURES == expected_features


# Note: Full drift detection tests require reference data and populated database
# These would be integration tests run separately
