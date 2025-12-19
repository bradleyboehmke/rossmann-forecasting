"""Pytest configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parents[1]


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing.

    Returns a small DataFrame mimicking train.csv structure (NO store metadata).
    """
    np.random.seed(42)
    n_stores = 5
    n_days = 30

    data = []
    for store in range(1, n_stores + 1):
        for day in range(n_days):
            date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=day)
            is_open = 1 if date.dayofweek < 6 else 0

            data.append(
                {
                    "Store": store,
                    "DayOfWeek": date.dayofweek + 1,
                    "Date": date,
                    "Sales": np.random.randint(3000, 8000) if is_open else 0,
                    "Customers": np.random.randint(400, 1200) if is_open else 0,
                    "Open": is_open,
                    "Promo": np.random.choice([0, 1]) if is_open else 0,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_store_data():
    """Create sample store data for testing.

    Returns a small DataFrame mimicking store.csv structure.
    """
    stores = []
    for store_id in range(1, 6):
        stores.append(
            {
                "Store": store_id,
                "StoreType": np.random.choice(["a", "b", "c", "d"]),
                "Assortment": np.random.choice(["a", "b", "c"]),
                "CompetitionDistance": np.random.randint(100, 5000),
                "CompetitionOpenSinceMonth": np.random.randint(1, 13),
                "CompetitionOpenSinceYear": np.random.randint(2010, 2015),
                "Promo2": np.random.choice([0, 1]),
                "Promo2SinceWeek": np.random.randint(1, 53),
                "Promo2SinceYear": 2014,
                "PromoInterval": "Feb,May,Aug,Nov",
            }
        )

    return pd.DataFrame(stores)


@pytest.fixture
def sample_features_data(sample_train_data, sample_store_data):
    """Create sample feature-engineered data.

    Merges train and store data with basic features.
    """
    # Parse dates first
    df = sample_train_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Merge with store data
    df = df.merge(sample_store_data, on="Store", how="left")

    # Add calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Day"] = df["Date"].dt.day

    return df


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters for ensemble models.

    Returns
    -------
    dict
        Hyperparameters for LightGBM, XGBoost, and CatBoost
    """
    return {
        "lightgbm": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "gamma": 0.01,
        },
        "catboost": {
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.5,
            "border_count": 128,
        },
    }


@pytest.fixture
def mock_model_version():
    """Create a mock MLflow ModelVersion object.

    Returns
    -------
    Mock
        Mock ModelVersion with common attributes
    """
    from unittest.mock import Mock

    mv = Mock()
    mv.name = "rossmann-ensemble"
    mv.version = "1"
    mv.current_stage = "Production"
    mv.description = "Test ensemble model"
    mv.run_id = "test-run-id-123"
    mv.status = "READY"
    mv.creation_timestamp = 1609459200000  # 2021-01-01
    mv.last_updated_timestamp = 1609459200000
    return mv


@pytest.fixture
def mock_registered_model():
    """Create a mock MLflow RegisteredModel object.

    Returns
    -------
    Mock
        Mock RegisteredModel with common attributes
    """
    from unittest.mock import Mock

    rm = Mock()
    rm.name = "rossmann-ensemble"
    rm.description = "Ensemble model for Rossmann sales forecasting"
    rm.creation_timestamp = 1609459200000
    rm.last_updated_timestamp = 1609459200000
    return rm
