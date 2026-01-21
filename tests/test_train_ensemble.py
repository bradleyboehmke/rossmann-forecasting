"""Tests for production ensemble training pipeline.

This module tests the training workflow in src/models/train_ensemble.py including data loading,
hyperparameter loading, individual model training, and ensemble registration.
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.models import train_ensemble


class TestLoadTrainingData:
    """Tests for load_training_data()."""

    def test_load_training_data_basic(self, sample_features_data, tmp_path):
        """Test loading and splitting training data."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act - use smaller holdout since sample data is only 30 days
        train_df, holdout_df = train_ensemble.load_training_data(str(data_path), holdout_days=7)

        # Assert
        assert len(train_df) > 0
        assert len(holdout_df) > 0
        assert "Date" in train_df.columns
        assert "Date" in holdout_df.columns

    def test_load_training_data_correct_split(self, sample_features_data, tmp_path):
        """Test that train/holdout split is correct."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)
        holdout_days = 14

        # Calculate expected split
        max_date = sample_features_data["Date"].max()
        expected_holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)

        # Act
        train_df, holdout_df = train_ensemble.load_training_data(
            str(data_path), holdout_days=holdout_days
        )

        # Assert
        assert train_df["Date"].max() < expected_holdout_start
        assert holdout_df["Date"].min() >= expected_holdout_start
        assert holdout_df["Date"].max() == max_date

    def test_load_training_data_no_overlap(self, sample_features_data, tmp_path):
        """Test that train and holdout sets don't overlap."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act
        train_df, holdout_df = train_ensemble.load_training_data(str(data_path))

        # Assert
        train_dates = set(train_df["Date"].unique())
        holdout_dates = set(holdout_df["Date"].unique())
        assert len(train_dates & holdout_dates) == 0  # No intersection

    def test_load_training_data_custom_holdout_days(self, sample_features_data, tmp_path):
        """Test loading with custom holdout period."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act
        train_df, holdout_df = train_ensemble.load_training_data(str(data_path), holdout_days=7)

        # Assert
        assert len(train_df) > len(holdout_df)


class TestLoadBestHyperparameters:
    """Tests for load_best_hyperparameters()."""

    def test_load_best_hyperparameters_valid_json(self, sample_hyperparameters, tmp_path):
        """Test loading valid hyperparameters from JSON."""
        # Arrange
        config_path = tmp_path / "best_hyperparameters.json"

        # Create test hyperparameters with metadata
        test_params = {
            "metadata": {
                "best_model": "xgboost",
                "best_rmspe": 0.09123,
            },
            "lightgbm": sample_hyperparameters["lightgbm"],
            "xgboost": sample_hyperparameters["xgboost"],
            "catboost": sample_hyperparameters["catboost"],
        }

        with open(config_path, "w") as f:
            json.dump(test_params, f)

        # Act
        params = train_ensemble.load_best_hyperparameters(str(config_path))

        # Assert
        assert "metadata" in params
        assert "lightgbm" in params
        assert "xgboost" in params
        assert "catboost" in params
        assert params["metadata"]["best_model"] == "xgboost"

    def test_load_best_hyperparameters_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            train_ensemble.load_best_hyperparameters("nonexistent.json")


class TestPrepareTrainingData:
    """Tests for prepare_training_data()."""

    def test_prepare_training_data_basic(self, sample_features_data):
        """Test basic data preparation."""
        # Act
        X_train, y_train, feature_cols, cat_features = train_ensemble.prepare_training_data(
            sample_features_data
        )

        # Assert
        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(feature_cols) > 0
        assert "Sales" not in feature_cols
        assert "Date" not in feature_cols
        assert "Store" not in feature_cols
        assert "Customers" not in feature_cols

    def test_prepare_training_data_filters_closed_stores(self, sample_features_data):
        """Test that closed stores are filtered out."""
        # Arrange
        # Set some stores to closed
        df_with_closed = sample_features_data.copy()
        df_with_closed.loc[0:5, "Open"] = 0

        # Act
        X_train, y_train, feature_cols, cat_features = train_ensemble.prepare_training_data(
            df_with_closed
        )

        # Assert
        # Should have fewer rows than original (closed stores removed)
        assert len(X_train) < len(df_with_closed)
        assert len(X_train) == len(y_train)

    def test_prepare_training_data_identifies_categorical_features(self, sample_features_data):
        """Test that categorical features are correctly identified."""
        # Arrange - convert some columns to categorical
        df_with_cats = sample_features_data.copy()
        df_with_cats["StoreType"] = df_with_cats["StoreType"].astype("category")
        df_with_cats["Assortment"] = df_with_cats["Assortment"].astype("category")

        # Act
        X_train, y_train, feature_cols, cat_features = train_ensemble.prepare_training_data(
            df_with_cats
        )

        # Assert
        # StoreType, Assortment should be categorical
        assert len(cat_features) > 0
        # Verify they are actually category dtype in X_train
        for cat_col in cat_features:
            assert X_train[cat_col].dtype.name == "category"

    def test_prepare_training_data_removes_missing_features(self, sample_features_data):
        """Test that rows with missing features are dropped."""
        # Arrange
        df_with_nans = sample_features_data.copy()
        df_with_nans.loc[0, "Year"] = np.nan
        df_with_nans.loc[1, "Month"] = np.nan

        # Act
        X_train, y_train, feature_cols, cat_features = train_ensemble.prepare_training_data(
            df_with_nans
        )

        # Assert
        # Should have dropped rows with NaNs
        assert not X_train.isna().any().any()


class TestTrainLightGBM:
    """Tests for train_lightgbm()."""

    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_train_lightgbm_basic(
        self,
        mock_lgb_dataset,
        mock_lgb_train,
        mock_log_param,
        mock_log_metric,
        sample_hyperparameters,
    ):
        """Test LightGBM training with mocked dependencies."""
        # Arrange
        X_train = pd.DataFrame({"Feature1": [1, 2, 3], "Feature2": [4, 5, 6]})
        y_train = np.array([100, 200, 300])
        cat_features = []

        # Prepare hyperparameters in expected format
        best_params = {"lightgbm": {"hyperparameters": sample_hyperparameters["lightgbm"]}}

        mock_model = Mock()
        mock_lgb_train.return_value = mock_model

        # Act
        model = train_ensemble.train_lightgbm(X_train, y_train, best_params, cat_features)

        # Assert
        mock_lgb_dataset.assert_called_once()
        mock_lgb_train.assert_called_once()
        assert model == mock_model
        assert mock_log_param.called
        assert mock_log_metric.called

    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_train_lightgbm_logs_parameters(
        self,
        mock_lgb_dataset,
        mock_lgb_train,
        mock_log_param,
        mock_log_metric,
        sample_hyperparameters,
    ):
        """Test that LightGBM parameters are logged to MLflow."""
        # Arrange
        X_train = pd.DataFrame({"Feature1": [1, 2]})
        y_train = np.array([100, 200])
        cat_features = []

        best_params = {"lightgbm": {"hyperparameters": sample_hyperparameters["lightgbm"]}}

        mock_lgb_train.return_value = Mock()

        # Act
        _ = train_ensemble.train_lightgbm(X_train, y_train, best_params, cat_features)

        # Assert
        # Check that parameters were logged
        logged_params = [call[0][0] for call in mock_log_param.call_args_list]
        assert any("lgb_" in param for param in logged_params)


class TestTrainXGBoost:
    """Tests for train_xgboost()."""

    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("xgboost.train")
    @patch("xgboost.DMatrix")
    def test_train_xgboost_basic(
        self,
        mock_xgb_dmatrix,
        mock_xgb_train,
        mock_log_param,
        mock_log_metric,
        sample_hyperparameters,
    ):
        """Test XGBoost training with mocked dependencies."""
        # Arrange
        X_train = pd.DataFrame({"Feature1": [1, 2, 3]})
        y_train = np.array([100, 200, 300])

        best_params = {"xgboost": {"hyperparameters": sample_hyperparameters["xgboost"]}}

        mock_model = Mock()
        mock_xgb_train.return_value = mock_model

        # Act
        model = train_ensemble.train_xgboost(X_train, y_train, best_params)

        # Assert
        mock_xgb_dmatrix.assert_called_once()
        mock_xgb_train.assert_called_once()
        assert model == mock_model
        assert mock_log_param.called

    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("xgboost.train")
    @patch("xgboost.DMatrix")
    def test_train_xgboost_encodes_categorical_features(
        self,
        mock_xgb_dmatrix,
        mock_xgb_train,
        mock_log_param,
        mock_log_metric,
        sample_hyperparameters,
    ):
        """Test that categorical features are encoded for XGBoost."""
        # Arrange
        X_train = pd.DataFrame(
            {"NumericFeature": [1, 2, 3], "CategoricalFeature": pd.Categorical(["a", "b", "c"])}
        )
        y_train = np.array([100, 200, 300])

        best_params = {"xgboost": {"hyperparameters": sample_hyperparameters["xgboost"]}}

        mock_xgb_train.return_value = Mock()

        # Act
        _ = train_ensemble.train_xgboost(X_train, y_train, best_params)

        # Assert
        # XGBoost should have been called (categorical encoding happens inside)
        mock_xgb_train.assert_called_once()


class TestTrainCatBoost:
    """Tests for train_catboost()."""

    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("catboost.CatBoost")
    @patch("catboost.Pool")
    def test_train_catboost_basic(
        self,
        mock_cb_pool,
        mock_cb_class,
        mock_log_param,
        mock_log_metric,
        sample_hyperparameters,
    ):
        """Test CatBoost training with mocked dependencies."""
        # Arrange
        X_train = pd.DataFrame({"Feature1": [1, 2, 3]})
        y_train = np.array([100, 200, 300])
        cat_features = []

        best_params = {"catboost": {"hyperparameters": sample_hyperparameters["catboost"]}}

        mock_model = Mock()
        mock_cb_class.return_value = mock_model

        # Act
        model = train_ensemble.train_catboost(X_train, y_train, best_params, cat_features)

        # Assert
        mock_cb_pool.assert_called_once()
        mock_cb_class.assert_called_once()
        mock_model.fit.assert_called_once()
        assert model == mock_model

    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("catboost.CatBoost")
    @patch("catboost.Pool")
    def test_train_catboost_logs_parameters(
        self,
        mock_cb_pool,
        mock_cb_class,
        mock_log_param,
        mock_log_metric,
        sample_hyperparameters,
    ):
        """Test that CatBoost parameters are logged to MLflow."""
        # Arrange
        X_train = pd.DataFrame({"Feature1": [1, 2]})
        y_train = np.array([100, 200])
        cat_features = []

        best_params = {"catboost": {"hyperparameters": sample_hyperparameters["catboost"]}}

        mock_model = Mock()
        mock_cb_class.return_value = mock_model

        # Act
        _ = train_ensemble.train_catboost(X_train, y_train, best_params, cat_features)

        # Assert
        logged_params = [call[0][0] for call in mock_log_param.call_args_list]
        assert any("cb_" in param for param in logged_params)


class TestMainTrainingPipeline:
    """Tests for the main() training pipeline."""

    @patch("src.models.train_ensemble.register_ensemble_model")
    @patch("src.models.train_ensemble.create_ensemble")
    @patch("src.models.train_ensemble.train_catboost")
    @patch("src.models.train_ensemble.train_xgboost")
    @patch("src.models.train_ensemble.train_lightgbm")
    @patch("src.models.train_ensemble.prepare_training_data")
    @patch("src.models.train_ensemble.load_best_hyperparameters")
    @patch("src.models.train_ensemble.load_training_data")
    @patch("src.models.train_ensemble.setup_mlflow")
    @patch("mlflow.start_run")
    @patch("mlflow.log_param")
    def test_main_pipeline_basic(
        self,
        mock_log_param,
        mock_start_run,
        mock_setup_mlflow,
        mock_load_data,
        mock_load_params,
        mock_prepare_data,
        mock_train_lgb,
        mock_train_xgb,
        mock_train_cb,
        mock_create_ensemble,
        mock_register,
        sample_features_data,
        sample_hyperparameters,
        tmp_path,
    ):
        """Test main training pipeline orchestration."""
        # Arrange
        mock_setup_mlflow.return_value = "test-experiment-id"

        # Mock MLflow run context
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = Mock(return_value=False)

        # Mock data loading
        train_df = sample_features_data.iloc[:100]
        holdout_df = sample_features_data.iloc[100:]
        mock_load_data.return_value = (train_df, holdout_df)

        # Mock hyperparameters with proper structure
        best_params = {
            "metadata": {"best_model": "xgboost", "best_rmspe": 0.09},
            "lightgbm": {"hyperparameters": sample_hyperparameters["lightgbm"]},
            "xgboost": {"hyperparameters": sample_hyperparameters["xgboost"]},
            "catboost": {"hyperparameters": sample_hyperparameters["catboost"]},
        }
        mock_load_params.return_value = best_params

        # Mock prepared data
        X_train = sample_features_data[["DayOfWeek", "Promo"]]
        y_train = np.random.rand(len(X_train)) * 1000
        feature_cols = ["DayOfWeek", "Promo"]
        cat_features = []
        mock_prepare_data.return_value = (X_train, y_train, feature_cols, cat_features)

        # Mock trained models
        mock_lgb_model = Mock()
        mock_xgb_model = Mock()
        mock_cb_model = Mock()
        mock_train_lgb.return_value = mock_lgb_model
        mock_train_xgb.return_value = mock_xgb_model
        mock_train_cb.return_value = mock_cb_model

        # Mock ensemble
        mock_ensemble = Mock()
        mock_create_ensemble.return_value = mock_ensemble

        # Mock registration
        mock_register.return_value = "7"

        # Create temporary config file
        config_path = tmp_path / "best_hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(best_params, f)

        data_path = tmp_path / "features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act
        version = train_ensemble.main(
            data_path=str(data_path),
            config_path=str(config_path),
            model_name="rossmann-ensemble",
        )

        # Assert
        mock_setup_mlflow.assert_called_once()
        mock_load_data.assert_called_once()
        mock_load_params.assert_called_once()
        mock_prepare_data.assert_called_once()
        mock_train_lgb.assert_called_once()
        mock_train_xgb.assert_called_once()
        mock_train_cb.assert_called_once()
        mock_create_ensemble.assert_called_once()
        mock_register.assert_called_once()
        assert version == "7"

    @patch("src.models.train_ensemble.register_ensemble_model")
    @patch("src.models.train_ensemble.create_ensemble")
    @patch("src.models.train_ensemble.train_catboost")
    @patch("src.models.train_ensemble.train_xgboost")
    @patch("src.models.train_ensemble.train_lightgbm")
    @patch("src.models.train_ensemble.prepare_training_data")
    @patch("src.models.train_ensemble.load_best_hyperparameters")
    @patch("src.models.train_ensemble.load_training_data")
    @patch("src.models.train_ensemble.setup_mlflow")
    @patch("mlflow.start_run")
    @patch("mlflow.log_param")
    def test_main_pipeline_uses_custom_weights(
        self,
        mock_log_param,
        mock_start_run,
        mock_setup_mlflow,
        mock_load_data,
        mock_load_params,
        mock_prepare_data,
        mock_train_lgb,
        mock_train_xgb,
        mock_train_cb,
        mock_create_ensemble,
        mock_register,
        sample_features_data,
        sample_hyperparameters,
        tmp_path,
    ):
        """Test that custom ensemble weights are used."""
        # Arrange - similar setup as above
        mock_setup_mlflow.return_value = "test-experiment-id"

        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = Mock(return_value=False)

        train_df = sample_features_data.iloc[:100]
        holdout_df = sample_features_data.iloc[100:]
        mock_load_data.return_value = (train_df, holdout_df)

        best_params = {
            "metadata": {"best_model": "xgboost", "best_rmspe": 0.09},
            "lightgbm": {"hyperparameters": sample_hyperparameters["lightgbm"]},
            "xgboost": {"hyperparameters": sample_hyperparameters["xgboost"]},
            "catboost": {"hyperparameters": sample_hyperparameters["catboost"]},
        }
        mock_load_params.return_value = best_params

        X_train = sample_features_data[["DayOfWeek"]]
        y_train = np.random.rand(len(X_train)) * 1000
        feature_cols = ["DayOfWeek"]
        cat_features = []
        mock_prepare_data.return_value = (X_train, y_train, feature_cols, cat_features)

        mock_train_lgb.return_value = Mock()
        mock_train_xgb.return_value = Mock()
        mock_train_cb.return_value = Mock()

        mock_ensemble = Mock()
        mock_create_ensemble.return_value = mock_ensemble

        mock_register.return_value = "7"

        config_path = tmp_path / "best_hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(best_params, f)

        data_path = tmp_path / "features.parquet"
        sample_features_data.to_parquet(data_path)

        custom_weights = {"lightgbm": 0.5, "xgboost": 0.3, "catboost": 0.2}

        # Act
        _ = train_ensemble.main(
            data_path=str(data_path),
            config_path=str(config_path),
            ensemble_weights=custom_weights,
        )

        # Assert
        # Verify create_ensemble was called with custom weights
        call_kwargs = mock_create_ensemble.call_args[1]
        assert call_kwargs["weights"] == custom_weights


class TestTrainingIntegration:
    """Integration tests for training workflow."""

    @patch("src.models.train_ensemble.register_ensemble_model")
    @patch("src.models.train_ensemble.create_ensemble")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("lightgbm.train")
    @patch("xgboost.train")
    @patch("catboost.CatBoost")
    def test_all_models_trained_in_sequence(
        self,
        mock_cb_class,
        mock_xgb_train,
        mock_lgb_train,
        mock_log_param,
        mock_log_metric,
        mock_create_ensemble,
        mock_register,
        sample_features_data,
        sample_hyperparameters,
    ):
        """Test that all three models are trained in the correct sequence."""
        # Arrange
        X_train = sample_features_data[["DayOfWeek", "Promo", "Year"]]
        y_train = np.random.rand(len(X_train)) * 1000
        cat_features = []

        best_params = {
            "lightgbm": {"hyperparameters": sample_hyperparameters["lightgbm"]},
            "xgboost": {"hyperparameters": sample_hyperparameters["xgboost"]},
            "catboost": {"hyperparameters": sample_hyperparameters["catboost"]},
        }

        # Mock model returns
        mock_lgb_model = Mock()
        mock_xgb_model = Mock()
        mock_cb_model = Mock()

        mock_lgb_train.return_value = mock_lgb_model
        mock_xgb_train.return_value = mock_xgb_model
        mock_cb_class.return_value = mock_cb_model

        # Act
        lgb_result = train_ensemble.train_lightgbm(X_train, y_train, best_params, cat_features)
        xgb_result = train_ensemble.train_xgboost(X_train, y_train, best_params)
        cb_result = train_ensemble.train_catboost(X_train, y_train, best_params, cat_features)

        # Assert
        assert lgb_result == mock_lgb_model
        assert xgb_result == mock_xgb_model
        assert cb_result == mock_cb_model
        mock_lgb_train.assert_called_once()
        mock_xgb_train.assert_called_once()
        mock_cb_class.assert_called_once()
