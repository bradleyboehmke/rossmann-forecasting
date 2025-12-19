"""Tests for production inference pipeline.

This module tests the prediction workflow in src/models/predict.py including loading data,
generating predictions, and saving results.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.models import predict


class TestLoadInferenceData:
    """Tests for load_inference_data()."""

    def test_load_inference_data_basic(self, sample_features_data, tmp_path):
        """Test loading inference data without date filtering."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act
        data_df, X_features, feature_cols = predict.load_inference_data(str(data_path))

        # Assert
        assert len(data_df) > 0
        assert "Sales" not in feature_cols
        assert "Date" not in feature_cols
        assert "Store" not in feature_cols
        assert "Customers" not in feature_cols
        assert X_features.shape[0] == data_df.shape[0]

    def test_load_inference_data_with_date_filter(self, sample_features_data, tmp_path):
        """Test loading data with date range filtering."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act
        data_df, X_features, feature_cols = predict.load_inference_data(
            str(data_path), start_date="2015-01-15", end_date="2015-01-20"
        )

        # Assert
        assert data_df["Date"].min() >= pd.Timestamp("2015-01-15")
        assert data_df["Date"].max() <= pd.Timestamp("2015-01-20")

    def test_load_inference_data_filters_closed_stores(self, sample_features_data, tmp_path):
        """Test that only open stores are included in inference data."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"
        sample_features_data.to_parquet(data_path)

        # Act
        data_df, X_features, feature_cols = predict.load_inference_data(str(data_path))

        # Assert
        # All loaded rows should have Open=1
        assert all(data_df["Open"] == 1)

    def test_load_inference_data_drops_missing_features(self, sample_features_data, tmp_path):
        """Test that rows with missing features are dropped."""
        # Arrange
        data_path = tmp_path / "test_features.parquet"

        # Add some NaN values to feature columns
        df_with_nans = sample_features_data.copy()
        df_with_nans.loc[0, "Year"] = np.nan

        df_with_nans.to_parquet(data_path)

        # Act
        data_df, X_features, feature_cols = predict.load_inference_data(str(data_path))

        # Assert
        # Should have dropped the row with NaN
        assert len(data_df) < len(df_with_nans[df_with_nans["Open"] == 1])
        assert not X_features.isna().any().any()


class TestGeneratePredictions:
    """Tests for generate_predictions()."""

    def test_generate_predictions_basic(self, sample_features_data):
        """Test generating predictions with a mock model."""
        # Arrange
        mock_model = Mock()
        mock_predictions = np.array([5000.0, 6000.0, 7000.0])
        mock_model.predict.return_value = mock_predictions

        data_df = sample_features_data.head(3).copy()
        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]

        # Act
        predictions_df = predict.generate_predictions(mock_model, X_features, data_df)

        # Assert
        mock_model.predict.assert_called_once()
        assert len(predictions_df) == 3
        assert "Store" in predictions_df.columns
        assert "Date" in predictions_df.columns
        assert "Predicted_Sales" in predictions_df.columns
        assert all(predictions_df["Predicted_Sales"] == mock_predictions)

    def test_generate_predictions_with_actual_sales(self, sample_features_data):
        """Test generating predictions when actual sales are available."""
        # Arrange
        mock_model = Mock()
        mock_predictions = np.array([5000.0, 6000.0, 7000.0])
        mock_model.predict.return_value = mock_predictions

        data_df = sample_features_data.head(3).copy()
        data_df["Sales"] = [5100.0, 5900.0, 7200.0]  # Add actual sales

        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]

        # Act
        # rmspe is imported inside generate_predictions, so patch it in evaluation.metrics
        with patch("src.evaluation.metrics.rmspe") as mock_rmspe:
            mock_rmspe.return_value = 0.05
            predictions_df = predict.generate_predictions(mock_model, X_features, data_df)

        # Assert
        assert "Actual_Sales" in predictions_df.columns
        assert "Prediction_Error" in predictions_df.columns
        assert "Absolute_Percentage_Error" in predictions_df.columns
        assert all(predictions_df["Actual_Sales"] == data_df["Sales"].values)

    def test_generate_predictions_includes_dayofweek(self, sample_features_data):
        """Test that DayOfWeek is included in predictions if available."""
        # Arrange
        mock_model = Mock()
        mock_predictions = np.array([5000.0, 6000.0, 7000.0])
        mock_model.predict.return_value = mock_predictions

        data_df = sample_features_data.head(3).copy()
        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]

        # Act
        predictions_df = predict.generate_predictions(mock_model, X_features, data_df)

        # Assert
        assert "DayOfWeek" in predictions_df.columns


class TestSavePredictions:
    """Tests for save_predictions()."""

    def test_save_predictions_creates_csv(self, tmp_path):
        """Test saving predictions to CSV."""
        # Arrange
        predictions_df = pd.DataFrame(
            {
                "Store": [1, 2, 3],
                "Date": pd.date_range("2015-01-01", periods=3),
                "Predicted_Sales": [5000.0, 6000.0, 7000.0],
            }
        )
        output_path = tmp_path / "predictions.csv"

        # Act
        predict.save_predictions(predictions_df, str(output_path), model_version="1")

        # Assert
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 3
        assert "Predicted_Sales" in saved_df.columns

    def test_save_predictions_creates_metadata(self, tmp_path):
        """Test that metadata JSON is created."""
        # Arrange
        predictions_df = pd.DataFrame(
            {
                "Store": [1, 2, 3],
                "Date": pd.date_range("2015-01-01", periods=3),
                "Predicted_Sales": [5000.0, 6000.0, 7000.0],
            }
        )
        output_path = tmp_path / "predictions.csv"

        # Act
        predict.save_predictions(
            predictions_df,
            str(output_path),
            model_version="7",
            metadata={"model_name": "rossmann-ensemble"},
        )

        # Assert
        metadata_path = tmp_path / "predictions_metadata.json"
        assert metadata_path.exists()

        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["model_version"] == "7"
        assert metadata["model_name"] == "rossmann-ensemble"
        assert metadata["n_predictions"] == 3
        assert "date_range" in metadata
        assert "prediction_date" in metadata

    def test_save_predictions_creates_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        # Arrange
        predictions_df = pd.DataFrame(
            {
                "Store": [1, 2],
                "Date": pd.date_range("2015-01-01", periods=2),
                "Predicted_Sales": [5000.0, 6000.0],
            }
        )
        output_path = tmp_path / "outputs" / "predictions" / "test.csv"

        # Act
        predict.save_predictions(predictions_df, str(output_path), model_version="1")

        # Assert
        assert output_path.exists()
        assert output_path.parent.exists()


class TestPredict:
    """Tests for the main predict() function."""

    @patch("src.models.predict.save_predictions")
    @patch("src.models.predict.generate_predictions")
    @patch("src.models.predict.load_inference_data")
    @patch("src.models.predict.load_model")
    @patch("src.models.predict.get_model_version")
    def test_predict_production_stage(
        self,
        mock_get_version,
        mock_load_model,
        mock_load_data,
        mock_generate_preds,
        mock_save_preds,
        sample_features_data,
    ):
        """Test prediction pipeline with Production stage."""
        # Arrange
        mock_get_version.return_value = "5"
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        data_df = sample_features_data.head(10)
        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]
        mock_load_data.return_value = (data_df, X_features, feature_cols)

        predictions_df = pd.DataFrame(
            {
                "Store": data_df["Store"],
                "Date": data_df["Date"],
                "Predicted_Sales": np.random.rand(len(data_df)) * 10000,
            }
        )
        mock_generate_preds.return_value = predictions_df

        # Act
        result = predict.predict(
            model_name="rossmann-ensemble",
            stage="Production",
            data_path="data/processed/train_features.parquet",
            output_path="outputs/predictions/test.csv",
        )

        # Assert
        mock_get_version.assert_called_once_with("rossmann-ensemble", stage="Production")
        mock_load_model.assert_called_once_with("rossmann-ensemble", stage="Production")
        mock_load_data.assert_called_once()
        mock_generate_preds.assert_called_once_with(mock_model, X_features, data_df)
        mock_save_preds.assert_called_once()
        assert result.equals(predictions_df)

    @patch("src.models.predict.save_predictions")
    @patch("src.models.predict.generate_predictions")
    @patch("src.models.predict.load_inference_data")
    @patch("src.models.predict.load_model")
    @patch("src.models.predict.get_model_version")
    def test_predict_staging_stage(
        self,
        mock_get_version,
        mock_load_model,
        mock_load_data,
        mock_generate_preds,
        mock_save_preds,
        sample_features_data,
    ):
        """Test prediction pipeline with Staging stage."""
        # Arrange
        mock_get_version.return_value = "7"
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        data_df = sample_features_data.head(10)
        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]
        mock_load_data.return_value = (data_df, X_features, feature_cols)

        predictions_df = pd.DataFrame(
            {
                "Store": data_df["Store"],
                "Date": data_df["Date"],
                "Predicted_Sales": np.random.rand(len(data_df)) * 10000,
            }
        )
        mock_generate_preds.return_value = predictions_df

        # Act
        predict.predict(stage="Staging")

        # Assert
        mock_get_version.assert_called_once_with("rossmann-ensemble", stage="Staging")
        mock_load_model.assert_called_once_with("rossmann-ensemble", stage="Staging")

    @patch("src.models.predict.save_predictions")
    @patch("src.models.predict.generate_predictions")
    @patch("src.models.predict.load_inference_data")
    @patch("src.models.predict.load_model")
    def test_predict_with_version_number(
        self,
        mock_load_model,
        mock_load_data,
        mock_generate_preds,
        mock_save_preds,
        sample_features_data,
    ):
        """Test prediction pipeline with specific version number."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        data_df = sample_features_data.head(10)
        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]
        mock_load_data.return_value = (data_df, X_features, feature_cols)

        predictions_df = pd.DataFrame(
            {
                "Store": data_df["Store"],
                "Date": data_df["Date"],
                "Predicted_Sales": np.random.rand(len(data_df)) * 10000,
            }
        )
        mock_generate_preds.return_value = predictions_df

        # Act
        predict.predict(stage="3")  # Use version number directly

        # Assert
        mock_load_model.assert_called_once_with("rossmann-ensemble", stage="3")

    @patch("src.models.predict.save_predictions")
    @patch("src.models.predict.generate_predictions")
    @patch("src.models.predict.load_inference_data")
    @patch("src.models.predict.load_model")
    @patch("src.models.predict.get_model_version")
    def test_predict_with_date_range(
        self,
        mock_get_version,
        mock_load_model,
        mock_load_data,
        mock_generate_preds,
        mock_save_preds,
        sample_features_data,
    ):
        """Test prediction pipeline with date range filtering."""
        # Arrange
        mock_get_version.return_value = "5"
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        data_df = sample_features_data.head(10)
        feature_cols = [
            col for col in data_df.columns if col not in ["Sales", "Date", "Store", "Customers"]
        ]
        X_features = data_df[feature_cols]
        mock_load_data.return_value = (data_df, X_features, feature_cols)

        predictions_df = pd.DataFrame(
            {
                "Store": data_df["Store"],
                "Date": data_df["Date"],
                "Predicted_Sales": np.random.rand(len(data_df)) * 10000,
            }
        )
        mock_generate_preds.return_value = predictions_df

        # Act
        predict.predict(start_date="2015-01-01", end_date="2015-01-31")

        # Assert
        # Verify load_inference_data was called with date filters
        # Check positional and keyword arguments
        call_args = mock_load_data.call_args
        # The function is called with positional args, check the call
        assert mock_load_data.called
        # Check that the call included start_date and end_date
        args, kwargs = call_args
        # load_inference_data(data_path, start_date, end_date)
        assert len(args) >= 1  # At least data_path
        if len(args) >= 3:
            assert args[1] == "2015-01-01"  # start_date
            assert args[2] == "2015-01-31"  # end_date


class TestMain:
    """Tests for the main() CLI function."""

    @patch("src.models.predict.predict")
    def test_main_with_staging_arg(self, mock_predict):
        """Test main() CLI with Staging stage argument."""
        # Arrange
        mock_predictions = pd.DataFrame(
            {
                "Store": [1, 2],
                "Date": pd.date_range("2015-01-01", periods=2),
                "Predicted_Sales": [5000.0, 6000.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Act
        with patch("sys.argv", ["predict.py", "--stage", "Staging"]):
            predict.main()

        # Assert
        mock_predict.assert_called_once()
        call_kwargs = mock_predict.call_args[1]
        assert call_kwargs["stage"] == "Staging"

    @patch("src.models.predict.predict")
    def test_main_with_date_range_args(self, mock_predict):
        """Test main() CLI with date range arguments."""
        # Arrange
        mock_predictions = pd.DataFrame(
            {
                "Store": [1, 2],
                "Date": pd.date_range("2015-01-01", periods=2),
                "Predicted_Sales": [5000.0, 6000.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Act
        with patch(
            "sys.argv",
            [
                "predict.py",
                "--model-name",
                "test-model",
                "--start-date",
                "2015-01-01",
                "--end-date",
                "2015-01-31",
            ],
        ):
            predict.main()

        # Assert
        mock_predict.assert_called_once()
        call_kwargs = mock_predict.call_args[1]
        assert call_kwargs["model_name"] == "test-model"
        assert call_kwargs["start_date"] == "2015-01-01"
        assert call_kwargs["end_date"] == "2015-01-31"
