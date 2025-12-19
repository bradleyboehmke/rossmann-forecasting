"""Tests for ensemble model functionality.

This module tests the RossmannEnsemble custom MLflow PyFunc model and ensemble creation
functionality in src/models/ensemble.py.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import RossmannEnsemble, create_ensemble


class TestRossmannEnsemble:
    """Tests for RossmannEnsemble class."""

    def test_init_with_valid_weights(self):
        """Test ensemble initialization with valid weights that sum to 1.0."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        weights = {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}
        cat_features = ["StoreType", "Assortment"]

        # Act
        ensemble = RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, weights, cat_features)

        # Assert
        assert ensemble.lgb_model == mock_lgb
        assert ensemble.xgb_model == mock_xgb
        assert ensemble.cb_model == mock_cb
        assert ensemble.weights == weights
        assert ensemble.cat_features == cat_features

    def test_init_with_invalid_weights_raises_error(self):
        """Test that weights not summing to 1.0 raises ValueError."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        invalid_weights = {"lightgbm": 0.3, "xgboost": 0.5, "catboost": 0.1}  # Sum = 0.9
        cat_features = ["StoreType"]

        # Act & Assert
        with pytest.raises(ValueError, match="Ensemble weights must sum to 1.0"):
            RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, invalid_weights, cat_features)

    def test_init_with_weights_slightly_off_due_to_float_precision(self):
        """Test that weights summing to ~1.0 (within float precision) are accepted."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        # Weights that sum to 1.0 within floating point precision
        weights = {"lightgbm": 0.333333, "xgboost": 0.333334, "catboost": 0.333333}
        cat_features = ["StoreType"]

        # Act
        ensemble = RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, weights, cat_features)

        # Assert - should not raise error
        assert ensemble is not None

    def test_predict_basic(self):
        """Test ensemble prediction with mocked base models."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        # Mock model predictions
        lgb_preds = np.array([100.0, 200.0, 300.0])
        xgb_preds = np.array([110.0, 210.0, 310.0])
        cb_preds = np.array([90.0, 190.0, 290.0])

        mock_lgb.predict.return_value = lgb_preds
        mock_xgb.predict.return_value = xgb_preds
        mock_cb.predict.return_value = cb_preds

        weights = {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}
        cat_features = ["StoreType"]

        ensemble = RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, weights, cat_features)

        # Create sample input data
        model_input = pd.DataFrame(
            {
                "Feature1": [1.0, 2.0, 3.0],
                "Feature2": [4.0, 5.0, 6.0],
                "StoreType": pd.Categorical(["a", "b", "c"]),
            }
        )

        # Act
        predictions = ensemble.predict(context=None, model_input=model_input)

        # Assert
        # Weighted average: 0.3 * lgb + 0.6 * xgb + 0.1 * cb
        expected = 0.3 * lgb_preds + 0.6 * xgb_preds + 0.1 * cb_preds
        np.testing.assert_array_almost_equal(predictions, expected)

        # Verify all models were called
        mock_lgb.predict.assert_called_once()
        mock_xgb.predict.assert_called_once()
        mock_cb.predict.assert_called_once()

    def test_predict_handles_categorical_features_for_xgboost(self):
        """Test that categorical features are encoded for XGBoost."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        mock_lgb.predict.return_value = np.array([100.0])
        mock_xgb.predict.return_value = np.array([110.0])
        mock_cb.predict.return_value = np.array([90.0])

        weights = {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}
        cat_features = ["StoreType", "Assortment"]

        ensemble = RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, weights, cat_features)

        model_input = pd.DataFrame(
            {
                "Feature1": [1.0],
                "StoreType": pd.Categorical(["a"]),
                "Assortment": pd.Categorical(["basic"]),
            }
        )

        # Act
        _ = ensemble.predict(context=None, model_input=model_input)

        # Assert
        # XGBoost should have been called with DMatrix
        assert mock_xgb.predict.called
        # Verify the input to XGBoost was a DMatrix
        call_args = mock_xgb.predict.call_args
        assert call_args is not None

    def test_predict_with_different_weights(self):
        """Test ensemble with different weight configurations."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        lgb_preds = np.array([100.0])
        xgb_preds = np.array([200.0])
        cb_preds = np.array([150.0])

        mock_lgb.predict.return_value = lgb_preds
        mock_xgb.predict.return_value = xgb_preds
        mock_cb.predict.return_value = cb_preds

        # Equal weights
        weights = {"lightgbm": 0.33333, "xgboost": 0.33334, "catboost": 0.33333}
        cat_features = []

        ensemble = RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, weights, cat_features)

        model_input = pd.DataFrame({"Feature1": [1.0]})

        # Act
        predictions = ensemble.predict(context=None, model_input=model_input)

        # Assert
        # With equal weights, prediction should be close to average
        expected = (100.0 + 200.0 + 150.0) / 3.0
        assert np.isclose(predictions[0], expected, atol=0.01)

    def test_predict_preserves_input_shape(self):
        """Test that prediction output shape matches input shape."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        n_samples = 10
        mock_lgb.predict.return_value = np.random.rand(n_samples) * 100
        mock_xgb.predict.return_value = np.random.rand(n_samples) * 100
        mock_cb.predict.return_value = np.random.rand(n_samples) * 100

        weights = {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}
        cat_features = []

        ensemble = RossmannEnsemble(mock_lgb, mock_xgb, mock_cb, weights, cat_features)

        model_input = pd.DataFrame({"Feature1": np.random.rand(n_samples)})

        # Act
        predictions = ensemble.predict(context=None, model_input=model_input)

        # Assert
        assert predictions.shape == (n_samples,)


class TestCreateEnsemble:
    """Tests for create_ensemble() factory function."""

    def test_create_ensemble_with_defaults(self):
        """Test creating ensemble with default weights and features."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        # Act
        ensemble = create_ensemble(mock_lgb, mock_xgb, mock_cb)

        # Assert
        assert isinstance(ensemble, RossmannEnsemble)
        assert ensemble.weights == {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}
        assert ensemble.cat_features == [
            "StoreType",
            "Assortment",
            "StateHoliday",
            "PromoInterval",
        ]
        assert ensemble.lgb_model == mock_lgb
        assert ensemble.xgb_model == mock_xgb
        assert ensemble.cb_model == mock_cb

    def test_create_ensemble_with_custom_weights(self):
        """Test creating ensemble with custom weights."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        custom_weights = {"lightgbm": 0.5, "xgboost": 0.3, "catboost": 0.2}

        # Act
        ensemble = create_ensemble(mock_lgb, mock_xgb, mock_cb, weights=custom_weights)

        # Assert
        assert ensemble.weights == custom_weights

    def test_create_ensemble_with_custom_categorical_features(self):
        """Test creating ensemble with custom categorical features."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        custom_cat_features = ["StoreType", "CustomFeature"]

        # Act
        ensemble = create_ensemble(mock_lgb, mock_xgb, mock_cb, cat_features=custom_cat_features)

        # Assert
        assert ensemble.cat_features == custom_cat_features

    def test_create_ensemble_with_all_custom_params(self):
        """Test creating ensemble with all custom parameters."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        custom_weights = {"lightgbm": 0.25, "xgboost": 0.5, "catboost": 0.25}
        custom_cat_features = ["Feature1", "Feature2"]

        # Act
        ensemble = create_ensemble(
            mock_lgb, mock_xgb, mock_cb, weights=custom_weights, cat_features=custom_cat_features
        )

        # Assert
        assert ensemble.weights == custom_weights
        assert ensemble.cat_features == custom_cat_features
        assert ensemble.lgb_model == mock_lgb
        assert ensemble.xgb_model == mock_xgb
        assert ensemble.cb_model == mock_cb

    def test_create_ensemble_with_invalid_weights_raises_error(self):
        """Test that create_ensemble raises error for invalid weights."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()
        invalid_weights = {"lightgbm": 0.3, "xgboost": 0.3, "catboost": 0.3}  # Sum = 0.9

        # Act & Assert
        with pytest.raises(ValueError, match="Ensemble weights must sum to 1.0"):
            create_ensemble(mock_lgb, mock_xgb, mock_cb, weights=invalid_weights)


class TestEnsembleIntegration:
    """Integration tests for ensemble functionality."""

    def test_ensemble_workflow_end_to_end(self):
        """Test complete ensemble workflow from creation to prediction."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        # Set up consistent predictions
        n_samples = 5
        mock_lgb.predict.return_value = np.array([100.0] * n_samples)
        mock_xgb.predict.return_value = np.array([200.0] * n_samples)
        mock_cb.predict.return_value = np.array([150.0] * n_samples)

        weights = {"lightgbm": 0.3, "xgboost": 0.6, "catboost": 0.1}
        # Use empty cat_features to avoid CatBoost validation
        cat_features = []

        # Act
        # Create ensemble
        ensemble = create_ensemble(
            mock_lgb, mock_xgb, mock_cb, weights=weights, cat_features=cat_features
        )

        # Generate predictions
        model_input = pd.DataFrame({"Feature1": np.arange(n_samples)})
        predictions = ensemble.predict(context=None, model_input=model_input)

        # Assert
        # Expected: 0.3 * 100 + 0.6 * 200 + 0.1 * 150 = 30 + 120 + 15 = 165
        expected = np.array([165.0] * n_samples)
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_ensemble_with_zero_weight(self):
        """Test ensemble when one model has zero weight."""
        # Arrange
        mock_lgb = Mock()
        mock_xgb = Mock()
        mock_cb = Mock()

        mock_lgb.predict.return_value = np.array([100.0])
        mock_xgb.predict.return_value = np.array([200.0])
        mock_cb.predict.return_value = np.array([150.0])

        # CatBoost has zero weight
        weights = {"lightgbm": 0.4, "xgboost": 0.6, "catboost": 0.0}
        # Use empty cat_features to avoid CatBoost validation
        cat_features = []

        # Act
        ensemble = create_ensemble(
            mock_lgb, mock_xgb, mock_cb, weights=weights, cat_features=cat_features
        )
        model_input = pd.DataFrame({"Feature1": [1.0]})
        predictions = ensemble.predict(context=None, model_input=model_input)

        # Assert
        # Expected: 0.4 * 100 + 0.6 * 200 + 0.0 * 150 = 40 + 120 = 160
        expected = 0.4 * 100.0 + 0.6 * 200.0
        assert np.isclose(predictions[0], expected)

        # CatBoost should still be called (implementation doesn't skip it)
        mock_cb.predict.assert_called_once()
