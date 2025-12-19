"""Tests for MLflow Model Registry utilities.

This module tests model registration, promotion, loading, and metadata retrieval functionality in
src/models/model_registry.py.
"""

from unittest.mock import Mock, call, patch

from src.models import model_registry


class TestGetMLflowClient:
    """Tests for get_mlflow_client()."""

    @patch("src.models.model_registry.MlflowClient")
    def test_get_mlflow_client_returns_client(self, mock_client_class):
        """Test that get_mlflow_client returns an MlflowClient instance."""
        # Arrange
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Act
        client = model_registry.get_mlflow_client()

        # Assert
        mock_client_class.assert_called_once()
        assert client == mock_client_instance


class TestRegisterEnsembleModel:
    """Tests for register_ensemble_model()."""

    @patch("mlflow.pyfunc.log_model")
    @patch("src.models.model_registry.get_mlflow_client")
    def test_register_ensemble_model_basic(self, mock_get_client, mock_log_model):
        """Test basic model registration without description."""
        # Arrange
        mock_ensemble = Mock()
        mock_model_info = Mock()
        mock_model_info.registered_model_version = "1"
        mock_log_model.return_value = mock_model_info

        # Act
        version = model_registry.register_ensemble_model(
            ensemble_model=mock_ensemble, model_name="rossmann-ensemble"
        )

        # Assert
        mock_log_model.assert_called_once_with(
            artifact_path="ensemble_model",
            python_model=mock_ensemble,
            registered_model_name="rossmann-ensemble",
            conda_env=None,
            signature=None,
            input_example=None,
        )
        assert version == "1"

    @patch("mlflow.pyfunc.log_model")
    @patch("src.models.model_registry.get_mlflow_client")
    def test_register_ensemble_model_with_description(self, mock_get_client, mock_log_model):
        """Test model registration with description update."""
        # Arrange
        mock_ensemble = Mock()
        mock_model_info = Mock()
        mock_model_info.registered_model_version = "2"
        mock_log_model.return_value = mock_model_info

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.register_ensemble_model(
            ensemble_model=mock_ensemble,
            model_name="rossmann-ensemble",
            description="Test description",
        )

        # Assert
        mock_client.update_model_version.assert_called_once_with(
            name="rossmann-ensemble", version="2", description="Test description"
        )
        assert version == "2"

    @patch("mlflow.pyfunc.log_model")
    @patch("src.models.model_registry.get_mlflow_client")
    def test_register_ensemble_model_with_all_params(self, mock_get_client, mock_log_model):
        """Test model registration with all optional parameters."""
        # Arrange
        mock_ensemble = Mock()
        mock_conda_env = {"python": "3.10"}
        mock_signature = Mock()
        mock_input_example = Mock()

        mock_model_info = Mock()
        mock_model_info.registered_model_version = "3"
        mock_log_model.return_value = mock_model_info

        # Act
        version = model_registry.register_ensemble_model(
            ensemble_model=mock_ensemble,
            model_name="rossmann-ensemble",
            conda_env=mock_conda_env,
            signature=mock_signature,
            input_example=mock_input_example,
            description="Full registration test",
        )

        # Assert
        mock_log_model.assert_called_once_with(
            artifact_path="ensemble_model",
            python_model=mock_ensemble,
            registered_model_name="rossmann-ensemble",
            conda_env=mock_conda_env,
            signature=mock_signature,
            input_example=mock_input_example,
        )
        assert version == "3"


class TestPromoteModel:
    """Tests for promote_model()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_to_staging(self, mock_get_client):
        """Test promoting a model to Staging stage."""
        # Arrange
        mock_client = Mock()
        mock_client.get_latest_versions.return_value = []  # No existing models
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(model_name="rossmann-ensemble", version="1", stage="Staging")

        # Assert
        mock_client.get_latest_versions.assert_called_once_with(
            "rossmann-ensemble", stages=["Staging"]
        )
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="rossmann-ensemble", version="1", stage="Staging"
        )

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_to_production_with_archival(self, mock_get_client):
        """Test promoting to Production archives existing Production model."""
        # Arrange
        mock_client = Mock()

        # Existing Production model
        mock_existing = Mock()
        mock_existing.version = "5"
        mock_client.get_latest_versions.return_value = [mock_existing]
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(
            model_name="rossmann-ensemble", version="7", stage="Production", archive_existing=True
        )

        # Assert
        # Should archive existing model first
        assert mock_client.transition_model_version_stage.call_count == 2
        calls = mock_client.transition_model_version_stage.call_args_list

        # First call: archive existing
        assert calls[0] == call(name="rossmann-ensemble", version="5", stage="Archived")

        # Second call: promote new model
        assert calls[1] == call(name="rossmann-ensemble", version="7", stage="Production")

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_without_archival(self, mock_get_client):
        """Test promoting without archiving existing models."""
        # Arrange
        mock_client = Mock()
        mock_existing = Mock()
        mock_existing.version = "3"
        mock_client.get_latest_versions.return_value = [mock_existing]
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(
            model_name="rossmann-ensemble",
            version="4",
            stage="Production",
            archive_existing=False,
        )

        # Assert
        # Should only promote, not archive
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="rossmann-ensemble", version="4", stage="Production"
        )

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_to_archived(self, mock_get_client):
        """Test promoting directly to Archived stage."""
        # Arrange
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(model_name="rossmann-ensemble", version="2", stage="Archived")

        # Assert
        # Should not try to archive existing Archived models
        mock_client.get_latest_versions.assert_not_called()
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="rossmann-ensemble", version="2", stage="Archived"
        )


class TestGetModelVersion:
    """Tests for get_model_version()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_version_production(self, mock_get_client, mock_model_version):
        """Test retrieving Production model version."""
        # Arrange
        mock_client = Mock()
        mock_model_version.current_stage = "Production"
        mock_model_version.version = "5"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.get_model_version("rossmann-ensemble", stage="Production")

        # Assert
        mock_client.get_latest_versions.assert_called_once_with(
            "rossmann-ensemble", stages=["Production"]
        )
        assert version == "5"

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_version_staging(self, mock_get_client, mock_model_version):
        """Test retrieving Staging model version."""
        # Arrange
        mock_client = Mock()
        mock_model_version.current_stage = "Staging"
        mock_model_version.version = "7"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.get_model_version("rossmann-ensemble", stage="Staging")

        # Assert
        mock_client.get_latest_versions.assert_called_once_with(
            "rossmann-ensemble", stages=["Staging"]
        )
        assert version == "7"

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_version_no_model_found(self, mock_get_client):
        """Test get_model_version returns None when no model in stage."""
        # Arrange
        mock_client = Mock()
        mock_client.get_latest_versions.return_value = []  # No models
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.get_model_version("rossmann-ensemble", stage="Production")

        # Assert
        assert version is None


class TestLoadModel:
    """Tests for load_model()."""

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_production_stage(self, mock_load_model):
        """Test loading model by Production stage."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-ensemble", stage="Production")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-ensemble/Production")
        assert model == mock_model

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_staging_stage(self, mock_load_model):
        """Test loading model by Staging stage."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-ensemble", stage="Staging")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-ensemble/Staging")
        assert model == mock_model

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_version_number(self, mock_load_model):
        """Test loading model by specific version number."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-ensemble", stage="3")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-ensemble/3")
        assert model == mock_model

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_archived_stage(self, mock_load_model):
        """Test loading model by Archived stage."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-ensemble", stage="Archived")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-ensemble/Archived")
        assert model == mock_model


class TestGetModelInfo:
    """Tests for get_model_info()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_info_specific_version(self, mock_get_client, mock_model_version):
        """Test getting info for a specific model version."""
        # Arrange
        mock_client = Mock()
        mock_client.get_model_version.return_value = mock_model_version
        mock_get_client.return_value = mock_client

        # Act
        info = model_registry.get_model_info("rossmann-ensemble", version="1")

        # Assert
        mock_client.get_model_version.assert_called_once_with(name="rossmann-ensemble", version="1")
        assert info["name"] == "rossmann-ensemble"
        assert info["version"] == "1"
        assert info["current_stage"] == "Production"
        assert info["run_id"] == "test-run-id-123"
        assert info["status"] == "READY"

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_info_all_versions(
        self, mock_get_client, mock_registered_model, mock_model_version
    ):
        """Test getting info for all versions of a model."""
        # Arrange
        mock_client = Mock()
        mock_client.get_registered_model.return_value = mock_registered_model

        # Create multiple versions
        mv1 = Mock()
        mv1.version = "1"
        mv1.current_stage = "Archived"
        mv1.run_id = "run-1"
        mv1.status = "READY"

        mv2 = Mock()
        mv2.version = "2"
        mv2.current_stage = "Production"
        mv2.run_id = "run-2"
        mv2.status = "READY"

        mock_client.search_model_versions.return_value = [mv1, mv2]
        mock_get_client.return_value = mock_client

        # Act
        info = model_registry.get_model_info("rossmann-ensemble")

        # Assert
        mock_client.get_registered_model.assert_called_once_with("rossmann-ensemble")
        mock_client.search_model_versions.assert_called_once_with("name='rossmann-ensemble'")

        assert info["name"] == "rossmann-ensemble"
        assert len(info["versions"]) == 2
        assert info["versions"][0]["version"] == "1"
        assert info["versions"][0]["stage"] == "Archived"
        assert info["versions"][1]["version"] == "2"
        assert info["versions"][1]["stage"] == "Production"


class TestListRegisteredModels:
    """Tests for list_registered_models()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_list_registered_models(self, mock_get_client):
        """Test listing all registered models."""
        # Arrange
        mock_client = Mock()

        rm1 = Mock()
        rm1.name = "rossmann-ensemble"

        rm2 = Mock()
        rm2.name = "rossmann-baseline"

        rm3 = Mock()
        rm3.name = "rossmann-xgboost"

        mock_client.search_registered_models.return_value = [rm1, rm2, rm3]
        mock_get_client.return_value = mock_client

        # Act
        models = model_registry.list_registered_models()

        # Assert
        mock_client.search_registered_models.assert_called_once()
        assert models == ["rossmann-ensemble", "rossmann-baseline", "rossmann-xgboost"]
        assert len(models) == 3

    @patch("src.models.model_registry.get_mlflow_client")
    def test_list_registered_models_empty(self, mock_get_client):
        """Test listing when no models are registered."""
        # Arrange
        mock_client = Mock()
        mock_client.search_registered_models.return_value = []
        mock_get_client.return_value = mock_client

        # Act
        models = model_registry.list_registered_models()

        # Assert
        assert models == []
        assert len(models) == 0
