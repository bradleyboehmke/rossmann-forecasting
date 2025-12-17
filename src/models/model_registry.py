"""MLflow Model Registry utilities for Rossmann forecasting.

This module provides helper functions for managing models in MLflow Model Registry, including
registration, promotion, and loading models.
"""

import logging
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def get_mlflow_client() -> MlflowClient:
    """Get an MLflow tracking client.

    Returns
    -------
    MlflowClient
        Initialized MLflow client
    """
    return MlflowClient()


def register_ensemble_model(
    ensemble_model,
    model_name: str,
    run_id: Optional[str] = None,
    conda_env: Optional[dict[str, Any]] = None,
    signature=None,
    input_example=None,
    description: Optional[str] = None,
) -> str:
    """Register an ensemble model to MLflow Model Registry.

    Parameters
    ----------
    ensemble_model : RossmannEnsemble
        Trained ensemble model instance
    model_name : str
        Name for the registered model (e.g., 'rossmann-ensemble')
    run_id : str, optional
        MLflow run ID. If None, uses active run
    conda_env : dict, optional
        Conda environment specification
    signature : mlflow.models.ModelSignature, optional
        Model signature
    input_example : pd.DataFrame, optional
        Example input for model testing
    description : str, optional
        Model description

    Returns
    -------
    str
        Model version number

    Examples
    --------
    >>> from models.ensemble import create_ensemble
    >>> ensemble = create_ensemble(lgb_model, xgb_model, cb_model)
    >>> version = register_ensemble_model(
    ...     ensemble_model=ensemble,
    ...     model_name='rossmann-ensemble',
    ...     signature=signature,
    ...     input_example=X_sample
    ... )
    >>> print(f"Registered model version: {version}")
    """
    logger.info(f"Registering ensemble model: {model_name}")

    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="ensemble_model",
        python_model=ensemble_model,
        registered_model_name=model_name,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )

    # Get version from model info
    version = model_info.registered_model_version

    # Update description if provided
    if description:
        client = get_mlflow_client()
        client.update_model_version(name=model_name, version=version, description=description)

    logger.info(f"✓ Registered {model_name} version {version}")
    return version


def promote_model(
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> None:
    """Promote a model version to a specific stage.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    version : str
        Model version to promote
    stage : str
        Target stage ('Staging', 'Production', or 'Archived')
    archive_existing : bool, default=True
        If True, archive existing models in the target stage

    Examples
    --------
    >>> # Promote to Staging
    >>> promote_model('rossmann-ensemble', version='1', stage='Staging')
    >>> # Promote to Production
    >>> promote_model('rossmann-ensemble', version='2', stage='Production')
    """
    client = get_mlflow_client()

    # Archive existing models in the target stage if requested
    if archive_existing and stage != "Archived":
        existing_versions = client.get_latest_versions(model_name, stages=[stage])
        for mv in existing_versions:
            logger.info(f"Archiving {model_name} version {mv.version} (was in {stage})")
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )

    # Promote the specified version
    logger.info(f"Promoting {model_name} version {version} to {stage}")
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)
    logger.info(f"✓ {model_name} version {version} is now in {stage}")


def get_model_version(model_name: str, stage: str = "Production") -> Optional[str]:
    """Get the version number of a model in a specific stage.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    stage : str, default='Production'
        Model stage ('Staging', 'Production', or 'None')

    Returns
    -------
    str or None
        Model version number, or None if no model in stage

    Examples
    --------
    >>> version = get_model_version('rossmann-ensemble', stage='Production')
    >>> print(f"Production version: {version}")
    """
    client = get_mlflow_client()
    versions = client.get_latest_versions(model_name, stages=[stage])

    if not versions:
        logger.warning(f"No model found for {model_name} in {stage} stage")
        return None

    version = versions[0].version
    logger.info(f"{model_name} {stage} version: {version}")
    return version


def load_model(model_name: str, stage: str = "Production"):
    """Load a model from MLflow Model Registry.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    stage : str, default='Production'
        Model stage to load ('Staging', 'Production', or version number)

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        Loaded model ready for predictions

    Examples
    --------
    >>> # Load production model
    >>> model = load_model('rossmann-ensemble', stage='Production')
    >>> predictions = model.predict(X_test)
    >>>
    >>> # Load specific version
    >>> model = load_model('rossmann-ensemble', stage='1')
    >>> predictions = model.predict(X_test)
    """
    if stage in ["Staging", "Production", "Archived"]:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Assume it's a version number
        model_uri = f"models:/{model_name}/{stage}"

    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✓ Model loaded successfully")

    return model


def get_model_info(model_name: str, version: Optional[str] = None) -> dict[str, Any]:
    """Get detailed information about a registered model.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    version : str, optional
        Model version. If None, gets info for all versions

    Returns
    -------
    dict
        Model information including version, stage, metrics, etc.

    Examples
    --------
    >>> info = get_model_info('rossmann-ensemble', version='1')
    >>> print(f"Model stage: {info['current_stage']}")
    >>> print(f"Run ID: {info['run_id']}")
    """
    client = get_mlflow_client()

    if version:
        mv = client.get_model_version(name=model_name, version=version)
        return {
            "name": mv.name,
            "version": mv.version,
            "current_stage": mv.current_stage,
            "description": mv.description,
            "run_id": mv.run_id,
            "status": mv.status,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
        }
    else:
        # Get all versions
        model = client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name='{model_name}'")

        return {
            "name": model.name,
            "description": model.description,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "versions": [
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "status": mv.status,
                }
                for mv in versions
            ],
        }


def list_registered_models() -> list:
    """List all registered models in MLflow Model Registry.

    Returns
    -------
    list
        List of registered model names

    Examples
    --------
    >>> models = list_registered_models()
    >>> for model_name in models:
    ...     print(model_name)
    """
    client = get_mlflow_client()
    registered_models = client.search_registered_models()

    model_names = [rm.name for rm in registered_models]
    logger.info(f"Found {len(model_names)} registered models")

    return model_names
