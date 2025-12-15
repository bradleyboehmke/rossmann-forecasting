"""MLflow utilities for experiment tracking and model registry.

This module provides utilities to initialize and manage MLflow tracking server, experiments, and
model registry for the Rossmann forecasting project.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import mlflow
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def load_mlflow_config(config_path: str = "config/params.yaml") -> dict[str, Any]:
    """Load MLflow configuration from params.yaml.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration file, by default "config/params.yaml"

    Returns
    -------
    dict[str, Any]
        MLflow configuration dictionary
    """
    # Try the provided path first
    config_file = Path(config_path)

    # If not found, try relative to current directory and parent directories
    if not config_file.exists():
        # Search up the directory tree to find the config file
        current = Path.cwd()
        found = False
        for _ in range(4):  # Try up to 4 levels up
            candidate = current / config_path
            if candidate.exists():
                config_file = candidate
                found = True
                break
            current = current.parent

        if not found:
            raise FileNotFoundError(
                f"Could not find {config_path}. Searched from {Path.cwd()} up to "
                f"{Path.cwd().parents[3] if len(Path.cwd().parents) > 3 else Path.cwd().parent}"
            )

    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config.get("mlflow", {})


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """Set up MLflow tracking server and experiment.

    This function initializes MLflow by:

    1. Creating necessary directories for tracking and artifacts
    2. Setting the tracking URI
    3. Creating or getting the experiment
    4. Setting the experiment as active

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking URI. If None, loads from config.
    experiment_name : str, optional
        Name of the MLflow experiment. If None, loads from config.
    artifact_location : str, optional
        Location to store artifacts. If None, loads from config.

    Returns
    -------
    str
        Experiment ID

    Examples
    --------
    >>> # Use default configuration
    >>> experiment_id = setup_mlflow()
    >>> # Use custom configuration
    >>> experiment_id = setup_mlflow(
    ...     tracking_uri="./mlruns",
    ...     experiment_name="my-experiment",
    ...     artifact_location="./mlartifacts"
    ... )
    """
    # Load config if parameters not provided
    if tracking_uri is None or experiment_name is None or artifact_location is None:
        config = load_mlflow_config()
        tracking_uri = tracking_uri or config.get("tracking_uri", "./mlruns")
        experiment_name = experiment_name or config.get("experiment_name", "rossmann-forecasting")
        artifact_location = artifact_location or config.get("artifact_location", "./mlartifacts")

    # Convert relative paths to absolute paths based on project root
    # Find project root by searching for config/params.yaml
    project_root = Path.cwd()
    found_root = False
    for _ in range(4):  # Search up to 4 levels
        if (project_root / "config" / "params.yaml").exists():
            found_root = True
            break
        project_root = project_root.parent

    if not found_root:
        # Fallback to current directory if can't find project root
        project_root = Path.cwd()

    # Create absolute paths
    tracking_path = Path(tracking_uri.replace("file://", ""))
    if not tracking_path.is_absolute():
        tracking_path = project_root / tracking_path

    artifact_path = Path(artifact_location.replace("file://", ""))
    if not artifact_path.is_absolute():
        artifact_path = project_root / artifact_path

    tracking_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Project root: {project_root}")
    logger.info(f"Created MLflow directories: {tracking_path}, {artifact_path}")

    # Set tracking URI using absolute path
    mlflow.set_tracking_uri(str(tracking_path))
    logger.info(f"MLflow tracking URI set to: {tracking_path}")

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=str(artifact_path)
        )
        logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")

    # Set active experiment
    mlflow.set_experiment(experiment_name)

    return experiment_id


def get_mlflow_client(tracking_uri: Optional[str] = None) -> MlflowClient:
    """Get MLflow client for programmatic access to tracking server.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking URI. If None, loads from config.

    Returns
    -------
    MlflowClient
        MLflow client instance

    Examples
    --------
    >>> client = get_mlflow_client()
    >>> experiments = client.search_experiments()
    """
    if tracking_uri is None:
        config = load_mlflow_config()
        tracking_uri = config.get("tracking_uri", "./mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def get_or_create_experiment(experiment_name: str, artifact_location: Optional[str] = None) -> str:
    """Get existing experiment or create new one.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    artifact_location : str, optional
        Location to store artifacts

    Returns
    -------
    str
        Experiment ID
    """
    client = get_mlflow_client()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            logger.info(f"Found existing experiment: {experiment_name}")
            return experiment.experiment_id
    except Exception:
        pass

    # Create new experiment
    if artifact_location is None:
        config = load_mlflow_config()
        artifact_location = config.get("artifact_location", "./mlartifacts")

    experiment_id = client.create_experiment(
        name=experiment_name, artifact_location=artifact_location
    )
    logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id


def log_params_from_dict(params: dict[str, Any], prefix: str = "") -> None:
    """Log parameters from dictionary to MLflow.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameters to log
    prefix : str, optional
        Prefix to add to parameter names, by default ""

    Examples
    --------
    >>> params = {"learning_rate": 0.01, "max_depth": 5}
    >>> log_params_from_dict(params, prefix="model.")
    """
    for key, value in params.items():
        param_name = f"{prefix}{key}" if prefix else key

        # Handle nested dictionaries
        if isinstance(value, dict):
            log_params_from_dict(value, prefix=f"{param_name}.")
        else:
            # Convert to string for MLflow
            mlflow.log_param(param_name, value)


def log_model_info(model_name: str, model_params: dict[str, Any]) -> None:
    """Log model information to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the model
    model_params : dict[str, Any]
        Model hyperparameters

    Examples
    --------
    >>> log_model_info(
    ...     model_name="lightgbm",
    ...     model_params={"num_leaves": 50, "learning_rate": 0.03}
    ... )
    """
    mlflow.log_param("model_name", model_name)
    log_params_from_dict(model_params, prefix=f"{model_name}.")


def cleanup_old_runs(experiment_name: str, keep_last_n: int = 10) -> None:
    """Clean up old MLflow runs, keeping only the most recent N runs.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment to clean up
    keep_last_n : int, optional
        Number of recent runs to keep, by default 10
    """
    client = get_mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        logger.warning(f"Experiment {experiment_name} not found")
        return

    # Get all runs sorted by start time
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    # Delete old runs
    if len(runs) > keep_last_n:
        runs_to_delete = runs[keep_last_n:]
        for run in runs_to_delete:
            client.delete_run(run.info.run_id)
            logger.info(f"Deleted old run: {run.info.run_id}")

        logger.info(f"Cleaned up {len(runs_to_delete)} old runs from {experiment_name}")


def log_dvc_data_version(data_file_path: str) -> None:
    """Log DVC data version information to MLflow.

    This function logs the DVC metadata for a data file to track which version
    of the data was used for training. It logs:

    - Git commit hash (current HEAD)
    - DVC file hash (md5 hash of the .dvc file)
    - Data file path

    This allows tracing any experiment back to the exact data version without
    duplicating data storage (DVC handles versioning, MLflow references it).

    Parameters
    ----------
    data_file_path : str
        Path to the data file tracked by DVC (e.g., "data/processed/train_features.parquet")

    Examples
    --------
    >>> with mlflow.start_run():
    ...     log_dvc_data_version("data/processed/train_features.parquet")
    ...     # Logs: git_commit_hash, dvc_file_hash, data_file_path
    """
    # Find project root
    project_root = Path.cwd()
    for _ in range(4):
        if (project_root / ".git").exists():
            break
        project_root = project_root.parent

    # Get git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit = result.stdout.strip()
        mlflow.log_param("git_commit_hash", git_commit)
        logger.info(f"Logged git commit hash: {git_commit[:8]}")
    except subprocess.CalledProcessError:
        logger.warning("Could not get git commit hash")
        git_commit = "unknown"

    # Get DVC file hash (from .dvc file)
    dvc_file_path = project_root / f"{data_file_path}.dvc"
    if dvc_file_path.exists():
        try:
            with open(dvc_file_path) as f:
                dvc_content = yaml.safe_load(f)
                # DVC stores file hash under 'outs' -> 'md5' or 'md5-cache'
                if "outs" in dvc_content and len(dvc_content["outs"]) > 0:
                    dvc_hash = dvc_content["outs"][0].get("md5", "unknown")
                    mlflow.log_param("dvc_file_hash", dvc_hash)
                    logger.info(f"Logged DVC file hash: {dvc_hash[:8]}")
                else:
                    logger.warning(f"Could not parse DVC hash from {dvc_file_path}")
        except Exception as e:
            logger.warning(f"Could not read DVC file {dvc_file_path}: {e}")
    else:
        logger.warning(f"DVC file not found: {dvc_file_path}")

    # Log data file path
    mlflow.log_param("data_file_path", data_file_path)
    logger.info(f"Logged data file path: {data_file_path}")
