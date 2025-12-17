"""Model validation and promotion for Rossmann forecasting.

This module validates a candidate model against performance thresholds and promotes it to Staging if
it passes all checks.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from evaluation.cv import remove_missing_features
from evaluation.metrics import rmspe
from utils.io import read_parquet

from models.model_registry import (
    get_model_version,
    load_model,
    promote_model,
)

logger = logging.getLogger(__name__)


def load_holdout_data(
    data_path: str = "data/processed/train_features.parquet",
    holdout_days: int = 42,
) -> tuple:
    """Load holdout validation data.

    Parameters
    ----------
    data_path : str
        Path to processed features
    holdout_days : int
        Number of days for holdout set

    Returns
    -------
    tuple
        (X_holdout, y_holdout, feature_cols)
    """
    logger.info(f"Loading holdout data from: {data_path}")
    df = read_parquet(data_path)

    # Create holdout split
    max_date = df["Date"].max()
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)
    holdout_df = df[df["Date"] >= holdout_start].copy()

    # Filter to open stores
    holdout_data = holdout_df[holdout_df["Open"] == 1].copy()

    # Define features
    exclude_cols = ["Sales", "Date", "Store", "Customers"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove missing features
    holdout_data, _ = remove_missing_features(holdout_data, feature_cols)

    X_holdout = holdout_data[feature_cols]
    y_holdout = holdout_data["Sales"].values

    logger.info(f"Holdout data: {len(holdout_data):,} rows (open stores)")
    logger.info(f"Date range: {holdout_data['Date'].min()} to {holdout_data['Date'].max()}")

    return X_holdout, y_holdout, feature_cols


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate model on test data.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        Loaded model
    X_test : pd.DataFrame
        Test features
    y_test : np.ndarray
        Test targets

    Returns
    -------
    dict
        Evaluation metrics
    """
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)

    logger.info("Calculating metrics...")
    metrics = {
        "rmspe": float(rmspe(y_test, predictions)),
        "rmse": float(np.sqrt(np.mean((y_test - predictions) ** 2))),
        "mae": float(np.mean(np.abs(y_test - predictions))),
        "mape": float(np.mean(np.abs((y_test - predictions) / y_test)) * 100),
    }

    logger.info("Model Performance:")
    logger.info(f"  RMSPE: {metrics['rmspe']:.6f}")
    logger.info(f"  RMSE:  {metrics['rmse']:.2f}")
    logger.info(f"  MAE:   {metrics['mae']:.2f}")
    logger.info(f"  MAPE:  {metrics['mape']:.2f}%")

    return metrics


def check_performance_threshold(
    metrics: dict[str, float],
    threshold_rmspe: float = 0.10,
    strict: bool = False,
) -> bool:
    """Check if model meets performance thresholds.

    Parameters
    ----------
    metrics : dict
        Model evaluation metrics
    threshold_rmspe : float, default=0.10
        Maximum acceptable RMSPE (10% error threshold)
    strict : bool, default=False
        If True, use stricter threshold (0.09856 - top 50 leaderboard)

    Returns
    -------
    bool
        True if model passes threshold checks
    """
    if strict:
        threshold_rmspe = 0.09856  # Top 50 Kaggle leaderboard

    logger.info("=" * 70)
    logger.info("PERFORMANCE THRESHOLD CHECKS")
    logger.info("=" * 70)

    # Check RMSPE
    rmspe_pass = metrics["rmspe"] <= threshold_rmspe
    logger.info(
        f"RMSPE: {metrics['rmspe']:.6f} <= {threshold_rmspe:.6f} "
        f"{'✓ PASS' if rmspe_pass else '✗ FAIL'}"
    )

    # Check for reasonable predictions (not extreme values)
    predictions_reasonable = metrics["mae"] < 2000  # Average error less than $2000 seems reasonable
    logger.info(
        f"MAE: {metrics['mae']:.2f} < 2000 " f"{'✓ PASS' if predictions_reasonable else '✗ FAIL'}"
    )

    all_pass = rmspe_pass and predictions_reasonable

    logger.info("=" * 70)
    if all_pass:
        logger.info("✓ ALL CHECKS PASSED")
    else:
        logger.info("✗ VALIDATION FAILED")
    logger.info("=" * 70)

    return all_pass


def validate_and_promote(
    model_name: str = "rossmann-ensemble",
    version: Optional[str] = None,
    data_path: str = "data/processed/train_features.parquet",
    threshold_rmspe: float = 0.10,
    strict: bool = False,
    auto_promote: bool = True,
) -> dict[str, any]:
    """Validate a model and promote to Staging if passing.

    Parameters
    ----------
    model_name : str
        Name of registered model
    version : str, optional
        Model version to validate. If None, uses latest version
    data_path : str
        Path to processed data
    threshold_rmspe : float
        Performance threshold
    strict : bool
        Use strict threshold (0.09856)
    auto_promote : bool
        Automatically promote to Staging if passing

    Returns
    -------
    dict
        Validation results including metrics and promotion status
    """
    logger.info("=" * 70)
    logger.info("MODEL VALIDATION & PROMOTION")
    logger.info("=" * 70)

    # Get version if not specified
    if version is None:
        # Find the latest version (any stage)
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")
        version = max([int(mv.version) for mv in versions])
        logger.info(f"Using latest version: {version}")
    else:
        logger.info(f"Validating version: {version}")

    # Load model
    logger.info(f"\nLoading model: {model_name} version {version}")
    model = load_model(model_name, stage=str(version))

    # Load holdout data
    logger.info("\nLoading holdout validation data...")
    X_holdout, y_holdout, feature_cols = load_holdout_data(data_path)

    # Evaluate model
    logger.info("\nEvaluating model performance...")
    metrics = evaluate_model(model, X_holdout, y_holdout)

    # Check thresholds
    logger.info("\nChecking performance thresholds...")
    passed = check_performance_threshold(metrics, threshold_rmspe, strict)

    # Promotion decision
    promoted = False
    if passed and auto_promote:
        logger.info("\n✓ Model passed validation, promoting to Staging...")
        promote_model(model_name, version=str(version), stage="Staging")
        promoted = True
        logger.info(f"✓ Model {model_name} v{version} promoted to Staging")
    elif passed:
        logger.info("\n✓ Model passed validation (auto_promote=False, manual promotion required)")
    else:
        logger.info("\n✗ Model did not pass validation, not promoting")

    results = {
        "model_name": model_name,
        "version": version,
        "metrics": metrics,
        "threshold_rmspe": threshold_rmspe if not strict else 0.09856,
        "passed": passed,
        "promoted": promoted,
        "stage": "Staging" if promoted else "None",
    }

    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name} v{version}")
    logger.info(f"RMSPE: {metrics['rmspe']:.6f}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Promoted to Staging: {promoted}")
    logger.info("=" * 70)

    return results


def promote_to_production(
    model_name: str = "rossmann-ensemble",
    version: Optional[str] = None,
) -> None:
    """Promote a Staging model to Production.

    This should typically be done manually after testing the Staging model.

    Parameters
    ----------
    model_name : str
        Name of registered model
    version : str, optional
        Model version to promote. If None, uses current Staging version
    """
    logger.info("=" * 70)
    logger.info("PROMOTE TO PRODUCTION")
    logger.info("=" * 70)

    # Get Staging version if not specified
    if version is None:
        version = get_model_version(model_name, stage="Staging")
        if version is None:
            raise ValueError(f"No Staging version found for {model_name}")
        logger.info(f"Promoting Staging version: {version}")

    # Promote to Production
    logger.info(f"\nPromoting {model_name} v{version} to Production...")
    promote_model(model_name, version=version, stage="Production")

    logger.info("=" * 70)
    logger.info(f"✓ {model_name} v{version} is now in Production")
    logger.info("=" * 70)


def main():
    """Main validation workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and promote Rossmann ensemble model")
    parser.add_argument("--model-name", default="rossmann-ensemble", help="Model name in registry")
    parser.add_argument("--version", type=str, help="Model version to validate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="RMSPE threshold (default: 0.10)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict threshold (0.09856 - top 50)",
    )
    parser.add_argument(
        "--no-auto-promote",
        action="store_true",
        help="Disable automatic promotion to Staging",
    )
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Promote Staging model to Production (manual step)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.promote_to_production:
        promote_to_production(model_name=args.model_name, version=args.version)
    else:
        results = validate_and_promote(
            model_name=args.model_name,
            version=args.version,
            threshold_rmspe=args.threshold,
            strict=args.strict,
            auto_promote=not args.no_auto_promote,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(f"Model: {results['model_name']} v{results['version']}")
        print(f"RMSPE: {results['metrics']['rmspe']:.6f}")
        print(f"Threshold: {results['threshold_rmspe']:.6f}")
        print(f"Passed: {'✓ YES' if results['passed'] else '✗ NO'}")
        print(f"Stage: {results['stage']}")
        print("=" * 70)

        if results["passed"] and results["promoted"]:
            print("\n✓ Model validated and promoted to Staging!")
            print("\nNext steps:")
            print("  1. Test Staging model in production-like environment")
            print("  2. If satisfied, promote to Production:")
            print(
                f"     python src/models/validate_model.py --promote-to-production --version {results['version']}"
            )
        elif results["passed"]:
            print("\n✓ Model passed validation!")
            print("\nTo promote to Staging:")
            print(f"  python src/models/validate_model.py --version {results['version']}")
        else:
            print("\n✗ Model did not pass validation")
            print("\nConsider:")
            print("  - Retraining with more data")
            print("  - Tuning hyperparameters further")
            print("  - Adjusting ensemble weights")


if __name__ == "__main__":
    main()
