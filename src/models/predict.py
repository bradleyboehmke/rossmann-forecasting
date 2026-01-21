"""Production inference pipeline for Rossmann forecasting.

This module provides functionality to load the production model from MLflow and generate predictions
on new data.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from utils.io import ensure_dir, read_parquet

from models.model_registry import get_model_version, load_model

logger = logging.getLogger(__name__)


def load_inference_data(
    data_path: str = "data/processed/train_features.parquet",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple:
    """Load data for inference.

    Parameters
    ----------
    data_path : str
        Path to processed features parquet file
    start_date : str, optional
        Start date for predictions (YYYY-MM-DD). If None, uses all data
    end_date : str, optional
        End date for predictions (YYYY-MM-DD). If None, uses all data

    Returns
    -------
    tuple
        (data_df, X_features, feature_cols)
    """
    logger.info(f"Loading data from: {data_path}")
    df = read_parquet(data_path)

    # Filter by date range if specified
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
        logger.info(f"Filtered to start_date >= {start_date}")

    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]
        logger.info(f"Filtered to end_date <= {end_date}")

    # Filter to open stores only (predictions only needed for open stores)
    df_open = df[df["Open"] == 1].copy()

    logger.info(f"Loaded {len(df_open):,} rows (open stores)")
    logger.info(f"Date range: {df_open['Date'].min()} to {df_open['Date'].max()}")

    # Define feature columns
    exclude_cols = ["Sales", "Date", "Store", "Customers"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Handle missing features (if any)
    df_open = df_open.dropna(subset=feature_cols)

    X_features = df_open[feature_cols]

    logger.info(f"Features: {len(feature_cols)}")

    return df_open, X_features, feature_cols


def generate_predictions(
    model,
    X_features: pd.DataFrame,
    data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions using loaded model.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        Loaded production model
    X_features : pd.DataFrame
        Feature matrix for predictions
    data_df : pd.DataFrame
        Original data with Store, Date, etc.

    Returns
    -------
    pd.DataFrame
        Predictions with metadata (Store, Date, Predicted_Sales)
    """
    logger.info(f"Generating predictions for {len(X_features):,} rows...")

    # Generate predictions
    predictions = model.predict(X_features)

    # Create predictions dataframe
    predictions_df = pd.DataFrame(
        {
            "Store": data_df["Store"].values,
            "Date": data_df["Date"].values,
            "DayOfWeek": data_df["DayOfWeek"].values if "DayOfWeek" in data_df.columns else None,
            "Predicted_Sales": predictions,
        }
    )

    # Add actual sales if available (for comparison)
    if "Sales" in data_df.columns:
        predictions_df["Actual_Sales"] = data_df["Sales"].values
        predictions_df["Prediction_Error"] = (
            predictions_df["Actual_Sales"] - predictions_df["Predicted_Sales"]
        )
        predictions_df["Absolute_Percentage_Error"] = (
            np.abs(predictions_df["Prediction_Error"]) / predictions_df["Actual_Sales"]
        ) * 100

    logger.info(f"✓ Generated {len(predictions):,} predictions")
    logger.info(f"  Prediction range: ${predictions.min():.2f} to ${predictions.max():.2f}")

    if "Actual_Sales" in predictions_df.columns:
        from evaluation.metrics import rmspe

        rmspe_score = rmspe(
            predictions_df["Actual_Sales"].values,
            predictions_df["Predicted_Sales"].values,
        )
        logger.info(f"  RMSPE: {rmspe_score:.6f}")

    return predictions_df


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: str,
    model_version: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save predictions to CSV with metadata.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions dataframe
    output_path : str
        Output file path
    model_version : str
        Model version used for predictions
    metadata : dict, optional
        Additional metadata to save
    """
    # Ensure output directory exists
    output_file = Path(output_path)
    ensure_dir(output_file.parent)

    # Save predictions
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved predictions to: {output_path}")

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata.update(
        {
            "model_version": model_version,
            "prediction_date": datetime.now().isoformat(),
            "n_predictions": len(predictions_df),
            "date_range": {
                "start": str(predictions_df["Date"].min()),
                "end": str(predictions_df["Date"].max()),
            },
        }
    )

    metadata_path = output_file.parent / f"{output_file.stem}_metadata.json"
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved metadata to: {metadata_path}")


def predict(
    model_name: str = "rossmann-ensemble",
    stage: str = "Production",
    data_path: str = "data/processed/train_features.parquet",
    output_path: str = "outputs/predictions/production_predictions.csv",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Main prediction pipeline.

    Parameters
    ----------
    model_name : str
        Name of registered model in MLflow
    stage : str
        Model stage to use ('Production', 'Staging', or version number)
    data_path : str
        Path to input data
    output_path : str
        Path to save predictions
    start_date : str, optional
        Start date for predictions (YYYY-MM-DD)
    end_date : str, optional
        End date for predictions (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        Predictions dataframe

    Examples
    --------
    >>> # Predict using Production model
    >>> predictions = predict(
    ...     model_name='rossmann-ensemble',
    ...     stage='Production',
    ...     data_path='data/processed/train_features.parquet'
    ... )
    >>>
    >>> # Predict for specific date range
    >>> predictions = predict(
    ...     model_name='rossmann-ensemble',
    ...     stage='Production',
    ...     start_date='2015-07-01',
    ...     end_date='2015-07-31'
    ... )
    """
    logger.info("=" * 70)
    logger.info("PRODUCTION INFERENCE PIPELINE")
    logger.info("=" * 70)

    # Get model version
    if stage in ["Production", "Staging"]:
        version = get_model_version(model_name, stage=stage)
        logger.info(f"Using {stage} model version: {version}")
    else:
        version = stage  # Assume it's a version number
        logger.info(f"Using model version: {version}")

    # Load model
    logger.info(f"\nLoading model: {model_name} ({stage})")
    model = load_model(model_name, stage=stage)

    # Load inference data
    logger.info("\nLoading inference data...")
    data_df, X_features, feature_cols = load_inference_data(data_path, start_date, end_date)

    # Generate predictions
    logger.info("\nGenerating predictions...")
    predictions_df = generate_predictions(model, X_features, data_df)

    # Save predictions
    logger.info("\nSaving predictions...")
    save_predictions(
        predictions_df,
        output_path,
        model_version=version,
        metadata={
            "model_name": model_name,
            "model_stage": stage,
            "data_path": data_path,
        },
    )

    logger.info("=" * 70)
    logger.info("✓ PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Predictions saved to: {output_path}")
    logger.info(f"Total predictions: {len(predictions_df):,}")

    return predictions_df


def main():
    """Main inference workflow with CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate predictions using production Rossmann model"
    )
    parser.add_argument("--model-name", default="rossmann-ensemble", help="Model name in registry")
    parser.add_argument(
        "--stage",
        default="Production",
        help="Model stage ('Production', 'Staging', or version number)",
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/train_features.parquet",
        help="Input data path",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/predictions/production_predictions.csv",
        help="Output predictions path",
    )
    parser.add_argument("--start-date", help="Start date for predictions (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for predictions (YYYY-MM-DD)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    predictions_df = predict(
        model_name=args.model_name,
        stage=args.stage,
        data_path=args.data_path,
        output_path=args.output_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total predictions: {len(predictions_df):,}")
    print(f"Date range: {predictions_df['Date'].min()} to {predictions_df['Date'].max()}")
    print(f"Stores: {predictions_df['Store'].nunique():,} unique stores")
    print(
        f"Predicted sales range: ${predictions_df['Predicted_Sales'].min():.2f} to ${predictions_df['Predicted_Sales'].max():.2f}"
    )

    if "Actual_Sales" in predictions_df.columns:
        print("\nActual vs Predicted:")
        print(f"  Mean Actual: ${predictions_df['Actual_Sales'].mean():.2f}")
        print(f"  Mean Predicted: ${predictions_df['Predicted_Sales'].mean():.2f}")
        print(f"  Mean APE: {predictions_df['Absolute_Percentage_Error'].mean():.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
