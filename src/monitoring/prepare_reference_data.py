"""Prepare reference data for drift detection.

This script creates a stratified sample of training data to use as the reference distribution for
comparing against production predictions.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.io import read_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_reference_sample(
    features_path: Path,
    output_path: Path,
    sample_size: int = 10000,
    random_state: int = 42,
):
    """Create a stratified reference sample from training data.

    Parameters
    ----------
    features_path : Path
        Path to processed features parquet file
    output_path : Path
        Path to save reference sample
    sample_size : int, optional
        Number of samples to include, by default 10000
    random_state : int, optional
        Random seed for reproducibility, by default 42
    """
    logger.info(f"Loading training features from {features_path}")
    df = read_parquet(features_path)

    # Remove target variable and date/store identifiers
    # Keep only features used for drift detection
    drop_cols = ["Sales", "Date", "Store", "Customers"]
    drop_cols = [col for col in drop_cols if col in df.columns]

    df_features = df.drop(columns=drop_cols)

    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Features for drift detection: {df_features.shape[1]} columns")

    # Create stratified sample based on Store and month
    # This ensures we have representation across stores and time periods
    df_with_context = df.copy()

    # Add month for stratification (if Date column exists)
    if "Date" in df.columns:
        df_with_context["Date"] = pd.to_datetime(df_with_context["Date"])
        df_with_context["_month"] = df_with_context["Date"].dt.month
    elif "month" in df.columns:
        df_with_context["_month"] = df["month"]
    else:
        # Fallback: just random sample
        df_with_context["_month"] = 1

    # Create stratification groups (Store × Month)
    # Limit to reasonable number of strata
    if "Store" in df_with_context.columns:
        # Sample stores if too many
        unique_stores = df_with_context["Store"].unique()
        if len(unique_stores) > 100:
            sampled_stores = pd.Series(unique_stores).sample(n=100, random_state=random_state)
            df_with_context = df_with_context[df_with_context["Store"].isin(sampled_stores)]

        df_with_context["_strata"] = (
            df_with_context["Store"].astype(str) + "_" + df_with_context["_month"].astype(str)
        )
    else:
        df_with_context["_strata"] = df_with_context["_month"].astype(str)

    # Sample from each stratum
    logger.info(f"Stratified sampling {sample_size} records...")
    sampled = df_with_context.groupby("_strata", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), max(1, sample_size // len(df_with_context["_strata"].unique()))),
            random_state=random_state,
        )
    )

    # Take exact sample_size (may be slightly over due to stratification)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)

    # Extract feature columns only (remove stratification columns)
    feature_cols = [col for col in df_features.columns]
    reference_sample = sampled[feature_cols].reset_index(drop=True)

    # Also save Sales column for target drift detection
    if "Sales" in df.columns:
        reference_sample["Sales"] = sampled["Sales"].values

    logger.info(f"Reference sample shape: {reference_sample.shape}")

    # Save reference data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(reference_sample, output_path)

    logger.info(f"✓ Reference data saved to {output_path}")

    # Print summary statistics
    logger.info("\nReference Sample Summary:")
    logger.info(f"  Total records: {len(reference_sample)}")
    logger.info(f"  Total features: {reference_sample.shape[1]}")
    if "Sales" in reference_sample.columns:
        logger.info(
            f"  Sales range: [{reference_sample['Sales'].min():.2f}, "
            f"{reference_sample['Sales'].max():.2f}]"
        )
        logger.info(f"  Sales mean: {reference_sample['Sales'].mean():.2f}")


def main():
    """Main entry point for reference data preparation."""
    parser = argparse.ArgumentParser(description="Prepare reference data for drift detection")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train_features.parquet",
        help="Path to training features parquet file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "monitoring" / "reference_data" / "training_sample.parquet",
        help="Path to save reference sample",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples in reference data",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    create_reference_sample(
        features_path=args.features_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
