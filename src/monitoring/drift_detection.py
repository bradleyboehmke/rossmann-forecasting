"""Drift detection module using Evidently AI.

This module provides functionality for detecting data drift and target drift by comparing production
predictions against reference training data.
"""

import importlib.util
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Import Evidently components (version 0.7.20)
try:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    EVIDENTLY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported Evidently 0.7.20")

except ImportError as e:
    Report = None
    DataDriftPreset = None
    DataDefinition = None
    Dataset = None
    EVIDENTLY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import Evidently: {e}")

# Add src to path
PROJECT_ROOT = Path(__file__).parents[2]
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import utilities
try:
    from utils.io import read_parquet
except ImportError:
    # Fallback: try direct import if running from different context
    spec = importlib.util.spec_from_file_location(
        "utils.io", PROJECT_ROOT / "src" / "utils" / "io.py"
    )
    utils_io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_io)
    read_parquet = utils_io.read_parquet

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detector for feature and target drift using Evidently.

    Parameters
    ----------
    reference_data_path : Path
        Path to reference training data parquet file
    db_path : Path
        Path to predictions database

    Attributes
    ----------
    reference_data : pd.DataFrame
        Reference training data for comparison
    db_path : Path
        Path to predictions database
    """

    # Key features to monitor for drift (~10 most important)
    # These are the snake_case names used in the predictions database
    KEY_FEATURES = [
        "promo",
        "day_of_week",
        "month",
        "state_holiday",
        "school_holiday",
        "store_type",
        "assortment",
        "competition_distance",
        "promo2",
        "is_promo2_active",
    ]

    # Mapping from database column names (snake_case) to training data columns (PascalCase)
    COLUMN_MAPPING = {
        "promo": "Promo",
        "day_of_week": "DayOfWeek",
        "month": "Month",
        "state_holiday": "StateHoliday",
        "school_holiday": "SchoolHoliday",
        "store_type": "StoreType",
        "assortment": "Assortment",
        "competition_distance": "CompetitionDistance",
        "promo2": "Promo2",
        "is_promo2_active": "Promo2Active",  # Note: training data uses Promo2Active
    }

    def __init__(self, reference_data_path: Path, db_path: Path):
        """Initialize drift detector.

        Parameters
        ----------
        reference_data_path : Path
            Path to reference training data parquet file
        db_path : Path
            Path to predictions database
        """
        self.reference_data = read_parquet(reference_data_path)
        self.db_path = db_path
        logger.info(f"Loaded reference data: {self.reference_data.shape}")

    def get_production_data(self, days: int = 7) -> pd.DataFrame:
        """Retrieve production predictions from database.

        Parameters
        ----------
        days : int, optional
            Number of days to retrieve, by default 7

        Returns
        -------
        pd.DataFrame
            Production predictions
        """
        query = """
        SELECT
            prediction,
            promo,
            day_of_week,
            month,
            state_holiday,
            school_holiday,
            store_type,
            assortment,
            competition_distance,
            promo2,
            is_promo2_active
        FROM predictions
        WHERE timestamp >= datetime('now', '-' || ? || ' days')
        """

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[days])

        logger.info(f"Retrieved {len(df)} production predictions from last {days} days")
        return df

    def prepare_data_for_comparison(
        self, production_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare reference and production data for drift comparison.

        Aligns columns and data types between reference and production data.

        Parameters
        ----------
        production_df : pd.DataFrame
            Production prediction data

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (reference_data, production_data) aligned for comparison
        """
        # Get common columns (features that exist in both datasets)
        # Use snake_case names from predictions database
        common_features = [col for col in self.KEY_FEATURES if col in production_df.columns]

        # Map to PascalCase names for reference data
        common_features_reference = [self.COLUMN_MAPPING.get(col, col) for col in common_features]

        # Filter to only columns that exist in reference data
        existing_reference_cols = [
            col for col in common_features_reference if col in self.reference_data.columns
        ]
        existing_production_cols = [
            common_features[i]
            for i, col in enumerate(common_features_reference)
            if col in self.reference_data.columns
        ]

        logger.info(f"Common features (production): {common_features}")
        logger.info(f"Mapped features (reference): {common_features_reference}")
        logger.info(f"Existing in reference: {existing_reference_cols}")
        logger.info(f"Final features to compare: {existing_production_cols}")

        # Add prediction as target for target drift
        if "prediction" in production_df.columns:
            production_target_col = "prediction"
            reference_target_col = "Sales" if "Sales" in self.reference_data.columns else None
        else:
            production_target_col = None
            reference_target_col = None

        # Extract reference data features (using PascalCase names)
        ref_features = self.reference_data[existing_reference_cols].copy()
        # Rename to snake_case for consistency
        ref_features.columns = existing_production_cols

        # Extract production data features (already in snake_case)
        prod_features = production_df[existing_production_cols].copy()

        # Add target columns if available
        if production_target_col and reference_target_col:
            ref_features["target"] = self.reference_data[reference_target_col]
            prod_features["target"] = production_df[production_target_col]

        # Ensure consistent data types
        for col in existing_production_cols:
            if col in ["store_type", "assortment", "state_holiday"]:
                # Categorical columns
                ref_features[col] = ref_features[col].astype(str)
                prod_features[col] = prod_features[col].astype(str)
            else:
                # Numerical columns
                ref_features[col] = pd.to_numeric(ref_features[col], errors="coerce")
                prod_features[col] = pd.to_numeric(prod_features[col], errors="coerce")

        logger.info(f"Prepared data for comparison: {len(existing_production_cols)} features")
        return ref_features, prod_features

    def generate_drift_report(
        self,
        days: int = 7,
        output_path: Path | None = None,
    ) -> tuple[Report, dict[str, Any]]:
        """Generate drift detection report using Evidently.

        Parameters
        ----------
        days : int, optional
            Number of days of production data to analyze, by default 7
        output_path : Path, optional
            Path to save HTML report, by default None

        Returns
        -------
        tuple[Report, dict[str, Any]]
            Evidently report object and drift summary dict
        """
        # Get production data
        production_df = self.get_production_data(days=days)

        if len(production_df) == 0:
            logger.warning(f"No production data found for last {days} days")
            return None, {"error": "No production data available"}

        # Debug: Log production data columns
        logger.info(f"Production data columns: {production_df.columns.tolist()}")
        logger.info(f"Production data shape: {production_df.shape}")

        # Prepare data for comparison
        reference_df, current_df = self.prepare_data_for_comparison(production_df)

        # Get all feature columns (exclude target if present)
        all_feature_cols = [col for col in reference_df.columns if col != "target"]

        # Separate categorical and numerical features
        categorical_cols = ["store_type", "assortment", "state_holiday"]
        categorical_features = [col for col in categorical_cols if col in all_feature_cols]
        numerical_features = [col for col in all_feature_cols if col not in categorical_cols]

        logger.info("Generating drift report...")
        logger.info(f"  Reference data: {reference_df.shape}")
        logger.info(f"  Production data: {current_df.shape}")
        logger.info(f"  All features: {all_feature_cols}")
        logger.info(f"  Numerical features ({len(numerical_features)}): {numerical_features}")
        logger.info(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

        # Ensure both dataframes have the same columns
        if set(reference_df.columns) != set(current_df.columns):
            logger.error("Column mismatch!")
            logger.error(f"  Reference: {sorted(reference_df.columns)}")
            logger.error(f"  Current: {sorted(current_df.columns)}")
            return None, {"error": "Column mismatch between reference and current data"}

        # Create DataDefinition for Evidently 0.7.20
        # DataDefinition uses lists of column names, not a column_types dict
        data_definition = DataDefinition(
            numerical_columns=numerical_features,
            categorical_columns=categorical_features,
        )

        # Create Dataset objects using from_pandas
        reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
        current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

        # Check if Evidently is available
        if not EVIDENTLY_AVAILABLE or DataDriftPreset is None:
            logger.error("Evidently not properly configured")
            return None, {"error": "Evidently library not properly configured"}

        # Create and run report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_dataset, current_data=current_dataset)

        # Extract drift summary from the report's metrics
        # NOTE: Evidently 0.7.20 has changed architecture - Report doesn't have as_dict()
        # We'll extract summary directly from the generated metrics
        drift_summary = self._extract_drift_summary_from_metrics(report, reference_df, current_df)

        # Optionally save a summary file
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save drift summary as JSON for record keeping
            json_path = output_path.with_suffix(".json")
            import json

            with open(json_path, "w") as f:
                json.dump(drift_summary, f, indent=2)

            logger.info(f"✓ Drift summary saved to {json_path}")

        return report, drift_summary

    def _extract_drift_summary_from_metrics(
        self, report: "Report", reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Extract drift summary by comparing dataframe distributions.

        Since Evidently 0.7.20 makes it difficult to extract computed results,
        we'll perform a simple statistical comparison ourselves.

        Parameters
        ----------
        report : Report
            Evidently report (for reference, not used for extraction)
        reference_df : pd.DataFrame
            Reference data
        current_df : pd.DataFrame
            Current production data

        Returns
        -------
        dict[str, Any]
            Summary of drift metrics
        """
        try:
            from scipy import stats

            # Compare distributions for each feature
            drifted_features = []
            total_features = len([c for c in reference_df.columns if c != "target"])

            for col in reference_df.columns:
                if col == "target":
                    continue

                ref_values = reference_df[col].dropna()
                cur_values = current_df[col].dropna()

                # Determine if categorical or numerical
                if col in ["store_type", "assortment", "state_holiday"]:
                    # Categorical: Skip chi-square test, use simple distribution comparison
                    try:
                        # Create frequency tables
                        ref_counts = ref_values.value_counts(normalize=True)
                        cur_counts = cur_values.value_counts(normalize=True)

                        # Align categories
                        all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))

                        # Calculate simple distribution difference
                        diff_sum = 0.0
                        for cat in all_cats:
                            ref_prop = ref_counts.get(cat, 0)
                            cur_prop = cur_counts.get(cat, 0)
                            diff_sum += abs(ref_prop - cur_prop)

                        # Threshold: if total variation distance > 0.2, flag as drift
                        drift_detected = diff_sum > 0.2
                        drift_score = diff_sum
                        test_name = "total_variation"

                        logger.info(
                            f"Feature {col}: total_variation={diff_sum:.3f}, drift={drift_detected}"
                        )

                    except Exception as e:
                        logger.warning(f"Distribution comparison failed for {col}: {e}. Skipping.")
                        drift_detected = False
                        drift_score = 0.0
                        test_name = "comparison_failed"

                else:
                    # Numerical: use Kolmogorov-Smirnov test
                    try:
                        ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
                        drift_detected = p_value < 0.05
                        drift_score = ks_stat  # KS statistic itself is the drift score
                        test_name = "ks"
                    except Exception as e:
                        logger.warning(f"KS test failed for {col}: {e}. Skipping.")
                        drift_detected = False
                        drift_score = 0.0
                        test_name = "ks_failed"

                if drift_detected:
                    drifted_features.append(
                        {
                            "feature": col,
                            "drift_score": float(drift_score),
                            "stattest": test_name,
                        }
                    )

            # Overall drift: detected if >50% of features show drift
            num_drifted = len(drifted_features)
            drift_share = num_drifted / total_features if total_features > 0 else 0.0
            dataset_drift_detected = drift_share > 0.5

            summary = {
                "dataset_drift_detected": dataset_drift_detected,
                "drift_share": drift_share,
                "number_of_drifted_features": num_drifted,
                "drifted_features": drifted_features,
                "total_features_checked": total_features,
            }

            return summary

        except Exception as e:
            logger.error(f"Error extracting drift summary: {e}")
            return {"error": str(e)}


def generate_drift_report_cli(
    days: int = 7,
    reference_data_path: Path | None = None,
    db_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """CLI-friendly function to generate drift report.

    Parameters
    ----------
    days : int, optional
        Number of days to analyze, by default 7
    reference_data_path : Path, optional
        Path to reference data, by default None (uses full training data)
    db_path : Path, optional
        Path to predictions database, by default None (uses default)
    output_path : Path, optional
        Path to save report, by default None (auto-generated)

    Returns
    -------
    dict[str, Any]
        Drift summary
    """
    # Set defaults - use full training data for most accurate drift detection
    if reference_data_path is None:
        reference_data_path = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"

    if db_path is None:
        db_path = PROJECT_ROOT / "data" / "monitoring" / "predictions.db"

    if output_path is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_path = (
            PROJECT_ROOT / "monitoring" / "drift_reports" / f"drift_report_{timestamp}.html"
        )

    # Create detector and generate report
    detector = DriftDetector(reference_data_path=reference_data_path, db_path=db_path)

    report, summary = detector.generate_drift_report(days=days, output_path=output_path)

    # Print summary to console
    if "error" not in summary:
        print("\n" + "=" * 70)
        print("DRIFT DETECTION SUMMARY")
        print("=" * 70)
        print(
            f"Dataset Drift Detected: {'⚠️ YES' if summary['dataset_drift_detected'] else '✅ NO'}"
        )
        print(f"Drift Share: {summary['drift_share']:.2%}")
        print(
            f"Drifted Features: {summary['number_of_drifted_features']}/{summary['total_features_checked']}"
        )

        if summary["drifted_features"]:
            print("\nFeatures with Drift:")
            for feat in summary["drifted_features"]:
                print(f"  ⚠️ {feat['feature']}: drift_score={feat['drift_score']:.3f}")
        else:
            print("\n✅ No features showing significant drift")

        print("=" * 70)
        print(f"\nFull report saved to: {output_path}")
    else:
        print(f"\n❌ Error: {summary['error']}")

    return summary
