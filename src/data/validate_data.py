"""Data validation using Great Expectations.

This module validates data quality at different stages of the pipeline:
- Raw data validation (train.csv, store.csv)
- Processed data validation (train_clean.parquet)
- Feature data validation (train_features.parquet)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from great_expectations.core.batch import RuntimeBatchRequest

# Try to import Great Expectations
try:
    import great_expectations as gx
    from great_expectations.checkpoint import SimpleCheckpoint
    from great_expectations.core.batch import RuntimeBatchRequest

    GX_AVAILABLE = True
except ImportError:
    print("Warning: Great Expectations not installed. Validation will be skipped.")
    GX_AVAILABLE = False
    # Define dummy classes to avoid NameErrors
    RuntimeBatchRequest = None
    SimpleCheckpoint = None
    gx = None


class DataValidator:
    """Data validation orchestrator using Great Expectations."""

    def __init__(self, context_root_dir: Path | None = None):
        """Initialize validator with Great Expectations context.

        Args:
            context_root_dir: Root directory for GX context. Defaults to project root.
        """
        if not GX_AVAILABLE:
            self.context = None
            return

        if context_root_dir is None:
            context_root_dir = Path(__file__).parents[2] / "great_expectations"

        self.context_root_dir = context_root_dir
        self.context = None
        self._initialize_context()

    def _initialize_context(self):
        """Initialize or load Great Expectations context."""
        if not GX_AVAILABLE:
            return

        try:
            # Try to load existing context
            self.context = gx.get_context(context_root_dir=str(self.context_root_dir))
            print(f"✓ Loaded existing GX context from {self.context_root_dir}")
        except Exception as e:
            print(f"Warning: Could not load GX context: {e}")
            print("Validation will be skipped.")
            self.context = None

    def validate_raw_train_data(self, file_path: Path) -> dict:
        """Validate raw training data (train.csv).

        Args:
            file_path: Path to train.csv

        Returns:
            Validation results dictionary
        """
        if not GX_AVAILABLE or self.context is None:
            return self._skip_validation("raw_train")

        print(f"\n{'=' * 60}")
        print(f"Validating raw training data: {file_path}")
        print(f"{'=' * 60}\n")

        # Load data
        df = pd.read_csv(file_path)

        # Create runtime batch
        batch_request = RuntimeBatchRequest(
            datasource_name="rossmann_datasource",
            data_connector_name="raw_data_connector",
            data_asset_name="train_raw",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"file_name": "train.csv"},
        )

        # Run validation
        results = self._run_validation(
            batch_request=batch_request,
            expectation_suite_name="raw_train_suite",
            checkpoint_name="raw_train_checkpoint",
        )

        return results

    def validate_raw_store_data(self, file_path: Path) -> dict:
        """Validate raw store data (store.csv).

        Args:
            file_path: Path to store.csv

        Returns:
            Validation results dictionary
        """
        if not GX_AVAILABLE or self.context is None:
            return self._skip_validation("raw_store")

        print(f"\n{'=' * 60}")
        print(f"Validating raw store data: {file_path}")
        print(f"{'=' * 60}\n")

        # Load data
        df = pd.read_csv(file_path)

        # Create runtime batch
        batch_request = RuntimeBatchRequest(
            datasource_name="rossmann_datasource",
            data_connector_name="raw_data_connector",
            data_asset_name="store_raw",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"file_name": "store.csv"},
        )

        # Run validation
        results = self._run_validation(
            batch_request=batch_request,
            expectation_suite_name="raw_store_suite",
            checkpoint_name="raw_store_checkpoint",
        )

        return results

    def validate_processed_data(self, file_path: Path) -> dict:
        """Validate processed/cleaned data.

        Args:
            file_path: Path to train_clean.parquet

        Returns:
            Validation results dictionary
        """
        if not GX_AVAILABLE or self.context is None:
            return self._skip_validation("processed")

        print(f"\n{'=' * 60}")
        print(f"Validating processed data: {file_path}")
        print(f"{'=' * 60}\n")

        # Load data
        df = pd.read_parquet(file_path)

        # Create runtime batch
        batch_request = RuntimeBatchRequest(
            datasource_name="rossmann_datasource",
            data_connector_name="processed_data_connector",
            data_asset_name="train_clean",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"file_name": "train_clean.parquet"},
        )

        # Run validation
        results = self._run_validation(
            batch_request=batch_request,
            expectation_suite_name="processed_data_suite",
            checkpoint_name="processed_data_checkpoint",
        )

        return results

    def validate_features(self, file_path: Path) -> dict:
        """Validate feature-engineered data.

        Args:
            file_path: Path to train_features.parquet

        Returns:
            Validation results dictionary
        """
        if not GX_AVAILABLE or self.context is None:
            return self._skip_validation("features")

        print(f"\n{'=' * 60}")
        print(f"Validating feature data: {file_path}")
        print(f"{'=' * 60}\n")

        # Load data
        df = pd.read_parquet(file_path)

        # Create runtime batch
        batch_request = RuntimeBatchRequest(
            datasource_name="rossmann_datasource",
            data_connector_name="processed_data_connector",
            data_asset_name="train_features",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"file_name": "train_features.parquet"},
        )

        # Run validation
        results = self._run_validation(
            batch_request=batch_request,
            expectation_suite_name="features_suite",
            checkpoint_name="features_checkpoint",
        )

        return results

    def _run_validation(
        self, batch_request: RuntimeBatchRequest, expectation_suite_name: str, checkpoint_name: str
    ) -> dict:
        """Run validation using a checkpoint.

        Args:
            batch_request: Runtime batch request
            expectation_suite_name: Name of expectation suite
            checkpoint_name: Name of checkpoint

        Returns:
            Validation results
        """
        try:
            # Get validator using batch_request
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=expectation_suite_name,
            )

            # Run validation
            results = validator.validate()

            # Extract success status
            success = results.success

            # Print summary
            print(f"\n{'=' * 60}")
            print(f"Validation Results: {'✓ PASSED' if success else '✗ FAILED'}")
            print(f"{'=' * 60}\n")

            # Get statistics
            stats = results.statistics

            print(f"Total expectations: {stats['evaluated_expectations']}")
            print(f"Successful: {stats['successful_expectations']}")
            print(f"Failed: {stats['unsuccessful_expectations']}")
            print(f"Success rate: {stats['success_percent']:.1f}%\n")

            # If there are failures, show which ones failed
            if not success:
                print("Failed Expectations:")
                for result in results.results:
                    if not result.success:
                        exp_type = result.expectation_config.expectation_type
                        kwargs = result.expectation_config.kwargs
                        print(f"  ✗ {exp_type}")
                        if "column" in kwargs:
                            print(f"    Column: {kwargs['column']}")
                        if "value_set" in kwargs:
                            print(f"    Expected values: {kwargs['value_set']}")
                        if "min_value" in kwargs or "max_value" in kwargs:
                            print(
                                f"    Expected range: [{kwargs.get('min_value', 'N/A')}, {kwargs.get('max_value', 'N/A')}]"
                            )
                        if hasattr(result, "result") and "observed_value" in result.result:
                            print(f"    Observed: {result.result['observed_value']}")
                        print()

            return {
                "success": success,
                "statistics": stats,
                "checkpoint_name": checkpoint_name,
            }

        except Exception as e:
            print(f"✗ Validation failed with error: {e}")
            return {
                "success": False,
                "error": str(e),
                "checkpoint_name": checkpoint_name,
            }

    def _skip_validation(self, stage: str) -> dict:
        """Return skip result when GX is not available."""
        print(f"⊗ Skipping {stage} validation (Great Expectations not configured)")
        return {
            "success": True,
            "skipped": True,
            "stage": stage,
        }


def main():
    """CLI entry point for data validation."""
    parser = argparse.ArgumentParser(description="Validate data with Great Expectations")
    parser.add_argument(
        "--stage",
        choices=["raw", "processed", "features", "all"],
        default="all",
        help="Which data stage to validate",
    )
    parser.add_argument(
        "--fail-on-error", action="store_true", help="Exit with error code if validation fails"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = DataValidator()

    # Define data paths
    project_root = Path(__file__).parents[2]
    raw_train = project_root / "data" / "raw" / "train.csv"
    raw_store = project_root / "data" / "raw" / "store.csv"
    processed = project_root / "data" / "processed" / "train_clean.parquet"
    features = project_root / "data" / "processed" / "train_features.parquet"

    # Run validations based on stage
    all_results = {}

    if args.stage in ["raw", "all"]:
        if raw_train.exists():
            all_results["raw_train"] = validator.validate_raw_train_data(raw_train)
        else:
            print(f"⊗ Skipping raw train validation (file not found: {raw_train})")

        if raw_store.exists():
            all_results["raw_store"] = validator.validate_raw_store_data(raw_store)
        else:
            print(f"⊗ Skipping raw store validation (file not found: {raw_store})")

    if args.stage in ["processed", "all"]:
        if processed.exists():
            all_results["processed"] = validator.validate_processed_data(processed)
        else:
            print(f"⊗ Skipping processed validation (file not found: {processed})")

    if args.stage in ["features", "all"]:
        if features.exists():
            all_results["features"] = validator.validate_features(features)
        else:
            print(f"⊗ Skipping features validation (file not found: {features})")

    # Print summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}\n")

    all_passed = all(result.get("success", False) for result in all_results.values())

    for stage, result in all_results.items():
        status = "✓ PASSED" if result.get("success") else "✗ FAILED"
        skipped = " (skipped)" if result.get("skipped") else ""
        print(f"{stage:20s}: {status}{skipped}")

    print(f"\n{'=' * 60}\n")

    # Exit with error if requested and validation failed
    if args.fail_on_error and not all_passed:
        print("✗ Validation failed. Exiting with error code.")
        sys.exit(1)

    print("✓ Validation complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
