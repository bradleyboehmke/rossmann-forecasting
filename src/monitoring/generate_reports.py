"""CLI tool for generating monitoring reports.

This script generates drift detection reports comparing production predictions against reference
training data.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from monitoring.drift_detection import generate_drift_report_cli

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate drift detection reports for model monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for last 7 days
  python src/monitoring/generate_reports.py --days 7

  # Generate report for last 30 days
  python src/monitoring/generate_reports.py --days 30

  # Custom output path
  python src/monitoring/generate_reports.py --days 14 --output monitoring/custom_report.html

  # Custom reference data
  python src/monitoring/generate_reports.py --reference-data path/to/reference.parquet
        """,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of production data to analyze (default: 7)",
    )

    parser.add_argument(
        "--reference-data",
        type=Path,
        default=None,
        help="Path to reference training data (default: data/processed/train_features.parquet)",
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to predictions database (default: data/monitoring/predictions.db)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save HTML report (default: monitoring/drift_reports/drift_report_<date>.html)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("DRIFT DETECTION REPORT GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Analyzing last {args.days} days of production data")

    # Generate report
    try:
        summary = generate_drift_report_cli(
            days=args.days,
            reference_data_path=args.reference_data,
            db_path=args.db_path,
            output_path=args.output,
        )

        # Check for errors
        if "error" in summary:
            logger.error(f"Report generation failed: {summary['error']}")
            sys.exit(1)

        # Create symlink to latest report
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            output_path = (
                PROJECT_ROOT / "monitoring" / "drift_reports" / f"drift_report_{timestamp}.html"
            )

        latest_link = PROJECT_ROOT / "monitoring" / "drift_reports" / "latest.html"
        if output_path.exists():
            # Remove old symlink if exists
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()

            # Create new symlink
            try:
                latest_link.symlink_to(output_path.name)
                logger.info(f"✓ Created symlink: {latest_link} -> {output_path.name}")
            except OSError as e:
                logger.warning(f"Could not create symlink: {e}")

        logger.info("=" * 70)
        logger.info("✓ Report generation complete!")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(
            "\nMake sure you have:\n"
            "  1. Generated reference data: python src/monitoring/prepare_reference_data.py\n"
            "  2. Made some predictions via the API to populate the database"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
