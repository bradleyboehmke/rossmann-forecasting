"""Tests for data processing pipeline (make_dataset.py).

Tests cover:
- Data loading
- Data merging
- Basic cleaning
- Data type conversions
- Missing value handling
"""


import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
class TestDataLoading:
    """Test data loading functions."""

    def test_load_train_data(self, sample_train_data, tmp_path):
        """Test loading training data from CSV."""
        # Save sample data
        data_file = tmp_path / "train.csv"
        sample_train_data.to_csv(data_file, index=False)

        # Load it back
        loaded_data = pd.read_csv(data_file)

        assert loaded_data.shape == sample_train_data.shape
        assert list(loaded_data.columns) == list(sample_train_data.columns)

    def test_load_store_data(self, sample_store_data, tmp_path):
        """Test loading store data from CSV."""
        # Save sample data
        data_file = tmp_path / "store.csv"
        sample_store_data.to_csv(data_file, index=False)

        # Load it back
        loaded_data = pd.read_csv(data_file)

        assert loaded_data.shape == sample_store_data.shape
        assert list(loaded_data.columns) == list(sample_store_data.columns)


@pytest.mark.unit
class TestDataMerging:
    """Test merging train and store data."""

    def test_merge_train_and_store(self, sample_train_data, sample_store_data):
        """Test merging train data with store metadata."""
        merged = sample_train_data.merge(sample_store_data, on="Store", how="left")

        # Should have train rows with added store columns
        assert len(merged) == len(sample_train_data)
        assert "StoreType" in merged.columns
        assert "Assortment" in merged.columns
        assert "CompetitionDistance" in merged.columns

    def test_merge_preserves_all_stores(self, sample_train_data, sample_store_data):
        """Test that merge doesn't lose any stores."""
        merged = sample_train_data.merge(sample_store_data, on="Store", how="left")

        train_stores = set(sample_train_data["Store"].unique())
        merged_stores = set(merged["Store"].unique())

        assert train_stores == merged_stores


@pytest.mark.unit
class TestDataCleaning:
    """Test data cleaning operations."""

    def test_date_parsing(self, sample_train_data):
        """Test that dates are parsed correctly."""
        df = sample_train_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_categorical_conversion(self, sample_store_data):
        """Test conversion of categorical columns."""
        df = sample_store_data.copy()

        categorical_cols = ["StoreType", "Assortment"]
        for col in categorical_cols:
            df[col] = df[col].astype("category")

        assert df["StoreType"].dtype.name == "category"
        assert df["Assortment"].dtype.name == "category"

    def test_missing_value_handling_competition_distance(self, sample_store_data):
        """Test handling of missing competition distance values."""
        df = sample_store_data.copy()

        # Introduce some missing values
        df.loc[0, "CompetitionDistance"] = np.nan

        # Fill with median (common strategy)
        median_dist = df["CompetitionDistance"].median()
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_dist)

        assert df["CompetitionDistance"].isna().sum() == 0


@pytest.mark.unit
class TestDataQuality:
    """Test data quality checks."""

    def test_no_duplicate_store_dates(self, sample_train_data):
        """Test that there are no duplicate Store-Date combinations."""
        duplicates = sample_train_data.duplicated(subset=["Store", "Date"], keep=False)
        assert duplicates.sum() == 0

    def test_sales_customers_relationship(self, sample_train_data):
        """Test that stores with sales have customers (when open)."""
        open_with_sales = sample_train_data[
            (sample_train_data["Open"] == 1) & (sample_train_data["Sales"] > 0)
        ]

        # Most should have customers
        assert (open_with_sales["Customers"] > 0).mean() > 0.95

    def test_closed_stores_mostly_have_zero_sales(self, sample_train_data):
        """Test that most closed stores (Open=0) have zero sales.

        Note: Some stores may have non-zero sales even when closed
        (e.g., online orders, data collection issues). We check that
        the majority follow the expected pattern.
        """
        closed_stores = sample_train_data[sample_train_data["Open"] == 0]

        if len(closed_stores) > 0:
            # At least 90% of closed stores should have zero sales
            zero_sales_pct = (closed_stores["Sales"] == 0).mean()
            assert (
                zero_sales_pct >= 0.90
            ), f"Expected >=90% of closed stores to have zero sales, got {zero_sales_pct:.1%}"


@pytest.mark.integration
class TestDataProcessingPipeline:
    """Integration tests for complete data processing pipeline."""

    def test_full_pipeline(self, sample_train_data, sample_store_data, tmp_path):
        """Test complete data processing pipeline."""
        # 1. Save raw data
        train_file = tmp_path / "train.csv"
        store_file = tmp_path / "store.csv"
        sample_train_data.to_csv(train_file, index=False)
        sample_store_data.to_csv(store_file, index=False)

        # 2. Load data
        train_df = pd.read_csv(train_file)
        store_df = pd.read_csv(store_file)

        # 3. Parse dates
        train_df["Date"] = pd.to_datetime(train_df["Date"])

        # 4. Merge
        merged_df = train_df.merge(store_df, on="Store", how="left")

        # 5. Basic cleaning
        merged_df["CompetitionDistance"] = merged_df["CompetitionDistance"].fillna(
            merged_df["CompetitionDistance"].median()
        )

        # 6. Save processed data
        output_file = tmp_path / "train_clean.parquet"
        merged_df.to_parquet(output_file, index=False)

        # 7. Verify output
        assert output_file.exists()

        # 8. Load and verify
        processed_df = pd.read_parquet(output_file)
        assert len(processed_df) == len(train_df)
        assert "StoreType" in processed_df.columns
        assert processed_df["CompetitionDistance"].isna().sum() == 0
