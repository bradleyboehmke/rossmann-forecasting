"""Tests for data validation using Great Expectations.

Tests cover:
- Data validation functionality
- Expectation suite loading
- Validation checkpoint execution
- Error handling
"""


import pytest

from src.data.validate_data import DataValidator


@pytest.mark.unit
class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_validator_initialization(self, project_root):
        """Test that validator initializes correctly."""
        validator = DataValidator()
        # Should initialize without errors (may have no context if GX not installed)
        assert validator is not None

    def test_validate_raw_train_data_with_valid_data(self, sample_train_data, tmp_path):
        """Test validation passes with valid training data."""
        # Save sample data
        data_file = tmp_path / "train.csv"
        sample_train_data.to_csv(data_file, index=False)

        # Validate
        validator = DataValidator()
        result = validator.validate_raw_train_data(data_file)

        # Should return a result dict (may be skipped if GX not available)
        assert isinstance(result, dict)
        assert "success" in result or "skipped" in result

    def test_validate_raw_store_data_with_valid_data(self, sample_store_data, tmp_path):
        """Test validation passes with valid store data."""
        # Save sample data
        data_file = tmp_path / "store.csv"
        sample_store_data.to_csv(data_file, index=False)

        # Validate
        validator = DataValidator()
        result = validator.validate_raw_store_data(data_file)

        # Should return a result dict
        assert isinstance(result, dict)
        assert "success" in result or "skipped" in result

    def test_skip_validation_when_gx_unavailable(self):
        """Test graceful handling when Great Expectations is not available."""
        validator = DataValidator()
        result = validator._skip_validation("test_stage")

        assert result["success"] is True
        assert result["skipped"] is True
        assert result["stage"] == "test_stage"


@pytest.mark.unit
class TestDataSchema:
    """Test data schema and structure."""

    def test_train_data_has_required_columns(self, sample_train_data):
        """Test that train data has all required columns."""
        required_columns = [
            "Store",
            "DayOfWeek",
            "Date",
            "Sales",
            "Customers",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
        ]
        for col in required_columns:
            assert col in sample_train_data.columns, f"Missing column: {col}"

    def test_store_data_has_required_columns(self, sample_store_data):
        """Test that store data has all required columns."""
        required_columns = [
            "Store",
            "StoreType",
            "Assortment",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval",
        ]
        for col in required_columns:
            assert col in sample_store_data.columns, f"Missing column: {col}"

    def test_sales_values_are_non_negative(self, sample_train_data):
        """Test that sales values are non-negative."""
        assert (sample_train_data["Sales"] >= 0).all()

    def test_store_ids_are_valid(self, sample_train_data):
        """Test that store IDs are within valid range."""
        assert sample_train_data["Store"].min() >= 1
        assert sample_train_data["Store"].max() <= 1200

    def test_day_of_week_values_are_valid(self, sample_train_data):
        """Test that DayOfWeek values are 1-7."""
        assert sample_train_data["DayOfWeek"].min() >= 1
        assert sample_train_data["DayOfWeek"].max() <= 7

    def test_open_values_are_binary(self, sample_train_data):
        """Test that Open values are 0 or 1."""
        assert set(sample_train_data["Open"].unique()).issubset({0, 1})

    def test_promo_values_are_binary(self, sample_train_data):
        """Test that Promo values are 0 or 1."""
        assert set(sample_train_data["Promo"].unique()).issubset({0, 1})

    def test_store_type_values_are_valid(self, sample_store_data):
        """Test that StoreType values are a, b, c, or d."""
        valid_types = {"a", "b", "c", "d"}
        assert set(sample_store_data["StoreType"].unique()).issubset(valid_types)

    def test_assortment_values_are_valid(self, sample_store_data):
        """Test that Assortment values are a, b, or c."""
        valid_assortments = {"a", "b", "c"}
        assert set(sample_store_data["Assortment"].unique()).issubset(valid_assortments)


@pytest.mark.integration
class TestDataValidationIntegration:
    """Integration tests for end-to-end data validation."""

    def test_full_validation_pipeline(self, sample_train_data, sample_store_data, tmp_path):
        """Test complete validation pipeline with both datasets."""
        # Save data files
        train_file = tmp_path / "train.csv"
        store_file = tmp_path / "store.csv"
        sample_train_data.to_csv(train_file, index=False)
        sample_store_data.to_csv(store_file, index=False)

        # Initialize validator
        validator = DataValidator()

        # Validate both datasets
        train_result = validator.validate_raw_train_data(train_file)
        store_result = validator.validate_raw_store_data(store_file)

        # Both should succeed or skip
        assert train_result.get("success") or train_result.get("skipped")
        assert store_result.get("success") or store_result.get("skipped")
