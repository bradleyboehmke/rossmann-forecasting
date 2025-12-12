"""Tests for feature engineering (standard/proven features in DataOps workflow).

These tests validate the standard features that are automatically created
in the dataops_workflow.sh script. Features tested include:
- Calendar features (year, month, quarter, season, weekend)
- Promotion features
- Competition features
- Lag features (time-series safe)
- Rolling features (time-series safe)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.build_features import (
    add_calendar_features,
    add_competition_features,
    add_lag_features,
    add_promo_features,
    add_rolling_features,
    build_all_features,
)


class TestCalendarFeatures:
    """Test calendar-based feature engineering."""

    def test_calendar_features_created(self, sample_train_data):
        """Test that all expected calendar features are created."""
        result = add_calendar_features(sample_train_data)

        expected_features = [
            "Year",
            "Month",
            "Week",
            "Day",
            "DayOfMonth",
            "Quarter",
            "IsMonthStart",
            "IsMonthEnd",
            "IsQuarterStart",
            "IsQuarterEnd",
            "Season",
            "IsWeekend",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_year_extracted_correctly(self, sample_train_data):
        """Test that year is extracted from Date."""
        result = add_calendar_features(sample_train_data)
        assert result["Year"].min() >= 2013
        assert result["Year"].max() <= 2025

    def test_month_in_valid_range(self, sample_train_data):
        """Test that month is between 1 and 12."""
        result = add_calendar_features(sample_train_data)
        assert result["Month"].min() >= 1
        assert result["Month"].max() <= 12

    def test_quarter_in_valid_range(self, sample_train_data):
        """Test that quarter is between 1 and 4."""
        result = add_calendar_features(sample_train_data)
        assert result["Quarter"].min() >= 1
        assert result["Quarter"].max() <= 4

    def test_season_values(self, sample_train_data):
        """Test that season contains valid values."""
        result = add_calendar_features(sample_train_data)
        # Season is encoded as integers: 0=Winter, 1=Spring, 2=Summer, 3=Fall
        valid_seasons = {0, 1, 2, 3}
        assert set(result["Season"].unique()).issubset(valid_seasons)

    def test_weekend_flag(self, sample_train_data):
        """Test that weekend flag is correctly set."""
        result = add_calendar_features(sample_train_data)
        # DayOfWeek: Monday=1, Sunday=7
        # IsWeekend should be 1 for Saturday(6) and Sunday(7)
        weekend_rows = result[result["DayOfWeek"].isin([6, 7])]
        assert (weekend_rows["IsWeekend"] == 1).all()

    def test_month_start_flag(self, sample_train_data):
        """Test that month start flag is set for first day of month."""
        result = add_calendar_features(sample_train_data)
        month_starts = result[result["DayOfMonth"] == 1]
        assert (month_starts["IsMonthStart"] == 1).all()


class TestPromoFeatures:
    """Test promotion-related features."""

    def test_promo_features_created(self, sample_features_data):
        """Test that promotion features are created."""
        result = add_promo_features(sample_features_data)

        expected_features = [
            "Promo2Active",
            "Promo2Duration",
            "PromoInterval_JAJO",
            "PromoInterval_FMAN",
            "PromoInterval_MJSD",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_promo2_active_binary(self, sample_features_data):
        """Test that Promo2Active is binary."""
        result = add_promo_features(sample_features_data)
        assert set(result["Promo2Active"].unique()).issubset({0, 1})

    def test_promo2_duration_non_negative(self, sample_features_data):
        """Test that Promo2Duration is non-negative."""
        result = add_promo_features(sample_features_data)
        assert (result["Promo2Duration"] >= 0).all()


class TestCompetitionFeatures:
    """Test competition-related features."""

    def test_competition_features_created(self, sample_features_data):
        """Test that competition features are created."""
        result = add_competition_features(sample_features_data)

        expected_features = ["CompetitionDistance_log", "CompetitionAge", "HasCompetition"]

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_has_competition_binary(self, sample_features_data):
        """Test that HasCompetition is binary."""
        result = add_competition_features(sample_features_data)
        assert set(result["HasCompetition"].unique()).issubset({0, 1})

    def test_competition_age_non_negative(self, sample_features_data):
        """Test that competition age is non-negative."""
        result = add_competition_features(sample_features_data)
        assert (result["CompetitionAge"] >= 0).all()

    def test_competition_distance_log_handled(self, sample_features_data):
        """Test that log distance handles zero/missing values."""
        result = add_competition_features(sample_features_data)
        # Should not have inf or NaN values
        assert not result["CompetitionDistance_log"].isnull().any()
        assert not np.isinf(result["CompetitionDistance_log"]).any()


class TestLagFeatures:
    """Test lag features (critical for time-series safety)."""

    def test_lag_features_created(self, sample_train_data):
        """Test that lag features are created for specified lags."""
        lags = [1, 7, 14]
        result = add_lag_features(sample_train_data, lags=lags, target_col="Sales")

        for lag in lags:
            feature_name = f"Sales_Lag_{lag}"
            assert feature_name in result.columns, f"Missing feature: {feature_name}"

    def test_lag_values_shifted_correctly(self):
        """Test that lag values are correctly shifted per store."""
        # Create simple test data with known pattern
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Store": [1] * 10,
                "Date": dates,
                "Sales": range(100, 110),  # 100, 101, 102, ...
                "Open": [1] * 10,
            }
        )

        result = add_lag_features(df, lags=[1], target_col="Sales")

        # First row should have NaN for lag
        assert pd.isna(result.iloc[0]["Sales_Lag_1"])
        # Second row should have first row's sales value
        assert result.iloc[1]["Sales_Lag_1"] == 100
        # Third row should have second row's sales value
        assert result.iloc[2]["Sales_Lag_1"] == 101

    def test_lags_grouped_by_store(self):
        """CRITICAL: Test that lags are calculated per store (no leakage)."""
        # Create data with two stores
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Store": [1, 1, 1, 2, 2],
                "Date": list(dates[:3]) + list(dates[:2]),
                "Sales": [10, 20, 30, 100, 200],
                "Open": [1, 1, 1, 1, 1],
            }
        )
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        result = add_lag_features(df, lags=[1], target_col="Sales")

        # Store 1's second row should have Store 1's first value (10), NOT Store 2's
        store1_data = result[result["Store"] == 1]
        assert store1_data.iloc[1]["Sales_Lag_1"] == 10

        # Store 2's first row should have NaN (not carry over from Store 1)
        store2_data = result[result["Store"] == 2]
        assert pd.isna(store2_data.iloc[0]["Sales_Lag_1"])


class TestRollingFeatures:
    """Test rolling window features (critical for time-series safety)."""

    def test_rolling_features_created(self, sample_train_data):
        """Test that rolling features are created."""
        windows = [7, 14]
        result = add_rolling_features(sample_train_data, windows=windows, target_col="Sales")

        for window in windows:
            mean_feature = f"Sales_RollingMean_{window}"
            std_feature = f"Sales_RollingStd_{window}"
            assert mean_feature in result.columns
            assert std_feature in result.columns

    def test_rolling_mean_calculated_correctly(self):
        """Test that rolling mean is calculated correctly."""
        # Simple test case
        df = pd.DataFrame(
            {
                "Store": [1] * 10,
                "Date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "Sales": [10] * 10,  # All same value
                "Open": [1] * 10,
            }
        )

        result = add_rolling_features(df, windows=[3], target_col="Sales")

        # Rolling mean of constant should be constant (after window fills)
        non_null_means = result["Sales_RollingMean_3"].dropna()
        assert (non_null_means == 10).all()

    def test_rolling_grouped_by_store(self):
        """CRITICAL: Test that rolling is calculated per store (no leakage)."""
        # Two stores with different sales patterns
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Store": [1, 1, 1, 2, 2, 2],
                "Date": list(dates[:3]) + list(dates[:3]),
                "Sales": [10, 10, 10, 100, 100, 100],
                "Open": [1, 1, 1, 1, 1, 1],
            }
        )
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        result = add_rolling_features(df, windows=[2], target_col="Sales")

        # Store 1 rolling mean should be around 10
        store1_means = result[result["Store"] == 1]["Sales_RollingMean_2"].dropna()
        assert (store1_means == 10).all()

        # Store 2 rolling mean should be around 100
        store2_means = result[result["Store"] == 2]["Sales_RollingMean_2"].dropna()
        assert (store2_means == 100).all()


class TestBuildAllFeatures:
    """Integration tests for the complete feature pipeline."""

    def test_all_features_created(self, sample_features_data):
        """Test that build_all_features creates all expected features."""
        result = build_all_features(sample_features_data)

        # Should have original columns plus new features
        assert len(result.columns) > len(sample_features_data.columns)

        # Check for presence of each feature category
        assert "Year" in result.columns  # Calendar
        assert "Promo2Active" in result.columns  # Promo
        assert "CompetitionAge" in result.columns  # Competition
        assert "Sales_Lag_1" in result.columns  # Lag
        assert "Sales_RollingMean_7" in result.columns  # Rolling

    def test_no_data_leakage_in_full_pipeline(self):
        """CRITICAL: Test that full pipeline maintains time-series integrity."""
        # Create simple multi-store dataset
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        stores = [1, 2]
        data = []

        for store in stores:
            for date in dates:
                data.append(
                    {
                        "Store": store,
                        "Date": date,
                        "Sales": store * 100 + np.random.randint(0, 10),
                        "Open": 1,
                        "DayOfWeek": date.dayofweek + 1,
                        "Promo": 0,
                        "StateHoliday": "0",
                        "SchoolHoliday": 0,
                        "StoreType": "a",
                        "Assortment": "a",
                        "CompetitionDistance": 1000,
                        "CompetitionOpenSinceMonth": 1,
                        "CompetitionOpenSinceYear": 2019,
                        "Promo2": 0,
                        "Promo2SinceWeek": 0,
                        "Promo2SinceYear": 0,
                        "PromoInterval": "",
                    }
                )

        df = pd.DataFrame(data)
        result = build_all_features(df)

        # Verify lags don't cross stores
        for store in stores:
            store_data = result[result["Store"] == store].sort_values("Date")
            # First row should have NaN lags
            assert pd.isna(store_data.iloc[0]["Sales_Lag_1"])

    def test_output_has_no_unexpected_nulls(self, sample_features_data):
        """Test that feature engineering doesn't introduce unexpected nulls."""
        result = build_all_features(sample_features_data)

        # Original non-null columns should remain non-null
        for col in ["Store", "Date", "Sales"]:
            assert not result[col].isnull().any()

        # Calendar features should never be null
        assert not result["Year"].isnull().any()
        assert not result["Month"].isnull().any()

    def test_feature_count(self, sample_features_data):
        """Test that expected number of features are created."""
        result = build_all_features(sample_features_data)

        # Count new features added
        # Note: sample_features_data already has 4 calendar features (Year, Month, Week, Day)
        # Calendar: 12 total, 4 already present = 8 new (DayOfMonth, Quarter, IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd, Season, IsWeekend)
        # Promo: 5 (Promo2Active, Promo2Duration, PromoInterval_JAJO, PromoInterval_FMAN, PromoInterval_MJSD)
        # Competition: 3 (CompetitionDistance_log, HasCompetition, CompetitionAge)
        # Lags: 4 (Sales_Lag_1, Sales_Lag_7, Sales_Lag_14, Sales_Lag_28)
        # Rolling: 8 (mean + std for windows 7, 14, 28, 60)
        # Total new features: 8 + 5 + 3 + 4 + 8 = 28
        new_features = len(result.columns) - len(sample_features_data.columns)
        assert new_features >= 28, f"Expected ~28 new features, got {new_features}"
