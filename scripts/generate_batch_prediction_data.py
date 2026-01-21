"""Generate batch prediction data for testing monitoring system.

This script creates realistic prediction data for Aug-Dec 2015 with:
- Multiple stores (50 stores)
- All days in the date range
- Realistic patterns based on training data
- Intentional drift in 1-2 features to test drift detection
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.io import read_csv

# Set random seed for reproducibility
np.random.seed(42)


def generate_date_range(start_date: str, end_date: str) -> list[str]:
    """Generate list of dates in range.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format

    Returns
    -------
    list of str
        List of dates
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


def get_day_of_week(date_str: str) -> int:
    """Get day of week (1=Monday, 7=Sunday).

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format

    Returns
    -------
    int
        Day of week (1-7)
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    # Python weekday: 0=Monday, 6=Sunday
    # We want: 1=Monday, 7=Sunday
    return date.weekday() + 1


def is_state_holiday(date_str: str) -> str:
    """Determine if date is a state holiday.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format

    Returns
    -------
    str
        Holiday type: '0', 'a' (public), 'b' (easter), 'c' (christmas)
    """
    _ = datetime.strptime(date_str, "%Y-%m-%d")  # Parse date for validation

    # German public holidays in 2015
    public_holidays = [
        "2015-10-03",  # German Unity Day
    ]

    # Christmas season
    christmas_holidays = [
        "2015-12-24",  # Christmas Eve
        "2015-12-25",  # Christmas Day
        "2015-12-26",  # Boxing Day
        "2015-12-31",  # New Year's Eve
    ]

    if date_str in public_holidays:
        return "a"
    elif date_str in christmas_holidays:
        return "c"
    else:
        return "0"


def is_school_holiday(date_str: str) -> int:
    """Determine if date is a school holiday.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format

    Returns
    -------
    int
        1 if school holiday, 0 otherwise
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # German school holidays (approximate)
    # Summer holidays: late July - early September
    # Fall break: mid-October
    # Christmas holidays: late December

    if (date.month == 8) or (date.month == 10 and 12 <= date.day <= 23):
        return 1
    elif date.month == 12 and date.day >= 21:
        return 1
    else:
        return 0


def generate_batch_prediction_data(
    start_date: str = "2015-08-01",
    end_date: str = "2015-12-31",
    num_stores: int = 50,
    introduce_drift: bool = True,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Generate batch prediction data with optional drift.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    num_stores : int
        Number of stores to include
    introduce_drift : bool
        Whether to introduce intentional drift in some features
    output_path : str, optional
        Path to save CSV file

    Returns
    -------
    pd.DataFrame
        Generated prediction data
    """
    print("=" * 70)
    print("BATCH PREDICTION DATA GENERATOR")
    print("=" * 70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of stores: {num_stores}")
    print(f"Introduce drift: {introduce_drift}")
    print()

    # Load store metadata to get valid store IDs
    store_path = PROJECT_ROOT / "data" / "raw" / "store.csv"
    stores_df = read_csv(store_path)

    # Select random stores (but consistently)
    selected_stores = sorted(stores_df["Store"].sample(n=num_stores, random_state=42).tolist())

    print(f"Selected stores: {selected_stores[:10]}... (showing first 10)")
    print()

    # Generate dates
    dates = generate_date_range(start_date, end_date)
    print(f"Generated {len(dates)} dates")
    print()

    # Create all combinations of stores and dates
    records = []
    for store_id in selected_stores:
        for date in dates:
            day_of_week = get_day_of_week(date)
            state_holiday = is_state_holiday(date)
            school_holiday = is_school_holiday(date)

            # Determine if store is open
            # Stores are typically closed on Sundays and state holidays
            if day_of_week == 7 or state_holiday != "0":
                open_status = 0
            else:
                open_status = 1

            # Determine promo status
            # Base promo probability: 30%
            # Higher on weekends (Fridays/Saturdays): 50%
            # Lower during holidays: 10%
            if state_holiday != "0":
                promo_prob = 0.10
            elif day_of_week in [5, 6]:  # Friday, Saturday
                promo_prob = 0.50
            else:
                promo_prob = 0.30

            # DRIFT INJECTION: Increase promo rate in November-December
            if introduce_drift:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if date_obj.month >= 11:  # November-December
                    # Increase promo rate by 25 percentage points (30% -> 55%)
                    promo_prob += 0.25
                    promo_prob = min(promo_prob, 0.90)  # Cap at 90%

            promo = 1 if np.random.random() < promo_prob else 0

            # DRIFT INJECTION: Increase school_holiday rate slightly in December
            if introduce_drift:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if date_obj.month == 12 and school_holiday == 0:
                    # Add some noise - randomly mark 20% of non-holidays as holidays
                    if np.random.random() < 0.20:
                        school_holiday = 1

            records.append(
                {
                    "Store": store_id,
                    "DayOfWeek": day_of_week,
                    "Date": date,
                    "Open": open_status,
                    "Promo": promo,
                    "StateHoliday": state_holiday,
                    "SchoolHoliday": school_holiday,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(records)

    print(f"Generated {len(df):,} total records")
    print()

    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 70)
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Stores: {df['Store'].nunique()} unique stores")
    print(f"Open rate: {df['Open'].mean():.1%}")
    print(f"Promo rate: {df['Promo'].mean():.1%}")
    print(f"School holiday rate: {df['SchoolHoliday'].mean():.1%}")
    print(f"State holiday rate: {(df['StateHoliday'] != '0').mean():.1%}")
    print()

    # Print drift indicators if enabled
    if introduce_drift:
        print("Drift Indicators:")
        print("-" * 70)

        # Promo drift by month
        df_temp = df.copy()
        df_temp["Month"] = pd.to_datetime(df_temp["Date"]).dt.month
        monthly_promo = df_temp.groupby("Month")["Promo"].mean()
        print("Promo rate by month:")
        for month, rate in monthly_promo.items():
            month_name = datetime(2015, month, 1).strftime("%B")
            print(f"  {month_name}: {rate:.1%}")
        print()

        # School holiday drift by month
        monthly_school = df_temp.groupby("Month")["SchoolHoliday"].mean()
        print("School holiday rate by month:")
        for month, rate in monthly_school.items():
            month_name = datetime(2015, month, 1).strftime("%B")
            print(f"  {month_name}: {rate:.1%}")
        print()

    # Save to file if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
        print()

    print("=" * 70)
    print("Generation complete!")
    print("=" * 70)

    return df


def main():
    """Generate batch prediction data for testing."""
    # Generate data with drift
    output_path = PROJECT_ROOT / "data" / "batch_predictions" / "batch_aug_dec_2015.csv"

    _ = generate_batch_prediction_data(
        start_date="2015-08-01",
        end_date="2015-12-31",
        num_stores=50,
        introduce_drift=True,
        output_path=output_path,
    )

    print()
    print("Next steps:")
    print("-" * 70)
    print("1. Upload this file to the Streamlit dashboard Batch Upload page:")
    print(f"   {output_path.relative_to(PROJECT_ROOT)}")
    print()
    print("2. After predictions are made, check the Monitoring page for drift detection")
    print()
    print("3. Generate drift report from command line:")
    print("   python src/monitoring/generate_reports.py --days 30")
    print()


if __name__ == "__main__":
    main()
