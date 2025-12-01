"""
Feature engineering functions for the Rossmann forecasting project.

CRITICAL: All lag and rolling features MUST use proper groupby to prevent data leakage.
- Lag features: df.groupby("Store")["column"].shift(lag)
- Rolling features: df.groupby("Store")["column"].rolling(window).agg()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.io import read_parquet, save_parquet
from utils.log import get_logger

logger = get_logger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features derived from Date.

    Features created:
    - Year, Month, Week, Day, DayOfMonth
    - Quarter, IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd
    - Season (meteorological: Winter, Spring, Summer, Fall)
    - IsWeekend

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Date column (datetime)

    Returns
    -------
    pd.DataFrame
        Dataframe with added calendar features
    """
    logger.info("Adding calendar features")

    df = df.copy()

    # Basic date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['DayOfMonth'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter

    # Month flags
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype('int8')
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype('int8')
    df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype('int8')
    df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype('int8')

    # Weekend flag (DayOfWeek: 1=Mon, 7=Sun)
    df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype('int8')

    # Season (meteorological)
    # Winter: 12, 1, 2
    # Spring: 3, 4, 5
    # Summer: 6, 7, 8
    # Fall: 9, 10, 11
    df['Season'] = df['Month'].map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    }).astype('int8')

    # Convert to memory-efficient dtypes
    int_cols = ['Year', 'Month', 'Week', 'Day', 'DayOfMonth', 'Quarter']
    for col in int_cols:
        df[col] = df[col].astype('int16')

    logger.info(f"Added {11} calendar features")

    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add promotion-related features.

    Features created:
    - Promo2Active: Whether Promo2 is active in current month
    - Promo2Duration: How long store has been in Promo2 (in days)
    - PromoInterval_<Month>: One-hot encoding of PromoInterval months

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Promo, Promo2, PromoInterval columns

    Returns
    -------
    pd.DataFrame
        Dataframe with added promo features
    """
    logger.info("Adding promotion features")

    df = df.copy()

    # Promo2 active in current month
    # PromoInterval format: "Jan,Apr,Jul,Oct" or "Feb,May,Aug,Nov" or "Mar,Jun,Sept,Dec"
    month_abbr = df['Date'].dt.strftime('%b')

    # Check if current month is in PromoInterval
    def is_promo2_active(row):
        if pd.isna(row['PromoInterval']) or row['PromoInterval'] == '' or row['Promo2'] == 0:
            return 0
        # Handle "Sept" vs "Sep" inconsistency
        promo_months = row['PromoInterval'].replace('Sept', 'Sep')
        current_month = row['Month_Abbr'].replace('Sep', 'Sep')
        return 1 if current_month in promo_months else 0

    df['Month_Abbr'] = month_abbr
    df['Promo2Active'] = df.apply(is_promo2_active, axis=1).astype('int8')
    df = df.drop('Month_Abbr', axis=1)

    # Promo2 duration (days since Promo2 started)
    def calc_promo2_duration(row):
        if row['Promo2'] == 0 or row['Promo2SinceYear'] == 0:
            return 0

        # Convert Promo2SinceWeek and Promo2SinceYear to date
        try:
            promo2_start = pd.to_datetime(
                f"{int(row['Promo2SinceYear'])}-W{int(row['Promo2SinceWeek'])}-1",
                format='%Y-W%W-%w'
            )
            duration = (row['Date'] - promo2_start).days
            return max(0, duration)  # Return 0 if negative
        except:
            return 0

    df['Promo2Duration'] = df.apply(calc_promo2_duration, axis=1).astype('int32')

    # One-hot encode PromoInterval
    # Common patterns: "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"
    promo_patterns = {
        'Jan,Apr,Jul,Oct': 'PromoInterval_JAJO',
        'Feb,May,Aug,Nov': 'PromoInterval_FMAN',
        'Mar,Jun,Sept,Dec': 'PromoInterval_MJSD'
    }

    for pattern, col_name in promo_patterns.items():
        df[col_name] = (df['PromoInterval'] == pattern).astype('int8')

    logger.info(f"Added {5} promotion features")

    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add competition-related features.

    Features created:
    - CompetitionDistance_log: Log-scaled competition distance
    - CompetitionAge: Days since competition opened
    - HasCompetition: Binary flag for presence of competition

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with CompetitionDistance, CompetitionOpenSince columns

    Returns
    -------
    pd.DataFrame
        Dataframe with added competition features
    """
    logger.info("Adding competition features")

    df = df.copy()

    # Log-scaled competition distance
    # Use log1p to handle zeros and avoid -inf
    df['CompetitionDistance_log'] = np.log1p(df['CompetitionDistance']).astype('float32')

    # Has competition flag (distance < 100000, which was our fill value)
    df['HasCompetition'] = (df['CompetitionDistance'] < 100000).astype('int8')

    # Competition age (days since competition opened)
    def calc_competition_age(row):
        if row['CompetitionOpenSinceYear'] == 0 or row['CompetitionOpenSinceMonth'] == 0:
            return 0

        try:
            comp_open_date = pd.to_datetime(
                f"{int(row['CompetitionOpenSinceYear'])}-{int(row['CompetitionOpenSinceMonth'])}-01"
            )
            age = (row['Date'] - comp_open_date).days
            return max(0, age)  # Return 0 if negative
        except:
            return 0

    df['CompetitionAge'] = df.apply(calc_competition_age, axis=1).astype('int32')

    logger.info(f"Added {3} competition features")

    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 7, 14, 28],
    target_col: str = 'Sales'
) -> pd.DataFrame:
    """
    Add lag features at the store level.

    CRITICAL: Uses groupby("Store").shift(lag) to prevent data leakage.

    Features created:
    - Sales_Lag_{lag}: Sales from {lag} days ago for each store

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by Store and Date
    lags : list of int, default=[1, 7, 14, 28]
        Lag periods in days
    target_col : str, default='Sales'
        Column to create lags for

    Returns
    -------
    pd.DataFrame
        Dataframe with added lag features
    """
    logger.info(f"Adding lag features for {target_col} with lags: {lags}")

    df = df.copy()

    # CRITICAL: Must group by Store to prevent leakage across stores
    for lag in lags:
        col_name = f'{target_col}_Lag_{lag}'
        df[col_name] = df.groupby('Store')[target_col].shift(lag).astype('float32')
        logger.info(f"  Created {col_name}")

    logger.info(f"Added {len(lags)} lag features")

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [7, 14, 28, 60],
    target_col: str = 'Sales'
) -> pd.DataFrame:
    """
    Add rolling window features at the store level.

    CRITICAL: Uses groupby("Store").rolling(window) to prevent data leakage.

    Features created:
    - Sales_RollingMean_{window}: Rolling mean over {window} days
    - Sales_RollingStd_{window}: Rolling std over {window} days

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by Store and Date
    windows : list of int, default=[7, 14, 28, 60]
        Rolling window sizes in days
    target_col : str, default='Sales'
        Column to create rolling features for

    Returns
    -------
    pd.DataFrame
        Dataframe with added rolling features
    """
    logger.info(f"Adding rolling features for {target_col} with windows: {windows}")

    df = df.copy()

    # CRITICAL: Must group by Store to prevent leakage across stores
    for window in windows:
        # Rolling mean
        col_mean = f'{target_col}_RollingMean_{window}'
        df[col_mean] = (
            df.groupby('Store')[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype('float32')
        )
        logger.info(f"  Created {col_mean}")

        # Rolling std
        col_std = f'{target_col}_RollingStd_{window}'
        df[col_std] = (
            df.groupby('Store')[target_col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .astype('float32')
        )
        logger.info(f"  Created {col_std}")

    logger.info(f"Added {len(windows) * 2} rolling features")

    return df


def build_all_features(
    df: pd.DataFrame,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (from Phase 1)
    config : dict, optional
        Configuration with feature parameters
        Expected keys:
        - lags: list of lag periods
        - rolling_windows: list of window sizes
        - include_promo_features: bool
        - include_competition_features: bool

    Returns
    -------
    pd.DataFrame
        Dataframe with all engineered features
    """
    logger.info("="*60)
    logger.info("Starting feature engineering")
    logger.info("="*60)

    # Default config
    if config is None:
        config = {
            'lags': [1, 7, 14, 28],
            'rolling_windows': [7, 14, 28, 60],
            'include_promo_features': True,
            'include_competition_features': True
        }

    df = df.copy()
    initial_cols = len(df.columns)

    # Ensure data is sorted by Store and Date
    logger.info("Ensuring data is sorted by Store and Date")
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # Add calendar features
    df = add_calendar_features(df)

    # Add promotion features
    if config.get('include_promo_features', True):
        df = add_promo_features(df)
    else:
        logger.info("Skipping promotion features (disabled in config)")

    # Add competition features
    if config.get('include_competition_features', True):
        df = add_competition_features(df)
    else:
        logger.info("Skipping competition features (disabled in config)")

    # Add lag features
    lags = config.get('lags', [1, 7, 14, 28])
    df = add_lag_features(df, lags=lags)

    # Add rolling features
    windows = config.get('rolling_windows', [7, 14, 28, 60])
    df = add_rolling_features(df, windows=windows)

    final_cols = len(df.columns)
    added_cols = final_cols - initial_cols

    logger.info("="*60)
    logger.info(f"Feature engineering complete!")
    logger.info(f"Added {added_cols} new features")
    logger.info(f"Total columns: {final_cols}")
    logger.info("="*60)

    return df


def main():
    """
    Main function to run the feature engineering pipeline.
    """
    import yaml

    logger.info("="*60)
    logger.info("Starting feature engineering pipeline")
    logger.info("="*60)

    # Load configuration
    config_path = Path('config/params.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        feature_config = params.get('features', {})
    else:
        logger.warning("Config file not found, using defaults")
        feature_config = None

    # Load cleaned data
    logger.info("Loading cleaned data from data/processed/train_clean.parquet")
    df = read_parquet('data/processed/train_clean.parquet')
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Build features
    df_featured = build_all_features(df, config=feature_config)

    # Save featured data
    output_path = 'data/processed/train_features.parquet'
    logger.info(f"Saving featured data to {output_path}")
    save_parquet(df_featured, output_path)

    # Report file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(df_featured):,} rows to {output_path} ({file_size_mb:.2f} MB)")

    logger.info("="*60)
    logger.info("Feature engineering pipeline complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
