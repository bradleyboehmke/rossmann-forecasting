"""Data loading and cleaning functions for the Rossmann forecasting project."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from utils.io import read_csv, save_parquet
from utils.log import get_logger

logger = get_logger(__name__)


def load_raw_data(raw_path: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train.csv and store.csv files.

    Parameters
    ----------
    raw_path : str, default="data/raw"
        Path to directory containing raw data files

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        train_df, store_df
    """
    logger.info(f"Loading raw data from {raw_path}")

    train_path = Path(raw_path) / "train.csv"
    store_path = Path(raw_path) / "store.csv"

    # Load train data
    logger.info(f"Reading {train_path}")
    train_df = read_csv(train_path)
    logger.info(f"Loaded train data: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")

    # Load store data
    logger.info(f"Reading {store_path}")
    store_df = read_csv(store_path)
    logger.info(f"Loaded store data: {store_df.shape[0]:,} rows, {store_df.shape[1]} columns")

    return train_df, store_df


def merge_store_info(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merge train and store dataframes on Store column.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with daily sales
    store_df : pd.DataFrame
        Store metadata

    Returns
    -------
    pd.DataFrame
        Merged dataframe
    """
    logger.info("Merging train and store data on 'Store' column")

    # Perform left join to preserve all train records
    merged_df = train_df.merge(store_df, on="Store", how="left")

    logger.info(f"Merged data shape: {merged_df.shape[0]:,} rows, {merged_df.shape[1]} columns")

    # Check for any stores in train that don't have store metadata
    missing_stores = merged_df[merged_df["StoreType"].isna()]["Store"].unique()
    if len(missing_stores) > 0:
        logger.warning(f"Found {len(missing_stores)} stores in train data without store metadata")
    else:
        logger.info("All stores have metadata")

    return merged_df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning steps.

    Steps include:

    - Convert Date to datetime
    - Handle missing values in competition and promo fields
    - Convert categorical fields to appropriate dtypes
    - Sort by Store and Date

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    logger.info("Starting basic data cleaning")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Convert Date to datetime
    logger.info("Converting Date column to datetime")
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by Store and Date for time-series operations
    logger.info("Sorting by Store and Date")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Handle missing values in competition fields
    logger.info("Handling missing values in competition fields")

    # CompetitionDistance: Fill with a large value to indicate no nearby competition
    comp_dist_missing = df["CompetitionDistance"].isna().sum()
    if comp_dist_missing > 0:
        logger.info(f"Filling {comp_dist_missing:,} missing CompetitionDistance values with 100000")
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(100000)

    # CompetitionOpenSince: Fill with date values indicating no competition
    comp_month_missing = df["CompetitionOpenSinceMonth"].isna().sum()
    comp_year_missing = df["CompetitionOpenSinceYear"].isna().sum()
    if comp_month_missing > 0 or comp_year_missing > 0:
        logger.info(
            f"Filling {comp_month_missing:,} missing CompetitionOpenSinceMonth and "
            f"{comp_year_missing:,} missing CompetitionOpenSinceYear with 0"
        )
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)

    # Promo2 fields: Fill missing values
    promo2_week_missing = df["Promo2SinceWeek"].isna().sum()
    promo2_year_missing = df["Promo2SinceYear"].isna().sum()
    if promo2_week_missing > 0 or promo2_year_missing > 0:
        logger.info(
            f"Filling {promo2_week_missing:,} missing Promo2SinceWeek and "
            f"{promo2_year_missing:,} missing Promo2SinceYear with 0"
        )
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
        df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)

    # PromoInterval: Fill with empty string
    promo_interval_missing = df["PromoInterval"].isna().sum()
    if promo_interval_missing > 0:
        logger.info(f"Filling {promo_interval_missing:,} missing PromoInterval with empty string")
        df["PromoInterval"] = df["PromoInterval"].fillna("")

    # Convert categorical fields to category dtype
    logger.info("Converting categorical fields to category dtype")
    categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert integer fields
    logger.info("Ensuring correct dtypes for integer fields")
    int_cols = ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "Promo2"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    # Convert float fields
    float_cols = ["Sales", "Customers", "CompetitionDistance"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Convert competition and promo2 since fields to int
    comp_promo_cols = [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]
    for col in comp_promo_cols:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    logger.info(f"Cleaning complete. Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Number of unique stores: {df['Store'].nunique()}")

    return df


def save_processed_data(
    df: pd.DataFrame, output_path: str = "data/processed/train_clean.parquet"
) -> None:
    """Save cleaned dataframe to parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe to save
    output_path : str, default="data/processed/train_clean.parquet"
        Output file path
    """
    logger.info(f"Saving cleaned data to {output_path}")

    save_parquet(df, output_path)

    # Report file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Saved {df.shape[0]:,} rows to {output_path} ({file_size_mb:.2f} MB)")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to summarize

    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_stores": df["Store"].nunique(),
        "date_range": (df["Date"].min(), df["Date"].max()),
        "n_days": (df["Date"].max() - df["Date"].min()).days,
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.to_dict(),
    }

    return summary


def main():
    """Main function to run the full data loading and cleaning pipeline."""
    logger.info("=" * 60)
    logger.info("Starting data loading and cleaning pipeline")
    logger.info("=" * 60)

    # Load raw data
    train_df, store_df = load_raw_data()

    # Merge store information
    merged_df = merge_store_info(train_df, store_df)

    # Clean data
    clean_df = basic_cleaning(merged_df)

    # Save processed data
    save_processed_data(clean_df)

    logger.info("=" * 60)
    logger.info("Data cleaning pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
