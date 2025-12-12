"""I/O utilities for reading and writing data files."""

from pathlib import Path
from typing import Union

import pandas as pd


def read_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read CSV file into pandas DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    **kwargs
        Additional arguments passed to pd.read_csv

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    return pd.read_csv(filepath, **kwargs)


def read_parquet(filepath: Union[str, Path], categorize: bool = True, **kwargs) -> pd.DataFrame:
    """Read Parquet file into pandas DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to Parquet file
    categorize : bool, default=True
        If True, convert known categorical string columns back to category dtype.
        Useful for columns like StoreType, Assortment, StateHoliday, PromoInterval.
    **kwargs
        Additional arguments passed to pd.read_parquet

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    df = pd.read_parquet(filepath, **kwargs)

    if categorize:
        # List of columns that should be categorical in Rossmann dataset
        categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

        for col in categorical_cols:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype("category")

    return df


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str or Path
        Output path
    **kwargs
        Additional arguments passed to df.to_csv
    """
    # Create parent directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Default to not writing index
    if "index" not in kwargs:
        kwargs["index"] = False

    df.to_csv(filepath, **kwargs)


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str or Path
        Output path
    **kwargs
        Additional arguments passed to df.to_parquet
    """
    # Create parent directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Default to not writing index
    if "index" not in kwargs:
        kwargs["index"] = False

    # Make a copy to avoid modifying original
    df_to_save = df.copy()

    # Convert categorical columns to string to avoid Arrow conversion issues
    # This is especially important for categories with mixed string/numeric values
    categorical_columns = df_to_save.select_dtypes(include=["category"]).columns
    for col in categorical_columns:
        df_to_save[col] = df_to_save[col].astype(str)

    df_to_save.to_parquet(filepath, **kwargs)


def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    dirpath : str or Path
        Directory path to ensure exists

    Returns
    -------
    Path
        Path object for the directory
    """
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    return path
