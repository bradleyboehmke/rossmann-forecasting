"""Input validation helpers for prediction data.

This module provides validation functions to ensure user inputs meet the requirements before sending
to the API.
"""

from datetime import datetime

import pandas as pd


def validate_store_id(store: int) -> tuple[bool, str]:
    """Validate store ID is within valid range.

    Parameters
    ----------
    store : int
        Store ID to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    if not isinstance(store, int):
        return False, "Store ID must be an integer"

    if store < 1 or store > 1115:
        return False, "Store ID must be between 1 and 1115"

    return True, ""


def validate_day_of_week(day: int) -> tuple[bool, str]:
    """Validate day of week is within valid range.

    Parameters
    ----------
    day : int
        Day of week to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    if not isinstance(day, int):
        return False, "Day of week must be an integer"

    if day < 1 or day > 7:
        return False, "Day of week must be between 1 (Monday) and 7 (Sunday)"

    return True, ""


def validate_date_format(date_str: str) -> tuple[bool, str]:
    """Validate date is in YYYY-MM-DD format.

    Parameters
    ----------
    date_str : str
        Date string to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True, ""
    except ValueError:
        return False, "Date must be in YYYY-MM-DD format"


def validate_binary_flag(value: int, field_name: str) -> tuple[bool, str]:
    """Validate binary flag is 0 or 1.

    Parameters
    ----------
    value : int
        Value to validate
    field_name : str
        Name of the field for error message

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    if value not in [0, 1]:
        return False, f"{field_name} must be 0 or 1"

    return True, ""


def validate_state_holiday(value: str) -> tuple[bool, str]:
    """Validate state holiday indicator.

    Parameters
    ----------
    value : str
        State holiday value to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    valid_values = ["0", "a", "b", "c"]
    if value not in valid_values:
        return False, f"StateHoliday must be one of: {', '.join(valid_values)}"

    return True, ""


def validate_single_prediction_input(
    store: int,
    day_of_week: int,
    date: str,
    open_flag: int,
    promo: int,
    state_holiday: str,
    school_holiday: int,
) -> tuple[bool, list[str]]:
    """Validate all inputs for a single prediction.

    Parameters
    ----------
    store : int
        Store ID
    day_of_week : int
        Day of week
    date : str
        Date string
    open_flag : int
        Open flag
    promo : int
        Promo flag
    state_holiday : str
        State holiday indicator
    school_holiday : int
        School holiday flag

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages)
    """
    errors = []

    # Validate each field
    is_valid, error = validate_store_id(store)
    if not is_valid:
        errors.append(error)

    is_valid, error = validate_day_of_week(day_of_week)
    if not is_valid:
        errors.append(error)

    is_valid, error = validate_date_format(date)
    if not is_valid:
        errors.append(error)

    is_valid, error = validate_binary_flag(open_flag, "Open")
    if not is_valid:
        errors.append(error)

    is_valid, error = validate_binary_flag(promo, "Promo")
    if not is_valid:
        errors.append(error)

    is_valid, error = validate_state_holiday(state_holiday)
    if not is_valid:
        errors.append(error)

    is_valid, error = validate_binary_flag(school_holiday, "SchoolHoliday")
    if not is_valid:
        errors.append(error)

    return len(errors) == 0, errors


def validate_batch_csv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate uploaded CSV for batch predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded DataFrame to validate

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages)
    """
    errors = []

    # Check required columns (DayOfWeek is now optional - will be auto-calculated)
    required_columns = [
        "Store",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors

    # Check data types and ranges
    if df.empty:
        errors.append("CSV file is empty")
        return False, errors

    # Validate Store IDs
    if not df["Store"].between(1, 1115).all():
        errors.append("Store IDs must be between 1 and 1115")

    # Validate Date format - try multiple common formats
    try:
        # Try to parse dates with flexible format detection
        pd.to_datetime(df["Date"], infer_datetime_format=True)
    except Exception:
        errors.append(
            "Date column contains invalid dates. Common formats: YYYY-MM-DD, MM/DD/YY, MM/DD/YYYY"
        )

    # Validate binary flags
    if not df["Open"].isin([0, 1]).all():
        errors.append("Open must be 0 or 1")

    if not df["Promo"].isin([0, 1]).all():
        errors.append("Promo must be 0 or 1")

    if not df["SchoolHoliday"].isin([0, 1]).all():
        errors.append("SchoolHoliday must be 0 or 1")

    # Validate StateHoliday
    valid_holidays = ["0", "a", "b", "c"]
    if not df["StateHoliday"].astype(str).isin(valid_holidays).all():
        errors.append(f"StateHoliday must be one of: {', '.join(valid_holidays)}")

    return len(errors) == 0, errors


def process_batch_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Process uploaded CSV: normalize dates and add DayOfWeek column.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded DataFrame to process

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with Date normalized to YYYY-MM-DD and DayOfWeek added
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Parse dates with flexible format detection
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    # Add DayOfWeek column if not present (or recalculate if present)
    # Python's weekday(): Monday=0, Sunday=6
    # We need: Monday=1, Sunday=7
    df["DayOfWeek"] = df["Date"].dt.weekday + 1

    # Convert Date back to string in YYYY-MM-DD format for API
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df


def get_csv_template() -> pd.DataFrame:
    """Generate a CSV template for batch predictions.

    Returns
    -------
    pd.DataFrame
        Template DataFrame with sample data (DayOfWeek auto-calculated from Date)
    """
    template_data = {
        "Store": [1, 2, 3, 4, 5],
        "Date": [
            "2015-08-01",
            "2015-08-02",
            "2015-08-03",
            "2015-08-04",
            "2015-08-05",
        ],
        "Open": [1, 1, 1, 1, 1],
        "Promo": [0, 1, 0, 1, 0],
        "StateHoliday": ["0", "0", "0", "a", "0"],
        "SchoolHoliday": [0, 0, 1, 1, 0],
    }

    return pd.DataFrame(template_data)
