"""Prediction logging module for monitoring and drift detection.

This module logs all predictions to a SQLite database for:
- Drift detection (feature and target drift)
- Usage analytics (prediction volume, patterns)
- Model performance tracking (when actuals become available)
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logger for ML predictions and features to SQLite database.

    Logs predictions along with key features for drift detection and monitoring.
    Uses SQLite for simplicity and portability.

    Parameters
    ----------
    db_path : Path
        Path to SQLite database file

    Attributes
    ----------
    db_path : Path
        Path to database file
    """

    def __init__(self, db_path: Path):
        """Initialize prediction logger.

        Parameters
        ----------
        db_path : Path
            Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        # Create directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    batch_id TEXT NOT NULL,

                    -- Store and prediction info
                    store_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    prediction REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    model_stage TEXT NOT NULL,

                    -- Raw input features (7 fields from train.csv format)
                    day_of_week INTEGER,
                    open INTEGER,
                    promo INTEGER,
                    state_holiday TEXT,
                    school_holiday INTEGER,

                    -- Key engineered features (~10 important ones for drift detection)
                    month INTEGER,
                    year INTEGER,
                    store_type TEXT,
                    assortment TEXT,
                    competition_distance REAL,
                    promo2 INTEGER,
                    is_promo2_active INTEGER,

                    -- Lag features (if available, may be NULL)
                    sales_lag_7 REAL,
                    sales_rolling_mean_7 REAL,

                    -- Metadata
                    response_time_ms REAL
                )
                """
            )

            # Create index for faster queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON predictions(timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_store_date
                ON predictions(store_id, date)
                """
            )

            # Optional: Table for actual sales (when ground truth becomes available)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actual_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    store_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    actual_sales REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    UNIQUE(store_id, date)
                )
                """
            )

            conn.commit()

        logger.info(f"Prediction database initialized at {self.db_path}")

    def log_predictions(
        self,
        raw_inputs: pd.DataFrame,
        features: pd.DataFrame,
        predictions: list[float],
        model_version: str,
        model_stage: str = "Production",
        response_time_ms: float | None = None,
    ) -> str:
        """Log predictions to database.

        Parameters
        ----------
        raw_inputs : pd.DataFrame
            Raw input data (train.csv format with Store, Date, DayOfWeek, etc.)
        features : pd.DataFrame
            Engineered features used for prediction
        predictions : list[float]
            Model predictions
        model_version : str
            Model version number
        model_stage : str, optional
            Model stage (Production/Staging), by default "Production"
        response_time_ms : float, optional
            API response time in milliseconds

        Returns
        -------
        str
            Batch ID for this set of predictions
        """
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Prepare records
        records = []
        for idx, pred in enumerate(predictions):
            # Get raw input fields
            raw_row = raw_inputs.iloc[idx]
            feature_row = features.iloc[idx]

            # Extract key features for drift detection
            record = {
                "timestamp": timestamp,
                "batch_id": batch_id,
                "store_id": int(raw_row["Store"]),
                "date": str(raw_row["Date"]),
                "prediction": float(pred),
                "model_version": model_version,
                "model_stage": model_stage,
                # Raw inputs
                "day_of_week": int(raw_row["DayOfWeek"]),
                "open": int(raw_row["Open"]),
                "promo": int(raw_row["Promo"]),
                "state_holiday": str(raw_row["StateHoliday"]),
                "school_holiday": int(raw_row["SchoolHoliday"]),
                # Key engineered features (use .get() for safety)
                "month": int(feature_row.get("Month", 0)),
                "year": int(feature_row.get("Year", 0)),
                "store_type": str(feature_row.get("StoreType", "")),
                "assortment": str(feature_row.get("Assortment", "")),
                "competition_distance": (
                    float(feature_row["CompetitionDistance"])
                    if "CompetitionDistance" in feature_row
                    and pd.notna(feature_row["CompetitionDistance"])
                    else None
                ),
                "promo2": int(feature_row.get("Promo2", 0)),
                "is_promo2_active": int(feature_row.get("Promo2Active", 0)),
                "sales_lag_7": (
                    float(feature_row["sales_lag_7"])
                    if "sales_lag_7" in feature_row and pd.notna(feature_row["sales_lag_7"])
                    else None
                ),
                "sales_rolling_mean_7": (
                    float(feature_row["sales_rolling_mean_7"])
                    if "sales_rolling_mean_7" in feature_row
                    and pd.notna(feature_row["sales_rolling_mean_7"])
                    else None
                ),
                "response_time_ms": response_time_ms,
            }
            records.append(record)

        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            df = pd.DataFrame(records)
            df.to_sql("predictions", conn, if_exists="append", index=False)

        logger.info(
            f"Logged {len(predictions)} predictions (batch_id={batch_id}, model_v{model_version})"
        )
        return batch_id

    def get_predictions(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve predictions from database.

        Parameters
        ----------
        start_date : str, optional
            Start date for filtering (YYYY-MM-DD)
        end_date : str, optional
            End date for filtering (YYYY-MM-DD)
        days : int, optional
            Retrieve last N days of predictions
        limit : int, optional
            Maximum number of records to return

        Returns
        -------
        pd.DataFrame
            Predictions dataframe
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if days is not None:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        else:
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        logger.info(f"Retrieved {len(df)} predictions from database")
        return df

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics about logged predictions.

        Returns
        -------
        dict[str, Any]
            Summary statistics including total predictions, date range, model versions
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total predictions
            total = pd.read_sql_query("SELECT COUNT(*) as count FROM predictions", conn).iloc[0][
                "count"
            ]

            # Date range
            date_range = pd.read_sql_query(
                "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM predictions", conn
            ).iloc[0]

            # Model versions
            versions = pd.read_sql_query(
                """
                SELECT model_version, model_stage, COUNT(*) as count
                FROM predictions
                GROUP BY model_version, model_stage
                ORDER BY count DESC
                """,
                conn,
            )

            # Daily prediction volume
            daily = pd.read_sql_query(
                """
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM predictions
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
                """,
                conn,
            )

        return {
            "total_predictions": int(total),
            "first_prediction": date_range["first"],
            "last_prediction": date_range["last"],
            "model_versions": versions.to_dict("records"),
            "daily_volume": daily.to_dict("records"),
        }
