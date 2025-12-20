"""FastAPI client wrapper for Streamlit application.

This module provides a clean interface for communicating with the FastAPI backend, handling
requests, responses, and error states.
"""

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


class APIClient:
    """Client for interacting with Rossmann FastAPI backend.

    Parameters
    ----------
    base_url : str
        Base URL of the FastAPI server (default: http://localhost:8000)
    timeout : int
        Request timeout in seconds (default: 30)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 30,
    ):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.timeout = timeout

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Make HTTP request to API with error handling.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.)
        endpoint : str
            API endpoint path
        **kwargs
            Additional arguments passed to requests

        Returns
        -------
        dict or None
            Response JSON if successful, None if error
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Cannot connect to API server at {self.base_url}. "
                "Please ensure the FastAPI server is running."
            )
            return None

        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out after {self.timeout} seconds.")
            return None

        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            st.error(f"❌ API Error: {error_detail}")
            return None

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

    def health_check(self) -> dict[str, Any] | None:
        """Check API health status.

        Returns
        -------
        dict or None
            Health status response
        """
        return self._make_request("GET", "/health")

    def get_model_info(self) -> dict[str, Any] | None:
        """Get model registry information.

        Returns
        -------
        dict or None
            Model information including versions and stages
        """
        return self._make_request("GET", "/model/info")

    def load_model(self, stage: str = "Production") -> dict[str, Any] | None:
        """Load a model into API memory.

        Parameters
        ----------
        stage : str
            Model stage to load (Production, Staging, or version number)

        Returns
        -------
        dict or None
            Model loading status
        """
        return self._make_request("POST", f"/model/load?stage={stage}")

    def predict_single(
        self,
        store: int,
        day_of_week: int,
        date: str,
        open_flag: int,
        promo: int,
        state_holiday: str,
        school_holiday: int,
        model_stage: str = "Production",
    ) -> dict[str, Any] | None:
        """Make a single prediction.

        Parameters
        ----------
        store : int
            Store ID (1-1115)
        day_of_week : int
            Day of week (1=Monday, 7=Sunday)
        date : str
            Date in YYYY-MM-DD format
        open_flag : int
            Is store open? (0=closed, 1=open)
        promo : int
            Is promotion running? (0=no, 1=yes)
        state_holiday : str
            State holiday indicator (0, a, b, c)
        school_holiday : int
            Is school holiday? (0=no, 1=yes)
        model_stage : str
            Model stage to use for prediction

        Returns
        -------
        dict or None
            Prediction response with predicted sales
        """
        payload = {
            "inputs": [
                {
                    "Store": store,
                    "DayOfWeek": day_of_week,
                    "Date": date,
                    "Open": open_flag,
                    "Promo": promo,
                    "StateHoliday": state_holiday,
                    "SchoolHoliday": school_holiday,
                }
            ],
            "model_stage": model_stage,
        }

        return self._make_request("POST", "/predict", json=payload)

    def predict_batch(
        self,
        data: pd.DataFrame,
        model_stage: str = "Production",
    ) -> dict[str, Any] | None:
        """Make batch predictions from DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns: Store, DayOfWeek, Date, Open, Promo,
            StateHoliday, SchoolHoliday
        model_stage : str
            Model stage to use for prediction

        Returns
        -------
        dict or None
            Prediction response with list of predicted sales
        """
        # Convert DataFrame to list of dicts
        inputs = data.to_dict(orient="records")

        payload = {
            "inputs": inputs,
            "model_stage": model_stage,
        }

        return self._make_request("POST", "/predict", json=payload)


@st.cache_resource
def get_api_client() -> APIClient:
    """Get cached API client instance.

    Returns
    -------
    APIClient
        Singleton API client instance
    """
    return APIClient()
