"""Streamlit Home page for Rossmann Sales Forecasting Dashboard.

This is the main entry point for the multi-page Streamlit application.
"""

import sys
from pathlib import Path

import streamlit as st

# Add utils to path
streamlit_dir = Path(__file__).parent
sys.path.insert(0, str(streamlit_dir / "utils"))

from api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">📊 Rossmann Sales Forecasting Dashboard</div>', unsafe_allow_html=True
)

# Introduction
st.markdown(
    """
    Welcome to the **Rossmann Sales Forecasting Dashboard**! This application provides a comprehensive
    interface for generating sales predictions, monitoring model performance, and managing the ML model
    lifecycle.
    """
)

st.divider()

# API Health Check
api_client = get_api_client()
health_response = api_client.health_check()

if health_response:
    api_status = "🟢 Online"
    api_color = "#4caf50"
else:
    api_status = "🔴 Offline"
    api_color = "#f44336"

# System Status Section
st.subheader("🖥️ System Status")

status_col1, status_col2 = st.columns(2)

with status_col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size: 1.2rem; margin: 0;">FastAPI Server</p>'
        f'<p style="font-size: 1.8rem; font-weight: bold; color: {api_color}; margin: 0;">{api_status}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with status_col2:
    if health_response:
        model_loaded = health_response.get("model_loaded", False)
        model_status = "✅ Loaded" if model_loaded else "⏳ Not Loaded"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size: 1.2rem; margin: 0;">Model Cache</p>'
            f'<p style="font-size: 1.8rem; font-weight: bold; margin: 0;">{model_status}</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size: 1.2rem; margin: 0;">Model Cache</p>'
            '<p style="font-size: 1.8rem; font-weight: bold; color: #999; margin: 0;">N/A</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Model Status Section
st.subheader("🤖 Model Registry")

col1, col2, col3 = st.columns(3)

try:
    # Get current model versions from API only
    if health_response:
        model_info = api_client.get_model_info()
        if model_info:
            production_version = model_info.get("production_version")
            staging_version = model_info.get("staging_version")
            models = model_info.get("registered_models", [])
        else:
            # API is online but model info unavailable
            production_version = None
            staging_version = None
            models = []
    else:
        # API is offline - cannot get model info
        production_version = None
        staging_version = None
        models = []

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Production Model",
            value=f"v{production_version}" if production_version else "Not deployed",
            delta="Active" if production_version else None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Staging Model",
            value=f"v{staging_version}" if staging_version else "Not deployed",
            delta="Testing" if staging_version else None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Registered Models", value=len(models), delta=None)
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading model status: {str(e)}")
    st.info("Make sure the FastAPI server is running and connected to MLflow.")

# API Connection Help
if not health_response:
    st.divider()
    st.warning(
        "⚠️ **FastAPI server is not running.** "
        "The Predictions page requires the API server to be active."
    )
    st.markdown("**To start the API server:**")
    st.code("cd deployment/api && python main.py", language="bash")
    st.info("The API will be available at http://localhost:8000")

st.divider()

# Features Overview
st.subheader("✨ Dashboard Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown(
        """
        ### 📈 Predictions
        - **Single Prediction**: Real-time forecasts for one store/date
        - **Batch Upload**: Upload CSV with just 6 fields (DayOfWeek auto-calculated!)
        - **Flexible Date Formats**: YYYY-MM-DD, MM/DD/YY, MM/DD/YYYY, etc.
        - Automatic feature engineering via FastAPI
        - Download results as CSV with predictions
        - Store-level summaries and analytics
        """
    )

with feature_col2:
    st.markdown(
        """
        ### 🚀 Easy to Use
        - Simple train.csv format (no feature engineering needed!)
        - Template CSV download for batch predictions
        - Client-side validation before API calls
        - Clear error messages and help text
        - Production & Staging model selection

        ### 📚 Documentation
        - API usage examples and curl commands
        - Model architecture overview
        - Feature engineering pipeline details
        - Deployment and integration guides
        """
    )

st.divider()

# Quick Start Guide
st.subheader("🚀 Quick Start")

st.markdown(
    """
    1. **Start FastAPI Server**: Run `cd deployment/api && python main.py` (if not already running)
    2. **Single Prediction**: Navigate to *Predictions* → *Single Prediction* tab for real-time forecasts
    3. **Batch Upload**: Use *Predictions* → *Batch Upload* tab to process multiple predictions from CSV
    4. **Download Template**: Get the CSV template to see the exact format required (6 fields)
    5. **View Documentation**: Check *Documentation* page for API examples and integration guides
    """
)

# System Information
with st.expander("ℹ️ System Information"):
    st.markdown(
        """
        **Model Architecture**: Weighted ensemble of LightGBM (30%), XGBoost (60%), and CatBoost (10%)

        **Target Metric**: RMSPE (Root Mean Square Percentage Error) < 0.10

        **Features**: 20+ engineered features including calendar, promotion, competition, and lag features

        **Data**: Historical sales data from 1,115 Rossmann stores across Europe
        """
    )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit • MLflow • FastAPI</p>
        <p>🤖 Rossmann Sales Forecasting Platform v1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True,
)
