"""Streamlit Home page for Rossmann Sales Forecasting Dashboard.

This is the main entry point for the multi-page Streamlit application.
"""

import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from models.model_registry import get_model_version, list_registered_models

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

# Model Status Section
st.subheader("🤖 Model Status")

col1, col2, col3 = st.columns(3)

try:
    # Get current model versions
    production_version = get_model_version("rossmann-ensemble", stage="Production")
    staging_version = get_model_version("rossmann-ensemble", stage="Staging")
    models = list_registered_models()

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
    st.info("Make sure MLflow tracking server is running and models are registered.")

st.divider()

# Features Overview
st.subheader("✨ Dashboard Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown(
        """
        ### 📈 Predictions
        - Generate single-store or batch predictions
        - Interactive feature input with validation
        - Real-time prediction results
        - Export predictions to CSV

        ### 📊 Model Monitoring
        - Track model performance metrics
        - View prediction history
        - Monitor data drift and model health
        - Compare model versions
        """
    )

with feature_col2:
    st.markdown(
        """
        ### 🔧 Model Management
        - View registered model versions
        - Promote models through lifecycle stages
        - Compare model performance
        - Model validation and testing

        ### 📚 Documentation
        - API usage examples
        - Model architecture overview
        - Feature engineering details
        - Deployment guide
        """
    )

st.divider()

# Quick Start Guide
st.subheader("🚀 Quick Start")

st.markdown(
    """
    1. **Generate Predictions**: Navigate to the *Predictions* page to make forecasts for specific stores and dates
    2. **Monitor Performance**: Check the *Model Monitoring* page to view metrics and performance trends
    3. **Manage Models**: Use the *Model Management* page to view and promote model versions
    4. **View Documentation**: Access the *Documentation* page for detailed guides and API references
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
