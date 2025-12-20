"""Streamlit Predictions page for generating sales forecasts.

This page provides an interactive interface for making single or batch predictions using the
deployed Rossmann forecasting model.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from models.model_registry import get_model_version, load_model

st.set_page_config(page_title="Predictions", page_icon="📈", layout="wide")

st.title("📈 Sales Predictions")

st.markdown(
    """
    Generate sales predictions for Rossmann stores using the production ensemble model.
    You can make single predictions or upload a CSV file for batch predictions.
    """
)

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")

    model_stage = st.selectbox("Select Model Stage", ["Production", "Staging"], index=0)

    try:
        current_version = get_model_version("rossmann-ensemble", stage=model_stage)
        if current_version:
            st.success(f"✓ Model v{current_version} loaded")
        else:
            st.error(f"No {model_stage} model found")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

st.divider()

# Prediction Mode Selection
prediction_mode = st.radio(
    "Prediction Mode",
    ["Single Prediction", "Batch Prediction (CSV)"],
    horizontal=True,
)

if prediction_mode == "Single Prediction":
    st.subheader("🔮 Single Store Prediction")

    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Store Information**")
            store = st.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
            store_type = st.selectbox("Store Type", ["a", "b", "c", "d"], index=0)
            assortment = st.selectbox("Assortment", ["a", "b", "c"], index=0)

        with col2:
            st.markdown("**Date & Calendar**")
            prediction_date = st.date_input("Prediction Date", value=datetime.now().date())
            day_of_week = st.number_input("Day of Week (1=Mon)", min_value=1, max_value=7, value=1)
            school_holiday = st.selectbox("School Holiday", [0, 1], index=0)
            state_holiday = st.selectbox("State Holiday", ["0", "a", "b", "c"], index=0)

        with col3:
            st.markdown("**Promotions**")
            promo = st.selectbox("Promo Active", [0, 1], index=0)
            promo2 = st.selectbox("Promo2 Participation", [0, 1], index=0)
            promo_interval = st.text_input("Promo Interval", value="")

        col4, col5 = st.columns(2)

        with col4:
            st.markdown("**Competition**")
            competition_distance = st.number_input(
                "Competition Distance (m)", min_value=0.0, value=1000.0, step=100.0
            )
            comp_open_since_month = st.number_input(
                "Competition Open Since Month", min_value=0, max_value=12, value=0
            )
            comp_open_since_year = st.number_input(
                "Competition Open Since Year", min_value=0, max_value=2025, value=0
            )

        with col5:
            st.markdown("**Additional Features**")
            is_open = st.selectbox("Store Open", [0, 1], index=1)
            promo2_since_week = st.number_input(
                "Promo2 Since Week", min_value=0, max_value=52, value=0
            )
            promo2_since_year = st.number_input(
                "Promo2 Since Year", min_value=0, max_value=2025, value=0
            )

        submit_button = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)

    if submit_button:
        try:
            # Load model
            with st.spinner("Loading model..."):
                model = load_model("rossmann-ensemble", stage=model_stage)

            # Prepare features
            features = pd.DataFrame(
                {
                    "Store": [store],
                    "DayOfWeek": [day_of_week],
                    "Open": [is_open],
                    "Promo": [promo],
                    "StateHoliday": [state_holiday],
                    "SchoolHoliday": [school_holiday],
                    "StoreType": [store_type],
                    "Assortment": [assortment],
                    "CompetitionDistance": [
                        competition_distance if competition_distance > 0 else None
                    ],
                    "CompetitionOpenSinceMonth": [
                        comp_open_since_month if comp_open_since_month > 0 else None
                    ],
                    "CompetitionOpenSinceYear": [
                        comp_open_since_year if comp_open_since_year > 0 else None
                    ],
                    "Promo2": [promo2],
                    "Promo2SinceWeek": [promo2_since_week if promo2_since_week > 0 else None],
                    "Promo2SinceYear": [promo2_since_year if promo2_since_year > 0 else None],
                    "PromoInterval": [promo_interval if promo_interval else None],
                    "Year": [prediction_date.year],
                    "Month": [prediction_date.month],
                    "Week": [prediction_date.isocalendar()[1]],
                    "Day": [prediction_date.day],
                }
            )

            # Convert categorical columns
            categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
            for col in categorical_cols:
                features[col] = features[col].astype("category")

            # Generate prediction
            with st.spinner("Generating prediction..."):
                prediction = model.predict(features)

            # Display result
            st.success("✅ Prediction Complete!")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    label="Predicted Sales",
                    value=f"${prediction[0]:,.2f}",
                    delta=None,
                )

            with result_col2:
                st.metric(label="Store ID", value=store)

            with result_col3:
                st.metric(label="Date", value=prediction_date.strftime("%Y-%m-%d"))

            # Show feature summary
            with st.expander("📊 View Input Features"):
                st.dataframe(features, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

else:
    # Batch Prediction Mode
    st.subheader("📦 Batch Predictions from CSV")

    st.markdown(
        """
        Upload a CSV file with the required features to generate batch predictions.
        The CSV must include all required columns with proper formatting.
        """
    )

    # Download sample CSV template
    if st.button("📥 Download CSV Template"):
        template_data = {
            "Store": [1, 2, 3],
            "DayOfWeek": [1, 2, 3],
            "Open": [1, 1, 1],
            "Promo": [0, 1, 0],
            "StateHoliday": ["0", "0", "a"],
            "SchoolHoliday": [0, 0, 1],
            "StoreType": ["a", "b", "c"],
            "Assortment": ["a", "b", "c"],
            "CompetitionDistance": [1000.0, 2000.0, 1500.0],
            "CompetitionOpenSinceMonth": [1, 2, 3],
            "CompetitionOpenSinceYear": [2015, 2015, 2015],
            "Promo2": [0, 1, 1],
            "Promo2SinceWeek": [0, 1, 1],
            "Promo2SinceYear": [0, 2015, 2015],
            "PromoInterval": ["", "Feb,May,Aug,Nov", "Jan,Apr,Jul,Oct"],
            "Year": [2015, 2015, 2015],
            "Month": [7, 7, 7],
            "Week": [28, 28, 28],
            "Day": [1, 2, 3],
        }
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="prediction_template.csv",
            mime="text/csv",
        )

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV
            input_df = pd.read_csv(uploaded_file)

            st.info(f"Loaded {len(input_df)} rows from CSV")

            # Show preview
            with st.expander("📋 Preview Input Data"):
                st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("🚀 Generate Batch Predictions", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    # Load model
                    model = load_model("rossmann-ensemble", stage=model_stage)

                    # Convert categorical columns
                    categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
                    for col in categorical_cols:
                        if col in input_df.columns:
                            input_df[col] = input_df[col].astype("category")

                    # Generate predictions
                    predictions = model.predict(input_df)

                    # Add predictions to dataframe
                    result_df = input_df.copy()
                    result_df["Predicted_Sales"] = predictions

                    st.success(f"✅ Generated {len(predictions)} predictions!")

                    # Display results
                    st.subheader("📊 Prediction Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Avg Predicted Sales", f"${predictions.mean():,.2f}")
                    with col3:
                        st.metric("Total Predicted Sales", f"${predictions.sum():,.2f}")

                    # Show results table
                    st.dataframe(result_df, use_container_width=True)

                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
