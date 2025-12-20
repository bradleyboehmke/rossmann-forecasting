"""Streamlit Predictions page for generating sales forecasts.

This page provides an interactive interface for making single or batch predictions using the FastAPI
backend and the deployed Rossmann forecasting model.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directories to path
streamlit_dir = Path(__file__).parents[1]
sys.path.insert(0, str(streamlit_dir / "utils"))

from api_client import get_api_client
from validation import (
    get_csv_template,
    process_batch_csv,
    validate_batch_csv,
    validate_single_prediction_input,
)

# Page configuration
st.set_page_config(page_title="Predictions", page_icon="📈", layout="wide")

# Initialize API client
api_client = get_api_client()

# Header
st.title("📈 Sales Predictions")
st.markdown(
    """
    Generate sales predictions for Rossmann stores using the production ensemble model.
    Upload data in **train.csv format** (7 fields only) - the API handles all feature engineering.
    """
)

# Sidebar - Model Configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")

    model_stage = st.selectbox(
        "Model Stage",
        ["Production", "Staging"],
        index=0,
        help="Select which model version to use for predictions",
    )

    # Get model info
    st.divider()
    st.subheader("📊 Model Status")

    with st.spinner("Checking model status..."):
        model_info = api_client.get_model_info()

    if model_info:
        if model_stage == "Production":
            version = model_info.get("production_version")
        else:
            version = model_info.get("staging_version")

        if version:
            st.success(f"✓ {model_stage} v{version}")
        else:
            st.error(f"❌ No {model_stage} model available")
            st.stop()

        # Show all registered models
        with st.expander("View All Models"):
            st.write("**Registered Models:**")
            for model in model_info.get("registered_models", []):
                st.write(f"- {model}")
    else:
        st.error("Cannot connect to API. Please start the FastAPI server.")
        st.code("cd deployment/api && python main.py", language="bash")
        st.stop()

st.divider()

# Prediction Mode Tabs
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📦 Batch Upload"])

# ============================================================================
# TAB 1: Single Real-Time Prediction
# ============================================================================
with tab1:
    st.subheader("🔮 Single Store Prediction")
    st.markdown(
        """
        Enter data for **one store and date** to get an instant sales forecast.
        Day of Week is **auto-calculated** from the date. The API automatically handles store metadata and feature engineering.
        """
    )

    # Create input form
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📍 Store & Date**")
            store_id = st.number_input(
                "Store ID",
                min_value=1,
                max_value=1115,
                value=1,
                step=1,
                help="Store identifier (1-1115)",
            )

            prediction_date = st.date_input(
                "Date to Predict",
                value=datetime(2015, 8, 1).date(),
                help="Date to predict sales for - Day of Week will be calculated automatically",
            )

        with col2:
            st.markdown("**🏪 Store Status**")
            is_open = st.selectbox(
                "Store Open?",
                options=[1, 0],
                format_func=lambda x: "✅ Open" if x == 1 else "❌ Closed",
                help="Is the store open on this date?",
            )

            state_holiday = st.selectbox(
                "State Holiday",
                options=["0", "a", "b", "c"],
                format_func=lambda x: {
                    "0": "No Holiday",
                    "a": "Public Holiday",
                    "b": "Easter Holiday",
                    "c": "Christmas",
                }[x],
                help="Type of state holiday",
            )

        with col3:
            st.markdown("**🎯 Promotions & School**")
            promo = st.selectbox(
                "Promotion Active?",
                options=[0, 1],
                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
                help="Is a promotion running?",
            )

            school_holiday = st.selectbox(
                "School Holiday?",
                options=[0, 1],
                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
                help="Is it a school holiday?",
            )

        # Submit button
        st.divider()
        submit_button = st.form_submit_button(
            "🚀 Generate Prediction",
            type="primary",
        )

    # Handle form submission
    if submit_button:
        # Convert date to string
        date_str = prediction_date.strftime("%Y-%m-%d")

        # Auto-calculate day of week from selected date
        # Python's weekday(): Monday=0, Sunday=6
        # We need: Monday=1, Sunday=7
        day_of_week = prediction_date.weekday() + 1

        # Day names for display
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # Validate inputs
        is_valid, errors = validate_single_prediction_input(
            store=store_id,
            day_of_week=day_of_week,
            date=date_str,
            open_flag=is_open,
            promo=promo,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
        )

        if not is_valid:
            st.error("❌ Validation Error:")
            for error in errors:
                st.error(f"- {error}")
        else:
            # Make API request
            with st.spinner("🔮 Generating prediction..."):
                response = api_client.predict_single(
                    store=store_id,
                    day_of_week=day_of_week,
                    date=date_str,
                    open_flag=is_open,
                    promo=promo,
                    state_holiday=state_holiday,
                    school_holiday=school_holiday,
                    model_stage=model_stage,
                )

            if response:
                st.success("✅ Prediction Complete!")

                # Display prediction result
                predictions = response["predictions"]
                predicted_sales = predictions[0]

                # Metrics display - 5 columns
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

                with metric_col1:
                    st.metric(
                        label="💰 Predicted Sales",
                        value=f"${predicted_sales:,.2f}",
                    )

                with metric_col2:
                    st.metric(label="🏪 Store ID", value=store_id)

                with metric_col3:
                    st.metric(label="📅 Date", value=date_str)

                with metric_col4:
                    st.metric(
                        label="📆 Day of Week",
                        value=f"{day_names[day_of_week - 1]} ({day_of_week})",
                    )

                with metric_col5:
                    st.metric(label="🤖 Model Version", value=f"v{response['model_version']}")

                # Show input summary
                st.divider()
                with st.expander("📋 View Input Details"):
                    input_df = pd.DataFrame(
                        {
                            "Store": [store_id],
                            "DayOfWeek": [day_of_week],
                            "Date": [date_str],
                            "Open": [is_open],
                            "Promo": [promo],
                            "StateHoliday": [state_holiday],
                            "SchoolHoliday": [school_holiday],
                        }
                    )
                    st.dataframe(input_df)

                # Interpretation guide
                with st.expander("💡 How to Interpret This Prediction"):
                    st.markdown(
                        f"""
                        **Predicted Sales:** ${predicted_sales:,.2f}

                        This prediction is based on:
                        - Historical sales patterns for Store {store_id}
                        - Day of week effects ({["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week-1]})
                        - Promotion status ({'Active' if promo else 'None'})
                        - Holiday effects ({state_holiday if state_holiday != '0' else 'None'})
                        - Store metadata (type, assortment, competition)
                        - Engineered time-series features (lags, rolling averages)

                        **Model:** {model_stage} ensemble (LightGBM 30% + XGBoost 60% + CatBoost 10%)
                        """
                    )


# ============================================================================
# TAB 2: Batch Upload
# ============================================================================
with tab2:
    st.subheader("📦 Batch Predictions from CSV")
    st.markdown(
        """
        Upload a CSV file with just **6 fields** - Day of Week is auto-calculated from Date!
        Dates can be in any common format (YYYY-MM-DD, MM/DD/YY, MM/DD/YYYY, etc.).
        The API will automatically merge store metadata, clean data, and engineer features.
        """
    )

    # CSV Template Download
    st.markdown("### 📥 Step 1: Download Template")
    st.info(
        "👉 Download the template CSV to see the exact format required. "
        "Fill in your data and upload it below."
    )

    template_df = get_csv_template()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(template_df)
    with col2:
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv_template,
            file_name="rossmann_prediction_template.csv",
            mime="text/csv",
        )

    st.divider()

    # CSV Upload
    st.markdown("### 📤 Step 2: Upload Your Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must have columns: Store, Date, Open, Promo, StateHoliday, SchoolHoliday (DayOfWeek auto-calculated)",
    )

    if uploaded_file is not None:
        try:
            # Read CSV
            input_df = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(input_df)} rows from CSV")

            # Validate CSV
            is_valid, errors = validate_batch_csv(input_df)

            if not is_valid:
                st.error("❌ CSV Validation Failed:")
                for error in errors:
                    st.error(f"- {error}")
            else:
                st.success("✅ CSV validation passed!")

                # Process CSV: normalize dates and add DayOfWeek
                try:
                    processed_df = process_batch_csv(input_df)
                    st.info(
                        "✨ Dates normalized to YYYY-MM-DD format and Day of Week auto-calculated!"
                    )
                except Exception as e:
                    st.error(f"Error processing dates: {str(e)}")
                    st.stop()

                # Show preview of processed data
                with st.expander("📋 Preview Processed Data (first 10 rows)"):
                    st.dataframe(processed_df.head(10))

                # Generate predictions button
                st.divider()
                if st.button(
                    f"🚀 Generate {len(processed_df)} Predictions",
                    type="primary",
                ):
                    # Make API request with processed data
                    with st.spinner(f"Generating predictions for {len(processed_df)} records..."):
                        response = api_client.predict_batch(
                            data=processed_df,
                            model_stage=model_stage,
                        )

                    if response:
                        st.success(f"✅ Successfully generated {response['count']} predictions!")

                        # Extract predictions
                        predictions = response["predictions"]
                        model_version = response["model_version"]

                        # Add predictions to processed dataframe (includes DayOfWeek)
                        result_df = processed_df.copy()
                        result_df["Predicted_Sales"] = predictions

                        # Summary metrics
                        st.divider()
                        st.subheader("📊 Prediction Summary")

                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                        with metric_col1:
                            st.metric("Total Predictions", f"{len(predictions):,}")

                        with metric_col2:
                            st.metric("Total Predicted Sales", f"${sum(predictions):,.2f}")

                        with metric_col3:
                            st.metric("Average Sales", f"${sum(predictions)/len(predictions):,.2f}")

                        with metric_col4:
                            st.metric("Model Version", f"v{model_version}")

                        # Results table
                        st.divider()
                        st.subheader("📋 Prediction Results")
                        st.dataframe(
                            result_df,
                            height=400,
                        )

                        # Download results
                        st.divider()
                        st.subheader("💾 Export Results")

                        csv_output = result_df.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_output,
                            file_name=f"rossmann_predictions_{timestamp}.csv",
                            mime="text/csv",
                        )

                        # Store-level summary
                        with st.expander("📊 View Store-Level Summary"):
                            store_summary = (
                                result_df.groupby("Store")["Predicted_Sales"]
                                .agg(["count", "sum", "mean"])
                                .round(2)
                            )
                            store_summary.columns = [
                                "Num Predictions",
                                "Total Sales",
                                "Avg Sales",
                            ]
                            st.dataframe(store_summary)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV matches the template format.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>💡 <b>Tip:</b> For large batch predictions, use the FastAPI directly via curl or Python requests</p>
        <p>See the <b>Documentation</b> page for API examples</p>
    </div>
    """,
    unsafe_allow_html=True,
)
