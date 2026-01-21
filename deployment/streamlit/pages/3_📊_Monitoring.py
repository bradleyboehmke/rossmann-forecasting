"""Streamlit Monitoring page for drift detection and usage analytics.

This page provides:
- Prediction volume charts
- Drift detection reports
- Model performance tracking
- API usage statistics
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add paths
streamlit_dir = Path(__file__).parents[1]
project_root = streamlit_dir.parents[1]
sys.path.insert(0, str(streamlit_dir / "utils"))
sys.path.insert(0, str(project_root / "src"))

# Page configuration
st.set_page_config(
    page_title="Monitoring - Rossmann Forecasting",
    page_icon="📊",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-warning {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .alert-success {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("📊 Model Monitoring Dashboard")
st.markdown(
    """
    Monitor model performance, detect data drift, and track prediction usage.
    """
)

st.divider()

# Check if monitoring database exists
db_path = project_root / "data" / "monitoring" / "predictions.db"

# Use full training data for most accurate drift detection (default)
# Fallback to sample if user created one
reference_path_full = project_root / "data" / "processed" / "train_features.parquet"
reference_path_sample = project_root / "monitoring" / "reference_data" / "training_sample.parquet"

if reference_path_full.exists():
    reference_path = reference_path_full
elif reference_path_sample.exists():
    reference_path = reference_path_sample
else:
    reference_path = reference_path_full  # Will be checked later

if not db_path.exists():
    st.warning(
        "⚠️ **No prediction data found.** "
        "Make some predictions via the Predictions page or API to populate monitoring data."
    )
    st.info("The prediction database will be created automatically at:\n" f"`{db_path}`")
    st.stop()

# Import monitoring modules
try:
    from monitoring.drift_detection import DriftDetector, generate_drift_report_cli

    # Also need prediction logger for stats
    sys.path.insert(0, str(project_root / "deployment" / "api"))
    from prediction_logger import PredictionLogger
except ImportError as e:
    st.error(f"Failed to import monitoring modules: {e}")
    st.stop()

# Initialize logger
pred_logger = PredictionLogger(db_path)

# =============================================================================
# SECTION 1: Summary Statistics
# =============================================================================
st.subheader("📈 Prediction Usage Statistics")

try:
    stats = pred_logger.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Predictions", f"{stats['total_predictions']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if stats["first_prediction"]:
            first_date = pd.to_datetime(stats["first_prediction"]).strftime("%Y-%m-%d")
            st.metric("First Prediction", first_date)
        else:
            st.metric("First Prediction", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if stats["last_prediction"]:
            last_date = pd.to_datetime(stats["last_prediction"]).strftime("%Y-%m-%d")
            st.metric("Last Prediction", last_date)
        else:
            st.metric("Last Prediction", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if stats["model_versions"]:
            current_version = stats["model_versions"][0]["model_version"]
            st.metric("Current Model", f"v{current_version}")
        else:
            st.metric("Current Model", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    # Daily volume chart
    if stats["daily_volume"]:
        st.markdown("### Daily Prediction Volume")
        daily_df = pd.DataFrame(stats["daily_volume"])
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date")

        # Convert date to string for discrete x-axis
        daily_df["date_str"] = daily_df["date"].dt.strftime("%Y-%m-%d")

        fig = px.bar(
            daily_df,
            x="date_str",
            y="count",
            title="Predictions per Day (Last 30 Days)",
            labels={"count": "Number of Predictions", "date_str": "Date"},
        )
        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,  # Angle labels for readability
            xaxis={"type": "category"},  # Force categorical x-axis
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model version breakdown
    if stats["model_versions"]:
        st.markdown("### Model Version Distribution")
        version_df = pd.DataFrame(stats["model_versions"])

        fig = px.pie(
            version_df,
            values="count",
            names="model_version",
            title="Predictions by Model Version",
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading statistics: {e}")

st.divider()

# =============================================================================
# SECTION 2: Drift Detection
# =============================================================================
st.subheader("🔍 Data Drift Detection")

# Check if reference data exists
if not reference_path.exists():
    st.warning(
        "⚠️ **Reference data not found.** " "Run the DataOps workflow to generate training features."
    )
    st.code(
        "bash scripts/dataops_workflow.sh",
        language="bash",
    )
    st.info(
        "💡 Drift detection uses full training data by default for most accurate comparison. "
        f"Expected path: `{reference_path_full.relative_to(project_root)}`"
    )
else:
    st.success(f"✅ Reference data loaded from `{reference_path.name}`")

    # Drift detection controls
    col1, col2 = st.columns([3, 1])

    with col1:
        days_to_analyze = st.slider(
            "Days of production data to analyze",
            min_value=1,
            max_value=90,
            value=7,
            help="Compare recent production predictions against reference training data",
        )

    with col2:
        generate_button = st.button(
            "🔄 Generate Drift Report", type="primary", use_container_width=True
        )

    if generate_button:
        with st.spinner(f"Analyzing drift for last {days_to_analyze} days..."):
            try:
                # Generate drift report
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_path = (
                    project_root / "monitoring" / "drift_reports" / f"drift_report_{timestamp}.html"
                )

                summary = generate_drift_report_cli(
                    days=days_to_analyze,
                    reference_data_path=reference_path,
                    db_path=db_path,
                    output_path=output_path,
                )

                # Display summary
                if "error" in summary:
                    st.error(f"❌ Drift detection failed: {summary['error']}")
                else:
                    # Overall drift status
                    if summary["dataset_drift_detected"]:
                        st.markdown(
                            '<div class="alert-warning">⚠️ <strong>Dataset Drift Detected!</strong> '
                            "Production data distribution has shifted significantly from training data.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="alert-success">✅ <strong>No Significant Drift</strong> '
                            "Production data distribution is consistent with training data.</div>",
                            unsafe_allow_html=True,
                        )

                    # Drift metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Drift Share", f"{summary['drift_share']:.1%}")

                    with col2:
                        st.metric("Drifted Features", summary["number_of_drifted_features"])

                    with col3:
                        st.metric("Total Features", summary["total_features_checked"])

                    # Drifted features table
                    if summary["drifted_features"]:
                        st.markdown("### Features with Detected Drift")
                        drift_df = pd.DataFrame(summary["drifted_features"])
                        drift_df = drift_df.sort_values("drift_score", ascending=False)

                        # Format drift score as percentage
                        drift_df["drift_score"] = drift_df["drift_score"].apply(
                            lambda x: f"{x:.3f}"
                        )

                        st.dataframe(
                            drift_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "feature": "Feature Name",
                                "drift_score": "Drift Score",
                                "stattest": "Statistical Test",
                            },
                        )

                        # Add distribution comparison plots for drifted features
                        st.markdown("### 📊 Distribution Comparisons")
                        st.markdown(
                            "Compare production data (current) vs. training data (reference) "
                            "for features showing drift."
                        )

                        # Get production and reference data for plotting
                        try:
                            from monitoring.drift_detection import DriftDetector

                            detector = DriftDetector(
                                reference_data_path=reference_path, db_path=db_path
                            )
                            production_data = detector.get_production_data(days=days_to_analyze)
                            ref_data, prod_data = detector.prepare_data_for_comparison(
                                production_data
                            )

                            # Plot all drifted features
                            all_drifted = summary["drifted_features"]

                            for feat_info in all_drifted:
                                feature = feat_info["feature"]
                                test_type = feat_info["stattest"]

                                st.markdown(f"#### {feature}")

                                # Check if feature is categorical or numerical
                                if test_type == "total_variation":
                                    # Categorical - use bar chart
                                    ref_counts = ref_data[feature].value_counts()
                                    prod_counts = prod_data[feature].value_counts()

                                    # Create long-form dataframe for plotly express
                                    all_categories = sorted(
                                        set(ref_counts.index) | set(prod_counts.index)
                                    )
                                    plot_data = []

                                    # Map categories to labeled strings to force discrete axis
                                    cat_labels = {cat: f"Value: {cat}" for cat in all_categories}

                                    for cat in all_categories:
                                        cat_label = cat_labels[cat]
                                        plot_data.append(
                                            {
                                                "Category": cat_label,
                                                "Count": ref_counts.get(cat, 0),
                                                "Dataset": "Reference (Training)",
                                            }
                                        )
                                        plot_data.append(
                                            {
                                                "Category": cat_label,
                                                "Count": prod_counts.get(cat, 0),
                                                "Dataset": "Production (Current)",
                                            }
                                        )

                                    plot_df = pd.DataFrame(plot_data)

                                    # Get ordered category labels
                                    ordered_labels = [cat_labels[c] for c in all_categories]

                                    # Use plotly express which handles categorical better
                                    fig = px.bar(
                                        plot_df,
                                        x="Category",
                                        y="Count",
                                        color="Dataset",
                                        barmode="group",
                                        title=f"{feature} Distribution Comparison",
                                        color_discrete_map={
                                            "Reference (Training)": "#1f77b4",
                                            "Production (Current)": "#ff7f0e",
                                        },
                                        category_orders={"Category": ordered_labels},
                                    )

                                    fig.update_layout(
                                        xaxis_title="Category",
                                        yaxis_title="Count",
                                        height=400,
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                else:
                                    # Numerical - use histogram overlay
                                    fig = go.Figure()

                                    # Reference data histogram
                                    fig.add_trace(
                                        go.Histogram(
                                            x=ref_data[feature].dropna(),
                                            name="Reference (Training)",
                                            opacity=0.6,
                                            marker_color="#1f77b4",
                                            nbinsx=30,
                                        )
                                    )

                                    # Production data histogram
                                    fig.add_trace(
                                        go.Histogram(
                                            x=prod_data[feature].dropna(),
                                            name="Production (Current)",
                                            opacity=0.6,
                                            marker_color="#ff7f0e",
                                            nbinsx=30,
                                        )
                                    )

                                    fig.update_layout(
                                        title=f"{feature} Distribution Comparison",
                                        xaxis_title=feature,
                                        yaxis_title="Count",
                                        barmode="overlay",
                                        height=400,
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.warning(f"Could not generate distribution plots: {e}")

                    else:
                        st.success("✅ No features showing significant drift")

                    # Link to full report
                    st.markdown("### 📄 Full Drift Report")

                    # Check if JSON report was created (Evidently 0.7.20 saves as JSON)
                    json_path = output_path.with_suffix(".json")
                    if json_path.exists():
                        st.info(
                            f"Drift report data saved to:\n\n`{json_path.relative_to(project_root)}`"
                        )
                        st.markdown(
                            "ℹ️ *Evidently 0.7.20 saves reports as JSON. Use the summary above for drift insights.*"
                        )
                    else:
                        st.info(
                            f"Detailed Evidently report saved to:\n\n`{output_path.relative_to(project_root)}`"
                        )

            except Exception as e:
                st.error(f"Error generating drift report: {e}")
                st.exception(e)

    # Show latest report if exists (check for both JSON and HTML)
    latest_report_json = project_root / "monitoring" / "drift_reports" / "latest.json"
    latest_report_html = project_root / "monitoring" / "drift_reports" / "latest.html"

    if (latest_report_json.exists() or latest_report_html.exists()) and not generate_button:
        st.markdown("### 📄 Latest Drift Report")

        if latest_report_json.exists():
            st.info(f"Most recent report: `{latest_report_json.relative_to(project_root)}`")
            st.markdown(
                "ℹ️ *Evidently 0.7.20 report (JSON format). Generate a new report to see current drift status.*"
            )
        elif latest_report_html.exists():
            st.info(f"Most recent report: `{latest_report_html.resolve()}`")

            with st.expander("📊 View Latest Report", expanded=False):
                with open(latest_report_html, encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)

st.divider()

# =============================================================================
# SECTION 3: Recommendations
# =============================================================================
st.subheader("💡 Monitoring Recommendations")

st.markdown(
    """
    ### When to Retrain the Model

    Consider retraining if you observe:

    - **Significant data drift** (>20% of features showing drift)
    - **Shift in prediction distribution** (target drift detected)
    - **Changes in business patterns** (new promotions, store openings, seasonal shifts)
    - **Regular schedule** (retrain monthly/quarterly even without drift)

    ### Best Practices

    - ✅ Run drift detection **weekly** to catch issues early
    - ✅ Monitor prediction volume for anomalies
    - ✅ Compare multiple time windows (7-day, 30-day)
    - ✅ Document any detected drift and retraining decisions
    - ✅ Keep reference data updated with recent training runs

    ### Next Steps

    If drift is detected:
    1. Investigate which features are drifting (check full Evidently report)
    2. Verify data quality (check for data pipeline issues)
    3. Assess business impact (is drift expected due to real-world changes?)
    4. Retrain model with recent data if drift is significant
    5. Update reference data after retraining
    """
)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Model Monitoring powered by Evidently AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)
