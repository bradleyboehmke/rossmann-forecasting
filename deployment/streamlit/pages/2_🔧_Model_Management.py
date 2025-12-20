"""Streamlit Model Management page for MLflow Model Registry operations.

This page provides interfaces for viewing model versions, promoting models through lifecycle stages,
and managing the model registry.
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from models.model_registry import (
    get_model_info,
    get_model_version,
    list_registered_models,
    promote_model,
)

st.set_page_config(page_title="Model Management", page_icon="🔧", layout="wide")

st.title("🔧 Model Management")

st.markdown(
    """
    Manage your ML models through their lifecycle: from initial registration through
    staging, production deployment, and eventual archival.
    """
)

st.divider()

# Model Selection
st.subheader("📋 Registered Models")

try:
    models = list_registered_models()

    if not models:
        st.warning("No models registered in MLflow. Please train and register a model first.")
        st.stop()

    selected_model = st.selectbox("Select Model", models, index=0 if models else None)

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Make sure MLflow tracking server is running.")
    st.stop()

st.divider()

# Model Lifecycle Status
st.subheader("🔄 Model Lifecycle Status")

lifecycle_col1, lifecycle_col2, lifecycle_col3, lifecycle_col4 = st.columns(4)

try:
    # Get versions for each stage
    production_version = get_model_version(selected_model, stage="Production")
    staging_version = get_model_version(selected_model, stage="Staging")

    # Get full model info
    model_info = get_model_info(selected_model)

    with lifecycle_col1:
        st.markdown("**None (Registered)**")
        # Count models in None stage
        none_versions = [v for v in model_info.get("versions", []) if v.get("stage") == "None"]
        st.metric("Versions", len(none_versions))

    with lifecycle_col2:
        st.markdown("**Staging**")
        if staging_version:
            st.metric("Version", f"v{staging_version}")
            st.success("Active")
        else:
            st.metric("Version", "—")
            st.info("Not deployed")

    with lifecycle_col3:
        st.markdown("**Production**")
        if production_version:
            st.metric("Version", f"v{production_version}")
            st.success("Active")
        else:
            st.metric("Version", "—")
            st.warning("Not deployed")

    with lifecycle_col4:
        st.markdown("**Archived**")
        archived_versions = [
            v for v in model_info.get("versions", []) if v.get("stage") == "Archived"
        ]
        st.metric("Versions", len(archived_versions))

except Exception as e:
    st.error(f"Error loading lifecycle status: {str(e)}")

st.divider()

# Model Versions Table
st.subheader("📊 Model Versions")

try:
    versions = model_info.get("versions", [])

    if versions:
        # Create table data
        table_data = []
        for v in versions:
            table_data.append(
                {
                    "Version": v.get("version", "—"),
                    "Stage": v.get("stage", "—"),
                    "Status": v.get("status", "—"),
                    "Run ID": v.get("run_id", "—")[:8] + "...",  # Truncate for display
                }
            )

        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("No versions found for this model.")

except Exception as e:
    st.error(f"Error displaying versions: {str(e)}")

st.divider()

# Model Promotion
st.subheader("🚀 Model Promotion")

st.markdown(
    """
    Promote models through lifecycle stages. Use **Staging** for testing and validation,
    then promote to **Production** when ready for deployment.
    """
)

promo_col1, promo_col2 = st.columns(2)

with promo_col1:
    st.markdown("### Promote to Staging")

    with st.form("promote_staging_form"):
        st.markdown("Select a model version to promote to Staging:")

        # Get available versions (exclude already in Staging/Production)
        available_for_staging = [
            v["version"] for v in versions if v.get("stage") not in ["Staging", "Production"]
        ]

        if available_for_staging:
            version_to_staging = st.selectbox(
                "Version", available_for_staging, key="staging_version"
            )

            if st.form_submit_button("➡️ Promote to Staging", use_container_width=True):
                try:
                    with st.spinner("Promoting model..."):
                        promote_model(
                            model_name=selected_model,
                            version=version_to_staging,
                            stage="Staging",
                            archive_existing=True,
                        )
                    st.success(f"✅ Model v{version_to_staging} promoted to Staging!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Promotion failed: {str(e)}")
        else:
            st.info("No versions available for Staging promotion.")
            st.form_submit_button("➡️ Promote to Staging", disabled=True, use_container_width=True)

with promo_col2:
    st.markdown("### Promote to Production")

    with st.form("promote_production_form"):
        st.markdown("Promote the Staging model to Production:")

        if staging_version:
            st.info(f"Current Staging version: **v{staging_version}**")

            st.warning(
                """
                **⚠️ Production Deployment**

                This will deploy the model to production. Ensure you have:
                - Validated model performance on holdout data
                - Reviewed model metrics and predictions
                - Tested the model in staging environment
                """
            )

            confirm = st.checkbox("I have validated this model and confirm deployment")

            if st.form_submit_button(
                "🚀 Promote to Production",
                disabled=not confirm,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Promoting to Production..."):
                        promote_model(
                            model_name=selected_model,
                            version=staging_version,
                            stage="Production",
                            archive_existing=True,
                        )
                    st.success(f"✅ Model v{staging_version} promoted to Production!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Production promotion failed: {str(e)}")
        else:
            st.info("No model in Staging. Promote a version to Staging first.")
            st.form_submit_button(
                "🚀 Promote to Production", disabled=True, use_container_width=True
            )

st.divider()

# Archive Model
st.subheader("📦 Archive Model")

with st.expander("Archive Model Version"):
    st.markdown(
        """
        Archive models that are no longer needed. Archived models are moved out of the
        active lifecycle but remain available for reference.
        """
    )

    with st.form("archive_form"):
        # Get versions that can be archived (Production/Staging/None)
        archivable_versions = [v["version"] for v in versions if v.get("stage") != "Archived"]

        if archivable_versions:
            version_to_archive = st.selectbox("Version to Archive", archivable_versions)

            if st.form_submit_button("📦 Archive Version"):
                try:
                    with st.spinner("Archiving model..."):
                        promote_model(
                            model_name=selected_model,
                            version=version_to_archive,
                            stage="Archived",
                        )
                    st.success(f"✅ Model v{version_to_archive} archived!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Archival failed: {str(e)}")
        else:
            st.info("All versions are already archived.")
            st.form_submit_button("📦 Archive Version", disabled=True)

# Model Registry Info
with st.expander("ℹ️ Model Registry Information"):
    st.markdown(
        f"""
        **Model Name**: {selected_model}

        **Total Versions**: {len(versions)}

        **Lifecycle Stages**:
        - **None**: Newly registered models, not yet deployed
        - **Staging**: Models under testing and validation
        - **Production**: Active deployed models serving predictions
        - **Archived**: Retired models no longer in use

        **Last Updated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
    )
