# Rossmann Forecasting MLOps Enhancement Specification

## Project Overview

**Repository**: https://github.com/bradleyboehmke/rossmann-forecasting **Purpose**: Transform the existing Rossmann sales forecasting project into a production-ready ML system that demonstrates end-to-end MLOps practices for BANA 7075.

**Current State**: The repo contains a well-structured Kaggle competition solution with:

- Modular code structure (`src/` with data, features, models, evaluation, utils)
- Comprehensive spec document
- Phase-based implementation (Phases 0-5)
- Jupyter notebooks for exploration
- Focus on achieving RMSPE \< 0.09856

**Goal**: Enhance this project to serve as the **Traditional ML reference project** for the course, demonstrating:

- DataOps practices (pipeline, validation, versioning)
- ModelOps practices (experiment tracking, versioning, deployment)
- DevOps practices (CI/CD, containerization, monitoring)
- Interactive user interface (Streamlit dashboard)

______________________________________________________________________

## Key Enhancements Required

### 1. **MLflow Integration** (ModelOps - Experiment Tracking & Model Registry)

**Current Gap**: No experiment tracking or model versioning infrastructure.

**Implementation**:

#### A. Experiment Tracking

- Track all model training runs with MLflow
- Log parameters: model hyperparameters, feature engineering config, CV strategy
- Log metrics: RMSPE per fold, mean RMSPE, training time
- Log artifacts: trained models, feature importance plots, CV predictions
- Tag runs: model type (lightgbm/xgboost/catboost), phase (baseline/advanced/ensemble)

**Files to modify**:

- `src/models/train_baselines.py` - Add MLflow logging
- `src/models/train_advanced.py` - Add MLflow logging
- `src/models/ensembles.py` - Add MLflow logging
- `notebooks/03-baseline-models.ipynb` - Initialize MLflow tracking
- `notebooks/04-advanced-models-and-ensembles.ipynb` - Track experiments

**New files**:

- `src/utils/mlflow_utils.py` - Helper functions for MLflow operations
- `config/mlflow.yaml` - MLflow configuration

**Example implementation**:

```python
import mlflow
from src.utils.mlflow_utils import log_model_metrics, log_cv_results

def train_lightgbm(X_train, y_train, params, cv_folds):
    with mlflow.start_run(run_name=f"lightgbm_baseline"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("phase", "baseline")

        # Train model
        model = lgb.LGBMRegressor(**params)
        cv_results = time_series_cv(model, X_train, y_train, cv_folds)

        # Log metrics
        mlflow.log_metric("mean_rmspe", cv_results['mean_rmspe'])
        mlflow.log_metric("std_rmspe", cv_results['std_rmspe'])

        # Log artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("feature_importance.png")

        return model, cv_results
```

#### B. Model Registry

- Register best models to MLflow Model Registry
- Version models: v1 (baseline), v2 (tuned), v3 (ensemble)
- Stage models: None → Staging → Production
- Track model lineage and metadata

**New functionality**:

- `src/models/register_model.py` - Register models to MLflow
- Model promotion workflow (promote best model to Production)

**Example**:

```python
def register_best_model(run_id, model_name="rossmann-forecaster"):
    # Register model from run
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    # Transition to staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
```

______________________________________________________________________

### 2. **Data Versioning & Validation** (DataOps)

**Current Gap**: No data versioning or validation framework.

**Implementation**:

#### A. DVC for Data Versioning

- Version raw data (`data/raw/train.csv`, `data/raw/store.csv`)
- Version processed data (`data/processed/train_clean.parquet`)
- Version feature engineered data (`data/processed/train_features.parquet`)
- Track data pipeline dependencies

**New files**:

- `.dvc/config` - DVC configuration
- `data/raw/.dvc` - DVC tracking files
- `data/processed/.dvc` - DVC tracking files
- `.dvcignore` - Files to ignore

**Commands to add**:

```bash
# Initialize DVC
dvc init

# Track raw data
dvc add data/raw/train.csv
dvc add data/raw/store.csv

# Track processed data
dvc add data/processed/train_clean.parquet
dvc add data/processed/train_features.parquet

# Commit .dvc files to git
git add data/raw/*.dvc data/processed/*.dvc
git commit -m "Track data with DVC"
```

#### B. Great Expectations for Data Validation

- Validate raw data schema and quality
- Validate feature engineering outputs
- Create expectation suites for each data stage
- Generate data quality reports

**New files**:

- `great_expectations/` - GX configuration directory
- `src/data/validate_data.py` - Data validation functions
- `expectations/raw_data.json` - Expectations for raw data
- `expectations/features.json` - Expectations for features

**Example expectations**:

```python
# Validate raw data
suite = context.create_expectation_suite("raw_train_suite")

validator.expect_column_values_to_not_be_null("Store")
validator.expect_column_values_to_not_be_null("Date")
validator.expect_column_values_to_be_between("Sales", min_value=0)
validator.expect_column_values_to_be_in_set("DayOfWeek", [1,2,3,4,5,6,7])
validator.expect_column_values_to_be_in_set("Open", [0, 1])

# Save suite
validator.save_expectation_suite()
```

**Integration**:

- Add validation step to `src/data/make_dataset.py`
- Run validation in notebooks before processing
- Fail pipeline if validation fails

______________________________________________________________________

### 3. **Model Deployment** (API + Streamlit Dashboard)

**Current Gap**: No deployment infrastructure.

**Implementation**:

#### A. FastAPI Backend

Create REST API for model predictions.

**New files**:

- `deployment/fastapi_app.py` - FastAPI application
- `deployment/schemas.py` - Pydantic schemas for request/response
- `deployment/model_loader.py` - Load model from MLflow registry

**API Endpoints**:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI(title="Rossmann Sales Forecasting API")

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/rossmann-forecaster/Production")

class PredictionRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    # ... other features

class PredictionResponse(BaseModel):
    store: int
    date: str
    predicted_sales: float

@app.post("/predict", response_model=PredictionResponse)
def predict_sales(request: PredictionRequest):
    # Convert request to features
    features = prepare_features(request)

    # Predict
    prediction = model.predict(features)

    return PredictionResponse(
        store=request.Store,
        date=request.Date,
        predicted_sales=float(prediction[0])
    )

@app.post("/predict_batch")
def predict_batch(requests: List[PredictionRequest]):
    # Batch prediction endpoint
    pass

@app.get("/model_info")
def get_model_info():
    # Return current model version, metrics, etc.
    pass
```

#### B. Streamlit Dashboard

Create interactive web interface for forecasting and exploration.

**New files**:

- `deployment/streamlit_app.py` - Streamlit application

**Dashboard Features**:

1. **Single Store Forecasting**

   - Select store from dropdown
   - Choose date range (next 6 weeks)
   - Display forecast with confidence intervals
   - Show historical sales for context

1. **Multi-Store Comparison**

   - Compare forecasts across stores
   - Filter by StoreType, Assortment
   - Visualize geographic patterns (if location data available)

1. **What-If Analysis**

   - Toggle promotional scenarios
   - Simulate holiday impacts
   - Compare forecast changes

1. **Model Performance Dashboard**

   - Display current model metrics (RMSPE)
   - Show feature importance
   - Compare model versions

1. **Data Quality Dashboard**

   - Show Great Expectations validation results
   - Data freshness indicators
   - Missing data reports

**Example Streamlit code**:

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.title("🏪 Rossmann Sales Forecasting Dashboard")

# Sidebar - Store selection
store_id = st.sidebar.selectbox("Select Store", range(1, 1116))

# Main panel - Forecast
st.header(f"Sales Forecast for Store {store_id}")

# Date range selector
forecast_days = st.slider("Forecast horizon (days)", 7, 42, 42)

# Promotional scenario
promo_enabled = st.checkbox("Enable promotional forecast")

# Get prediction from API
response = requests.post(
    "http://localhost:8000/predict",
    json={"Store": store_id, "days": forecast_days, "promo": promo_enabled}
)

forecast_data = response.json()

# Plot forecast
fig = px.line(forecast_data, x='date', y='predicted_sales',
              title=f'Sales Forecast - Store {store_id}')
st.plotly_chart(fig)

# Show metrics
col1, col2, col3 = st.columns(3)
col1.metric("Avg Daily Sales", f"${forecast_data['avg_sales']:.2f}")
col2.metric("Total Period Sales", f"${forecast_data['total_sales']:.2f}")
col3.metric("Model RMSPE", f"{forecast_data['model_rmspe']:.4f}")
```

______________________________________________________________________

### 4. **Model Monitoring** (Drift Detection & Performance Tracking)

**Current Gap**: No monitoring infrastructure.

**Implementation**:

#### A. Evidently AI for Drift Detection

Monitor data drift and model performance degradation.

**New files**:

- `src/monitoring/drift_detection.py` - Drift monitoring logic
- `src/monitoring/generate_reports.py` - Create monitoring reports
- `monitoring/` - Store monitoring artifacts

**Monitoring Components**:

1. **Data Drift Detection**

   - Compare new data distribution vs training data
   - Monitor feature drift (promotions, holidays, competition)
   - Alert on significant shifts

1. **Prediction Drift**

   - Track prediction distributions over time
   - Detect unusual prediction patterns

1. **Performance Monitoring**

   - Compare actual sales (when available) vs predictions
   - Calculate rolling RMSPE
   - Track prediction errors by store type, day of week

**Example implementation**:

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

# Create drift report
column_mapping = ColumnMapping(
    target='Sales',
    prediction='predicted_sales',
    numerical_features=['CompetitionDistance', 'Promo', ...],
    categorical_features=['StoreType', 'Assortment', ...]
)

report = Report(metrics=[
    DataDriftPreset(),
    RegressionPreset()
])

report.run(
    reference_data=train_data,  # Historical training data
    current_data=new_data,       # Recent production data
    column_mapping=column_mapping
)

# Save report
report.save_html("monitoring/drift_report.html")

# Check for drift
results = report.as_dict()
if results['metrics'][0]['result']['dataset_drift']:
    print("⚠️ Data drift detected! Consider retraining.")
```

#### B. Logging & Metrics

- Log all predictions to database/file
- Track prediction latency
- Monitor API usage patterns

**New files**:

- `deployment/prediction_logger.py` - Log predictions
- `monitoring/analyze_predictions.py` - Analyze prediction logs

______________________________________________________________________

### 5. **CI/CD Pipeline** (Automated Testing & Deployment)

**Current Gap**: No automated testing or deployment.

**Implementation**:

#### A. GitHub Actions Workflows

**New files**:

- `.github/workflows/test.yml` - Run tests on PR
- `.github/workflows/train-model.yml` - Automated retraining
- `.github/workflows/deploy.yml` - Deploy model to production

**Workflow 1: Testing** (`.github/workflows/test.yml`)

```yaml
name: Test Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install -e .

      - name: Run data validation tests
        run: pytest tests/test_data_validation.py

      - name: Run feature engineering tests
        run: pytest tests/test_features.py

      - name: Run model tests
        run: pytest tests/test_models.py

      - name: Check code quality
        run: |
          ruff check .
          black --check .
```

**Workflow 2: Automated Model Training** (`.github/workflows/train-model.yml`)

```yaml
name: Automated Model Retraining

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install -e .

      - name: Pull latest data with DVC
        run: dvc pull

      - name: Run data validation
        run: python src/data/validate_data.py

      - name: Train model
        run: python scripts/train_production_model.py

      - name: Evaluate model
        run: python scripts/evaluate_model.py

      - name: Register model to MLflow
        if: success()
        run: python scripts/register_model.py

      - name: Generate monitoring report
        run: python src/monitoring/generate_reports.py
```

**Workflow 3: Deployment** (`.github/workflows/deploy.yml`)

```yaml
name: Deploy Model

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t rossmann-api:${{ github.event.inputs.model_version }} .

      - name: Test API
        run: |
          docker run -d -p 8000:8000 rossmann-api:${{ github.event.inputs.model_version }}
          sleep 10
          pytest tests/test_api.py

      - name: Push to registry
        run: |
          # Push to Docker registry or deploy to cloud
```

#### B. Testing Infrastructure

**New test files**:

- `tests/test_data_validation.py` - Test data validation
- `tests/test_features.py` - Test feature engineering
- `tests/test_models.py` - Test model training/prediction
- `tests/test_api.py` - Test API endpoints
- `tests/conftest.py` - Pytest fixtures

**Example tests**:

```python
# tests/test_features.py
import pytest
from src.features.build_features import add_lag_features, add_rolling_features

def test_lag_features_no_leakage(sample_data):
    """Ensure lag features don't leak future information"""
    df_with_lags = add_lag_features(sample_data, lags=[7])

    # First 7 days should have NaN lags
    assert df_with_lags.iloc[:7]['sales_lag_7'].isna().all()

    # Lag should equal sales from 7 days ago
    assert df_with_lags.iloc[7]['sales_lag_7'] == df_with_lags.iloc[0]['Sales']

def test_rolling_features_correct_window(sample_data):
    """Ensure rolling features use correct window"""
    df_with_rolling = add_rolling_features(sample_data, windows=[7])

    # Check rolling mean calculation
    expected_mean = sample_data.iloc[:7]['Sales'].mean()
    assert abs(df_with_rolling.iloc[6]['sales_rolling_mean_7'] - expected_mean) < 0.01
```

______________________________________________________________________

### 6. **Containerization** (Docker)

**Current Gap**: No containerization.

**Implementation**:

**New files**:

- `Dockerfile` - Container for API/training
- `docker-compose.yml` - Multi-container orchestration
- `.dockerignore` - Files to exclude from image

**Dockerfile**:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY deployment/ deployment/
COPY models/ models/

# Install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD [".venv/bin/uvicorn", "deployment.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./models:/app/models
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - ./mlruns:/mlflow/mlruns

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

______________________________________________________________________

### 7. **Migration to `uv`** (Package Management)

**Current Gap**: Likely using pip/conda.

**Implementation**:

**New file**: `pyproject.toml`

```toml
[project]
name = "rossmann-forecasting"
version = "1.0.0"
description = "Production ML system for Rossmann sales forecasting"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "mlflow>=2.8.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
    "evidently>=0.4.0",
    "great-expectations>=0.18.0",
    "dvc>=3.0.0",
    "pyyaml>=6.0.0",
    "pydantic>=2.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "jupyter>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Remove**:

- `env/requirements.txt`
- `env/environment.yml`

**Update README** with uv setup:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
```

______________________________________________________________________

## Enhanced Repository Structure

```
rossmann-forecasting/
├── .github/
│   └── workflows/
│       ├── test.yml                    # NEW: CI testing
│       ├── train-model.yml             # NEW: Automated retraining
│       └── deploy.yml                  # NEW: Deployment workflow
│
├── config/
│   ├── params.yaml                     # Existing
│   └── mlflow.yaml                     # NEW: MLflow config
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── train.csv.dvc               # NEW: DVC tracking
│   │   ├── store.csv
│   │   └── store.csv.dvc               # NEW: DVC tracking
│   ├── processed/
│   │   ├── train_clean.parquet.dvc     # NEW: DVC tracking
│   │   └── train_features.parquet.dvc  # NEW: DVC tracking
│   └── external/
│
├── deployment/                         # NEW: Deployment code
│   ├── fastapi_app.py
│   ├── streamlit_app.py
│   ├── schemas.py
│   ├── model_loader.py
│   └── prediction_logger.py
│
├── expectations/                       # NEW: Great Expectations
│   ├── raw_data.json
│   └── features.json
│
├── great_expectations/                 # NEW: GX config
│   └── great_expectations.yml
│
├── mlruns/                            # NEW: MLflow tracking data
│   └── (generated by MLflow)
│
├── monitoring/                         # NEW: Monitoring artifacts
│   ├── drift_reports/
│   └── performance_reports/
│
├── notebooks/
│   ├── 01-eda-and-cleaning.ipynb       # UPDATED: Add validation
│   ├── 02-feature-engineering.ipynb    # UPDATED: Add MLflow tracking
│   ├── 03-baseline-models.ipynb        # UPDATED: Add MLflow tracking
│   ├── 04-advanced-models-and-ensembles.ipynb  # UPDATED: Add MLflow
│   ├── 05-final-eval-and-test-simulation.ipynb # UPDATED: Add monitoring
│   └── 06-deployment-demo.ipynb        # NEW: API/Streamlit demo
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py             # UPDATED: Add validation
│   │   └── validate_data.py            # NEW: Data validation
│   ├── features/
│   │   └── build_features.py           # Existing
│   ├── models/
│   │   ├── train_baselines.py          # UPDATED: Add MLflow
│   │   ├── train_advanced.py           # UPDATED: Add MLflow
│   │   ├── ensembles.py                # UPDATED: Add MLflow
│   │   └── register_model.py           # NEW: Model registration
│   ├── evaluation/
│   │   ├── metrics.py                  # Existing
│   │   ├── cv.py                       # Existing
│   │   └── reporting.py                # UPDATED: Add MLflow
│   ├── monitoring/                     # NEW: Monitoring module
│   │   ├── drift_detection.py
│   │   ├── generate_reports.py
│   │   └── analyze_predictions.py
│   └── utils/
│       ├── io.py                       # Existing
│       ├── log.py                      # Existing
│       └── mlflow_utils.py             # NEW: MLflow helpers
│
├── scripts/                            # NEW: Automation scripts
│   ├── train_production_model.py
│   ├── evaluate_model.py
│   └── register_model.py
│
├── tests/                              # NEW: Test suite
│   ├── conftest.py
│   ├── test_data_validation.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── .dvc/                              # NEW: DVC config
├── .dvcignore                         # NEW: DVC ignore
├── .dockerignore                      # NEW: Docker ignore
├── docker-compose.yml                 # NEW: Multi-container setup
├── Dockerfile                         # NEW: API container
├── Dockerfile.streamlit               # NEW: Streamlit container
├── pyproject.toml                     # NEW: uv/Python config
├── README.md                          # UPDATED: New setup instructions
├── spec-rossmann-forecasting.md       # Existing
└── spec-rossmann-mlops.md             # NEW: This document
```

______________________________________________________________________

## Implementation Roadmap

### Phase 1: MLOps Foundation (Week 1)

- [ ] Migrate to `uv` (create `pyproject.toml`, remove old requirements)
- [ ] Set up MLflow tracking server
- [ ] Add MLflow logging to all model training code
- [ ] Create MLflow model registry structure

### Phase 2: DataOps (Week 1-2)

- [ ] Initialize DVC for data versioning
- [ ] Set up Great Expectations
- [ ] Create data validation suites
- [ ] Integrate validation into data pipeline

### Phase 3: Deployment (Week 2-3)

- [ ] Build FastAPI application
- [ ] Create Streamlit dashboard
- [ ] Dockerize applications
- [ ] Set up docker-compose for local deployment

### Phase 4: Monitoring (Week 3)

- [ ] Implement drift detection with Evidently
- [ ] Create prediction logging
- [ ] Build monitoring dashboards
- [ ] Set up alerting (optional)

### Phase 5: CI/CD (Week 3-4)

- [ ] Create test suite
- [ ] Set up GitHub Actions workflows
- [ ] Implement automated testing
- [ ] Create automated retraining pipeline

### Phase 6: Documentation & Polish (Week 4)

- [ ] Update README with new setup instructions
- [ ] Create deployment guide
- [ ] Add API documentation (OpenAPI/Swagger)
- [ ] Create demo notebook showing full workflow

______________________________________________________________________

## Teaching Integration

This enhanced project will demonstrate concepts from each week of the course:

**Week 1 (Foundations + Git)**

- Repo structure and Git workflow
- Branching strategy for experiments

**Week 2 (DataOps)**

- DVC for data versioning
- Great Expectations for validation
- Data pipeline best practices

**Week 3 (ModelOps - Training)**

- MLflow experiment tracking
- Systematic hyperparameter tuning
- Model comparison

**Week 4 (ModelOps - Deployment)**

- FastAPI for model serving
- Streamlit for user interface
- Docker containerization

**Week 5 (Monitoring & CI/CD)**

- Evidently for drift detection
- GitHub Actions for automation
- Automated testing

**Week 6 (Advanced Topics)**

- Scalability considerations
- Cost optimization
- A/B testing patterns

**Week 7 (Human Elements)**

- Documentation for stakeholders
- Responsible AI considerations
- Project presentation

______________________________________________________________________

## Success Metrics

The enhanced project is successful when:

1. ✅ All model experiments tracked in MLflow
1. ✅ Data versions managed with DVC
1. ✅ Data validated with Great Expectations
1. ✅ Working FastAPI + Streamlit deployment
1. ✅ Drift monitoring reports generated
1. ✅ CI/CD pipeline running automated tests
1. ✅ All code containerized with Docker
1. ✅ Comprehensive test coverage (>80%)
1. ✅ Full documentation for setup and usage
1. ✅ Can be cloned and run locally with `uv` and `docker-compose up`

______________________________________________________________________

## Notes for Course Integration

### Simplified Examples (In-Book)

Extract simplified versions of each component for book chapters:

- `simplified/01-data-pipeline/` - Basic data validation example
- `simplified/02-mlflow-tracking/` - Simple experiment tracking
- `simplified/03-fastapi-deployment/` - Minimal API example
- `simplified/04-streamlit-dashboard/` - Basic dashboard

### Production Examples (Full Repo)

Link to full implementation in the main repo showing:

- Complete error handling
- Production-grade logging
- Security considerations
- Performance optimizations
- Comprehensive testing

### Student Adaptation

Students can use this as a template for their projects by:

1. Replacing data source and problem domain
1. Adapting feature engineering to their use case
1. Using same MLOps infrastructure (MLflow, DVC, etc.)
1. Following same deployment patterns
1. Applying same monitoring approaches
