# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Rossmann Sales Forecasting** project using the Kaggle Rossmann dataset. The goal is to develop a machine learning model that achieves **RMSPE \< 0.09856** (top 50 leaderboard performance) using rigorous time-series validation and broad exploration of modeling + feature engineering techniques.

The project forecasts daily sales for 3,000+ stores across Europe for a 6-week period, accounting for promotions, competition, holidays, seasonality, and locality.

## Key Commands

### Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from pyproject.toml
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (recommended)
pre-commit install
```

### Quality Assurance

```bash
# Run all tests with coverage
pytest -v

# Run specific test file
pytest tests/test_features.py -v

# Run pre-commit checks on all files
pre-commit run --all-files

# Run data validation
python src/data/validate_data.py --stage raw
python src/data/validate_data.py --stage processed

# Build and serve documentation locally
mkdocs serve
```

### DataOps Automation

```bash
# Run full DataOps workflow (data validation + feature engineering)
bash scripts/dataops_workflow.sh

# Run with DVC tracking
dvc repro
```

### Running Notebooks in Order

The project follows a sequential notebook workflow:

```bash
# 1. EDA and cleaning
jupyter notebook notebooks/01-eda-and-cleaning.ipynb

# 2. Feature engineering
jupyter notebook notebooks/02-feature-engineering.ipynb

# 3. Baseline models
jupyter notebook notebooks/03-baseline-models.ipynb

# 4. Advanced models and ensembles
jupyter notebook notebooks/04-advanced-models-and-ensembles.ipynb

# 5. Final evaluation
jupyter notebook notebooks/05-final-eval-and-test-simulation.ipynb
```

### ModelOps Production Workflows

```bash
# Launch MLflow UI to view experiments
bash scripts/start_mlflow.sh
# Opens at http://localhost:5000

# Train production ensemble model
python src/models/train_ensemble.py

# Validate and promote model to Staging
python src/models/validate_model.py

# Promote Staging model to Production (after manual review)
python src/models/validate_model.py --promote-to-production

# Generate predictions using Production model
python src/models/predict.py --stage Production

# Run complete retraining pipeline (automated)
bash scripts/retrain_pipeline.sh
```

### Deployment Services

```bash
# Launch all deployment services (MLflow, FastAPI, Streamlit)
bash scripts/launch_dashboard.sh
# Opens: MLflow (5000), FastAPI (8000), Streamlit (8501)

# Launch individual services
bash scripts/launch_api.sh        # FastAPI only
bash scripts/launch_streamlit.sh  # Streamlit only
bash scripts/start_mlflow.sh      # MLflow only

# Stop background services
lsof -ti:5000 | xargs kill -9  # Stop MLflow
lsof -ti:8000 | xargs kill -9  # Stop FastAPI
# Streamlit: Ctrl+C (runs in foreground)
```

### Monitoring Workflows

```bash
# Prepare reference data for drift detection (run once)
python src/monitoring/prepare_reference_data.py

# Generate drift detection report (run weekly/monthly)
python src/monitoring/generate_reports.py --days 7   # Last 7 days
python src/monitoring/generate_reports.py --days 30  # Last 30 days

# View monitoring dashboard in Streamlit
# Navigate to "📊 Monitoring" page in Streamlit UI

# Query prediction database directly (advanced)
sqlite3 data/monitoring/predictions.db "SELECT COUNT(*) FROM predictions;"
```

### Running Individual Modules

```bash
# Data preparation
python -m src.data.make_dataset

# Feature engineering
python -m src.features.build_features
```

## Architecture

### Data Flow

**DataOps Pipeline (Experimentation):**

1. **Raw data** (`data/raw/`) → train.csv, store.csv
1. **Cleaning** (`src/data/make_dataset.py`) → `data/processed/train_clean.parquet`
1. **Feature Engineering** (`src/features/build_features.py`) → `data/processed/train_features.parquet`
1. **Experimentation** (notebooks 03, 04, 05) → MLflow experiment tracking
1. **Hyperparameter tuning** (Optuna) → `config/best_hyperparameters.json`

**ModelOps Pipeline (Production):**

1. **Load features** → `data/processed/train_features.parquet`
1. **Load best hyperparameters** → `config/best_hyperparameters.json`
1. **Train ensemble** (`src/models/train_ensemble.py`) → Register to MLflow Model Registry
1. **Validate model** (`src/models/validate_model.py`) → Auto-promote to Staging if RMSPE \< 0.10
1. **Manual review** → MLflow UI inspection, inference testing
1. **Promote to Production** → `validate_model.py --promote-to-production`
1. **Generate predictions** (`src/models/predict.py`) → Load model by stage, save predictions

**Deployment Architecture (Production Serving):**

1. **MLflow Model Registry** (Port 5000) → Centralized model storage and versioning
1. **FastAPI Service** (Port 8000) → REST API for model predictions
    - Loads Production model on startup
    - Accepts 7 simple fields (train.csv format)
    - Auto-applies 46 feature transformations
    - Returns predictions with model version
1. **Streamlit Dashboard** (Port 8501) → User-friendly web interface
    - Single prediction: Interactive form for one store/date
    - Batch upload: CSV file processing for multiple predictions
    - System health monitoring and model registry status
1. **Unified Launcher** → `scripts/launch_dashboard.sh` starts all services with health checks

### Module Responsibilities

**src/utils/io.py**

- `read_parquet()`: Load parquet files with automatic categorical column restoration
    - By default, converts known categorical columns (StoreType, Assortment, StateHoliday, PromoInterval) back to category dtype
    - Use `categorize=False` to keep as strings if needed
- `save_parquet()`: Save parquet files with automatic categorical-to-string conversion
    - Prevents Arrow conversion errors with mixed string/numeric categorical values
    - Categorical columns are stored as strings and restored on load

**src/data/make_dataset.py**

- `load_raw_data()`: Load train.csv and store.csv
- `merge_store_info()`: Join on Store column
- `basic_cleaning()`: Handle missing values, convert dtypes, parse dates
- `save_processed_data()`: Output to parquet using `save_parquet()` utility

**src/data/prepare_predictions.py** ⭐ NEW

- `prepare_prediction_data()`: Unified pipeline for preparing raw data for predictions
    - Accepts raw data in train.csv format (7 fields: Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday)
    - Automatically merges with store metadata, cleans data, engineers features
    - Returns model-ready features (46 columns, excludes Sales/Date/Store/Customers)
    - Ensures consistency between training and inference data processing
- `validate_input_data()`: Validates raw input data for required columns, types, and ranges
- `prepare_prediction_data_from_csv()`: Convenience wrapper to load and prepare data from CSV
- **Used by**: FastAPI deployment, batch prediction scripts, any inference workflow

**src/features/build_features.py**

- `add_calendar_features()`: Year, month, week, day-of-week, seasonality
- `add_promo_features()`: Promo, Promo2, durations, intervals
- `add_competition_features()`: Competition distance, age
- `add_lag_features()`: Store-level lags \[1, 7, 14, 28\] using groupby().shift()
- `add_rolling_features()`: Rolling means/stds using groupby().rolling()
- `build_all_features()`: Orchestrates all feature functions

**src/evaluation/cv.py**

- `make_time_series_folds()`: Creates expanding window splits for time-series CV
- Each fold: train on historical data, validate on next 6-week period
- Must prevent leakage: validation data never influences train features

**src/evaluation/metrics.py**

- `rmspe()`: Root Mean Square Percentage Error (ignores Sales=0)
    ```python
    def rmspe(y_true, y_pred, ignore_zero_sales=True):
        mask = y_true != 0 if ignore_zero_sales else np.ones_like(y_true, dtype=bool)
        return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))
    ```

**src/models/ensemble.py**

- `RossmannEnsemble`: Custom MLflow PyFunc wrapper for ensemble models
- Combines LightGBM (30%), XGBoost (60%), CatBoost (10%) predictions
- Enables single-artifact deployment with all three models packaged together
- Handles feature preprocessing and weighted prediction averaging

**src/models/train_ensemble.py**

- Production training script for ensemble model
- Loads best hyperparameters from `config/best_hyperparameters.json`
- Trains LightGBM, XGBoost, CatBoost with optimized parameters
- Wraps models in `RossmannEnsemble` PyFunc for deployment
- Registers trained ensemble to MLflow Model Registry
- Logs hyperparameters, metrics, and artifacts to MLflow

**src/models/validate_model.py**

- Automated validation and stage promotion workflow
- Loads latest registered model from MLflow
- Evaluates on holdout validation set (RMSPE calculation)
- Auto-promotes to Staging if RMSPE \< 0.10 threshold
- Supports manual Production promotion: `--promote-to-production`
- Archives previous Production model when promoting new version

**src/models/model_registry.py**

- Utilities for interacting with MLflow Model Registry
- `load_model()`: Load model by stage (Staging/Production) or version number
- `promote_model()`: Transition model between stages with optional archival
- `get_model_version()`: Retrieve version number for a given stage
- `get_model_info()`: Fetch model metadata and version details
- `list_registered_models()`: List all models in registry

**src/models/predict.py**

- Production inference pipeline
- Loads model from MLflow Registry by stage (default: Production)
- Generates predictions for new data
- Saves predictions to CSV with timestamps
- Command-line interface: `--stage`, `--output`, `--data-path` options

### Deployment Services

**deployment/api/** - FastAPI Prediction Service

- `main.py`: FastAPI application with startup model loading and prediction endpoints
- `model_manager.py`: Abstraction for MLflow model loading and prediction orchestration
- `schemas.py`: Pydantic models for request/response validation (train.csv format)
- Endpoints: `/health`, `/model/info`, `/predict`, `/model/load`
- Loads Production model on startup, caches in memory for fast inference
- Accepts 7 simple fields, auto-applies 46 feature transformations via `prepare_prediction_data()`

**deployment/streamlit/** - Streamlit Dashboard

- `Home.py`: Landing page with system status, model registry, and quick start guide
- `pages/1_📈_Predictions.py`: Single and batch prediction interfaces
- `pages/2_📚_Documentation.py`: API reference and integration examples
- `pages/3_📊_Monitoring.py`: Prediction usage statistics and drift detection dashboard
- `utils/api_client.py`: FastAPI client abstraction with error handling
- `utils/validation.py`: Client-side input validation and CSV template generation
- Multi-page app architecture with automatic navigation

**Monitoring System** - Prediction Logging and Drift Detection

- `deployment/api/prediction_logger.py`: SQLite-based prediction logging
    - Logs predictions, key features (~10), and metadata to `data/monitoring/predictions.db`
    - Integrated into `/predict` endpoint for automatic logging
    - Supports querying by date range, model version, and batch ID
- `src/monitoring/drift_detection.py`: Custom drift detection using Evidently AI framework
    - Compares production predictions against training data reference
    - Uses KS test for numerical features, Total Variation Distance for categorical features
    - Monitors ~10 key features (promo, day_of_week, month, state_holiday, store_type, etc.)
    - Generates drift summary with feature-level drift scores
    - Flags dataset-level drift when >50% of features show drift

**scripts/** - Automation Scripts

- `launch_dashboard.sh`: Unified launcher for MLflow, FastAPI, and Streamlit with health checks
- `launch_api.sh`: Standalone FastAPI launcher with dependency checking
- `launch_streamlit.sh`: Standalone Streamlit launcher
- `start_mlflow.sh`: MLflow tracking server launcher
- Smart port detection, health check timeouts, and background process management

### Monitoring Infrastructure

The project implements a lightweight monitoring system for tracking predictions and detecting data drift:

**Components**:

- **Prediction Logging**: SQLite database (`data/monitoring/predictions.db`) logs all predictions with metadata
- **Drift Detection**: Statistical comparison of production data vs. training data distributions
- **Monitoring Dashboard**: Streamlit page showing usage statistics and drift analysis

**Key Features**:

- Logs ~10 key features per prediction (promo, day_of_week, month, store_type, competition_distance, etc.)
- Statistical tests: Kolmogorov-Smirnov for numerical, Total Variation Distance for categorical
- Drift thresholds: p-value \< 0.05 (KS test), TVD > 0.2 (categorical)
- Dataset drift flagged when >50% of monitored features show drift
- Reference data: Full training feature set (`data/processed/train_features.parquet`)

**Access**: Navigate to "📊 Monitoring" page in Streamlit app after making predictions

**Documentation**: See `docs/monitoring/` for complete monitoring documentation

### Configuration

**config/params.yaml** - Central configuration for:

- Data paths
- Feature engineering parameters (lags, windows)
- CV strategy (expanding, fold length, min train days)
- Model hyperparameters (deprecated in favor of `best_hyperparameters.json`)

**config/best_hyperparameters.json** - Production hyperparameters:

- Generated by Optuna tuning in notebook 04
- Contains optimal hyperparameters for LightGBM, XGBoost, CatBoost
- Includes metadata (best model, CV RMSPE scores)
- Used by `train_ensemble.py` for production training
- JSON format for easy parsing and version control

### Quality Infrastructure

**Testing** (`tests/`)

- `test_data_processing.py`: Data loading, merging, cleaning, quality checks
- `test_data_validation.py`: Great Expectations integration, schema validation
- `test_features.py`: Feature engineering validation, time-series safety checks
- `test_monitoring.py`: Prediction logging, drift detection, database operations
- All tests use proper fixtures (`sample_train_data`, `sample_store_data`, `sample_features_data`)
- Run with `pytest -v` for full test suite

**Pre-commit Hooks** (`.pre-commit-config.yaml`)

- **black**: Python code formatting (100-char line length)
- **ruff**: Fast linting with auto-fix (replaces flake8, isort)
- **ruff-format**: Ensures consistent code style
- **mdformat**: Markdown formatting with GFM support
- **docformatter**: Python docstring formatting (Google/NumPy style)
- **Standard hooks**: trailing whitespace, EOF fixer, YAML/JSON/TOML validation, large file detection, private key detection

**Data Validation** (`src/data/validate_data.py`)

- Great Expectations integration for data quality checks
- Validates raw data (train.csv, store.csv) and processed data
- Schema validation, range checks, uniqueness constraints
- Run with `python src/data/validate_data.py --stage [raw|processed]`

**Documentation** (`docs/` + MkDocs)

- Comprehensive documentation in `docs/` directory
- Auto-generated API docs from docstrings
- Getting started guides, architecture diagrams, tutorials
- Build with `mkdocs build`, serve with `mkdocs serve`

**DataOps Automation** (`scripts/dataops_workflow.sh`)

- Automated data validation → feature engineering pipeline
- DVC integration for data versioning
- Ensures reproducible data processing
- Run with `bash scripts/dataops_workflow.sh`

**MLflow Integration**

- Experiment tracking for all notebook-based model training (notebooks 03, 04, 05)
- Optuna-MLflow integration for hyperparameter tuning visualization
- Model Registry for versioning and stage-based lifecycle management
- Stages: None (Registered) → Staging → Production → Archived
- Launch UI: `bash scripts/start_mlflow.sh` or `mlflow ui`
- Web interface at http://localhost:5000

**ModelOps Automation** (`scripts/retrain_pipeline.sh`)

- End-to-end retraining workflow from data validation to model promotion
- Steps: Data validation → Feature engineering → Model training → Validation → Staging promotion
- Designed for scheduled execution (cron, Airflow) or manual triggers
- Requires manual review before Production promotion (human-in-the-loop safety)
- Run with `bash scripts/retrain_pipeline.sh`

**Documentation** (`docs/` + MkDocs)

- Comprehensive documentation in `docs/` directory organized by workflow phase
- **Deployment docs** (`docs/deployment/`):
    - `overview.md`: Architecture, data flow diagrams, prerequisites
    - `fastapi.md`: API endpoints, feature pipeline, error handling
    - `streamlit.md`: UI features with screenshots (home, single/batch predictions)
    - `launcher.md`: Unified launcher process flow and troubleshooting
- **Monitoring docs** (`docs/monitoring/`):
    - `overview.md`: Monitoring introduction, components, design decisions, limitations
    - `drift-detection.md`: Statistical methodology (KS test, TVD), feature monitoring, drift thresholds
    - `dashboard.md`: Dashboard usage guide with screenshots
- **API docs** (`docs/api/`): Auto-generated from code docstrings
- **DataOps/ModelOps docs**: Getting started guides and best practices
- Build with `mkdocs build`, serve with `mkdocs serve` (http://localhost:8000)

## ModelOps Workflow

The project implements a complete MLOps lifecycle from experimentation to production deployment:

### 1. Experimentation (Notebooks)

- **Notebook 03**: Baseline models (naive, simple LightGBM/XGBoost) to establish performance benchmarks
- **Notebook 04**: Hyperparameter tuning with Optuna (100+ trials per model), ensemble development
- **Notebook 05**: Final model training with best hyperparameters, evaluation on holdout test set
- All experiments tracked in MLflow with hyperparameters, metrics (RMSPE), and artifacts
- Best hyperparameters saved to `config/best_hyperparameters.json` for production use

### 2. Model Registry & Lifecycle

- **Registration**: Ensemble models registered to MLflow Model Registry with version numbers
- **Staging Validation**: Auto-promote to "Staging" if RMSPE \< 0.10 threshold
- **Manual Review**: Human reviews Staging model (predictions, metrics, business alignment)
- **Production Promotion**: Manual approval required (`--promote-to-production` flag)
- **Archival**: Previous Production model automatically archived when new version promoted
- **Lineage Tracking**: Each model linked to data version (DVC hash) and training parameters

### 3. Production Training & Inference

- **Training**: `train_ensemble.py` loads best hyperparameters and trains production ensemble
- **Validation**: `validate_model.py` evaluates performance and manages stage transitions
- **Prediction**: `predict.py` loads models by stage (Staging/Production) for inference
- **Retraining**: `retrain_pipeline.sh` orchestrates full workflow (data → features → training → validation)

### 4. Key Design Principles

- **Reproducibility**: All training uses versioned data (DVC) and tracked hyperparameters (MLflow)
- **Automation with Safety**: Automated Staging promotion, manual Production approval
- **Traceability**: Complete lineage from raw data → features → model → predictions
- **Rollback Capability**: Archived models retained for quick rollback if issues arise

## Critical Implementation Rules

### Time-Series Validation (CRITICAL)

- **NEVER use future data for features**: All lag/rolling features must use `groupby("Store").shift(lag)` or `groupby("Store").rolling(window)`
- **Expanding window CV**: Each fold trains on all historical data up to validation period
- **6-week validation windows**: Mimics Kaggle test period
- **No overlap**: Validation fold must not leak into training features

### Data Leakage Prevention

- `Customers` field is NOT available in test set - can only be used for EDA, not modeling
- Competition/Promo2 metadata may have missing values - handle with appropriate imputation
- `Sales=0` observations are ignored in RMSPE scoring
- Remove or flag stores when `Open=0`

### Target Variable

- **Sales** is the target
- Consider log transformation for stability
- RMSPE metric requires percentage error calculation
- Exclude `Sales=0` from evaluation (per Kaggle rules)

### Feature Engineering Requirements

- **Calendar**: Year, month, week, day-of-week, seasonality flags, IsMonthStart/End
- **Promotion**: Promo, Promo2, durations, intervals, active_this_month flags
- **Competition**: Distance (log-scaled), age derived from OpenSince fields
- **Lags**: \[1, 7, 14, 28\] days at store level
- **Rolling**: Means/stds for windows \[7, 14, 28, 60\]
- **Categoricals**: StoreType, Assortment, PromoInterval (one-hot or native handling)
- **Interactions**: holiday × promo, promo × season, competition × store type

## Phased Implementation Strategy

The project MUST be implemented phase by phase. Do not refactor working code unless needed.

### Phase 0: Project Skeleton ✅ COMPLETED

Create directory structure, README, requirements.txt, config/params.yaml, utility stubs, empty notebooks

### Phase 1: Data Cleaning & EDA ✅ COMPLETED

Implement `src/data/make_dataset.py`, populate notebook 01, output `train_clean.parquet`

### Phase 2: Feature Engineering ✅ COMPLETED

Implement `src/features/build_features.py`, populate notebook 02, output `train_features.parquet`

### Phase 3: Baseline Models ✅ COMPLETED

Implement CV framework, baseline models, populate notebook 03, save baseline metrics

### Phase 4: Advanced Models & Ensembles ✅ COMPLETED

Implement tuned models, ensemble methods, populate notebook 04, compare RMSPE

### Phase 5: Final Evaluation ✅ COMPLETED

Train final model on full data, evaluate on 6-week holdout, save predictions and metrics

### Phase 6: ModelOps Production Infrastructure ✅ COMPLETED

- MLflow experiment tracking integration (notebooks 03, 04, 05)
- Optuna hyperparameter optimization with MLflow callbacks
- Production training pipeline (`train_ensemble.py`)
- Model Registry with stage-based lifecycle management
- Automated validation and promotion workflows (`validate_model.py`)
- Production inference pipeline (`predict.py`)
- End-to-end retraining automation (`retrain_pipeline.sh`)
- Comprehensive ModelOps documentation (docs/modelops/)

## Model Families to Explore

**Time Series Models**: Prophet, SARIMA/SARIMAX, TBATS, optional deep learning (LSTM, GRU, NBEATS, NHITS, TCN)

**Tree-Based Models**: LightGBM, XGBoost, CatBoost, Random Forest/ExtraTrees

**General ML**: Linear/Elastic Net, SVR, kNN (low priority)

**Ensembles**: Weighted blending, stacking with meta-learner

## Success Criteria

- Achieve **RMSPE \< 0.09856** on holdout set
- Use strictly correct time-series validation methodology
- Document all feature engineering steps
- Compare baseline vs advanced models
- Save final predictions, model artifacts, and metrics
- Ensure final notebook runs end-to-end reproducibly

## Data Dictionary Summary

**Key fields from train.csv**:

- `Store`: Unique store ID (join key)
- `Date`: Observation date (convert to datetime)
- `Sales`: Target variable (ignore when 0 in RMSPE)
- `Customers`: Count (NOT available in test - EDA only)
- `Open`: Store open flag (0/1)
- `Promo`: Daily promo (0/1)
- `StateHoliday`: Holiday type (0, a, b, c)
- `SchoolHoliday`: School closure flag (0/1)
- `DayOfWeek`: Day of week (1-7)

**Key fields from store.csv**:

- `Store`: Join key
- `StoreType`: Store format (a, b, c, d)
- `Assortment`: Product assortment (a, b, c)
- `CompetitionDistance`: Distance to competitor (meters, may be missing)
- `CompetitionOpenSinceMonth/Year`: When competition opened
- `Promo2`: Long-running promo participation (0/1)
- `Promo2SinceYear/Week`: When Promo2 started
- `PromoInterval`: Months when Promo2 restarts (e.g., "Feb,May,Aug,Nov")

## Common Pitfalls to Avoid

1. **Using `Customers` field in models** - not available in test set
1. **Forward-looking features** - causes leakage and overfitting
1. **Ignoring `Sales=0` in RMSPE** - Kaggle ignores these observations
1. **Not grouping by Store** - lag/rolling features must be per-store
1. **Random CV splits** - must use time-based expanding window
1. **Training on closed stores** - filter `Open=0` or handle separately

## Code Quality Standards

### Code Formatting

- **Python**: Use black (100-char line length) and ruff for formatting
- **Docstrings**: Google/NumPy style with proper blank lines before lists (for MkDocs rendering)
- **Imports**: Organized alphabetically by ruff
- **Type hints**: Use modern Python 3.10+ style (`list[str]`, `dict[str, Any]`)

### Linting Rules

The project uses ruff with the following key rules:

- **E402 ignored**: Module-level imports after `sys.path.append()` (necessary for project structure)
- **E722 enforced**: Must use `except Exception:` instead of bare `except:`
- **B006 enforced**: No mutable default arguments (use `None` with initialization)
- **B007 enforced**: Unused loop variables must be prefixed with `_`
- **F841 enforced**: Remove unused variables

### Testing Requirements

- All data processing functions must have tests in `tests/test_data_processing.py`
- All feature engineering functions must have tests in `tests/test_features.py`
- Critical tests for time-series safety (lag/rolling features don't leak across stores)
- Test fixtures must match actual data structure:
    - `sample_train_data`: Train data only (no store metadata)
    - `sample_store_data`: Store metadata only
    - `sample_features_data`: Merged train + store data with basic features

### Documentation Requirements

- All functions must have docstrings with Parameters, Returns, and description
- Markdown files must have blank lines before/after lists (for proper rendering)
- Code references in docs should use `[filename.py:line](path/to/file.py#Lline)` format
- Update CLAUDE.md when adding new modules or workflows
