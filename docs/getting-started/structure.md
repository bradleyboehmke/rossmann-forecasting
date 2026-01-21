# Project Structure

Complete overview of the Rossmann forecasting project organization.

______________________________________________________________________

## Directory Tree

```
rossmann-forecasting/
├── .dvc/                       # DVC configuration and cache
│   ├── config                  # DVC remote storage config
│   └── .gitignore              # Ignore cache and tmp files
├── .github/                    # GitHub Actions workflows (TODO)
│   └── workflows/
│       ├── data-validation.yml
│       ├── model-training.yml
│       └── deploy.yml
├── data/                       # Data directory (NOT in git, DVC tracked)
│   ├── raw/                    # Original immutable data
│   │   ├── train.csv           # Training data from Kaggle
│   │   ├── store.csv           # Store metadata
│   │   ├── train.csv.dvc       # DVC pointer (IN git)
│   │   └── store.csv.dvc       # DVC pointer (IN git)
│   ├── processed/              # Cleaned and merged data
│   │   ├── train_clean.parquet
│   │   └── train_clean.parquet.dvc
│   └── features/               # Feature-engineered datasets
│       ├── train_features.parquet
│       └── train_features.parquet.dvc
├── deployment/                 # Deployment code (API, dashboard)
│   ├── __init__.py
│   ├── fastapi_app.py          # FastAPI prediction service
│   ├── streamlit_app.py        # Streamlit dashboard
│   ├── schemas.py              # Request/response schemas
│   ├── model_loader.py         # Load models from registry
│   └── prediction_logger.py    # Log predictions for monitoring
├── docs/                       # MkDocs documentation
│   ├── index.md                # Home page
│   ├── getting-started/
│   │   ├── setup.md            # Installation guide
│   │   ├── quickstart.md       # Quick commands
│   │   └── structure.md        # This file
│   ├── dataops/
│   │   ├── overview.md
│   │   ├── processing.md
│   │   ├── validation.md
│   │   ├── versioning.md
│   │   ├── workflow.md
│   │   └── pipeline.md
│   ├── modelops/
│   │   ├── overview.md
│   │   ├── tracking.md
│   │   ├── training.md
│   │   ├── registry.md
│   │   └── tuning.md
│   ├── deployment/
│   │   ├── overview.md
│   │   ├── fastapi.md
│   │   ├── streamlit.md
│   │   └── docker.md
│   ├── monitoring/
│   │   ├── overview.md
│   │   ├── drift.md
│   │   ├── performance.md
│   │   └── logging.md
│   ├── testing/
│   │   ├── overview.md
│   │   ├── data-tests.md
│   │   ├── model-tests.md
│   │   └── api-tests.md
│   ├── cicd/
│   │   ├── overview.md
│   │   ├── github-actions.md
│   │   └── workflows.md
│   └── api/
│       ├── data.md
│       ├── features.md
│       ├── models.md
│       ├── evaluation.md
│       └── monitoring.md
├── great_expectations/         # Great Expectations configuration
│   ├── great_expectations.yml  # Main GX config
│   ├── expectations/           # Expectation suites (validation rules)
│   │   ├── raw_train_suite.json
│   │   ├── raw_store_suite.json
│   │   ├── processed_data_suite.json
│   │   └── features_suite.json
│   ├── checkpoints/            # Validation checkpoints
│   │   ├── raw_data_checkpoint.yml
│   │   ├── processed_data_checkpoint.yml
│   │   └── features_checkpoint.yml
│   └── uncommitted/            # Validation results (NOT in git)
│       └── validations/
├── mlruns/                     # MLflow tracking data (NOT in git)
│   └── 0/
│       └── <experiment-runs>/
├── models/                     # Saved model artifacts (DVC tracked)
│   ├── baseline/
│   │   ├── naive_model.pkl
│   │   └── lgbm_baseline.pkl
│   ├── advanced/
│   │   ├── lgbm_tuned.pkl
│   │   ├── xgboost_tuned.pkl
│   │   └── catboost_tuned.pkl
│   ├── ensembles/
│   │   ├── weighted_blend.pkl
│   │   └── stacked_ensemble.pkl
│   └── final/
│       └── production_model.pkl
├── monitoring/                 # Monitoring outputs (NOT in git)
│   ├── drift_reports/
│   │   └── <timestamp>_drift_report.html
│   └── performance_reports/
│       └── <timestamp>_performance.json
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01-eda-and-cleaning.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-baseline-models.ipynb
│   ├── 04-advanced-models-and-ensembles.ipynb
│   └── 05-final-eval-and-test-simulation.ipynb
├── outputs/                    # Model outputs and artifacts
│   ├── predictions/
│   │   ├── baseline_predictions.csv
│   │   ├── ensemble_predictions.csv
│   │   └── final_test_predictions.csv
│   ├── metrics/
│   │   ├── cv_results.json
│   │   ├── model_comparison.json
│   │   └── final_metrics.json
│   └── visualizations/
│       ├── feature_importance.png
│       ├── cv_performance.png
│       └── predictions_vs_actual.png
├── scripts/                    # Automation scripts
│   ├── dataops_workflow.sh     # Complete DataOps workflow
│   ├── train_all_models.sh     # Train all model variants
│   └── deploy.sh               # Deployment automation
├── src/                        # Source code (Python package)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py     # Load and clean raw data
│   │   └── validate_data.py    # Great Expectations validation
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_baselines.py  # Baseline model training
│   │   ├── train_advanced.py   # Advanced model training
│   │   └── ensembles.py        # Ensemble methods
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── cv.py               # Time-series cross-validation
│   │   └── metrics.py          # RMSPE and other metrics
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detection.py  # Data drift monitoring
│   │   └── performance.py      # Model performance tracking
│   └── utils/
│       ├── __init__.py
│       ├── io.py               # Data I/O utilities
│       └── mlflow_utils.py     # MLflow helper functions
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_data_processing.py # Data processing tests
│   ├── test_data_validation.py # Validation framework tests
│   ├── test_features.py        # Feature engineering tests
│   ├── test_models.py          # Model training tests
│   ├── test_api.py             # API endpoint tests
│   └── test_monitoring.py      # Monitoring tests
├── .dvcignore                  # DVC ignore patterns
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git ignore patterns
├── CLAUDE.md                   # Instructions for Claude Code
├── dvc.yaml                    # DVC pipeline definition
├── Dockerfile                  # FastAPI container
├── Dockerfile.streamlit        # Streamlit container
├── docker-compose.yml          # Multi-container orchestration
├── mkdocs.yml                  # MkDocs configuration
├── pyproject.toml              # Python package configuration
└── README.md                   # Project overview

```

______________________________________________________________________

## Module Descriptions

### `src/data/`

**Data processing and validation**

- **`make_dataset.py`**: Loads raw CSV files, merges store metadata, performs basic cleaning, saves to parquet
- **`validate_data.py`**: Great Expectations validation at each stage (raw, processed, features)

### `src/features/`

**Feature engineering**

- **`build_features.py`**: Creates calendar, promo, competition, lag, and rolling features
    - Time-series safe (no data leakage)
    - Store-level grouping for lags/rolling
    - Output: `train_features.parquet`

### `src/models/`

**Model training**

- **`train_baselines.py`**: Naive and simple LightGBM baselines
- **`train_advanced.py`**: Tuned LightGBM, XGBoost, CatBoost with hyperparameter optimization
- **`ensembles.py`**: Weighted blending and stacking ensembles

### `src/evaluation/`

**Model evaluation**

- **`cv.py`**: Time-series cross-validation with expanding windows
- **`metrics.py`**: RMSPE calculation and other metrics

### `src/monitoring/`

**Production monitoring**

- **`drift_detection.py`**: Detect data drift using statistical tests
- **`performance.py`**: Track model performance over time

### `src/utils/`

**Shared utilities**

- **`io.py`**: Parquet I/O with automatic categorical handling
- **`mlflow_utils.py`**: MLflow experiment tracking helpers

### `deployment/`

**Deployment code**

- **`fastapi_app.py`**: REST API for predictions
- **`streamlit_app.py`**: Interactive dashboard
- **`schemas.py`**: Pydantic request/response schemas
- **`model_loader.py`**: Load models from MLflow registry
- **`prediction_logger.py`**: Log predictions for monitoring

### `tests/`

**Test suite**

- **`conftest.py`**: Shared pytest fixtures
- **`test_data_*.py`**: Data processing and validation tests
- **`test_features.py`**: Feature engineering tests
- **`test_models.py`**: Model training tests
- **`test_api.py`**: API endpoint tests
- **`test_monitoring.py`**: Monitoring tests

______________________________________________________________________

## Configuration Files

### `dvc.yaml`

Defines the complete data pipeline with dependencies and outputs. Enables reproducible runs with `dvc repro`.

**Key stages:**

1. `validate_raw_data` - Check raw data quality
1. `prepare_data` - Clean and merge
1. `validate_processed_data` - Validate cleaning
1. `build_features` - Feature engineering
1. `validate_features` - Validate features
1. `train_baselines` - Train baseline models
1. `train_advanced` - Train tuned models

### `pyproject.toml`

Modern Python packaging standard (PEP 621). Defines:

- Project metadata
- Dependencies (production, dev, docs)
- Build configuration
- Tool settings (pytest, black, ruff)

**Dependency groups:**

- **Production**: numpy, pandas, scikit-learn, lightgbm, xgboost, catboost
- **Dev**: pytest, black, ruff, mypy, pre-commit
- **Docs**: mkdocs, mkdocs-material, mkdocstrings

### `docker-compose.yml`

Multi-container orchestration:

- **api**: FastAPI prediction service (port 8000)
- **streamlit**: Dashboard (port 8501)
- **mlflow**: Tracking server (port 5000)

### `mkdocs.yml`

Documentation site configuration:

- Material theme with custom colors
- Navigation structure
- Plugins: search, git-revision-date
- Markdown extensions: admonitions, code highlighting, mermaid diagrams

______________________________________________________________________

## Version Control Strategy

### Git tracks:

- Source code (`src/`)
- Notebooks (`notebooks/`)
- Configuration files (`.yml`, `.toml`)
- DVC metadata files (`.dvc`)
- Tests (`tests/`)
- Documentation (`docs/`)

### DVC tracks:

- Raw data (`data/raw/`)
- Processed data (`data/processed/`)
- Features (`data/features/`)
- Models (`models/`)

### Not tracked (`.gitignore`):

- Virtual environments
- DVC cache (`.dvc/cache/`)
- MLflow runs (`mlruns/`)
- Monitoring outputs (`monitoring/`)
- Test coverage reports
- MkDocs build artifacts (`site/`)

______________________________________________________________________

## Next Steps

Now that you understand the project structure, explore the MLOps workflows:

- **[Quick Start](quickstart.md)** - Get up and running in 5 minutes
- **[DataOps Workflow](../dataops/overview.md)** - Data processing and validation
- **[Model Training](../modelops/overview.md)** - Train and track experiments
- **Deployment** - Deploy models to production (Coming Soon)
