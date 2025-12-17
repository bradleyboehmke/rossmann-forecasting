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

### Running Individual Modules

```bash
# Data preparation
python -m src.data.make_dataset

# Feature engineering
python -m src.features.build_features

# Train baseline models
python -m src.models.train_baselines

# Train advanced models
python -m src.models.train_advanced
```

## Architecture

### Data Flow

1. **Raw data** (`data/raw/`) → train.csv, store.csv
1. **Cleaning** (`src/data/make_dataset.py`) → `data/processed/train_clean.parquet`
1. **Feature Engineering** (`src/features/build_features.py`) → `data/processed/train_features.parquet`
1. **Time-Series CV** (`src/evaluation/cv.py`) → expanding window splits
1. **Model Training** (`src/models/`) → per-fold predictions
1. **Ensemble** (`src/models/ensembles.py`) → weighted blend or stacking
1. **Final Model** → predictions saved to `outputs/predictions/`

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

**src/models/train_baselines.py**

- Naive last-week model (baseline benchmark)
- Simple LightGBM with default params
- Returns per-fold RMSPE

**src/models/train_advanced.py**

- Tuned LightGBM, XGBoost, CatBoost
- Hyperparameter optimization
- Cross-validation with RMSPE metric

**src/models/ensembles.py**

- `weighted_blend()`: Weighted average of model predictions
- `stacked_ensemble()`: Meta-learner (Linear/LightGBM) on out-of-fold predictions

### Configuration

**config/params.yaml** - Central configuration for:

- Data paths
- Feature engineering parameters (lags, windows)
- CV strategy (expanding, fold length, min train days)
- Model hyperparameters

### Quality Infrastructure

**Testing** (`tests/`)

- `test_data_processing.py`: Data loading, merging, cleaning, quality checks
- `test_data_validation.py`: Great Expectations integration, schema validation
- `test_features.py`: Feature engineering validation, time-series safety checks
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

### Phase 0: Project Skeleton

Create directory structure, README, requirements.txt, config/params.yaml, utility stubs, empty notebooks

### Phase 1: Data Cleaning & EDA

Implement `src/data/make_dataset.py`, populate notebook 01, output `train_clean.parquet`

### Phase 2: Feature Engineering

Implement `src/features/build_features.py`, populate notebook 02, output `train_features.parquet`

### Phase 3: Baseline Models

Implement CV framework, baseline models, populate notebook 03, save baseline metrics

### Phase 4: Advanced Models & Ensembles

Implement tuned models, ensemble methods, populate notebook 04, compare RMSPE

### Phase 5: Final Evaluation

Train final model on full data, evaluate on 6-week holdout, save predictions and metrics

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
