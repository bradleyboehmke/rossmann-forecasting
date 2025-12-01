# Rossmann Sales Forecasting

A machine learning project to forecast daily sales for 3,000+ Rossmann stores across Europe using rigorous time-series validation and advanced modeling techniques.

## Project Goal

Develop a forecasting model that achieves **RMSPE < 0.09856** (top 50 Kaggle leaderboard performance) for a 6-week sales prediction period. The model must account for promotions, competition, holidays, seasonality, and locality while maintaining strict time-series validation to prevent data leakage.

## Evaluation Metric

**Root Mean Square Percentage Error (RMSPE)**:

$$RMSPE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$$

- Observations where Sales = 0 are ignored in scoring
- Lower scores indicate better predictions
- Target: **RMSPE < 0.09856**

## High-Level Workflow

### Phase 0: Project Skeleton & Environment
- Create directory structure
- Set up dependencies and configuration
- Implement utility functions and metrics

### Phase 1: Data Loading, Cleaning & EDA
- Load raw data (train.csv, store.csv)
- Merge and clean datasets
- Perform exploratory data analysis
- Output: `data/processed/train_clean.parquet`

### Phase 2: Feature Engineering
- Calendar features (year, month, week, day-of-week, seasonality)
- Promotion features (Promo, Promo2, durations, intervals)
- Competition features (distance, age)
- Lag features (1, 7, 14, 28 days)
- Rolling window features (means, stds)
- Output: `data/processed/train_features.parquet`

### Phase 3: Baseline Models & Time-Series CV
- Implement expanding window cross-validation
- Train naive baseline models
- Train simple LightGBM baseline
- Establish performance benchmarks

### Phase 4: Advanced Models & Ensembles
- Train tuned tree-based models (LightGBM, XGBoost, CatBoost)
- Experiment with time-series models (Prophet, SARIMA, optional deep learning)
- Build weighted blends and stacked ensembles
- Compare model performance

### Phase 5: Final Model & Holdout Evaluation
- Train final model on full training data
- Generate predictions for 6-week holdout period
- Compute final RMSPE
- Save model artifacts and predictions

## Repository Structure

```
rossmann-forecasting/
├── data/                   # Data storage
│   ├── raw/               # Original data files
│   ├── processed/         # Cleaned and featured data
│   └── external/          # Additional data sources
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01-eda-and-cleaning.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-baseline-models.ipynb
│   ├── 04-advanced-models-and-ensembles.ipynb
│   ├── 05-final-eval-and-test-simulation.ipynb
│   └── scratch/           # Experimental notebooks
├── src/                   # Source code modules
│   ├── data/              # Data loading and cleaning
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   ├── evaluation/        # CV and metrics
│   └── utils/             # Utility functions
├── models/                # Saved model artifacts
│   ├── baseline/
│   └── final/
├── outputs/               # Generated outputs
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── config/                # Configuration files
└── env/                   # Environment specifications
```

## Getting Started

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r env/requirements.txt
```

### Running the Pipeline

Execute notebooks sequentially:

```bash
jupyter notebook notebooks/01-eda-and-cleaning.ipynb
jupyter notebook notebooks/02-feature-engineering.ipynb
jupyter notebook notebooks/03-baseline-models.ipynb
jupyter notebook notebooks/04-advanced-models-and-ensembles.ipynb
jupyter notebook notebooks/05-final-eval-and-test-simulation.ipynb
```

Or run individual modules:

```bash
python -m src.data.make_dataset
python -m src.features.build_features
python -m src.models.train_baselines
python -m src.models.train_advanced
```

## Key Implementation Principles

### Time-Series Validation
- Expanding window cross-validation with 6-week validation periods
- Strict temporal ordering to prevent data leakage
- All lag and rolling features use proper grouping and shifting

### Feature Engineering
- Store-level lags using `groupby("Store").shift(lag)`
- Rolling statistics using `groupby("Store").rolling(window)`
- No future data in features

### Data Handling
- `Customers` field excluded from modeling (not available in test set)
- Sales = 0 observations handled appropriately
- Proper imputation for missing competition/promotion metadata

## Success Criteria

- [ ] Model achieves RMSPE < 0.09856
- [ ] Validation uses strictly correct time-series methodology
- [ ] Feature engineering is documented and reproducible
- [ ] Baseline and advanced models are compared
- [ ] Final predictions and artifacts are saved
- [ ] Final notebook runs end-to-end

## Dataset

The project uses the Kaggle Rossmann Store Sales dataset:
- **train.csv**: Historical daily sales and operational data
- **store.csv**: Static store metadata (competition, promotions, store type)

Files must be placed in `data/raw/` before running the pipeline.
