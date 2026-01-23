# Rossmann Sales Forecasting: Initial Exploration (Stage 1)

An example of what a **Stage 1 initial exploration** project looks like - exploratory analysis in Jupyter notebooks to understand the problem and establish baseline performance.

## Project Maturity: Stage 1

This repository represents **Stage 1** in the data science project maturity progression:

- **Stage 1** (Initial Exploration): ← *You are here*
- **Stage 2** (Mature Research): Structured organization with modular code
- **Stage 3** (Production): MLOps infrastructure with automated pipelines

### What is Stage 1?

Stage 1 projects are characterized by:
- **Exploratory notebooks**: All analysis done directly in Jupyter notebooks
- **Self-contained code**: Functions and logic written inline, not in separate modules
- **Iterative experimentation**: Trying different approaches to understand what works
- **Basic organization**: Sequential notebooks, simple directory structure
- **Minimal infrastructure**: No testing, CI/CD, or deployment code yet

This is the natural starting point for most data science projects - explore the data, try some models, see what's promising.

## The Problem

Forecasting daily sales for 3,000+ Rossmann drug stores across Europe for a 6-week period. This is based on the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales).

**Data:**
- Historical daily sales for 1,115 stores (2.5 years)
- Store metadata (type, assortment, competition, promotions)
- ~1 million observations

**Evaluation Metric:** RMSPE (Root Mean Square Percentage Error)

## Initial Results

After exploring the data and trying some baseline models:

| Approach | RMSPE | Notes |
|----------|-------|-------|
| Naive (last week's sales) | 0.468 | Simple baseline |
| LightGBM with basic features | 0.141 | 70% improvement! |

**Key Findings:**
- Store-level lag features (7, 14, 28 days) are highly predictive
- Promotions have significant impact on sales
- Day of week and seasonality matter
- Time-series cross-validation essential (can't use random splits)

## Repository Structure

```
rossmann-forecasting/
├── notebooks/              # Exploratory notebooks (run in order)
│   ├── 01-eda-and-cleaning.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-baseline-models.ipynb
│
├── data/
│   ├── raw/               # train.csv, store.csv (from Kaggle)
│   └── processed/         # Cleaned and featured data
│
├── outputs/
│   ├── figures/           # EDA visualizations
│   └── metrics/           # Model performance results
│
└── env/
    └── requirements.txt   # Python dependencies
```
