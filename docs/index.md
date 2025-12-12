# Rossmann Sales Forecasting

!!! info "Production MLOps System" This repository accompanies the [ML/AI System Design book](https://bradleyboehmke.github.io/uc-bana-7075/), providing a hands-on implementation of the concepts covered in the course.

```
This project demonstrates end-to-end MLOps best practices for traditional machine learning, using the Kaggle Rossmann Store Sales competition as a real-world example. The reason for this example is because many students and individuals new to MLOps are often exposed to Kaggle examples—this is a great way to learn how to build production ML systems using familiar datasets.
```

## Overview

### The Business Problem

This project is based on the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales), which challenges participants to forecast daily sales for Rossmann drugstore chains across Europe. Rossmann operates over 3,000 stores in 7 European countries, and accurate sales forecasting is critical for inventory management, staffing decisions, and financial planning.

The competition dataset includes:

- **Historical sales data** for 1,115 stores from 2013-2015
- **Store metadata** including type, assortment, competition distance
- **Promotional information** for regular and extended promotions
- **External factors** like holidays, school closures, and seasonality

The forecasting challenge requires predicting sales for a 6-week period, with performance measured by **RMSPE (Root Mean Square Percentage Error)**.

### The Hypothetical Scenario

**Imagine you work as a data scientist at Rossmann.** You've successfully developed a forecasting model that achieves strong performance (RMSPE \< 0.10) on historical data. Now, your director has tasked you with **productionizing this system** to support ongoing business operations.

**The requirements:**

- **Automated daily/weekly updates**: New sales data arrives regularly and must be processed automatically
- **Data quality guarantees**: Invalid data must be caught before it corrupts the system
- **Model retraining**: The model should retrain automatically as new data becomes available
- **Production deployment**: Forecasts must be accessible via API and dashboard for business stakeholders
- **Monitoring and alerts**: System must detect data drift and performance degradation
- **Reproducibility**: All data, models, and predictions must be versioned and auditable

**This repository provides an exemplar approach** to productionizing a Kaggle-style model, assuming:

- New actual sales data arrives on a **daily or weekly basis**
- Historical data grows continuously, improving model performance over time
- The system must run **reliably with minimal human intervention**
- Quality and reproducibility are **more important than speed**

### What This Project Demonstrates

The Rossmann Sales Forecasting project is a **production-ready ML system** that showcases modern MLOps practices:

- 📊 **DataOps**: Data versioning (DVC), validation (Great Expectations), reproducible pipelines
- 🤖 **ModelOps**: Experiment tracking (MLflow), model registry, hyperparameter tuning
- 🚀 **Deployment**: REST API (FastAPI), interactive dashboard (Streamlit), containerization (Docker)
- 📈 **Monitoring**: Data drift detection (Evidently), model performance tracking
- ✅ **CI/CD**: Automated testing (pytest), GitHub Actions workflows

**This is not just a Kaggle competition submission**—it's a blueprint for building production ML systems that handle the messy realities of real-world data and operational constraints.

## Key Features

### Production-Grade Architecture

- Modular code structure with clear separation of concerns
- Comprehensive test suite with >80% coverage
- Type hints and documentation throughout
- Modern Python packaging with `uv`

### MLOps Infrastructure

- **Experiment Tracking**: All runs logged to MLflow
- **Data Validation**: Automated quality checks with Great Expectations
- **Version Control**: Data and models versioned with DVC
- **Containerization**: Docker for reproducible deployments
- **Monitoring**: Drift detection and performance tracking

## Quick Links

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } __Getting Started__

  ______________________________________________________________________

  Install and set up the project in minutes

  [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

- :material-database:{ .lg .middle } __DataOps__

  ______________________________________________________________________

  Learn data processing, validation, and versioning

  [:octicons-arrow-right-24: DataOps Guide](dataops/overview.md)

- :material-brain:{ .lg .middle } __ModelOps__

  ______________________________________________________________________

  Train, track, and register models with MLflow

  [:octicons-arrow-right-24: ModelOps Guide](modelops/overview.md)

- :material-cloud-upload:{ .lg .middle } __Deployment__

  ______________________________________________________________________

  Deploy models via API and dashboard

  [:octicons-arrow-right-24: Deployment Guide](deployment/overview.md)

</div>

## Technology Stack

### Core ML

- **Python 3.10+** with modern packaging (`uv`, `pyproject.toml`)
- **Pandas** for data manipulation
- **Scikit-learn** for preprocessing and metrics
- **LightGBM, XGBoost, CatBoost** for gradient boosting

### MLOps Tools

- **MLflow** - Experiment tracking and model registry
- **DVC** - Data and pipeline versioning
- **Great Expectations** - Data validation
- **Evidently** - Drift detection

### Deployment

- **FastAPI** - REST API for predictions
- **Streamlit** - Interactive dashboard
- **Docker** - Containerization
- **GitHub Actions** - CI/CD automation

### Development

- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting
- **mypy** - Type checking
- **MkDocs Material** - Documentation
