# Rossmann Sales Forecasting: A Data Science Maturity Journey

<!-- badges: start -->

[![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental) [![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://bradleyboehmke.github.io/rossmann-forecasting/) [![Python version](https://img.shields.io/badge/python-3.10-blue)](https://docs.python.org/3.10/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

<!-- badges: end -->

**A demonstration repository showing how data science projects evolve from initial exploration to production-ready MLOps systems.**

This repository uses the Kaggle Rossmann Store Sales forecasting problem to illustrate three distinct stages of data science maturity. By exploring different branches, you can see how the same problem is approached at different levels of sophistication—from exploratory notebooks to a full MLOps deployment with monitoring, CI/CD, and automated retraining.

______________________________________________________________________

## The Progression

```mermaid
graph TB
    subgraph "Stage 1: Initial Exploration"
        A1[Jupyter Notebooks Only] --> A2[Ad-hoc Analysis]
        A2 --> A3[Quick Insights]
        A3 --> A4[Manual Experimentation]
        style A1 fill:#fff4e1
    end

    subgraph "Stage 2: Research Mature"
        B1[Modular Code Structure] --> B2[Reusable Functions]
        B2 --> B3[Automated Pipelines]
        B3 --> B4[Comprehensive Testing]
        B4 --> B5[Quality Standards]
        style B1 fill:#e1f5ff
    end

    subgraph "Stage 3: Production Ready"
        C1[MLOps Infrastructure] --> C2[Model Registry & Versioning]
        C2 --> C3[Automated Deployment]
        C3 --> C4[Monitoring & Drift Detection]
        C4 --> C5[CI/CD Pipelines]
        C5 --> C6[API & Dashboard]
        style C1 fill:#e8f5e9
    end

    A4 --> B1
    B5 --> C1

    style A4 fill:#ffe1e1
    style B5 fill:#ffe1e1
    style C6 fill:#d4edda
```

______________________________________________________________________

## Three Branches, Three Maturity Levels

### [Stage 1: Initial Exploration](https://github.com/bradleyboehmke/rossmann-forecasting/tree/1-initial-exploration)

**Branch:** `1-initial-exploration`

**Characteristics:**

- Notebook-centric workflow
- Exploratory data analysis
- Rapid prototyping and experimentation
- Manual model training and evaluation

**Best for:** Early-stage research, proof-of-concept work, learning

### [Stage 2: Research Mature](https://github.com/bradleyboehmke/rossmann-forecasting/tree/2-research-mature)

**Branch:** `2-research-mature`

**Characteristics:**

- Modular code structure (`src/` directory)
- Reusable data processing and feature engineering
- Automated pipelines and testing
- Code quality standards and documentation

**Best for:** Collaborative projects, reproducible research, maintained codebases

### [Stage 3: Production Ready](https://github.com/bradleyboehmke/rossmann-forecasting/tree/main) ← **You are here**

**Branch:** `main`

**Characteristics:**

- Full MLOps infrastructure with MLflow
- Model registry and automated validation
- FastAPI prediction service + Streamlit dashboard
- Prediction logging and data drift detection
- GitHub Actions CI/CD pipelines

**Best for:** Production systems, real-world deployment, business applications

______________________________________________________________________

## What You'll Learn

If you explore each branch, you'll see how data science practices evolve. Key transitions include:

### From Stage 1 → Stage 2

**Key Transitions:**

- Moving from notebooks to modular code
- Creating reusable data pipelines
- Implementing automated testing
- Establishing code quality standards

**Skills:** Software engineering best practices, test-driven development, documentation

### From Stage 2 → Stage 3

**Key Transitions:**

- Experiment tracking with MLflow
- Model registry and lifecycle management
- API deployment with FastAPI
- Monitoring and drift detection
- CI/CD automation

**Skills:** MLOps infrastructure, production deployment, API design, DevOps practices

______________________________________________________________________

## 📚 Complete Documentation

For detailed implementation guides, API reference, and architecture diagrams, visit:

### **[https://bradleyboehmke.github.io/rossmann-forecasting/](https://bradleyboehmke.github.io/rossmann-forecasting/)**

**Documentation includes:**

- Getting Started guides
- DataOps workflows (data processing, validation, versioning)
- ModelOps workflows (MLflow tracking, model registry, lifecycle management)
- Deployment guides (FastAPI, Streamlit, Docker)
- Monitoring and drift detection
- Testing and CI/CD
- Complete API reference

______________________________________________________________________

## 🔑 Feature Comparison

| Feature                | Stage 1 | Stage 2 | Stage 3 |
| ---------------------- | ------- | ------- | ------- |
| Jupyter Notebooks      | ✅      | ✅      | ✅      |
| Modular Code (`src/`)  | ❌      | ✅      | ✅      |
| Automated Testing      | ❌      | ✅      | ✅      |
| Code Quality Standards | ❌      | ✅      | ✅      |
| Documentation (MkDocs) | ❌      | ❌      | ✅      |
| MLflow Tracking        | ❌      | ❌      | ✅      |
| Model Registry         | ❌      | ❌      | ✅      |
| FastAPI Deployment     | ❌      | ❌      | ✅      |
| Streamlit Dashboard    | ❌      | ❌      | ✅      |
| Drift Detection        | ❌      | ❌      | ✅      |
| CI/CD Pipelines        | ❌      | ❌      | ✅      |

______________________________________________________________________

## 🛠️ Technology Stack

- **Core ML:** pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna
- **MLOps:** MLflow, FastAPI, Streamlit, Evidently AI
- **Quality:** pytest, Great Expectations, pre-commit (black, ruff, mdformat)
- **DevOps:** GitHub Actions, Docker, DVC, uv

______________________________________________________________________

## 🤝 Contributing

This is a demonstration repository showing data science maturity progression. Feel free to:

- Fork and adapt for your own projects
- Submit issues for bugs or suggestions
- Use as a template for your ML systems

______________________________________________________________________

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

______________________________________________________________________

## 📞 Contact

**Author:** Bradley Boehmke

**Repository:** [github.com/bradleyboehmke/rossmann-forecasting](https://github.com/bradleyboehmke/rossmann-forecasting)

**Documentation:** [bradleyboehmke.github.io/rossmann-forecasting](https://bradleyboehmke.github.io/rossmann-forecasting/)

______________________________________________________________________

**Ready to explore?** Start with [Stage 1: Initial Exploration](https://github.com/bradleyboehmke/rossmann-forecasting/tree/1-initial-exploration) and work your way up to production! 🚀
