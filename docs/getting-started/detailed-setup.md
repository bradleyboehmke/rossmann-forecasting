# Detailed Setup Guide

Complete installation and setup instructions for the Rossmann forecasting project. This guide provides step-by-step explanations for users new to MLOps projects.

!!! tip "Already Familiar with Python Projects?"
    If you're comfortable with Python development, try the [Quick Start Guide](quickstart.md) for a faster setup.

## Prerequisites

- **Python 3.10 or higher**
- **Git**
- **8GB+ RAM** (recommended for processing full dataset)
- **~5GB disk space** (for data and models)

## Installation

### Step 1: Install `uv` Package Manager

`uv` is a fast Python package manager that we use for dependency management.

=== "pip (Recommended)" `bash     pip install uv     `

=== "macOS/Linux (Standalone)" \`\`\`bash curl -LsSf https://astral.sh/uv/install.sh | sh

````
# After installation, restart your terminal or run:
source $HOME/.cargo/env
```
````

=== "Windows PowerShell (Standalone)" `powershell     irm https://astral.sh/uv/install.ps1 | iex     `

Verify installation:

```bash
uv --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/bradleyboehmke/rossmann-forecasting.git
cd rossmann-forecasting
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

You should see `(.venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
# Install production dependencies
uv pip install -e .

# Or install with dev dependencies (recommended)
uv pip install -e ".[dev]"

# Or install everything (dev + docs)
uv pip install -e ".[dev,docs]"
```

### Step 5: Verify Installation

```bash
# Run tests to verify setup
pytest tests/ -v

# Check Python packages
python -c "
import pandas
import lightgbm
import xgboost
import catboost
print('✓ All packages installed successfully')
"
```

## Data Setup

### Verify Data Files

**Good news!** The raw data files are already included in this repository, so you can start working immediately.

Verify data files are present:

```bash
ls -lh data/raw/train.csv data/raw/store.csv
```

Expected output:

```
-rw-r--r--  1 user  staff   17M  train.csv
-rw-r--r--  1 user  staff   45K  store.csv
```

### Download Fresh Data (Optional)

If you want to download the latest data directly from Kaggle, you can use these methods:

=== "Kaggle CLI" \`\`\`bash # Install Kaggle CLI pip install kaggle

````
# Configure API credentials (~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/settings

# Download competition data
kaggle competitions download -c rossmann-store-sales

# Extract to data/raw/
unzip rossmann-store-sales.zip -d data/raw/
```
````

=== "Manual Download" 1. Visit [Kaggle competition page](https://www.kaggle.com/c/rossmann-store-sales/data) 2. Download `train.csv` and `store.csv` 3. Place in `data/raw/` directory

!!! note "Data Source" This project uses data from the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales/data). The included data is for educational purposes.

## Exploring MLOps Workflows

Now that setup is complete, you can explore different MLOps workflows:

### 1. DataOps Workflow

Process and validate data using production-grade practices:

```bash
# Run complete DataOps pipeline
bash scripts/dataops_workflow.sh

# Or run individual steps:
python src/data/validate_data.py --stage raw
python -m src.data.make_dataset
python -m src.features.build_features
```

**Learn more**: [DataOps Workflow Guide](../dataops/overview.md)

### 2. Model Experimentation

Explore different modeling approaches:

```bash
# Option A: Jupyter Notebooks (recommended for learning)
jupyter lab

# Navigate to notebooks/ and run in order:
# - 01-eda-and-cleaning.ipynb
# - 02-feature-engineering.ipynb
# - 03-baseline-models.ipynb
# - 04-advanced-models-and-ensembles.ipynb
```

```bash
# Option B: Python Scripts (production approach)
python -m src.models.train_baselines
python -m src.models.train_advanced
```

**Learn more**: [Model Training Guide](../modelops/training.md)

### 3. Reproducible Pipelines with DVC

Run automated, cacheable pipelines:

```bash
# Run entire pipeline (only reruns changed stages)
dvc repro

# Run specific stage
dvc repro build_features

# Visualize pipeline
dvc dag
```

**Learn more**: [DVC Pipeline Guide](../dataops/overview.md#automation-options-orchestrating-the-full-pipeline)

### 4. Deployment (Optional)

Deploy models via API and dashboard:

```bash
# Start all services with Docker
docker-compose up --build

# Access services:
# - FastAPI: http://localhost:8000
# - Streamlit: http://localhost:8501
# - MLflow: http://localhost:5000
```

**Learn more**: Deployment guide coming soon

## Next Steps

After setup is complete:

1. ✅ [Quick Start Guide](quickstart.md) - Run your first workflow
1. ✅ [Project Structure](structure.md) - Understand the codebase
1. ✅ [DataOps Workflow](../dataops/overview.md) - Process and validate data
1. ✅ [Model Training](../modelops/training.md) - Train your first model

## Additional Resources

- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) - Documentation theme
- [uv documentation](https://github.com/astral-sh/uv) - Package manager
- [DVC documentation](https://dvc.org/doc) - Data version control
- [MLflow documentation](https://mlflow.org/docs/latest/index.html) - Experiment tracking
