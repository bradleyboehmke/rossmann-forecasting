# Quick Start

Get up and running with the Rossmann forecasting project in 5 minutes.

!!! tip "New to MLOps Projects?" If you're unfamiliar with Python project setup, check the [Detailed Setup Guide](detailed-setup.md) for step-by-step instructions.

## Installation (3 minutes)

```bash
# 1. Install uv package manager
pip install uv

# 2. Clone repository
git clone https://github.com/bradleyboehmke/rossmann-forecasting.git
cd rossmann-forecasting

# 3. Create and activate virtual environment
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 4. Install dependencies
uv pip install -e .

# 5. (Optional) Set up pre-commit hooks for code quality
uv pip install -e ".[dev]"
pre-commit install
```

## Verify Setup (1 minute)

```bash
# Check data files (already included!)
ls -lh data/raw/train.csv data/raw/store.csv

# Run quick test
python -c "import pandas, lightgbm; print('✓ Setup successful!')"
```

## Try a Workflow (1 minute)

Choose one to explore:

=== "DataOps Pipeline" `bash     # Run complete data processing pipeline     bash scripts/dataops_workflow.sh     `

=== "Jupyter Notebooks" `bash     # Launch interactive notebooks     jupyter lab     # Open: notebooks/01-eda-and-cleaning.ipynb     `

=== "Model Training" `bash     # Train baseline models     python -m src.models.train_baselines     `

## Next Steps

**Explore MLOps workflows:**

1. 📊 [DataOps Workflow](../dataops/overview.md) - Data validation, processing, versioning
1. 🤖 [Model Training](../modelops/training.md) - Experiment tracking with MLflow
1. 🚀 [Deployment](../deployment/overview.md) - API and dashboard deployment
1. 📈 [Monitoring](../monitoring/overview.md) - Data drift and performance tracking

**Need help?** See the [Detailed Setup Guide](detailed-setup.md) for troubleshooting and advanced configuration.
