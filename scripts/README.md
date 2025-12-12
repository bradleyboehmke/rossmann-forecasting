# Scripts

Automation scripts for training, evaluation, and deployment workflows.

## Available Scripts

### Training Pipeline

- `train_production_model.py` - Train final production model
- `evaluate_model.py` - Evaluate model performance on holdout
- `register_model.py` - Register model to MLflow Model Registry

### Data Pipeline

- `prepare_data.py` - Run full data preparation pipeline
- `validate_data.py` - Run data validation checks

### Deployment

- `deploy_model.py` - Deploy model to production
- `promote_model.py` - Promote model from Staging to Production

### Utilities

- `generate_reports.py` - Generate monitoring and performance reports
- `cleanup.py` - Clean old artifacts and cache

## Usage

All scripts should be run from the project root:

```bash
# Example: Train production model
python scripts/train_production_model.py

# Example: Register best model
python scripts/register_model.py --run-id <mlflow_run_id>

# Example: Promote to production
python scripts/promote_model.py --version 3
```

## CI/CD Integration

These scripts are used in GitHub Actions workflows:

- `.github/workflows/test.yml` - Testing pipeline
- `.github/workflows/train-model.yml` - Automated training
- `.github/workflows/deploy.yml` - Deployment pipeline
