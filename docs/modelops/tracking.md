# Experiment Tracking

## MLflow Tracking

MLflow automatically tracks all experiments during model development, including:

- **Hyperparameters**: Learning rates, tree depths, regularization parameters
- **Metrics**: RMSPE, RMSE, MAE, MAPE per fold and overall CV scores
- **Artifacts**: Trained models, configuration files, prediction outputs
- **Data Versions**: DVC commit hashes linking experiments to data
- **Training Time**: Per-model and total training duration
- **System Info**: Python version, library versions, hardware specs

## Launching MLflow UI

```bash
# Start MLflow server
mlflow ui

# Custom port
mlflow ui --port 8080

# Open browser to http://localhost:5000
```

## Tracking in Notebooks

All experiments in notebooks are automatically tracked when using MLflow:

```python
import mlflow

# Start a run
with mlflow.start_run(run_name="experiment_name"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("max_depth", 6)

    # Train model
    model = train_model(params)

    # Log metrics
    mlflow.log_metric("cv_rmspe", 0.095)
    mlflow.log_metric("cv_rmse", 1234.56)

    # Log artifacts
    mlflow.log_artifact("outputs/predictions.csv")
```

## Experiment Organization

```
mlflow-experiments/
├── rossmann-forecasting/
│   ├── baseline-models/
│   ├── hyperparameter-tuning/
│   │   ├── lightgbm/
│   │   ├── xgboost/
│   │   └── catboost/
│   ├── ensemble-training/
│   └── production-retraining/
```

## Filtering and Comparing Runs

In the MLflow UI:

1. **Filter by metrics**: `metrics.cv_rmspe < 0.10`
1. **Sort by any column**: Click column headers
1. **Compare runs**: Select multiple runs and click "Compare"
1. **Download artifacts**: Click run → Artifacts tab

## Best Practices

1. Use descriptive run names
1. Tag runs with experiment type
1. Log DVC data versions
1. Document failures and debugging info
1. Archive failed experiments
