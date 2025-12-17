# Model Training

## Production Training Pipeline

The `src/models/train_ensemble.py` module provides automated training:

```bash
python src/models/train_ensemble.py
```

## Training Workflow

1. **Load Data**: Read processed features from parquet
1. **Load Hyperparameters**: Best params from Optuna tuning
1. **Train Models**: LightGBM, XGBoost, CatBoost
1. **Create Ensemble**: Weighted combination
1. **Register Model**: Add to MLflow Registry

## Training Code Example

```python
from models.train_ensemble import main

# Train with default settings
version = main(
    data_path="data/processed/train_features.parquet",
    config_path="config/best_hyperparameters.json",
    model_name="rossmann-ensemble"
)

print(f"Registered model version: {version}")
```

## Custom Ensemble Weights

```python
# Custom weights (must sum to 1.0)
version = main(
    ensemble_weights={
        "lightgbm": 0.25,
        "xgboost": 0.50,
        "catboost": 0.25
    }
)
```

## MLflow Logging

During training, the following are logged to MLflow:

- **Parameters**: All hyperparameters for each model
- **Metrics**: Training time per model
- **Artifacts**: Conda environment, model files
- **Tags**: Data version, training date
- **Model**: Registered ensemble model

## Next Steps

After training:

1. **Validate**: Run `python src/models/validate_model.py`
1. **Review**: Check MLflow UI for metrics
1. **Promote**: Move to Staging if validation passes
1. **Test**: Run inference with Staging model
