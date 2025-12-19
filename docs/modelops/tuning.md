# Hyperparameter Tuning

## Optuna Integration

Hyperparameter tuning is performed using Optuna with MLflow integration in `notebooks/04-advanced-models-and-ensembles.ipynb`.

**Training Time Considerations:**

Tuning models on a local computer can be **time-intensive**—each model may take over an hour with 100+ trials using 5-fold cross-validation. To reduce training time:

- **Reduce trials**: Use `n_trials=20` instead of `n_trials=100` for faster exploration
- **Narrow search space**: Limit hyperparameter ranges based on domain knowledge
- **Use fewer folds**: Reduce cross-validation from 5 folds to 3 folds
- **Parallel execution**: Run tuning on multiple cores with `n_jobs` parameter
- **Use pre-tuned parameters**: Load `config/best_hyperparameters.json` to skip tuning entirely

For full experimentation, consider using cloud compute resources or running overnight.

## Tuning Workflow

1. **Define objective function**: Function to minimize (CV RMSPE)
1. **Create Optuna study**: Specify optimization direction
1. **Run optimization**: Execute trials with MLflow callback
1. **Save best params**: Store in `config/best_hyperparameters.json`

## Example Code

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

# Create study
study = optuna.create_study(
    study_name="rossmann-lightgbm",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

# MLflow callback for tracking
mlflc = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="cv_rmspe"
)

# Optimize
study.optimize(
    objective_lightgbm,
    n_trials=100,
    callbacks=[mlflc]
)

# Get best parameters
best_params = study.best_params
print(f"Best RMSPE: {study.best_value:.6f}")
```

## Search Spaces

### LightGBM

```python
def objective_lightgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    # Train and evaluate
    cv_score = train_and_evaluate(params)
    return cv_score
```

### XGBoost

```python
def objective_xgboost(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }
    cv_score = train_and_evaluate(params)
    return cv_score
```

### CatBoost

```python
def objective_catboost(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    cv_score = train_and_evaluate(params)
    return cv_score
```

## MLflow Tracking

Every Optuna trial is automatically logged to MLflow:

- **Run per trial**: Each trial creates an MLflow run
- **Parameters logged**: All hyperparameters tested
- **Metrics logged**: Objective value (CV RMSPE)
- **Trial state**: COMPLETE, PRUNED, or FAIL
- **Intermediate values**: For pruning visualization

## Analyzing Results

### Parallel Coordinate Plot

View in MLflow UI after optimization completes.

### Hyperparameter Importance

```python
import optuna
from optuna.visualization import plot_param_importances

# Load study
study = optuna.load_study(study_name="rossmann-lightgbm")

# Plot importance
fig = plot_param_importances(study)
fig.show()
```

### Optimization History

```python
from optuna.visualization import plot_optimization_history

fig = plot_optimization_history(study)
fig.show()
```

## Best Parameters Storage

Best parameters are saved to `config/best_hyperparameters.json`:

```json
{
  "metadata": {
    "best_model": "xgboost",
    "best_rmspe": 0.098765
  },
  "lightgbm": {
    "hyperparameters": {...},
    "cv_rmspe": 0.099123
  },
  "xgboost": {
    "hyperparameters": {...},
    "cv_rmspe": 0.098765
  },
  "catboost": {
    "hyperparameters": {...},
    "cv_rmspe": 0.099456
  }
}
```

## Production Usage

The production training pipeline automatically loads best parameters:

```python
from models.train_ensemble import load_best_hyperparameters

# Load from config
best_params = load_best_hyperparameters("config/best_hyperparameters.json")

# Train with best params
model = train_model(best_params)
```
