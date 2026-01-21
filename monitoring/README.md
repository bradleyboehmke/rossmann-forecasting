# Monitoring

Model and data monitoring infrastructure for production ML systems.

## Directory Structure

```
monitoring/
├── drift_reports/          # Data and prediction drift reports
├── performance_reports/    # Model performance tracking
└── README.md              # This file
```

## Monitoring Components

### 1. Data Drift Detection (`src/monitoring/drift_detection.py`)

Monitors changes in input data distribution:

- Feature distribution shifts
- Categorical value changes
- Missing data patterns
- Statistical property changes

### 2. Prediction Drift (`src/monitoring/drift_detection.py`)

Tracks prediction distribution over time:

- Prediction value ranges
- Prediction patterns by store/time
- Anomaly detection

### 3. Performance Monitoring (`src/monitoring/generate_reports.py`)

Tracks model performance metrics:

- RMSPE on recent predictions vs actuals
- Error analysis by store type, day of week
- Performance degradation alerts

### 4. Prediction Logging (`deployment/prediction_logger.py`)

Logs all production predictions:

- Input features
- Predictions
- Timestamps
- Model version

## Usage

### Generate Drift Report

```python
from src.monitoring.drift_detection import generate_drift_report

generate_drift_report(
    reference_data=train_data,
    current_data=production_data,
    output_path="monitoring/drift_reports/drift_report.html"
)
```

### Generate Performance Report

```python
from src.monitoring.generate_reports import generate_performance_report

generate_performance_report(
    predictions_path="outputs/predictions/latest.csv",
    actuals_path="data/actuals/latest.csv",
    output_path="monitoring/performance_reports/performance.html"
)
```

### Analyze Predictions

```python
from src.monitoring.analyze_predictions import analyze_prediction_logs

results = analyze_prediction_logs(
    log_path="logs/predictions.jsonl",
    window_days=7
)
```

## Automated Monitoring

Monitoring reports are generated automatically:

- **Hourly**: Prediction logging
- **Daily**: Drift detection
- **Weekly**: Performance reports
- **On-demand**: Via GitHub Actions workflow

## Alerting

Configure alerts for:

- Significant data drift (p-value \< 0.05)
- Performance degradation (RMSPE > threshold)
- Unusual prediction patterns
- Missing features or data quality issues

## Evidently Integration

This module uses Evidently AI for drift detection. Reports include:

- Data drift analysis
- Regression performance metrics
- Interactive HTML visualizations
- JSON results for programmatic access
