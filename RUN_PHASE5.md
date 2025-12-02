# Quick Start: Running Phase 5

## Prerequisites

Ensure you've completed Phases 1-4:
- ✅ Phase 1: Data cleaning (`data/processed/train_clean.parquet`)
- ✅ Phase 2: Feature engineering (`data/processed/train_features.parquet`)
- ✅ Phase 3: Baseline models (metrics in `outputs/metrics/baseline/`)
- ✅ Phase 4: Advanced models (metrics in `outputs/metrics/advanced/`)

## Option 1: Run via Jupyter Notebook (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start Jupyter
cd notebooks
jupyter notebook

# Open: 05-final-eval-and-test-simulation.ipynb
# Click: Cell > Run All
```

**Expected runtime**: 5-10 minutes (depending on hardware)

## Option 2: Run via Python Script

```bash
# Activate virtual environment
source venv/bin/activate

# Run from project root
python run_phase5.py
```

## Option 3: Run via Python Module

```bash
# Activate virtual environment
source venv/bin/activate

# Run as module
python -m src.models.train_final
```

## What Happens During Execution

### Step 1: Data Loading
- Loads `train_features.parquet` (1M+ rows, 50 features)
- Creates 6-week holdout split (last 42 days)

### Step 2: Model Training
- Trains XGBoost on ~900K rows
- Trains CatBoost on ~900K rows
- Creates weighted ensemble (71% XGB + 29% CB)

**Training time**: 3-5 minutes total

### Step 3: Predictions
- Generates predictions for holdout set (~35K predictions)
- Filters open stores only

### Step 4: Evaluation
- Calculates RMSPE and other metrics
- Compares to target (0.09856)
- Analyzes error by store characteristics

### Step 5: Saving Artifacts
Creates outputs in:
- `models/final/`: Trained models + metadata
- `outputs/predictions/`: Holdout predictions CSV
- `outputs/metrics/final/`: Performance metrics JSON
- `outputs/figures/`: Visualization plots

## Expected Output

```
============================================================
PHASE 5: FINAL MODEL TRAINING & EVALUATION
============================================================

Creating Train/Holdout Split
============================================================
Full dataset: 1,017,209 rows
Date range: 2013-01-01 to 2015-07-31

Train set:
  Rows: 977,846
  Date range: 2013-01-01 to 2015-06-19

Holdout set:
  Rows: 39,363
  Date range: 2015-06-20 to 2015-07-31
  Duration: 42 days (6 weeks)
============================================================

Training Final Ensemble Model
============================================================
Ensemble weights: {'xgboost': 0.71, 'catboost': 0.29}
Training set size: 818,660
Features: 46
Categorical features: 4

Training XGBoost component...
XGBoost training complete

Training CatBoost component...
CatBoost training complete
============================================================

Evaluating Final Model Performance
============================================================
Holdout Performance:
  RMSPE: 0.127XXX
  RMSE:  9XX.XX
  MAE:   6XX.XX
  MAPE:  XX.XX%

Target Analysis:
  Target RMSPE: 0.098560
  Current RMSPE: 0.127XXX
  Gap: 0.02XXXX (+XX.XX%)

📊 Additional tuning needed to reach target
============================================================

============================================================
PHASE 5 COMPLETE!
============================================================
Final RMSPE: 0.127XXX
Gap to target: +XX.XX%
============================================================
```

## Outputs Created

### 1. Models (`models/final/`)
```
models/final/
├── xgboost_final.json          # XGBoost model
├── catboost_final.cbm           # CatBoost model
└── ensemble_metadata.json       # Weights + features + metrics
```

### 2. Predictions (`outputs/predictions/`)
```csv
Store,Date,Sales_Actual,Sales_Predicted
1,2015-06-20,5263,5124.32
1,2015-06-21,6064,6201.45
...
```

### 3. Metrics (`outputs/metrics/final/`)
```json
{
  "rmspe": 0.127XXX,
  "rmse": 9XX.XX,
  "mae": 6XX.XX,
  "mape": XX.XX,
  "target_rmspe": 0.098560,
  "gap": 0.02XXXX,
  "gap_pct": XX.XX,
  "target_achieved": false,
  "n_predictions": 35XXX
}
```

### 4. Figures (`outputs/figures/`)
- `15_final_predictions_scatter.png` - Predicted vs Actual scatter plot
- `16_final_predictions_timeseries.png` - Time series for sample stores
- `17_error_by_storetype.png` - Error distribution by store type

## Troubleshooting

### Issue: Import Error
```
ImportError: Unable to import required dependencies
```

**Solution**: Ensure you're in the project root and virtual environment is activated
```bash
cd /path/to/rossmann-forecasting
source venv/bin/activate
```

### Issue: Missing Data Files
```
FileNotFoundError: data/processed/train_features.parquet
```

**Solution**: Run previous phases first
```bash
# Run Phase 2 to create features
python -m src.features.build_features
```

### Issue: Out of Memory
```
MemoryError or Killed
```

**Solution**: Reduce data or use chunking (for systems with <8GB RAM)

## Verifying Success

Check that these files exist and are recent:
```bash
ls -lh models/final/
ls -lh outputs/predictions/final_holdout_predictions.csv
ls -lh outputs/metrics/final/final_metrics.json
ls -lh outputs/figures/15_*.png
```

## Next Steps

After Phase 5 completes:

1. **Review Performance**:
   - Check `outputs/metrics/final/final_metrics.json`
   - View visualizations in `outputs/figures/`

2. **Analyze Results**:
   - Compare holdout RMSPE to CV results
   - Identify stores with high errors
   - Review prediction patterns

3. **If Gap to Target Exists**:
   - Run hyperparameter optimization (Optuna results in `outputs/tuning/`)
   - Add advanced features
   - Try stacked ensemble

4. **Production Deployment**:
   - Load model: `xgboost.Booster.load_model('models/final/xgboost_final.json')`
   - Apply same preprocessing pipeline
   - Generate predictions for new data

## Performance Benchmarks

Expected performance on different hardware:

| Hardware | Training Time | Total Runtime |
|----------|---------------|---------------|
| MacBook Pro M1 | 3-4 min | 5-6 min |
| Modern laptop (Intel i7) | 4-6 min | 7-10 min |
| Cloud (2 vCPU) | 8-12 min | 12-15 min |

## Questions?

See `PHASE5_SUMMARY.md` for detailed implementation notes and design decisions.
