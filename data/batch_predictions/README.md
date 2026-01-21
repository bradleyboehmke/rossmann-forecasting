# Batch Prediction Data

This directory contains batch prediction datasets for testing the model deployment and monitoring system.

## batch_aug_dec_2015.csv

**Purpose**: Test dataset for batch predictions and drift detection monitoring

**Specifications**:

- **Date Range**: August 1 - December 31, 2015 (153 days)
- **Number of Stores**: 50 stores (randomly selected from store.csv)
- **Total Records**: 7,650 observations
- **File Size**: ~187 KB

**Features**:

- `Store`: Store ID (50 unique stores)
- `DayOfWeek`: Day of week (1=Monday, 7=Sunday)
- `Date`: Date in YYYY-MM-DD format
- `Open`: Whether store is open (0=closed, 1=open)
- `Promo`: Whether daily promotion is active (0=no, 1=yes)
- `StateHoliday`: State holiday indicator ('0', 'a', 'b', 'c')
- `SchoolHoliday`: School holiday indicator (0=no, 1=yes)

**Data Characteristics**:

- **Open Rate**: 82.4% (stores closed on Sundays and state holidays)
- **Overall Promo Rate**: 45.1% (includes drift)
- **School Holiday Rate**: 38.0% (includes drift)
- **State Holiday Rate**: 3.3%

## Intentional Data Drift

This dataset includes **intentional drift** in two features to test drift detection:

### 1. Promo Feature Drift (Strong Drift)

**Normal baseline** (Aug-Oct): ~35-36% promo rate **Drifted period** (Nov-Dec): ~57-62% promo rate

**Rationale**: Simulates increased promotional activity during holiday shopping season (Black Friday, Christmas)

| Month        | Promo Rate | Change from Baseline |
| ------------ | ---------- | -------------------- |
| August       | 36.1%      | Baseline             |
| September    | 35.2%      | Baseline             |
| October      | 35.9%      | Baseline             |
| **November** | **61.8%**  | **+26.6 pp**         |
| **December** | **56.8%**  | **+21.6 pp**         |

### 2. SchoolHoliday Feature Drift (Moderate Drift)

**Normal baseline**: Follows German school holiday calendar **Drifted period** (December): 20% noise injection

**Rationale**: Simulates data quality issues or labeling errors that might occur in production

| Month        | School Holiday Rate | Notes                           |
| ------------ | ------------------- | ------------------------------- |
| August       | 100.0%              | Summer holidays                 |
| September    | 0.0%                | School in session               |
| October      | 38.7%               | Fall break period               |
| November     | 0.0%                | School in session               |
| **December** | **48.6%**           | **Christmas break + 20% noise** |

## Usage

### 1. Upload via Streamlit Dashboard

Navigate to the **Batch Upload** page in the Streamlit dashboard and upload this CSV file.

```bash
# Launch dashboard
bash scripts/launch_dashboard.sh

# Navigate to: http://localhost:8501
# Go to: Predictions → Batch Upload tab
# Upload: data/batch_predictions/batch_aug_dec_2015.csv
```

### 2. Make Predictions via API

```bash
# Start API server
bash scripts/launch_api.sh

# Make batch predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @data/batch_predictions/batch_aug_dec_2015.csv
```

### 3. Monitor for Drift

After making predictions, check for data drift:

```bash
# Generate drift report (last 30 days)
python src/monitoring/generate_reports.py --days 30

# View report
open monitoring/drift_reports/latest.html
```

Or use the **Monitoring** page in the Streamlit dashboard.

## Expected Drift Detection Results

When analyzing this dataset, drift detection should identify:

1. **Strong drift in Promo feature**:

    - Drift detected: ✅ YES
    - Drift score: High (likely 0.8-1.0)
    - Affected period: November-December 2015

1. **Moderate drift in SchoolHoliday feature**:

    - Drift detected: ⚠️ POSSIBLY (depending on threshold)
    - Drift score: Moderate (likely 0.3-0.6)
    - Affected period: December 2015

1. **No drift in other features**:

    - Store, DayOfWeek, StateHoliday, Open: No significant drift
    - These follow natural calendar patterns

## Regenerating the Dataset

To regenerate this dataset with different parameters:

```bash
python scripts/generate_batch_prediction_data.py
```

Edit the script to customize:

- Date range
- Number of stores
- Drift magnitude
- Drift features
