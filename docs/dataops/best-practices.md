# DataOps Best Practices

Production-grade recommendations for building reliable, maintainable data pipelines.

______________________________________________________________________

## 1. Validate Early, Validate Often

**Principle:** Catch data quality issues as early as possible in the pipeline.

### Implementation

- ✅ **Raw data validation** - Catches 80% of issues before processing
- ✅ **Post-processing validation** - Catches remaining 20% of bugs
- ✅ **Feature validation** - Ensures correctness of complex transformations

### Example

```python
# Bad: Only validate once at the end
def bad_pipeline():
    data = load_raw()
    processed = process(data)
    features = engineer_features(processed)
    validate(features)  # Too late to catch data source issues!

# Good: Validate at each stage
def good_pipeline():
    data = load_raw()
    validate_raw(data)  # ✅ Fail fast if source data is bad

    processed = process(data)
    validate_processed(processed)  # ✅ Catch processing bugs

    features = engineer_features(processed)
    validate_features(features)  # ✅ Verify feature correctness
```

______________________________________________________________________

## 2. Version Everything

**Principle:** Track all data artifacts with the same rigor as code.

### What to Version

| Asset              | Tool         | Why                                 |
| ------------------ | ------------ | ----------------------------------- |
| **Raw data**       | DVC          | Reproduce exact input for any model |
| **Processed data** | DVC          | Debug processing bugs               |
| **Features**       | DVC          | Track feature evolution             |
| **Code**           | Git          | Standard version control            |
| **Models**         | MLflow + DVC | Link model to exact data version    |
| **Expectations**   | Git          | Document data quality assumptions   |

### Example

```bash
# Version data with DVC
dvc add data/processed/train_features.parquet

# Commit metadata to Git
git add data/processed/train_features.parquet.dvc .gitignore
git commit -m "data: add holiday interaction features

- Created 5 new holiday × promo interaction features
- Validation: PASSED
- Date range: 2013-01-01 to 2015-08-15
- Rows: 1,017,209

Links to:
- Code: commit e4f5g6h
- Model: run_id abc123 in MLflow"
```

**Benefits:**

- Return to exact data state for any model version
- Debug issues by comparing data versions
- Roll back bad changes quickly

______________________________________________________________________

## 3. Make Pipelines Idempotent

**Principle:** Running the pipeline twice with the same input should produce identical output.

### What to Avoid

```python
# ❌ Bad: Non-deterministic operations
df["random_feature"] = np.random.rand(len(df))  # Different every run
df["timestamp"] = datetime.now()  # Time-dependent

# ❌ Bad: Undocumented side effects
def process_data(df):
    global PROCESSED_COUNT
    PROCESSED_COUNT += len(df)  # Mutates global state
```

### What to Do

```python
# ✅ Good: Deterministic with fixed seed
np.random.seed(42)
df["random_feature"] = np.random.rand(len(df))

# ✅ Good: Use input data timestamp
df["processing_date"] = df["Date"].max()  # Derived from data

# ✅ Good: Pure functions
def process_data(df):
    """Process dataframe without side effects."""
    return df.copy()  # No mutation, no global state
```

______________________________________________________________________

## 4. Design for Failure

**Principle:** Expect validation to fail sometimes—it's a feature, not a bug.

### Failure Handling Strategy

```python
def run_pipeline():
    try:
        validate_raw_data()
    except ValidationError as e:
        # Log failure
        logger.error(f"Raw data validation failed: {e}")

        # Alert humans
        send_alert(
            "Data validation failed. Manual review required.",
            details=str(e)
        )

        # Don't proceed
        sys.exit(1)  # Fail the pipeline

    # Only reach here if validation passed
    process_data()
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def fetch_new_data():
    """Retry transient network failures."""
    response = requests.get(DATA_URL)
    response.raise_for_status()
    return response.json()
```

### Alert Thresholds

- **Critical**: Schema changes, missing required columns
- **Warning**: Statistical anomalies, outliers
- **Info**: Successful runs, metrics

______________________________________________________________________

## 5. Monitor Data Quality Over Time

**Principle:** Track data quality metrics to detect drift and degradation.

### Metrics to Track

| Metric                    | What It Detects                |
| ------------------------- | ------------------------------ |
| **Validation pass rate**  | Increasing failure frequency   |
| **Row count**             | Unexpected data volume changes |
| **Null percentage**       | Data quality degradation       |
| **Feature distributions** | Statistical drift              |
| **Processing time**       | Performance issues             |

### Example Dashboard

```python
# Track validation metrics
metrics = {
    "timestamp": datetime.now(),
    "validation_pass_rate": 13/13,
    "row_count": 1_017_209,
    "null_percentage": df.isnull().sum().sum() / df.size,
    "mean_sales": df["Sales"].mean(),
    "processing_time_seconds": 42.3
}

# Log to monitoring system
logger.info("DataOps metrics", extra=metrics)

# Alert on anomalies
if metrics["null_percentage"] > 0.05:
    send_alert("High null percentage detected")
```

______________________________________________________________________

## 6. Document Data Decisions

**Principle:** Every data transformation decision should be documented.

### Where to Document

| Decision Type               | Documentation Location            |
| --------------------------- | --------------------------------- |
| **Missing value strategy**  | Inline code comments + docstrings |
| **Feature transformations** | Function docstrings               |
| **Business logic**          | CLAUDE.md or README               |
| **Validation rules**        | Great Expectations metadata       |

### Example

```python
def fill_competition_distance(df):
    """Fill missing CompetitionDistance with median.

    Why median?
    - Mean is sensitive to outliers (some stores very far from competitors)
    - Median represents "typical" competitive distance
    - Zero would imply no competition (incorrect assumption)

    Historical context:
    - 2,642 stores (23%) have missing CompetitionDistance
    - Likely means competition data not available, not that there's no competition
    - Decision made: 2024-01-15, reviewed with business stakeholders
    """
    median_distance = df["CompetitionDistance"].median()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_distance)
    return df
```

______________________________________________________________________

## 7. Separate Standard vs. Experimental Features

**Principle:** Create clear boundaries between proven and experimental features.

### DataOps (Standard Features)

- Automated, tested, always created
- Required for all models
- Thoroughly validated
- Examples: lags, rolling stats, calendar features

### ModelOps (Experimental Features)

- Model-specific, optional
- Experimental, may not improve performance
- Examples: complex interactions, model-specific encodings

### Benefits

- **Faster experimentation** - Modelers don't wait for data engineers
- **Better reliability** - Standard features are battle-tested
- **Clear ownership** - Data engineers own DataOps, ML engineers own ModelOps

______________________________________________________________________

## 8. Test Your Data Pipeline

**Principle:** Data pipelines deserve the same testing rigor as application code.

### Test Types

```python
# test_data_processing.py
def test_merge_preserves_all_sales_records():
    """Ensure merge doesn't drop any sales records."""
    raw_train = pd.read_csv("data/raw/train.csv")
    processed = process_data()

    assert len(processed) == len(raw_train), \
        "Merge dropped sales records!"

def test_no_future_leakage_in_lags():
    """Ensure lag features don't use future information."""
    df = build_features()

    # For each store, lag_1 should equal previous day's sales
    for store_id in df["Store"].unique():
        store_df = df[df["Store"] == store_id].sort_values("Date")
        store_df["expected_lag_1"] = store_df["Sales"].shift(1)

        # Allow NaN for first row
        assert store_df["lag_sales_1"].equals(store_df["expected_lag_1"]), \
            f"Lag feature has data leakage for store {store_id}"

def test_validation_catches_bad_data():
    """Ensure validation fails on known-bad data."""
    bad_data = pd.DataFrame({
        "Sales": [-10, 5, 10],  # Negative sales should fail
        "Store": [1, 2, 3]
    })

    with pytest.raises(ValidationError):
        validate_raw(bad_data)
```

______________________________________________________________________

## 9. Optimize for Maintainability

**Principle:** Code will be read more often than written.

### Good Practices

```python
# ✅ Good: Clear, self-documenting
def add_calendar_features(df):
    """Add year, month, quarter, day-of-week features."""
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df

# ❌ Bad: Cryptic, hard to maintain
def acf(d):
    d["y"], d["m"], d["q"], d["dw"] = \
        d["dt"].dt.y, d["dt"].dt.m, d["dt"].dt.q, d["dt"].dt.dw
    return d
```

### Configuration Over Code

```yaml
# config/features.yaml (Good!)
calendar_features:
  - year
  - month
  - quarter
  - day_of_week

lag_features:
  lags: [1, 7, 14, 28]
  groupby: Store

rolling_features:
  windows: [7, 14, 28, 60]
  functions: [mean, std]
  groupby: Store
```

______________________________________________________________________

## 10. Plan for Scale

**Principle:** Design pipelines that can handle 10x current data volume.

### Scalability Strategies

| Technique                  | When to Use                               |
| -------------------------- | ----------------------------------------- |
| **Sampling**               | Development, validation on large datasets |
| **Partitioning**           | Processing data in chunks                 |
| **Columnar formats**       | Parquet for efficient storage/retrieval   |
| **Lazy evaluation**        | Dask/Polars for larger-than-memory data   |
| **Distributed processing** | Spark for multi-TB datasets               |

### Example: Chunked Processing

```python
def process_large_dataset(input_path, output_path, chunk_size=100_000):
    """Process large CSV in chunks to avoid memory issues."""
    chunks = []

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        processed_chunk = process_data(chunk)
        chunks.append(processed_chunk)

    # Concatenate and save
    result = pd.concat(chunks, ignore_index=True)
    result.to_parquet(output_path)
```

______________________________________________________________________

## Troubleshooting Common Issues

### Issue: Validation fails after data update

**Diagnosis:**

```bash
# Review detailed validation results
cat great_expectations/uncommitted/validations/latest.json
```

**Solutions:**

1. **Fix data at source** (preferred)
1. **Update expectations** (if assumptions changed)
1. **Add data cleaning** (if pattern is expected)

### Issue: Feature engineering creates NaN values

**Cause:** Lag features create NaN for first N rows per store

**Solution:**

This is expected behavior. Handle in modeling:

- Drop rows with NaN lags, OR
- Impute with 0 or forward-fill, OR
- Use models that handle missing values (LightGBM, XGBoost)

### Issue: DVC push fails

**Diagnosis:**

```bash
# Check remotes
dvc remote list
```

**Solution:**

```bash
# If empty, add remote
dvc remote add -d myremote s3://bucket/path

# Configure credentials (AWS example)
dvc remote modify myremote access_key_id <KEY>
dvc remote modify myremote secret_access_key <SECRET>
```

______________________________________________________________________

## Summary: Production DataOps Checklist

Use this checklist to ensure your DataOps pipeline is production-ready:

**Validation:**

- [ ] Raw data validation in place
- [ ] Post-processing validation configured
- [ ] Feature validation implemented
- [ ] Validation failures stop the pipeline

**Versioning:**

- [ ] All data tracked with DVC
- [ ] DVC metadata committed to Git
- [ ] Remote storage configured for collaboration

**Automation:**

- [ ] Bash script or DVC pipeline for end-to-end execution
- [ ] Integration with orchestration tool (Airflow/GitHub Actions)
- [ ] Automatic model retraining trigger

**Monitoring:**

- [ ] Data quality metrics tracked
- [ ] Alerts configured for failures
- [ ] Dashboard for pipeline health

**Testing:**

- [ ] Unit tests for data transformations
- [ ] Integration tests for full pipeline
- [ ] Tests run in CI/CD

**Documentation:**

- [ ] README explains pipeline purpose
- [ ] Inline comments explain data decisions
- [ ] Expectations documented in Great Expectations

______________________________________________________________________

## Next Steps

- **[Individual Steps](steps.md)** - Review detailed pipeline walkthrough
- **[Validation Guide](validation.md)** - Customize Great Expectations
- **[Automation](automation.md)** - Set up orchestration
- **[Real-World Scenarios](scenarios.md)** - Learn from practical examples
- **[Model Training](../modelops/training.md)** - Move to ModelOps phase
