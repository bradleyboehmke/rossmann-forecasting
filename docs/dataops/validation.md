# Data Validation with Great Expectations

Data validation is a critical component of production ML pipelines. This page explains how we use Great Expectations to ensure data quality throughout the DataOps workflow.

______________________________________________________________________

## What is Great Expectations?

[Great Expectations](https://greatexpectations.io/) is an open-source Python library for **data validation, documentation, and profiling**. It allows you to:

- Define **expectations** (assertions about your data)
- **Validate** data against those expectations
- Generate **data quality reports**
- **Fail pipelines** when data doesn't meet quality standards

**Think of it as unit tests for your data.**

### Why Great Expectations?

**Without data validation:**

- ❌ Bad data silently enters your pipeline
- ❌ Model performance degrades mysteriously
- ❌ Debugging requires manual data inspection
- ❌ Data quality assumptions are undocumented

**With Great Expectations:**

- ✅ Bad data is caught immediately
- ✅ Pipeline fails fast with clear error messages
- ✅ Data quality checks are automated
- ✅ Expectations serve as executable documentation

______________________________________________________________________

## How It Works

Great Expectations operates in three stages:

### 1. Define Expectations

Create a suite of expectations (validation rules) for your data:

```python
# Example: Raw train data expectations
validator.expect_table_columns_to_match_set(
    column_set=["Store", "DayOfWeek", "Date", "Sales", "Customers", ...]
)

validator.expect_column_values_to_not_be_null(
    column="Store"
)

validator.expect_column_values_to_be_between(
    column="Sales",
    min_value=0,
    max_value=1000000
)
```

### 2. Run Validation

Execute expectations against your data:

```bash
python src/data/validate_data.py --stage raw --fail-on-error
```

### 3. Review Results

Get immediate feedback on data quality:

```
============================================================
Validating raw training data: data/raw/train.csv
============================================================
✓ PASSED
Total expectations: 13
Successful: 13
Failed: 0
Success rate: 100.0%
```

______________________________________________________________________

## Our Validation Strategy

We validate data at **three critical stages** in the DataOps pipeline:

### Stage 1: Raw Data Validation

**Location:** `great_expectations/expectations/raw_train_suite.json`

**Purpose:** Catch issues in incoming data before any processing

**Key Expectations:**

| Expectation                           | What It Checks                              |
| ------------------------------------- | ------------------------------------------- |
| `expect_table_columns_to_match_set`   | All required columns are present            |
| `expect_column_values_to_not_be_null` | Critical fields have no missing values      |
| `expect_column_values_to_be_of_type`  | Data types are correct (int, float, string) |
| `expect_column_values_to_be_between`  | Numeric values are in valid ranges          |
| `expect_column_values_to_be_in_set`   | Categorical values match expected set       |

**Example Expectations:**

```json
{
  "expectation_type": "expect_column_values_to_be_between",
  "kwargs": {
    "column": "Sales",
    "min_value": 0,
    "max_value": 1000000
  }
},
{
  "expectation_type": "expect_column_values_to_be_in_set",
  "kwargs": {
    "column": "DayOfWeek",
    "value_set": [1, 2, 3, 4, 5, 6, 7]
  }
}
```

### Stage 2: Processed Data Validation

**Location:** `great_expectations/expectations/processed_data_suite.json`

**Purpose:** Verify processing didn't introduce errors

**Additional Expectations:**

| Expectation                            | What It Checks                            |
| -------------------------------------- | ----------------------------------------- |
| `expect_table_row_count_to_be_between` | No rows were lost during processing       |
| `expect_column_values_to_not_be_null`  | Processing didn't create unexpected nulls |
| `expect_column_mean_to_be_between`     | Statistical properties remain reasonable  |

### Stage 3: Feature Validation

**Location:** `great_expectations/expectations/features_suite.json`

**Purpose:** Ensure feature engineering produced valid features

**Feature-Specific Expectations:**

| Expectation                               | What It Checks                            |
| ----------------------------------------- | ----------------------------------------- |
| `expect_column_values_to_not_contain_inf` | No infinite values from log transforms    |
| `expect_column_values_to_not_be_null`     | Features don't have unexpected NaN values |
| `expect_column_values_to_be_between`      | Feature ranges are reasonable             |

______________________________________________________________________

## Common Expectation Types

Great Expectations provides dozens of built-in expectations. Here are the most commonly used:

### Schema Expectations

```python
# Column presence
validator.expect_table_columns_to_match_set(
    column_set=["Store", "Date", "Sales"]
)

# Data types
validator.expect_column_values_to_be_of_type(
    column="Date",
    type_="datetime64[ns]"
)

# Unique values
validator.expect_column_values_to_be_unique(
    column="Store"
)
```

### Value Range Expectations

```python
# Numeric ranges
validator.expect_column_values_to_be_between(
    column="Sales",
    min_value=0,
    max_value=1000000
)

# Categorical sets
validator.expect_column_values_to_be_in_set(
    column="StateHoliday",
    value_set=["0", "a", "b", "c"]
)

# No nulls
validator.expect_column_values_to_not_be_null(
    column="Store"
)
```

### Statistical Expectations

```python
# Mean value
validator.expect_column_mean_to_be_between(
    column="Sales",
    min_value=5000,
    max_value=7000
)

# Standard deviation
validator.expect_column_stdev_to_be_between(
    column="Sales",
    min_value=3000,
    max_value=5000
)

# Quantile ranges
validator.expect_column_quantile_values_to_be_between(
    column="Sales",
    quantile_ranges={
        "quantiles": [0.5, 0.95],
        "value_ranges": [[5000, 7000], [15000, 20000]]
    }
)
```

______________________________________________________________________

## Validation Workflow

### Running Validations

Our validation script (`src/data/validate_data.py`) provides a unified interface:

```bash
# Validate raw data
python src/data/validate_data.py --stage raw --fail-on-error

# Validate processed data
python src/data/validate_data.py --stage processed --fail-on-error

# Validate features
python src/data/validate_data.py --stage features --fail-on-error
```

**Key options:**

- `--stage` - Which validation suite to run (raw, processed, features)
- `--fail-on-error` - Exit with error code if validation fails (recommended for CI/CD)

### Understanding Validation Results

**Success output:**

```
============================================================
Validating processed data: data/processed/train_clean.parquet
============================================================
✓ PASSED
Total expectations: 15
Successful: 15
Failed: 0
Success rate: 100.0%
```

**Failure output:**

```
============================================================
Validating raw data: data/raw/train.csv
============================================================
❌ FAILED
Total expectations: 13
Successful: 11
Failed: 2
Success rate: 84.6%

Failed expectations:
  ❌ expect_column_values_to_be_between (Sales)
     - Found 3 values outside range [0, 1000000]
     - Example failures: [-10, -5, -2]

  ❌ expect_column_values_to_not_be_null (Store)
     - Found 5 null values in Store column
```

______________________________________________________________________

## Creating Custom Expectations

You can add new expectations to suit your project's needs:

### 1. Interactive Exploration

Use Great Expectations CLI to explore your data and generate expectations:

```bash
# Initialize Great Expectations
great_expectations init

# Create a checkpoint for validation
great_expectations checkpoint new my_checkpoint
```

### 2. Programmatic Creation

Add expectations directly in Python:

```python
import great_expectations as gx

# Load data context
context = gx.get_context()

# Create validator
validator = context.sources.pandas_default.read_csv(
    "data/raw/train.csv"
)

# Add custom expectations
validator.expect_column_unique_value_count_to_be_between(
    column="Store",
    min_value=1000,
    max_value=1200
)

# Save suite
validator.save_expectation_suite()
```

### 3. Modify Existing Suites

Edit expectation suite JSON files directly:

```json
{
  "expectation_type": "expect_column_values_to_match_regex",
  "kwargs": {
    "column": "Date",
    "regex": "^\\d{4}-\\d{2}-\\d{2}$"
  }
}
```

______________________________________________________________________

## Best Practices

### 1. Start Simple, Add Incrementally

Don't try to validate everything at once:

```python
# ✅ Good: Start with critical checks
validator.expect_column_values_to_not_be_null("Store")
validator.expect_column_values_to_not_be_null("Date")
validator.expect_column_values_to_not_be_null("Sales")

# ❌ Bad: 100 expectations on day 1
# (overwhelming, hard to maintain)
```

### 2. Focus on Business Logic

Validate assumptions that reflect business requirements:

```python
# ✅ Good: Business rule
validator.expect_column_values_to_be_between(
    column="Sales",
    min_value=0,  # Sales can't be negative
    max_value=1000000  # No single-day sales > $1M
)

# ❌ Less useful: Overly specific
validator.expect_column_mean_to_equal(
    column="Sales",
    value=5912.34567  # Too precise, likely to break
)
```

### 3. Use Tolerances for Statistical Checks

Allow for natural variation:

```python
# ✅ Good: Reasonable tolerance
validator.expect_column_mean_to_be_between(
    column="Sales",
    min_value=5000,
    max_value=7000,
    meta={"notes": "Historical mean ~6000, allow ±1000 variation"}
)
```

### 4. Document Expectations

Add metadata to explain why expectations exist:

```python
validator.expect_column_values_to_be_in_set(
    column="StateHoliday",
    value_set=["0", "a", "b", "c"],
    meta={
        "notes": "StateHoliday codes from Rossmann spec",
        "reference": "https://www.kaggle.com/c/rossmann-store-sales/data"
    }
)
```

### 5. Version Control Expectation Suites

Track changes to expectations like code:

```bash
# Commit expectation changes
git add great_expectations/expectations/
git commit -m "expectations: tighten sales range validation

- Reduce max_value from 2M to 1M
- Add outlier detection for sales > 100K
- Rationale: historical data shows 1M is realistic max"
```

______________________________________________________________________

## Troubleshooting

### Validation Fails After Data Update

**Symptom:** Validation that previously passed now fails

**Diagnosis:**

```bash
# Review detailed validation results
cat great_expectations/uncommitted/validations/latest.json
```

**Solutions:**

1. **Fix data at source** (preferred)

   - Contact data provider
   - Correct upstream ETL process

1. **Update expectations** (if assumptions changed)

   ```python
   # If sales range legitimately increased
   validator.expect_column_values_to_be_between(
       column="Sales",
       min_value=0,
       max_value=2000000  # Increased from 1M
   )
   ```

1. **Add data cleaning** (if pattern is expected)

   ```python
   # Add outlier capping in make_dataset.py
   df["Sales"] = df["Sales"].clip(upper=1000000)
   ```

### Performance Issues with Large Datasets

**Symptom:** Validation takes too long

**Solutions:**

1. **Sample data** for validation

   ```python
   # Validate on sample
   sample = df.sample(n=100000, random_state=42)
   validator = context.get_validator(sample)
   ```

1. **Reduce expectation complexity**

   - Remove statistical expectations on large columns
   - Focus on schema and critical value checks

1. **Parallelize validation**

   - Run different suites concurrently
   - Use Great Expectations' batch mode

______________________________________________________________________

## Next Steps

- **[Individual Steps](steps.md)** - See validation in context of full pipeline
- **[Automation](automation.md)** - Integrate validation into automated workflows
- **[Best Practices](best-practices.md)** - Production-grade validation strategies

**External Resources:**

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Expectation Gallery](https://greatexpectations.io/expectations/) - Full list of built-in expectations
- [Great Expectations GitHub](https://github.com/great-expectations/great_expectations)
