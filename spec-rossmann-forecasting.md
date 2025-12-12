# **spec-rossmann-forecasting.md**

**Project:** Rossmann Sales Forecasting (Time Series ML) **Owner:** Brad Boehmke **Goal:** Develop a machine learning model that achieves **RMSPE \< 0.09856** using rigorous **time-series validation** and a broad exploration of modeling + feature engineering techniques.

______________________________________________________________________

# **1. Project Overview**

Rossmann operates over 3,000 stores across Europe and tasks managers with predicting six weeks of daily sales. Forecast accuracy varies widely due to differences in promotions, competition, holidays, seasonality, and locality. This project will develop a robust forecasting system using the Kaggle Rossmann dataset to produce accurate predictions while following strict time-series validation procedures.

The primary objective is to produce a model scoring in the **top 50** on the Kaggle leaderboard, corresponding to **RMSPE \< 0.09856**.

Models of all types—time-series, tree-based ML, deep learning, and ensembles—are in scope. Feature engineering, proper data cleaning, and leakage-free validation will be essential.

______________________________________________________________________

# **2. Success Criteria**

### **Primary Metric**

The evaluation metric is **Root Mean Square Percentage Error (RMSPE)**:

$$RMSPE=\\sqrt{\\frac{1}{n}\\sum\_{i=1}^{n}\\left(\\frac{y_i - \\hat{y}\_i}{y_i}\\right)^2}$$

- Observations where **Sales = 0** are **ignored** in scoring.
- Lower scores indicate better predictions.

### **Performance Target**

- Achieve **RMSPE \< 0.09856**, placing the solution in the **top 50**.

______________________________________________________________________

# **3. Data Assets**

The dataset consists of:

- **train.csv** — historical daily sales plus operational data
- **store.csv** — static store metadata including competition and promotion info

These files must be merged on **Store**.

______________________________________________________________________

# **4. Data Cleaning & Preparation**

### Required cleaning steps

- Convert `Date` to proper datetime.
- Join `train.csv` and `store.csv` on `Store`.
- Handle missing values in competition and promo metadata.
- Convert categorical fields to consistent dtypes.
- Remove or flag observations where stores are closed (`Open=0`).
- Ensure **no future data is used in features** (no leakage).

### Required feature types

- **Calendar**: year, month, week, day-of-week, seasonality flags
- **Promotion**: Promo, Promo2, Promo durations, Promo intervals
- **Competition**: competition distance, competition age
- **Lag Features**: store-level lags (1, 7, 14, 28)
- **Rolling Windows**: means, stds for relevant windows
- **Categoricals**: StoreType, Assortment, PromoInterval
- **Interactions**: holiday × promo, promo × season, competition × store type

______________________________________________________________________

# **5. Modeling Approaches**

The project includes experimentation with multiple modeling families:

### **Time Series Models**

- Prophet
- SARIMA / SARIMAX
- TBATS
- Deep learning (optional): LSTM, GRU, NBEATS, NHITS, TCN

### **Tree-Based Models**

- LightGBM
- XGBoost
- CatBoost
- Random Forest / ExtraTrees (baseline)

### **General ML Models**

- Linear/Elastic Net
- SVR
- kNN regression (low priority)

### **Ensembles (Required in Advanced Phase)**

- Weighted blending
- Stacking using meta-learning (e.g., LightGBM/Elastic Net)

The final deliverable may be a single best-performing model or an ensemble.

______________________________________________________________________

# **6. Time Series Validation Strategy**

Validation must mimic Kaggle conditions using **rolling-origin** or **expanding window** splits.

### Validation Structure Example

| Fold  | Train Window        | Validation Window             |
| ----- | ------------------- | ----------------------------- |
| 1     | Jan 2013 – Jun 2014 | Jul 2014                      |
| 2     | Jan 2013 – Jul 2014 | Aug 2014                      |
| 3     | Jan 2013 – Aug 2014 | Sep 2014                      |
| …     | …                   | …                             |
| Final | Full training       | Last 6 weeks (simulated test) |

- Each validation window = **6 weeks**
- Strictly no overlap from validation into future feature calculations

Final holdout should represent a realistic test period.

______________________________________________________________________

# **7. Experimentation Plan**

### **Phase 1: Baseline**

- Naïve models
- Prophet
- Linear regression
- Simple LightGBM baseline

### **Phase 2: Feature Engineering**

- Add lags, rolling statistics
- Add promo/competition/holiday features
- Add embeddings or advanced encodings

### **Phase 3: Model Zoo**

- LightGBM, XGBoost, CatBoost
- NBEATS/NHITS/TCN if feasible
- SARIMA variants for store clusters (optional)

### **Phase 4: Ensemble Construction**

- Weighted blends of top models
- Stacked ensemble using meta-learner
- Model comparison via CV RMSPE
- Select final modeling strategy

### **Phase 5: Final Training + Holdout Evaluation**

- Train chosen model(s) on full training window
- Forecast final 6-week holdout period
- Compute final RMSPE
- Evaluate against target \< 0.09856

______________________________________________________________________

# **8. Risks & Mitigation**

| Risk                | Mitigation                                                      |
| ------------------- | --------------------------------------------------------------- |
| Data leakage        | Strict time-based splits, lag-only features                     |
| Overfitting         | Cross-validation, early stopping, ensembling                    |
| Missing metadata    | Appropriate imputation strategies                               |
| Store heterogeneity | Store-level features, embeddings, grouping                      |
| Long model runtimes | Use efficient lightGBM/XGBoost, limit deep TS to optional stage |

______________________________________________________________________

# **9. Deliverables**

- Cleaned processed dataset
- Feature-engineered dataset
- Time-series CV framework
- Baseline model results
- Advanced model results
- Ensemble analysis
- Final model + artifacts
- Final holdout predictions
- Full documentation + final RMSPE

______________________________________________________________________

# **10. Definition of Done (DoD)**

The project is complete when:

- [ ] A model or ensemble achieves **RMSPE \< 0.09856**
- [ ] Validation uses strictly correct time-series methodology
- [ ] Feature engineering is documented and reproducible
- [ ] Baseline and advanced models are compared
- [ ] Final predictions + artifacts are saved
- [ ] Final notebook runs end-to-end

______________________________________________________________________

# **11. Repository Structure**

```text
rossmann-forecasting/
  README.md
  spec-rossmann-forecasting.md

  data/
    raw/
    external/
    processed/

  notebooks/
    01-eda-and-cleaning.ipynb
    02-feature-engineering.ipynb
    03-baseline-models.ipynb
    04-advanced-models-and-ensembles.ipynb
    05-final-eval-and-test-simulation.ipynb
    scratch/

  src/
    config.py

    data/
      make_dataset.py

    features/
      build_features.py

    models/
      train_baselines.py
      train_advanced.py
      ensembles.py

    evaluation/
      metrics.py
      cv.py
      reporting.py

    utils/
      io.py
      log.py

  models/
    baseline/
    final/

  outputs/
    figures/
    metrics/
    predictions/

  config/
    params.yaml

  env/
    requirements.txt
    environment.yml
```

______________________________________________________________________

# **12. Implementation Plan for Claude Code**

(Section content remains unchanged above this line.)

______________________________________________________________________

# **14. Full Data Dictionary**

The Rossmann dataset consists of two primary files: **train.csv** and **store.csv**. After merging them on `Store`, all fields become available for modeling. This section documents every field in the dataset.

______________________________________________________________________

## **14.1 train.csv Fields**

| Column            | Type                          | Description                                                                                                                          |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Store**         | int                           | Unique identifier for each store. Used to join with `store.csv`.                                                                     |
| **DayOfWeek**     | int (1–7)                     | Day of the week: Monday=1, …, Sunday=7.                                                                                              |
| **Date**          | string → datetime             | Observation date.                                                                                                                    |
| **Sales**         | float                         | **Target variable**. Total daily sales for the store. Days with `Sales = 0` are ignored in RMSPE scoring.                            |
| **Customers**     | float                         | Number of customers on the given day. Not available in the test set, so cannot be used directly for prediction. Useful only for EDA. |
| **Open**          | int (0/1)                     | Whether the store was open. `0 = closed`, `1 = open`.                                                                                |
| **Promo**         | int (0/1)                     | Short-term promotional activity on that day.                                                                                         |
| **StateHoliday**  | category (`0`, `a`, `b`, `c`) | Indicates a state holiday: `a` = public holiday, `b` = Easter, `c` = Christmas, `0` = none.                                          |
| **SchoolHoliday** | int (0/1)                     | Whether the store/day was affected by public school closure.                                                                         |

After merging `store.csv`, additional fields below also appear.

______________________________________________________________________

## **14.2 store.csv Fields**

| Column                        | Type                          | Description                                                                     |
| ----------------------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| **Store**                     | int                           | Unique store ID.                                                                |
| **StoreType**                 | category (`a`, `b`, `c`, `d`) | Differentiates four store formats. Influences assortment and traffic patterns.  |
| **Assortment**                | category (`a`, `b`, `c`)      | Level of product assortment: `a` = basic, `b` = extra, `c` = extended.          |
| **CompetitionDistance**       | float                         | Distance (meters) to nearest competitor. Missing values typically mean unknown. |
| **CompetitionOpenSinceMonth** | float                         | Month (1–12) when competition opened.                                           |
| **CompetitionOpenSinceYear**  | float                         | Year competition opened.                                                        |
| **Promo2**                    | int (0/1)                     | Participation in Promo2 (long-running recurring promo).                         |
| **Promo2SinceYear**           | float                         | Year Promo2 started for the store.                                              |
| **Promo2SinceWeek**           | float                         | Calendar week Promo2 began.                                                     |
| **PromoInterval**             | string (csv of months)        | Months when Promo2 restarts, e.g., `"Feb,May,Aug,Nov"`.                         |

______________________________________________________________________

## **14.3 Additional Modeling Notes**

### **Sales**

- Target variable.
- Consider log-transforming for stability.
- Remove or treat specially when stores are closed (`Open=0`).

### **Holidays**

- `StateHoliday` and `SchoolHoliday` can be combined to define holiday intensity.
- Holidays influence sales patterns strongly.

### **Competition Fields**

- `CompetitionOpenSince*` form a useful **competition age** feature.
- `CompetitionDistance` may require log-scaling.

### **Promo & Promo2 Fields**

- Promo2 cycles should be mapped to months.
- Use flags like `Promo2_active_this_month`.

### **Categorical Metadata**

- `StoreType` and `Assortment` are high-signal features.
- Can be one-hot encoded or used with model-native categorical handling.

### **Date Field**

Decompose into:

- Year, month, week-of-year, day-of-month
- IsMonthStart / IsMonthEnd
- Season indicator
- Weekday vs weekend

______________________________________________________________________

## **14.4 Summary Table (All Fields)**

| File  | Column                    | Description            |
| ----- | ------------------------- | ---------------------- |
| train | Store                     | Unique store ID        |
| train | DayOfWeek                 | Day of week (1–7)      |
| train | Date                      | Observation date       |
| train | Sales                     | Target variable        |
| train | Customers                 | Customer count         |
| train | Open                      | Store open flag        |
| train | Promo                     | Short-term daily promo |
| train | StateHoliday              | Holiday type           |
| train | SchoolHoliday             | School closure flag    |
| store | StoreType                 | Store model/type       |
| store | Assortment                | Assortment level       |
| store | CompetitionDistance       | Distance to competitor |
| store | CompetitionOpenSinceMonth | Competition open month |
| store | CompetitionOpenSinceYear  | Competition open year  |
| store | Promo2                    | Promo2 participation   |
| store | Promo2SinceYear           | Promo2 starting year   |
| store | Promo2SinceWeek           | Promo2 starting week   |
| store | PromoInterval             | Months in Promo2 cycle |

______________________________________________________________________

# **15. Summary**

This section completes the full data dictionary for the Rossmann forecasting project. It should be used by Claude Code and project contributors to understand raw fields, feature engineering requirements, and downstream modeling considerations.

---\*\*

Claude Code must implement the project **phase by phase**, only modifying the code required for the current phase. No refactoring of working code unless needed.

______________________________________________________________________

## **Phase 0 — Project Skeleton & Environment**

Claude should:

1. Create directory structure above.

1. Add `README.md` with basic project overview.

1. Create `requirements.txt` with:

   - numpy, pandas, sklearn, xgboost, lightgbm, catboost
   - matplotlib

______________________________________________________________________

## **Phase 0 — Project Skeleton & Environment**

Claude should:

1. Create the full directory structure defined in Section 11.

1. Add a `README.md` describing:

   - The project goal
   - RMSPE metric
   - High-level workflow (Phase 0–5)

1. Create an `env/requirements.txt` file including:

   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
   - xgboost
   - lightgbm
   - catboost
   - pyyaml

1. Create `config/params.yaml` with placeholder sections:

   ```yaml
   data:
     raw_path: data/raw
     processed_path: data/processed

   features:
     lags: [1, 7, 14, 28]
     rolling_windows: [7, 14, 28, 60]
     include_promo_features: true
     include_competition_features: true

   cv:
     method: expanding
     fold_length_days: 42  # 6 weeks
     min_train_days: 365   # 1 year minimum

   models:
     baseline_lightgbm: {}
     advanced_lightgbm: {}
     advanced_xgboost: {}
     advanced_catboost: {}
   ```

1. Implement **utility stubs**:

   - `src/utils/io.py` — simple read/write CSV/parquet helpers
   - `src/utils/log.py` — lightweight logging wrapper

1. Implement **RMSPE** metric in `src/evaluation/metrics.py`:

   ```python
   def rmspe(y_true, y_pred, ignore_zero_sales=True):
       mask = y_true != 0 if ignore_zero_sales else np.ones_like(y_true, dtype=bool)
       return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))
   ```

1. Create empty notebooks `01`–`05` with section headers but **no logic** yet.

______________________________________________________________________

## **Phase 1 — Data Loading, Cleaning & EDA**

Claude should:

- Implement `src/data/make_dataset.py` with functions:

  - `load_raw_data()`
  - `merge_store_info()`
  - `basic_cleaning()`
  - `save_processed_data()`

- Cleaning should include:

  - Proper datetime conversion
  - Joining store metadata
  - Handling missing competition/promo fields
  - Ensuring correct dtypes

- `notebooks/01-eda-and-cleaning.ipynb` should:

  - Call the above functions (no inline cleaning!)
  - Produce basic descriptive analytics
  - Save plots to `outputs/figures/`

**Output of Phase 1:** `data/processed/train_clean.parquet`

______________________________________________________________________

## **Phase 2 — Feature Engineering**

Claude should:

- Implement in `src/features/build_features.py`:

  - `add_calendar_features(df)`
  - `add_promo_features(df)`
  - `add_competition_features(df)`
  - `add_lag_features(df, lags)`
  - `add_rolling_features(df, windows)`
  - `build_all_features(df, config)`

- Ensure no leakage using:

  - `df.groupby("Store").shift(lag)` for lags
  - `df.groupby("Store").rolling(window)` for rolling windows

- `02-feature-engineering.ipynb` should:

  - Load cleaned data
  - Apply `build_all_features`
  - Save `train_features.parquet`

______________________________________________________________________

## **Phase 3 — Baseline Models & Time-Series CV**

Claude should:

- Implement in `src/evaluation/cv.py`:

  - `make_time_series_folds(df, config)`

- Implement in `src/models/train_baselines.py`:

  - Naive last-week model
  - Simple LightGBM baseline
  - Common training interface returning per-fold RMSPE

- Add reporting helpers in `src/evaluation/reporting.py`

- `03-baseline-models.ipynb` should:

  - Load featured data
  - Build CV splits
  - Train baseline models
  - Save CV metrics to `outputs/metrics/baseline/`

______________________________________________________________________

## **Phase 4 — Advanced Models & Ensembles**

Claude should:

### Implement advanced models

- In `src/models/train_advanced.py`:

  - Tuned LightGBM
  - Tuned XGBoost
  - Tuned CatBoost

### Implement required ensembles

- In `src/models/ensembles.py`:

  ````python
  def weighted_blend(preds_dict, weights_dict):
      # preds_dict: model name → np.array of preds
      # weights_dict: model name → weight
      ```

  ```python
  def stacked_ensemble(train_meta, y, valid_meta, config):
      # meta-learner can be linear or LightGBM
      ```
  ````

### Notebook

- `04-advanced-models-and-ensembles.ipynb` should:

  - Train advanced models
  - Generate out-of-fold predictions
  - Build weighted and stacked ensembles
  - Compare RMSPE vs baselines

______________________________________________________________________

## **Phase 5 — Final Model & Holdout Evaluation**

Claude should:

- Implement `train_final_model()` in a new file (`final_model.py` or inside advanced module):

  - Train on full training window (minus final holdout)
  - Use chosen model/ensemble

- `05-final-eval-and-test-simulation.ipynb` should:

  - Define final 6-week holdout period

  - Train final model

  - Generate predictions

  - Compute holdout RMSPE

  - Save:

    - Final model to `models/final/`
    - Predictions to `outputs/predictions/`
    - Metrics to `outputs/metrics/final/`
