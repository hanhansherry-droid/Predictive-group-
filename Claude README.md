# MSIN0097 Predictive Analytics — Group Coursework
## Predicting Corporate Bankruptcy Using Financial Ratios (Taiwan, 1999–2009)

---

## Overview

This project applies binary classification to predict whether a Taiwanese listed company will go bankrupt, using annual financial ratios sourced from the Taiwan Economic Journal (TEJ) database. The work covers the full supervised learning pipeline: exploratory data analysis, feature screening, preprocessing, model training, and final evaluation — with strict attention to data leakage prevention throughout.

---

## Research Objective

> **Can we predict whether a Taiwanese listed company will go bankrupt using financial ratios?**

- **Task:** Binary classification
- **Target variable:** `Bankrupt?` (1 = bankrupt, 0 = not bankrupt)
- **Class imbalance:** approximately 3.2% positive (220 bankrupt firms out of 6,819 observations)

---

## Dataset

| Property | Detail |
|---|---|
| Source | Taiwan Economic Journal (TEJ), via UCI ML Repository |
| Coverage | 1999–2009, Taiwanese listed companies |
| Raw dimensions | 6,819 rows × 96 columns |
| Features | 95 financial ratios (continuous and binary flags) |
| Target | `Bankrupt?` (column index 0) |

> **Note:** The dataset file is named ` bankrupt data.csv` (with a leading space). The notebook handles this automatically via `pathlib`.

---

## Repository Contents

```
predictive group cw/
│
├──  bankrupt data.csv              # Raw dataset (original TEJ file)
├── bankrupt_clean.csv              # After EDA-stage removal (4 columns dropped)
├── bankrupt_screened.csv           # After unsupervised feature screening only
│                                   # (variance + pairwise correlation filters)
│                                   # Starting point for the modelling pipeline
├── bankrupt_final.csv              # After full EDA feature selection (reference only;
│                                   # not loaded by the final modelling pipeline)
│
├── MSIN0097 Predictive Analysis Group CW.ipynb   # Main analysis notebook
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

---

## Environment and Dependencies

- **Python:** 3.13
- **Platform:** macOS (tested); should run on any platform with the packages below

Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| imbalanced-learn | 0.13.0 |

**macOS only:** XGBoost requires the OpenMP library. Install it once via Homebrew:

```bash
brew install libomp
```

---

## How to Run

1. Clone or download the repository into a local folder.
2. Place the dataset file (` bankrupt data.csv`) in the same directory as the notebook.
3. Install dependencies: `pip install -r requirements.txt`
4. Open the notebook in JupyterLab or VS Code:

```bash
jupyter lab "MSIN0097 Predictive Analysis Group CW.ipynb"
```

5. Run cells **in order from top to bottom**. Each modelling section (9, 10, 11) is self-contained and can also be run independently — it will rebuild the full preprocessing pipeline from scratch.

---

## Workflow Summary

### Sections 1–2 — Setup and EDA
- Load raw data; inspect shape, types, and summary statistics.
- Data quality checks: missing values, sentinel encodings (values > 1 × 10⁶), duplicate rows and columns, constant and near-constant features.
- Four columns removed: one constant flag, one near-constant flag, two exact duplicate columns.
- Target variable analysis: class distribution, imbalance quantification.

### Section 3 — Unsupervised Feature Screening (full dataset, exploratory)
- Remove three near-zero-variance features.
- Remove nine features with high pairwise correlation (|r| > 0.85), using semantic justification.
- Output saved as `bankrupt_screened.csv` (79 features + target).

### Section 4 — Leakage Assessment and Target Correlation *(exploratory only)*
- Manual audit of binary flags for direct leakage risk.
- Full-dataset Pearson correlation ranking against `Bankrupt?` for exploratory reference.

### Section 5 — Further Feature Selection *(exploratory only, full dataset)*
- Near-zero IQR + low target-correlation filter and high-pairwise-correlation removal documented on the full dataset for transparency.
- Output saved as `bankrupt_final.csv` (57 features) for reference only.
- **This file is not used by the final modelling pipeline.**

### Sections 6–7 — Feature EDA and Supplementary Analysis *(exploratory only)*
- Skewness, outlier analysis, correlation heatmap, scale audit.
- Mutual information scores, class-conditional distributions, stratified summary statistics, preprocessing preview.
- All computed on the full dataset for exploratory insight; none feed into the evaluated pipeline.

### Section 8 — Preprocessing Pipeline Documentation
- Documents the anti-leakage preprocessing order for reference.

### Sections 9–11 — Modelling Pipeline (post-split, leakage-free)

> All steps below are performed strictly after the train/validation/test split.

1. **Starting point:** `bankrupt_screened.csv` — features selected by unsupervised methods only.
2. **Stratified split:** 70% train / 15% validation / 15% test (`random_state=42`, `stratify=y`). Split is performed before any supervised step.
3. **Sentinel replacement:** Values > 1 × 10⁶ replaced with `NaN` (TEJ encoding for undefined ratios). Columns with > 50% sentinel values in the training set are dropped.
4. **Supervised feature selection (training set only):** Near-zero IQR + low target-correlation filter, then high-pairwise-correlation removal. Feature set applied unchanged to validation and test.
5. **Imputation:** Median imputation fitted on training set; applied to all splits.
6. **Winsorisation:** p1/p99 bounds from training set; clipped across all splits.
7. **RobustScaler:** Fitted on training set; applied to all splits.
8. **SMOTE:** Applied to training set only (k = 5, `random_state=42`).
9. **Model training and selection:** Logistic Regression (baseline), XGBoost, and LightGBM trained and compared on the validation set using ROC-AUC, PR-AUC, F1, precision, and recall.
10. **Hyperparameter tuning (Section 10.4):** Randomised search (30 iterations, 5-fold stratified CV) over XGBoost hyperparameters. SMOTE is applied **inside each CV fold** via `imblearn.Pipeline` to prevent oversampled points leaking into fold validation rows. Optimisation metric: Average Precision (PR-AUC). The held-out validation set is used only after the search to compare tuned vs untuned; the better configuration is carried forward.
11. **Final evaluation:** Best XGBoost configuration retrained with fixed `n_estimators` (determined by early stopping on validation); evaluated **once** on the untouched test set using the decision threshold derived from the validation PR curve.

---

## Leakage Prevention and Reproducibility

**No target information from validation or test rows influenced any step of the modelling pipeline.** Key safeguards:

- The train/validation/test split is performed before any supervised computation.
- All preprocessing parameters (imputation medians, Winsorisation bounds, scaler statistics, feature selection thresholds) are derived from the training set and applied without refitting to validation and test.
- SMOTE is applied to the training set only; validation and test sets preserve the real-world class distribution.
- The decision threshold is derived from the validation PR curve and applied unchanged to the test set.
- Hyperparameter tuning uses SMOTE-inside-fold cross-validation so no oversampled points appear in any fold's validation rows.
- The test set is used exactly once, for final evaluation only.
- Full-dataset analyses in Sections 4–7 are explicitly labelled as exploratory and do not feed into the evaluated pipeline.

**Reproducibility:** All random operations use `random_state=42`. The notebook is fully self-contained; running it top to bottom on the same dataset will reproduce all results exactly.

---

*MSIN0097 Predictive Analytics — Group Coursework*
