# Taiwanese Bankruptcy Prediction

This project predicts whether a Taiwanese listed company will go bankrupt using financial ratios.

The target variable is `Bankrupt?`, so the task is an imbalanced binary classification problem on tabular financial data.

## Project Summary

The project is built around a Jupyter notebook workflow that:

- cleans the raw dataset
- performs descriptive exploratory data analysis
- splits the data before any learned preprocessing
- applies training-only feature screening
- fits preprocessing with strict anti-leakage practice
- trains and compares baseline and modern classifiers
- tunes the final selected model
- evaluates the final model on an untouched test set

The workflow is designed to keep the test set fully isolated until the final evaluation stage.

## Research Question

Can we predict whether a Taiwanese listed company will go bankrupt using financial ratios?

## Repository Structure

- `code/analysis.ipynb`
  Main notebook containing the full analysis and modelling workflow.

- `data/data_2.csv`
  Original raw dataset.

- `code/data_2_filtered.csv`
  Dataset after unsupervised cleaning only.

- `code/data_2_screened.csv`
  Dataset after feature screening decisions derived from the training data.

- `requirements.txt`
  Python dependencies for the notebook workflow.

- `Agent log.md`
  Condensed record of the main tasks and outputs from the build process.

## Workflow

## Section 1: Setup

- import required libraries
- load the raw CSV file
- inspect the first rows of the data

## Section 2: Initial Data Audit and Unsupervised Cleaning

- create `df_eda`
- inspect structure and descriptive statistics
- detect standard missingness
- recode invalid extreme numeric values to missing
- remove structurally unusable columns such as duplicated, constant, or highly incomplete variables
- save the cleaned result as `data_2_filtered.csv`

Important:
No imputation is done in this section.

## Section 3: Descriptive EDA Before the Split

- inspect the target distribution
- review feature distributions, skewness, and outliers
- visualise broad patterns in the cleaned dataset

Important:
This section is descriptive only and does not fit preprocessing or perform target-driven feature selection.

## Section 4: Train / Validation / Test Split

- separate predictors and target
- create `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, and `y_test`
- preserve the test set for final evaluation only

## Section 5: Training-Only Feature Screening

- remove constant and near-zero variance features using training data only
- identify highly correlated features using training data only
- use `y_train` only where target relevance is needed
- apply the resulting retained feature set consistently to train, validation, and test
- save the screened dataset as `data_2_screened.csv`

## Section 6: Post-Screening EDA on the Training Set

- review the screened training features
- inspect distributions, outliers, and remaining correlation structure

## Section 7: Training-Only Diagnostics Before Preprocessing

- inspect remaining missingness in the screened training set
- justify the numeric imputation strategy
- review scale differences and zero-heavy features
- inspect class-conditional separation of key training features

## Section 8: Training-Only Preprocessing

- fit imputation on training only
- fit winsorisation thresholds on training only
- fit scaling on training only
- apply the fitted transformations to validation and test
- create final modelling-ready objects such as:
  - `X_train_final`
  - `X_val_final`
  - `X_test_final`

## Section 9: Baseline Model

- train a simple baseline classifier
- evaluate it on the validation set using imbalance-aware metrics

## Section 10: Modern Model Comparison

- train and compare stronger models such as LightGBM and XGBoost
- select the best untuned model using validation performance
- inspect key factors affecting predictions
- summarise the selected model in a model card

## Section 11: Hyperparameter Tuning

- tune the final selected model only
- use training data only for the hyperparameter search
- keep validation for model comparison
- keep the test set untouched
- compare tuned and untuned versions and keep the better one

## Section 12: Final Evaluation on the Untouched Test Set

- evaluate the final chosen model once on the test set
- report final metrics, classification report, and confusion matrix
- compare validation and test performance to assess generalisation

## Data Leakage Control

The notebook is structured to reduce leakage:

- unsupervised cleaning is allowed before the split
- train/validation/test split happens before any learned preprocessing
- feature screening decisions for modelling are derived from training data only
- imputation is fit on training only
- winsorisation thresholds are fit on training only
- scaling is fit on training only
- validation is used for model comparison and selection
- the test set is used only in the final evaluation section

## Main Notebook Objects

The notebook uses stable object names across sections. The most important are:

- `df`
  Raw dataset loaded from CSV.

- `df_eda`
  Cleaned full dataset used for EDA and splitting.

- `df_screen`
  Screened dataset after training-derived feature selection.

- `X_train`, `X_val`, `X_test`
  Screened feature splits.

- `y_train`, `y_val`, `y_test`
  Target splits.

- `X_train_final`, `X_val_final`, `X_test_final`
  Final preprocessed feature matrices.

These names are preserved so the modelling sections remain compatible with earlier notebook stages.

## Modelling Approach

Because this is an imbalanced classification problem, the project prioritises metrics such as:

- balanced accuracy
- precision
- recall
- F1 score
- ROC-AUC
- average precision

Plain accuracy is not sufficient on its own because it can look strong even when the model performs poorly on bankrupt firms.

## Why These Models

The project focuses on models suitable for structured financial data:

- Logistic Regression
  Used as a baseline.

- LightGBM
  Strong for tabular data, efficient, and well suited to non-linear relationships.

- XGBoost
  Robust, widely used, and competitive for structured classification tasks.

The final selected model is tuned only after the best untuned candidate has already been identified.

## How to Run

1. Open `code/analysis.ipynb`.
2. Run the notebook in section order from Section 1 onward.
3. Do not skip the split section.
4. Do not use the test set before the final evaluation stage.

Recommended practice:

- keep `data_2_filtered.csv` as the unsupervised-cleaned dataset only
- keep `data_2_screened.csv` as the training-derived screened dataset only

## Environment

The notebook uses these main libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `statsmodels`
- Jupyter / Notebook

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Limitations

- The project uses a single dataset, so external generalisation is not guaranteed.
- Financial ratio data can be skewed, noisy, and sensitive to data-quality issues.
- Results depend on the cleaning rules used for invalid extreme values.
- Class imbalance makes threshold choice and evaluation especially important.
- The best-performing model may still require calibration or further robustness checks.

## Notes

This repository appears to support a coursework or group predictive analysis project. Update contributor and project metadata as needed for submission or sharing.
