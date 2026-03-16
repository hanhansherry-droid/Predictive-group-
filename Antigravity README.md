# Taiwan Bankruptcy Prediction Project

Predicting corporate bankruptcy using financial ratios for firms listed on the Taiwan Stock Exchange.

## Project Overview
This project aims to develop a robust predictive model to identify firms at risk of bankruptcy. Due to the high cost of bankruptcy for stakeholders, accurate prediction is critical for risk management and investment decision-making.

## Dataset Description
- **Source:** Taiwan Economic Journal (TEJ) / UC Irvine Machine Learning Repository.
- **File:** `data 2.csv`
- **Samples:** 6819 instances.
- **Features:** 96 financial indicators (ratios) representing liquidity, profitability, debt, etc.
- **Target:** `Bankrupt?` (1 for bankrupt, 0 otherwise).
- **Imbalance:** The dataset is highly imbalanced, with approximately 3.23% of firms marked as bankrupt.

## Pipeline & Methodology

### 1. Data Cleaning & Preparation
- Initial screening for missing values and duplicates.
- Dropping constant or near-constant features (e.g., `Net Income Flag`).
- **Unicode Sanitation:** Renamed features with non-ASCII symbols (e.g., `¥` symbols in ratios) to ensure compatibility with parallel processing backends.

### 2. High-Fidelity Preprocessing (Anti-Leakage)
- **Strict Split:** Data is split into 70% Training, 15% Validation, and 15% Test **before any feature analysis**.
- All transformations are fitted **only** on the training set and then applied to validation and test sets.

### 3. Feature Engineering
- **Variance Analysis:** Removal of low-variance features using Coefficient of Variation (CV).
- **Multicollinearity Removal:** Pair-wise correlation analysis to drop redundant features (threshold: 0.90), using target correlation as a tie-breaker.
- **Skewness & Outlier Treatment:** 
  - Extreme values handled via **Adaptive Outlier Boundary Normalization** (Log-transform + MinMax scaling for high-magnitude features).
  - Winsorization (Clipping at 1st and 100th percentiles where appropriate) for moderate outliers.

### 4. Modeling & Evaluation
- **Models:** Logistic Regression (Baseline) and LightGBM (Gradient Boosting).
- **Addressing Imbalance:** Models utilize `class_weight='balanced'` for Logistic Regression and `is_unbalance=True` for LightGBM.
- **Metrics:** F1-Score, Precision-Recall AUC (Average Precision), and ROC-AUC.

### 5. Hyperparameter Tuning
- **Randomized Search:** Conducted on the champion LightGBM model using `RandomizedSearchCV` with 5-fold Stratified CV.
- **Optimization Goal:** Average Precision (PR-AUC) to maximize sensitivity to the minority bankruptcy class.

## Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project directory.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook:**
   Open `prediction_project.ipynb` in your preferred Jupyter environment (VS Code, JupyterLab, etc.) and run all cells.

## Results
The refined pipeline produces a feature set of 64 indicators. After hyperparameter optimization, the **LightGBM Champion Model** achieved:
- **Validation PR-AUC:** >0.48
- **Test Set PR-AUC:** >0.46
- **Test ROC-AUC:** >0.94

The model shows strong discriminative power even with severe class imbalance, confirming its reliability for corporate bankruptcy risk assessment.
