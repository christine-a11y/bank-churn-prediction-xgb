# 📊 Bank Customer Churn Prediction & Tableau Analytics

An end-to-end Machine Learning pipeline designed to predict customer churn for a banking institution. This repository features data cleaning, Exploratory Data Analysis (EDA), an optimized **XGBoost Classifier** tuned via **GridSearchCV**, and risk-stratified predictions evaluated on test data and exported directly for **Tableau Dashboard** integration.

## 🔗 Quick Links
* **Interactive Tableau Dashboard:** [Link to your Tableau Public Profile](https://public.tableau.com/views/BankChurnXGBoostAnalysis/BankChurnPredictiveAnalyticsDashboardXGBoostModel?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
* **Data Sources:** `train.csv` (Model Training) & `test.csv` (Out-of-sample Predictions)

---

## 🛠️ Tech Stack & Dependencies
* **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Machine Learning & Pipeline Architecture:** `scikit-learn`
  * `ColumnTransformer` & `Pipeline`
  * `SimpleImputer` (Median & Constant strategies)
  * `StandardScaler` & `OneHotEncoder`
  * `GridSearchCV` (5-Fold Cross-Validation)
* **Predictive Modeling:** `xgboost` (`XGBClassifier`)

---

## 📈 Exploratory Data Analysis (EDA) Highlights
Before model fitting, data distributions were thoroughly analyzed to extract descriptive features:
* **Target Imbalance:** Visualized the exact retention vs. attrition baseline ratios utilizing percentage-based count plots.
* **Demographic Triggers:** Boxplots uncovered that older age cohorts correlate significantly with higher exit rates. 
* **Financial Integrity:** Examined the interaction between customer account balances and target classes.
* **Multicollinearity Check:** Built a full-scale correlation heatmap across numerical segments to ensure clean feature gradients.

---

## 🧠 Machine Learning Architecture

The script processes data dynamically inside an isolated `scikit-learn` workflow to ensure zero data leakage:

### 1. Robust Feature Engineering
* **Numerical Columns:** Missing attributes are replaced via `median` imputation and scaled smoothly using `StandardScaler`.
* **Categorical Columns:** Any unseen missing categorical values are caught with a static placeholder (`"missing"`) and vectorized using `OneHotEncoder(handle_unknown='ignore')`.

### 2. Hyperparameter Optimization & Tuning
The model applies a stratified 5-fold cross-validation (`cv=5`) targeting the **ROC-AUC** metric to locate steady operational boundaries:
```python
param_xgb = {
    "model__n_estimators": [300],
    "model__learning_rate": [0.1],
    "model__max_depth": [5]
}


Tableau Integration & Operational Risk Mapping


The finalized pipeline fits the comprehensive grid setup onto the training environment, evaluates the test.csv entries, and maps continuous probabilities (predict_proba) into discrete actionable categories:

🔴 High Risk: Churn Probability $ \ge $ 0.75
🟡 Medium Risk: Churn Probability between 0.35 and 0.74
🟢 Low Risk: Churn Probability < 0.35

The structured outcome is automatically synchronized into a dedicated file: bank_churn_for_tableau.csv. This powers the final executive views, serving as a clean tactical dispatch list for automated CRM outreach.