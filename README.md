# Customer Churn Prediction - End-to-End Machine Learning Project

## Project Overview

This project focuses on predicting customer churn for a telecom company using multiple machine learning algorithms. The primary objective is to accurately predict churn probability and segment customers into risk groups, enabling the business to proactively reduce churn rates and improve revenue stability.

## Tools and Technologies Used

- Python
- Pandas, NumPy (Data Manipulation)
- Matplotlib, Seaborn (Data Visualization)
- Scikit-learn (Machine Learning Models, Evaluation Metrics, Cross-Validation, Preprocessing)
- XGBoost (Gradient Boosting Model)
- SHAP (Model Interpretability)

## Dataset

- Dataset used: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Total Records: 7,043
- Target Variable: Churn (Yes/No)

## Project Workflow

### Data Cleaning and Preprocessing

- Detected and handled invalid entries in the TotalCharges column.
- Converted non-numeric values to NaN and dropped affected rows.
- Final dataset shape after cleaning: 7,032 rows.

### Exploratory Data Analysis (EDA)

- Analyzed churn distribution across both numerical and categorical features.
- Used:
  - Histograms for feature distributions (tenure, MonthlyCharges, TotalCharges).
  - Box plots to compare distributions across churn categories.
  - Bar charts for categorical features like Partner, InternetService, Gender.
- Derived business insights from EDA patterns.

### Feature Engineering

- Performed binary encoding for binary categorical variables.
- Applied one-hot encoding for multi-class categorical variables.
- Dropped customerID column.
- Scaled numerical features using MinMaxScaler.

### Train-Test Split

- Applied stratified train-test split to maintain churn class balance.
- Train set: 80 percent (5,625 records)
- Test set: 20 percent (1,407 records)

## Model Development and Evaluation

### Logistic Regression (Baseline Model)

- Achieved 80.5 percent test accuracy.
- Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.
- Performed 5-fold cross-validation.

### Threshold Tuning

- Adjusted decision threshold to 0.3 to optimize recall for business needs.
- Boosted recall from 57 percent to 75.4 percent by lowering threshold.

### Decision Tree Classifier

- Built and evaluated Decision Tree model.
- ROC-AUC: 0.66
- Performance lower compared to logistic regression.

### Random Forest Classifier

- Trained ensemble Random Forest model.
- Achieved ROC-AUC: 0.81
- Conducted cross-validation for stability analysis.

### XGBoost Classifier

- Trained and evaluated XGBoost model.
- ROC-AUC: 0.81
- Performed hyperparameter tuning via GridSearchCV for best parameters.

## Model Comparison Summary

| Model                          | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression             | 80.5%    | 65.1%     | 57.0%  | 60.8%    | 0.835   |
| Decision Tree                   | 73.1%    | 49.5%     | 50.8%  | 50.1%    | 0.660   |
| Random Forest                   | 78.6%    | 62.3%     | 49.5%  | 55.1%    | 0.813   |
| XGBoost                          | 76.8%    | 57.1%     | 51.6%  | 54.2%    | 0.814   |
| Logistic Regression (Threshold=0.3) | 74.3% | 51.1% | 75.4% | 60.9% | 0.835 |

## Churn Risk Segmentation

- Segmented customers into:
  - High Risk (probability ≥ 0.7)
  - Medium Risk (0.4 ≤ probability < 0.7)
  - Low Risk (probability < 0.4)
- This segmentation enables targeted retention strategies for the business.

## Model Explainability

- Applied SHAP (SHapley Additive Explanations) to the Random Forest model.
- Identified key features driving churn predictions.
- Generated SHAP summary plots to assist business stakeholders in understanding feature contributions.

## Business Takeaways

- Logistic Regression delivered high interpretability with competitive accuracy.
- Threshold tuning significantly improved recall performance, which is valuable for business retention strategies.
- Customer segmentation provided actionable insights for targeted interventions.
- SHAP explainability enhanced transparency and trust in model predictions.

## Project Status

Project completed.

Project Prepared by: Ankit Hiremath