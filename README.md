# Kaggle House Prices — Top 5% Regression Solution with Databricks + MLflow

Achieved a **Top 5% score** on a Kaggle-style House Prices regression assignment by building an end-to-end machine learning pipeline for residential sale price prediction.

This repository includes both the original competition-style modeling workflow and a Databricks + MLflow version that demonstrates a more production-oriented machine learning process.

## Project Highlights

- **Result:** Top 5%
- **Final Kaggle RMSE:** 0.12564
- **Task:** Predict residential home sale prices
- **Metric:** Root Mean Squared Error
- **Models Used:** Ridge Regression, LightGBM, CatBoost
- **Techniques:** Feature engineering, log-target transformation, cross-validation, Optuna tuning, OOF stacking, multi-seed averaging
- **Platform:** Databricks
- **Experiment Tracking:** MLflow

## Project Overview

The goal of this project was to predict home sale prices using structured housing data. The dataset includes property, neighborhood, quality, basement, garage, sale condition, and area-related features.

The modeling pipeline includes:

1. Data loading and validation
2. Missing value handling
3. Feature engineering
4. Ordinal and categorical feature processing
5. Log transformation of the target variable
6. Cross-validation using RMSE
7. Hyperparameter tuning with Optuna
8. Model training with Ridge, LightGBM, and CatBoost
9. Out-of-fold stacking
10. Submission generation
11. MLflow experiment tracking in Databricks

## Why Databricks + MLflow?

The original Kaggle notebook focused on maximizing predictive performance.  
The Databricks version adds a more professional ML workflow by tracking:

- Model parameters
- Cross-validation metrics
- Fold-level RMSE
- Final stacked ensemble RMSE
- Submission artifacts
- Model bundle artifacts

This demonstrates how a competition-style machine learning notebook can be converted into a more reproducible and trackable workflow.

## Repository Structure

```text
kaggle-house-prices-top-5/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_original_kaggle_solution.ipynb
│   └── 02_databricks_mlflow_version.ipynb
│
├── databricks/
│   └── House_Prices_Databricks_MLflow_Version.py
│
├── images/
│   ├── leaderboard_score.png
│   ├── databricks_notebook.png
│   ├── mlflow_experiment.png
│   └── feature_importance.png
│
└── outputs/
    └── sample_submission_preview.csv
