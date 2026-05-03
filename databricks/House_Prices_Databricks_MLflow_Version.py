# Databricks notebook source
TRAIN_TABLE = "workspace.default.house_prices_train"
TEST_TABLE = "workspace.default.house_prices_test"

# COMMAND ----------

# ============================================================
# Kaggle House Prices — Improved Databricks + MLflow Version
# Ridge + LightGBM + CatBoost + XGBoost
# Interaction Features + OOF Target Encoding + Blending
# ============================================================

# COMMAND ----------
# Install dependencies
# COMMAND ----------

%pip install -q lightgbm catboost optuna mlflow xgboost

# COMMAND ----------
# Restart Python after install
# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# Imports
# ============================================================

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import mlflow
import mlflow.sklearn

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

# ============================================================
# 1. Configuration
# ============================================================

# Update these if your Databricks table names are different.
TRAIN_TABLE = "workspace.default.house_prices_train"
TEST_TABLE = "workspace.default.house_prices_test"

# Local driver output path.
# This works reliably on Databricks Serverless.
OUTPUT_DIR = "/tmp/kaggle_house_prices"
SUBMISSION_STACK_PATH = f"{OUTPUT_DIR}/submission_stacked_databricks_mlflow.csv"
SUBMISSION_BLEND_PATH = f"{OUTPUT_DIR}/submission_blend_databricks_mlflow.csv"
MODEL_BUNDLE_PATH = f"{OUTPUT_DIR}/model_bundle_improved.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# For debugging, use 5.
# For final stronger run, use 10.
N_SPLITS = 10

# For debugging, use small values like 5–10.
# For final run, use 75+ for LGBM and 50+ for CatBoost.
LGB_TRIALS = 75
CAT_TRIALS = 50

# For debugging, use [42].
# For final run, use multiple seeds.
TREE_SEEDS = [42, 2024, 777, 1337, 9001]

EXPERIMENT_NAME = "/Shared/Kaggle House Prices - Improved Top 5 Percent"

print("Output directory:", OUTPUT_DIR)
print("N_SPLITS:", N_SPLITS)
print("TREE_SEEDS:", TREE_SEEDS)

# COMMAND ----------

# ============================================================
# 2. Utility Functions
# ============================================================

def rmse_log(y_true_log, y_pred_log):
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def print_score(name, y_true_log, y_pred_log):
    score = rmse_log(y_true_log, y_pred_log)
    print(f"{name}: {score:.6f}")
    return score


def table_exists(table_name):
    try:
        spark.table(table_name).limit(1).collect()
        return True
    except Exception:
        return False


def load_train_test():
    if table_exists(TRAIN_TABLE) and table_exists(TEST_TABLE):
        print(f"Loading train from table: {TRAIN_TABLE}")
        print(f"Loading test from table : {TEST_TABLE}")
        train_df = spark.table(TRAIN_TABLE).toPandas()
        test_df = spark.table(TEST_TABLE).toPandas()
    else:
        raise ValueError(
            f"""
            Could not find Databricks tables.

            Expected:
            {TRAIN_TABLE}
            {TEST_TABLE}

            Go to Catalog/Data, find your uploaded CSV tables,
            copy the full table names, and update TRAIN_TABLE and TEST_TABLE.
            """
        )

    train_df = train_df.replace(["NA", "NaN", "nan", "None", ""], np.nan)
    test_df = test_df.replace(["NA", "NaN", "nan", "None", ""], np.nan)

    if "Id" in train_df.columns:
        train_df["Id"] = pd.to_numeric(train_df["Id"], errors="coerce").astype("Int64")
    if "Id" in test_df.columns:
        test_df["Id"] = pd.to_numeric(test_df["Id"], errors="coerce").astype("Int64")
    if "SalePrice" in train_df.columns:
        train_df["SalePrice"] = pd.to_numeric(train_df["SalePrice"], errors="coerce")

    return train_df, test_df

# COMMAND ----------

# ============================================================
# 3. Load Data
# ============================================================

train, test = load_train_test()

print("Train shape:", train.shape)
print("Test shape :", test.shape)

display(train.head())

# COMMAND ----------

# ============================================================
# 4. Remove Known Train-Only Outliers
# ============================================================

train = train.drop(
    train[(pd.to_numeric(train["GrLivArea"], errors="coerce") > 4000) &
          (pd.to_numeric(train["SalePrice"], errors="coerce") < 300000)].index
).reset_index(drop=True)

print("Train shape after outlier removal:", train.shape)

# COMMAND ----------

# ============================================================
# 5. Feature Engineering
# ============================================================

QUAL_MAP_5 = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BSMT_FIN_MAP = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
FINISH_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
FUNCTIONAL_MAP = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
PAVED_MAP = {"N": 0, "P": 1, "Y": 2}
FENCE_MAP = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

NONE_STRING_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]

ZERO_NUM_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]

EXPECTED_NUMERIC_COLS = [
    "Id", "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"
]


def build_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = train_df.replace(["NA", "NaN", "nan", "None", ""], np.nan)
    test_df = test_df.replace(["NA", "NaN", "nan", "None", ""], np.nan)

    train_df["SalePrice"] = pd.to_numeric(train_df["SalePrice"], errors="coerce")

    train_target = train_df["SalePrice"].copy()
    train_ids = pd.to_numeric(train_df["Id"], errors="coerce")
    test_ids = pd.to_numeric(test_df["Id"], errors="coerce")

    train_df = train_df.drop(columns=["SalePrice"])

    full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    for col in EXPECTED_NUMERIC_COLS:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    if "MSSubClass" in full.columns:
        full["MSSubClass"] = full["MSSubClass"].astype("Int64").astype(str)

    for col in NONE_STRING_COLS:
        if col in full.columns:
            full[col] = full[col].fillna("None")

    for col in ZERO_NUM_COLS:
        if col in full.columns:
            full[col] = full[col].fillna(0)

    obj_cols = full.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        full[col] = full[col].fillna("Missing")

    if "LotFrontage" in full.columns and "Neighborhood" in full.columns:
        full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        full["LotFrontage"] = full["LotFrontage"].fillna(full["LotFrontage"].median())

    num_cols_local = full.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols_local:
        full[col] = full[col].fillna(full[col].median())

    ord_5_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"
    ]

    for col in ord_5_cols:
        if col in full.columns:
            full[col + "_Ord"] = full[col].map(QUAL_MAP_5).fillna(0).astype(int)

    if "BsmtFinType1" in full.columns:
        full["BsmtFinType1_Ord"] = full["BsmtFinType1"].map(BSMT_FIN_MAP).fillna(0).astype(int)

    if "BsmtFinType2" in full.columns:
        full["BsmtFinType2_Ord"] = full["BsmtFinType2"].map(BSMT_FIN_MAP).fillna(0).astype(int)

    if "BsmtExposure" in full.columns:
        full["BsmtExposure_Ord"] = full["BsmtExposure"].map(EXPOSURE_MAP).fillna(0).astype(int)

    if "GarageFinish" in full.columns:
        full["GarageFinish_Ord"] = full["GarageFinish"].map(FINISH_MAP).fillna(0).astype(int)

    if "Functional" in full.columns:
        full["Functional_Ord"] = full["Functional"].map(FUNCTIONAL_MAP).fillna(0).astype(int)

    if "PavedDrive" in full.columns:
        full["PavedDrive_Ord"] = full["PavedDrive"].map(PAVED_MAP).fillna(0).astype(int)

    if "Fence" in full.columns:
        full["Fence_Ord"] = full["Fence"].map(FENCE_MAP).fillna(0).astype(int)

    for col in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
        if col in full.columns:
            full[f"{col}_MissingFlag"] = full[col].isna().astype(int)

    # Core aggregate features
    full["TotalSF"] = full["TotalBsmtSF"] + full["1stFlrSF"] + full["2ndFlrSF"]
    full["TotalArea"] = full["GrLivArea"] + full["TotalBsmtSF"]

    full["TotalPorchSF"] = (
        full.get("OpenPorchSF", 0)
        + full.get("EnclosedPorch", 0)
        + full.get("3SsnPorch", 0)
        + full.get("ScreenPorch", 0)
        + full.get("WoodDeckSF", 0)
    )

    full["Baths"] = (
        full.get("FullBath", 0)
        + 0.5 * full.get("HalfBath", 0)
        + full.get("BsmtFullBath", 0)
        + 0.5 * full.get("BsmtHalfBath", 0)
    )

    full["TotalRooms"] = full.get("TotRmsAbvGrd", 0) + full.get("KitchenAbvGr", 0)
    full["TotalHomeQuality"] = full["OverallQual"] + full["OverallCond"]
    full["OverallGrade"] = full["OverallQual"] * full["OverallCond"]

    full["BsmtFinSF"] = full["BsmtFinSF1"] + full["BsmtFinSF2"]
    full["BsmtFinRatio"] = full["BsmtFinSF"] / (full["TotalBsmtSF"] + 1)
    full["BsmtUnfRatio"] = full["BsmtUnfSF"] / (full["TotalBsmtSF"] + 1)

    full["HouseAge"] = full["YrSold"] - full["YearBuilt"]
    full["RemodAge"] = full["YrSold"] - full["YearRemodAdd"]
    full["GarageAge"] = full["YrSold"] - full["GarageYrBlt"]

    full["GarageAge"] = full["GarageAge"].clip(lower=0)
    full["HouseAge"] = full["HouseAge"].clip(lower=0)
    full["RemodAge"] = full["RemodAge"].clip(lower=0)

    full["IsRemodeled"] = (full["YearBuilt"] != full["YearRemodAdd"]).astype(int)
    full["IsNewHouse"] = (full["YrSold"] == full["YearBuilt"]).astype(int)

    full["HasPool"] = (full.get("PoolArea", 0) > 0).astype(int)
    full["Has2ndFloor"] = (full["2ndFlrSF"] > 0).astype(int)
    full["HasGarage"] = (full["GarageArea"] > 0).astype(int)
    full["HasBsmt"] = (full["TotalBsmtSF"] > 0).astype(int)
    full["HasFireplace"] = (full.get("Fireplaces", 0) > 0).astype(int)
    full["HasPorch"] = (full["TotalPorchSF"] > 0).astype(int)

    if "KitchenQual_Ord" in full.columns:
        full["KitchenScore"] = full["KitchenQual_Ord"] * full["OverallQual"]

    if "ExterQual_Ord" in full.columns:
        full["ExteriorScore"] = full["ExterQual_Ord"] * full["OverallQual"]

    if "GarageQual_Ord" in full.columns and "GarageCond_Ord" in full.columns:
        full["GarageScore"] = full["GarageQual_Ord"] * full["GarageCond_Ord"]

    if "BsmtQual_Ord" in full.columns and "BsmtExposure_Ord" in full.columns:
        full["BsmtScore"] = full["BsmtQual_Ord"] * full["BsmtExposure_Ord"]

    # Log features
    full["LotAreaLog"] = np.log1p(full["LotArea"])
    full["GrLivAreaLog"] = np.log1p(full["GrLivArea"])
    full["TotalSFLog"] = np.log1p(full["TotalSF"])
    full["LotFrontageLog"] = np.log1p(full["LotFrontage"])

    full["LivLotRatio"] = full["GrLivArea"] / (full["LotArea"] + 1)
    full["GarageAreaPerCar"] = full["GarageArea"] / (full["GarageCars"] + 1)
    full["BathPerRoom"] = full["Baths"] / (full.get("TotRmsAbvGrd", 0) + 1)

    full["MoSold"] = full["MoSold"].astype(int)
    full["SoldInSpring"] = full["MoSold"].isin([3, 4, 5]).astype(int)
    full["SoldInSummer"] = full["MoSold"].isin([6, 7, 8]).astype(int)
    full["SoldInFall"] = full["MoSold"].isin([9, 10, 11]).astype(int)
    full["SoldInWinter"] = full["MoSold"].isin([12, 1, 2]).astype(int)

    # ========================================================
    # New improvement features
    # ========================================================

    # Quality-area interactions
    full["OverallQual_TotalSF"] = full["OverallQual"] * full["TotalSF"]
    full["OverallQual_GrLivArea"] = full["OverallQual"] * full["GrLivArea"]
    full["OverallQual_TotalArea"] = full["OverallQual"] * full["TotalArea"]

    # Neighborhood quality categorical interaction
    if "Neighborhood" in full.columns:
        full["Neighborhood_OverallQual"] = (
            full["Neighborhood"].astype(str) + "_" + full["OverallQual"].astype(str)
        )

    # Age and remodel interactions
    full["Age_Quality"] = full["HouseAge"] * full["OverallQual"]
    full["RemodAge_Quality"] = full["RemodAge"] * full["OverallQual"]

    # Garage interactions
    full["GarageCars_Qual"] = full["GarageCars"] * full["OverallQual"]
    full["GarageArea_Qual"] = full["GarageArea"] * full["OverallQual"]

    # Basement interactions
    full["TotalBsmtSF_Qual"] = full["TotalBsmtSF"] * full["OverallQual"]
    full["BsmtFinSF1_Qual"] = full["BsmtFinSF1"] * full["OverallQual"]

    # Bathroom and room interactions
    full["Baths_Qual"] = full["Baths"] * full["OverallQual"]
    full["Rooms_Qual"] = full["TotRmsAbvGrd"] * full["OverallQual"]

    # Exterior and kitchen quality interactions
    if "KitchenQual_Ord" in full.columns:
        full["KitchenQual_TotalSF"] = full["KitchenQual_Ord"] * full["TotalSF"]

    if "ExterQual_Ord" in full.columns:
        full["ExterQual_TotalSF"] = full["ExterQual_Ord"] * full["TotalSF"]

    # Newer/remodeled premium
    full["RecentRemodel"] = (full["YearRemodAdd"] >= 2000).astype(int)
    full["RecentBuilt"] = (full["YearBuilt"] >= 2000).astype(int)

    # Sale condition premium
    if "SaleCondition" in full.columns:
        full["NewSale_OverallQual"] = (
            (full["SaleCondition"].astype(str) == "Partial").astype(int)
            * full["OverallQual"]
        )

    # Size-per-room features
    full["SFPerRoom"] = full["GrLivArea"] / (full["TotRmsAbvGrd"] + 1)
    full["TotalSFPerRoom"] = full["TotalSF"] / (full["TotRmsAbvGrd"] + 1)

    # Lot interactions
    full["LotArea_OverallQual"] = full["LotArea"] * full["OverallQual"]
    full["LotFrontage_OverallQual"] = full["LotFrontage"] * full["OverallQual"]

    X_train = full.iloc[:len(train_df)].copy()
    X_test = full.iloc[len(train_df):].copy()
    y_train = np.log1p(train_target.values)

    return X_train, X_test, y_train, train_ids, test_ids

# COMMAND ----------

# ============================================================
# 6. Leakage-Safe OOF Target Encoding
# ============================================================

def add_oof_target_encoding(
    X_train,
    X_test,
    y_log,
    cols,
    n_splits=5,
    seed=42,
    smoothing=10
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    global_mean = y_log.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        if col not in X_train.columns:
            continue

        new_col = f"{col}_TargetEnc"
        X_train[new_col] = np.nan

        for tr_idx, va_idx in kf.split(X_train):
            tr_col = X_train.iloc[tr_idx][col].astype(str)
            va_col = X_train.iloc[va_idx][col].astype(str)
            tr_y = y_log[tr_idx]

            stats = pd.DataFrame({
                col: tr_col.values,
                "target": tr_y
            }).groupby(col)["target"].agg(["mean", "count"])

            smooth = (
                (stats["mean"] * stats["count"] + global_mean * smoothing)
                / (stats["count"] + smoothing)
            )

            X_train.iloc[va_idx, X_train.columns.get_loc(new_col)] = (
                va_col.map(smooth).fillna(global_mean).values
            )

        full_stats = pd.DataFrame({
            col: X_train[col].astype(str).values,
            "target": y_log
        }).groupby(col)["target"].agg(["mean", "count"])

        full_smooth = (
            (full_stats["mean"] * full_stats["count"] + global_mean * smoothing)
            / (full_stats["count"] + smoothing)
        )

        X_test[new_col] = X_test[col].astype(str).map(full_smooth).fillna(global_mean)

        print(f"Added target encoding: {new_col}")

    return X_train, X_test

# COMMAND ----------

# ============================================================
# 7. Build Features + Add Target Encoding
# ============================================================

X, X_test, y, train_ids, test_ids = build_features(train, test)

target_encode_cols = [
    "Neighborhood",
    "MSSubClass",
    "Exterior1st",
    "Exterior2nd",
    "SaleCondition",
    "OverallQual",
    "Neighborhood_OverallQual"
]

target_encode_cols = [c for c in target_encode_cols if c in X.columns]

X, X_test = add_oof_target_encoding(
    X,
    X_test,
    y,
    cols=target_encode_cols,
    n_splits=N_SPLITS,
    seed=RANDOM_STATE,
    smoothing=10
)

print("Engineered train shape:", X.shape)
print("Engineered test shape :", X_test.shape)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

print(f"Categorical columns: {len(cat_cols)}")
print(f"Numeric columns    : {len(num_cols)}")

# COMMAND ----------

# ============================================================
# 8. Linear / XGBoost Preprocessing Pipelines
# ============================================================

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

linear_preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

ridge_model = Pipeline([
    ("prep", linear_preprocessor),
    ("ridge", Ridge(alpha=12.0, random_state=RANDOM_STATE))
])

elastic_model = Pipeline([
    ("prep", linear_preprocessor),
    ("elastic", ElasticNet(
        alpha=0.0005,
        l1_ratio=0.7,
        random_state=RANDOM_STATE,
        max_iter=50000
    ))
])

xgb_model = Pipeline([
    ("prep", linear_preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=3,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.0005,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# COMMAND ----------

# ============================================================
# 9. Tune LightGBM
# ============================================================

def tune_lgbm(X_df, y_log, cat_columns, n_trials=25, random_state=42):
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 10000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.55, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 0.1, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1
        }

        fold_scores = []

        for tr_idx, va_idx in cv.split(X_df, y_log):
            X_tr = X_df.iloc[tr_idx].copy()
            X_va = X_df.iloc[va_idx].copy()
            y_tr = y_log[tr_idx]
            y_va = y_log[va_idx]

            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                categorical_feature=cat_columns,
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )

            pred = model.predict(X_va)
            fold_scores.append(rmse_log(y_va, pred))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("Best LightGBM score :", study.best_value)
    print("Best LightGBM params:", study.best_params)

    best_params = {
        **study.best_params,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 10000,
        "n_jobs": -1,
        "verbosity": -1
    }

    return best_params, study.best_value

# COMMAND ----------

# ============================================================
# 10. Tune CatBoost
# ============================================================

def tune_catboost(X_df, y_log, cat_columns, n_trials=20, random_state=42):
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 10000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 0.95),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_seed": RANDOM_STATE,
            "verbose": False
        }

        fold_scores = []

        for tr_idx, va_idx in cv.split(X_df, y_log):
            X_tr = X_df.iloc[tr_idx].copy()
            X_va = X_df.iloc[va_idx].copy()
            y_tr = y_log[tr_idx]
            y_va = y_log[va_idx]

            model = CatBoostRegressor(**params)

            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_columns,
                eval_set=(X_va, y_va),
                use_best_model=True,
                early_stopping_rounds=200,
                verbose=False
            )

            pred = model.predict(X_va)
            fold_scores.append(rmse_log(y_va, pred))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("Best CatBoost score :", study.best_value)
    print("Best CatBoost params:", study.best_params)

    best_params = {
        **study.best_params,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 10000,
        "verbose": False
    }

    return best_params, study.best_value

# COMMAND ----------

# ============================================================
# 11. Train, Stack, Blend, Log to MLflow
# ============================================================

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="Improved_Ridge_Elastic_LGBM_CatBoost_XGBoost_Ensemble"):

    mlflow.log_param("project", "Kaggle House Prices")
    mlflow.log_param("platform", "Databricks")
    mlflow.log_param("model_type", "Improved Stacked Ensemble + Weighted Blend")
    mlflow.log_param("base_models", "Ridge, ElasticNet, LightGBM, CatBoost, XGBoost")
    mlflow.log_param("target_transform", "log1p(SalePrice)")
    mlflow.log_param("feature_improvements", "Interactions + OOF Target Encoding")
    mlflow.log_param("n_splits", N_SPLITS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("lgb_trials", LGB_TRIALS)
    mlflow.log_param("cat_trials", CAT_TRIALS)
    mlflow.log_param("tree_seeds", str(TREE_SEEDS))
    mlflow.log_param("train_rows_after_outlier_removal", len(X))
    mlflow.log_param("test_rows", len(X_test))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_categorical_features", len(cat_cols))
    mlflow.log_param("n_numeric_features", len(num_cols))

    print("Tuning LightGBM...")
    best_lgb_params, best_lgb_cv = tune_lgbm(
        X,
        y,
        cat_cols,
        n_trials=LGB_TRIALS,
        random_state=RANDOM_STATE
    )

    mlflow.log_metric("lightgbm_tuning_cv_rmse", best_lgb_cv)
    mlflow.log_params({
        f"lgb_{k}": v
        for k, v in best_lgb_params.items()
        if isinstance(v, (str, int, float, bool))
    })

    print("Tuning CatBoost...")
    best_cat_params, best_cat_cv = tune_catboost(
        X,
        y,
        cat_cols,
        n_trials=CAT_TRIALS,
        random_state=RANDOM_STATE
    )

    mlflow.log_metric("catboost_tuning_cv_rmse", best_cat_cv)
    mlflow.log_params({
        f"cat_{k}": v
        for k, v in best_cat_params.items()
        if isinstance(v, (str, int, float, bool))
    })

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_ridge = np.zeros(len(X))
    oof_elastic = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))

    test_ridge = np.zeros(len(X_test))
    test_elastic = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Ridge
        ridge_fold = clone(ridge_model)
        ridge_fold.fit(X_tr, y_tr)
        pred_ridge_va = ridge_fold.predict(X_va)
        pred_ridge_test = ridge_fold.predict(X_test)
        oof_ridge[va_idx] = pred_ridge_va
        test_ridge += pred_ridge_test / N_SPLITS

        # ElasticNet
        elastic_fold = clone(elastic_model)
        elastic_fold.fit(X_tr, y_tr)
        pred_elastic_va = elastic_fold.predict(X_va)
        pred_elastic_test = elastic_fold.predict(X_test)
        oof_elastic[va_idx] = pred_elastic_va
        test_elastic += pred_elastic_test / N_SPLITS

        # XGBoost
        xgb_fold = clone(xgb_model)
        xgb_fold.fit(X_tr, y_tr)
        pred_xgb_va = xgb_fold.predict(X_va)
        pred_xgb_test = xgb_fold.predict(X_test)
        oof_xgb[va_idx] = pred_xgb_va
        test_xgb += pred_xgb_test / N_SPLITS

        # LightGBM multi-seed
        lgb_va_seed_preds = []
        lgb_test_seed_preds = []

        for seed in TREE_SEEDS:
            lgb_params = best_lgb_params.copy()
            lgb_params["random_state"] = seed

            lgb_model = lgb.LGBMRegressor(**lgb_params)

            lgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )

            lgb_va_seed_preds.append(lgb_model.predict(X_va))
            lgb_test_seed_preds.append(lgb_model.predict(X_test))

        pred_lgb_va = np.mean(lgb_va_seed_preds, axis=0)
        pred_lgb_test = np.mean(lgb_test_seed_preds, axis=0)

        oof_lgb[va_idx] = pred_lgb_va
        test_lgb += pred_lgb_test / N_SPLITS

        # CatBoost multi-seed
        cat_va_seed_preds = []
        cat_test_seed_preds = []

        for seed in TREE_SEEDS:
            cat_params = best_cat_params.copy()
            cat_params["random_seed"] = seed

            cat_model = CatBoostRegressor(**cat_params)

            cat_model.fit(
                X_tr,
                y_tr,
                cat_features=cat_cols,
                eval_set=(X_va, y_va),
                use_best_model=True,
                early_stopping_rounds=200,
                verbose=False
            )

            cat_va_seed_preds.append(cat_model.predict(X_va))
            cat_test_seed_preds.append(cat_model.predict(X_test))

        pred_cat_va = np.mean(cat_va_seed_preds, axis=0)
        pred_cat_test = np.mean(cat_test_seed_preds, axis=0)

        oof_cat[va_idx] = pred_cat_va
        test_cat += pred_cat_test / N_SPLITS

        fold_scores = {
            "ridge": rmse_log(y_va, pred_ridge_va),
            "elastic": rmse_log(y_va, pred_elastic_va),
            "xgboost": rmse_log(y_va, pred_xgb_va),
            "lightgbm": rmse_log(y_va, pred_lgb_va),
            "catboost": rmse_log(y_va, pred_cat_va),
        }

        for model_name, score in fold_scores.items():
            mlflow.log_metric(f"fold_{model_name}_rmse", score, step=fold)
            print(f"Fold {model_name}: {score:.6f}")

    print("\n========== Base OOF Scores ==========")

    ridge_score = print_score("OOF Ridge    ", y, oof_ridge)
    elastic_score = print_score("OOF Elastic  ", y, oof_elastic)
    xgb_score = print_score("OOF XGBoost  ", y, oof_xgb)
    lgb_score = print_score("OOF LightGBM ", y, oof_lgb)
    cat_score = print_score("OOF CatBoost ", y, oof_cat)

    mlflow.log_metric("ridge_oof_rmse", ridge_score)
    mlflow.log_metric("elastic_oof_rmse", elastic_score)
    mlflow.log_metric("xgboost_oof_rmse", xgb_score)
    mlflow.log_metric("lightgbm_oof_rmse", lgb_score)
    mlflow.log_metric("catboost_oof_rmse", cat_score)

    # ========================================================
    # Ridge stacking
    # ========================================================

    stack_train = np.column_stack([
        oof_ridge,
        oof_elastic,
        oof_lgb,
        oof_cat,
        oof_xgb
    ])

    stack_test = np.column_stack([
        test_ridge,
        test_elastic,
        test_lgb,
        test_cat,
        test_xgb
    ])

    meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    meta_model.fit(stack_train, y)

    oof_stack = meta_model.predict(stack_train)
    test_stack = meta_model.predict(stack_test)

    print("\n========== Stacked OOF Score ==========")

    stack_score = print_score("OOF STACK", y, oof_stack)

    print("Meta coefficients:", meta_model.coef_)
    print("Meta intercept   :", meta_model.intercept_)

    mlflow.log_metric("stacked_oof_rmse", stack_score)
    mlflow.log_param("meta_coefficients", json.dumps(meta_model.coef_.tolist()))
    mlflow.log_param("meta_intercept", float(meta_model.intercept_))

    # ========================================================
    # Weighted blends
    # ========================================================

    blend_candidates = {
        "blend_1_balanced": {
            "ridge": 0.05,
            "elastic": 0.05,
            "lgb": 0.35,
            "cat": 0.35,
            "xgb": 0.20,
        },
        "blend_2_tree_heavy": {
            "ridge": 0.03,
            "elastic": 0.02,
            "lgb": 0.40,
            "cat": 0.40,
            "xgb": 0.15,
        },
        "blend_3_cat_heavy": {
            "ridge": 0.05,
            "elastic": 0.05,
            "lgb": 0.30,
            "cat": 0.45,
            "xgb": 0.15,
        },
        "blend_4_xgb_more": {
            "ridge": 0.05,
            "elastic": 0.05,
            "lgb": 0.30,
            "cat": 0.35,
            "xgb": 0.25,
        },
        "blend_5_linear_support": {
            "ridge": 0.10,
            "elastic": 0.10,
            "lgb": 0.30,
            "cat": 0.35,
            "xgb": 0.15,
        },
    }

    blend_results = {}

    for blend_name, w in blend_candidates.items():
        blend_oof = (
            w["ridge"] * oof_ridge +
            w["elastic"] * oof_elastic +
            w["lgb"] * oof_lgb +
            w["cat"] * oof_cat +
            w["xgb"] * oof_xgb
        )

        blend_test = (
            w["ridge"] * test_ridge +
            w["elastic"] * test_elastic +
            w["lgb"] * test_lgb +
            w["cat"] * test_cat +
            w["xgb"] * test_xgb
        )

        score = rmse_log(y, blend_oof)
        blend_results[blend_name] = {
            "score": score,
            "weights": w,
            "test_pred": blend_test
        }

        print(f"{blend_name}: {score:.6f} | {w}")
        mlflow.log_metric(f"{blend_name}_oof_rmse", score)
        mlflow.log_param(f"{blend_name}_weights", json.dumps(w))

    best_blend_name = min(blend_results, key=lambda k: blend_results[k]["score"])
    best_blend_score = blend_results[best_blend_name]["score"]
    best_blend_weights = blend_results[best_blend_name]["weights"]
    best_blend_test = blend_results[best_blend_name]["test_pred"]

    print("\n========== Best Blend ==========")
    print("Best blend:", best_blend_name)
    print("Best blend score:", best_blend_score)
    print("Best blend weights:", best_blend_weights)

    mlflow.log_param("best_blend_name", best_blend_name)
    mlflow.log_param("best_blend_weights", json.dumps(best_blend_weights))
    mlflow.log_metric("best_blend_oof_rmse", best_blend_score)

    # ========================================================
    # Save submissions
    # ========================================================

    submission_stack = pd.DataFrame({
        "Id": test_ids.astype(int),
        "SalePrice": np.expm1(test_stack)
    })

    submission_stack["SalePrice"] = submission_stack["SalePrice"].clip(lower=0)
    submission_stack.to_csv(SUBMISSION_STACK_PATH, index=False)

    submission_blend = pd.DataFrame({
        "Id": test_ids.astype(int),
        "SalePrice": np.expm1(best_blend_test)
    })

    submission_blend["SalePrice"] = submission_blend["SalePrice"].clip(lower=0)
    submission_blend.to_csv(SUBMISSION_BLEND_PATH, index=False)

    print("\nSaved stacked submission:", SUBMISSION_STACK_PATH)
    print("Saved blend submission  :", SUBMISSION_BLEND_PATH)

    display(submission_blend.head())

    # ========================================================
    # Save model bundle
    # ========================================================

    model_bundle = {
        "meta_model": meta_model,
        "best_lgb_params": best_lgb_params,
        "best_cat_params": best_cat_params,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "stacked_oof_rmse": stack_score,
        "best_blend_name": best_blend_name,
        "best_blend_oof_rmse": best_blend_score,
        "best_blend_weights": best_blend_weights,
        "ridge_oof_rmse": ridge_score,
        "elastic_oof_rmse": elastic_score,
        "xgboost_oof_rmse": xgb_score,
        "lightgbm_oof_rmse": lgb_score,
        "catboost_oof_rmse": cat_score,
    }

    with open(MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(model_bundle, f)

    mlflow.log_artifact(SUBMISSION_STACK_PATH, artifact_path="submissions")
    mlflow.log_artifact(SUBMISSION_BLEND_PATH, artifact_path="submissions")
    mlflow.log_artifact(MODEL_BUNDLE_PATH, artifact_path="model_bundle")
    mlflow.sklearn.log_model(meta_model, artifact_path="ridge_meta_model")

    print("\nMLflow logging complete.")

# COMMAND ----------

# ============================================================
# 12. Display Output Files
# ============================================================

print("Output files:")
print(os.listdir(OUTPUT_DIR))

print("\nStacked submission:")
print(SUBMISSION_STACK_PATH)

print("\nBest blend submission:")
print(SUBMISSION_BLEND_PATH)