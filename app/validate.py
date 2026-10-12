"""
Run once:  python validate.py
Checks that the app pipeline reproduces the notebook's test metrics
(expected: R2 ~ 0.9404, MAE ~ 89.52, RMSE ~ 304.67).
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_model_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

model = joblib.load(os.path.join(MODEL_DIR, "lgbm_sales_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
model_columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
onehot_enc = joblib.load(os.path.join(MODEL_DIR, "onehot_encoder.pkl"))

print("model_columns   :", len(model_columns))
print("feature_columns :", len(feature_columns))
print("scaler expects  :", scaler.n_features_in_)
print("model expects   :", model.n_features_in_)

cols = ["date", "store_nbr", "family", "sales", "onpromotion", "city", "state",
        "store_type", "cluster", "dcoilwtico", "holiday_type", "locale", "locale_name"]
csv_path = next(p for p in [os.path.join(ROOT_DIR, "df_merged.csv"),
                            os.path.join(ROOT_DIR, "data", "df_merged.csv"),
                            "df_merged.csv"] if os.path.exists(p))
df = pd.read_csv(csv_path, usecols=cols, keep_default_na=False)
df["date"] = pd.to_datetime(df["date"])

test = df[df["date"] > df["date"].max() - pd.Timedelta(days=16)]
print("test rows:", len(test))

X = build_model_input(test, onehot_enc, scaler, feature_columns, model_columns)
pred = np.maximum(0, np.expm1(model.predict(X)))
y = test["sales"].values

print(f"R2   : {r2_score(y, pred):.4f}   (notebook: 0.9404)")
print(f"MAE  : {mean_absolute_error(y, pred):.4f}   (notebook: 89.5189)")
print(f"RMSE : {np.sqrt(mean_squared_error(y, pred)):.4f}   (notebook: 304.6664)")
