"""
Shared feature pipeline: turns raw rows (same columns as df_merged.csv)
into the exact model input used during training in the notebook.

Used by BOTH app.py and validate.py so they can never drift apart.
"""
import numpy as np
import pandas as pd

# Same constants as the notebook
PROMO_THRESHOLD = 55.0                      # 99th percentile of onpromotion on the train set
EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-20")

REQUIRED_COLS = [
    "date", "store_nbr", "family", "onpromotion", "city", "state",
    "store_type", "cluster", "dcoilwtico", "holiday_type", "locale", "locale_name",
]


def add_time_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_mega_promo"] = (df["onpromotion"] >= PROMO_THRESHOLD).astype(int)
    df["is_earthquake_period"] = ((d >= EARTHQUAKE_START) & (d <= EARTHQUAKE_END)).astype(int)
    df["is_december"] = (df["month"] == 12).astype(int)
    return df


def build_model_input(df, onehot_enc, scaler, feature_columns, model_columns):
    """
    raw rows -> time features + flags -> saved OneHotEncoder -> training column order
    (all 116 features) -> saved StandardScaler -> keep the selected features.
    Returns a NumPy array ready for model.predict().
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    df = add_time_and_flags(df)

    cat_cols = list(onehot_enc.feature_names_in_)
    cat = pd.DataFrame(
        onehot_enc.transform(df[cat_cols]),
        columns=onehot_enc.get_feature_names_out(cat_cols),
        index=df.index,
    )

    X = pd.concat([df.drop(columns=cat_cols), cat], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0.0)   # exact training order

    n_in = scaler.n_features_in_
    if n_in == len(feature_columns):
        X_scaled = scaler.transform(X)
        idx = [feature_columns.index(c) for c in model_columns]
        return X_scaled[:, idx]
    if n_in == len(model_columns):
        return scaler.transform(X[model_columns])

    raise ValueError(
        f"Scaler expects {n_in} features, but feature_columns has {len(feature_columns)} "
        f"and model_columns has {len(model_columns)}. The saved files are from different "
        f"notebook runs. Re-run the 'Save Model Artifacts' cell."
    )
