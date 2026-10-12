"""
Run once (locally):  python make_app_data.py
Turns the big data/df_merged.csv into a compact data/df_app.parquet that the app reads.
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

COLS = ["date", "store_nbr", "family", "sales", "onpromotion", "city", "state",
        "store_type", "cluster", "dcoilwtico", "holiday_type", "locale", "locale_name"]
DTYPES = {
    "store_nbr": "int16", "onpromotion": "int32", "cluster": "int16",
    "sales": "float32", "dcoilwtico": "float32",
    "family": "category", "city": "category", "state": "category",
    "store_type": "category", "holiday_type": "category",
    "locale": "category", "locale_name": "category",
}

src = next(p for p in [os.path.join(ROOT_DIR, "data", "df_merged.csv"),
                       os.path.join(ROOT_DIR, "df_merged.csv")] if os.path.exists(p))
out = os.path.join(ROOT_DIR, "data", "df_app.parquet")

# keep_default_na=False keeps the literal string 'None' (a real category in training)
df = pd.read_csv(src, usecols=COLS, dtype=DTYPES, parse_dates=["date"], keep_default_na=False)
df.to_parquet(out, index=False, compression="zstd")

print(f"Rows: {len(df):,}")
print(f"Wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")