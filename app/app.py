import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import build_model_input

st.set_page_config(page_title="StockPulse", page_icon="📦", layout="wide")

try:
    import dashboard_utils as du
except ImportError as e:
    st.error(f"Missing package: {e}. Run:  pip install plotly scikit-learn")
    st.stop()

# ==========================================
# PATHS, ARTIFACTS, DATA
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

ARTIFACT_FILES = {
    "model": "lgbm_sales_model.joblib",
    "scaler": "scaler.pkl",
    "model_columns": "columns.pkl",
    "feature_columns": "feature_columns.pkl",
    "onehot_enc": "onehot_encoder.pkl",
    "store_metadata": "store_metadata.pkl",
}

USE_COLS = [
    "date", "store_nbr", "family", "sales", "onpromotion",
    "city", "state", "store_type", "cluster", "dcoilwtico",
    "holiday_type", "locale", "locale_name",
]


@st.cache_resource
def load_artifacts():
    try:
        return {k: joblib.load(os.path.join(MODEL_DIR, v)) for k, v in ARTIFACT_FILES.items()}
    except Exception as e:
        st.error(f"Artifacts load error: {e}")
        return None


@st.cache_resource(show_spinner="Loading historical data (first run only)...")
def load_historical_data():
    parquet_path = os.path.join(ROOT_DIR, "data", "df_app.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    candidates = [
        os.path.join(ROOT_DIR, "df_merged.csv"),
        os.path.join(ROOT_DIR, "data", "df_merged.csv"),
        "df_merged.csv",]

    dtypes = {
        "store_nbr": "int16", "onpromotion": "int32", "cluster": "int16",
        "sales": "float32", "dcoilwtico": "float32",
        "family": "category", "city": "category", "state": "category",
        "store_type": "category", "holiday_type": "category",
        "locale": "category", "locale_name": "category",
    }
    for path in candidates:
        if os.path.exists(path):
            # keep_default_na=False: the notebook uses the literal string 'None' for locale/locale_name
            return pd.read_csv(path, usecols=USE_COLS, dtype=dtypes, parse_dates=["date"],
                               keep_default_na=False, low_memory=True)
    return None


art = load_artifacts()
df_merged = load_historical_data()

if art is None or df_merged is None:
    st.warning("⚠️ Essential files (model artifacts or df_merged.csv) not found.")
    st.stop()

model = art["model"]
store_metadata = art["store_metadata"]
LAST_DATE = df_merged["date"].max()
TEST_START = LAST_DATE - pd.Timedelta(days=15)   # notebook: test = last 16 days


# ==========================================
# CACHED DASHBOARD DATA (computed once)
# ==========================================
@st.cache_resource(show_spinner="Preparing dashboard data...")
def get_daily(_df):
    return du.daily_totals(_df)


@st.cache_resource
def get_kpis(_df, _daily):
    return du.kpi_summary(_df, _daily)


@st.cache_resource(show_spinner="Scoring the test period...")
def get_test_results(_df, _art, test_start):
    return du.test_predictions(_df, _art, test_start)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 1: COMPANY OVERVIEW
# ==========================================
def page_overview():
    st.subheader("🏠 Company Overview")
    daily = get_daily(df_merged)
    k = get_kpis(df_merged, daily)
    st.caption(f"Data coverage: {k['first']:%Y-%m-%d} to {k['last']:%Y-%m-%d} · Corporación Favorita retail data")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total units sold", f"{k['total'] / 1e6:,.0f} M")
    c2.metric("Stores", k["stores"])
    c3.metric("Product families", k["families"])
    c4.metric("Avg daily sales (all stores)", f"{k['avg_daily']:,.0f}")
    c5.metric(f"Growth {k['last'].year} YTD vs {k['last'].year - 1}", f"{k['yoy']:+.1f}%")

    show_chart(du.fig_monthly(daily["sales"].resample("MS").mean()))

    left, right = st.columns(2)
    with left:
        show_chart(du.fig_bar(du.top_n(df_merged, "family"), "Top 10 product families"))
    with right:
        show_chart(du.fig_bar(du.top_n(df_merged, "store_nbr"), "Top 10 stores", label_prefix="Store "))

    left, right = st.columns(2)
    with left:
        show_chart(du.fig_yearly(daily))
    with right:
        show_chart(du.fig_donut(du.top_n(df_merged, "store_type", 10), "Sales share by store type"))


# ==========================================
# PAGE 2: DEMAND PATTERNS
# ==========================================
def page_patterns():
    st.subheader("📊 Demand Patterns")
    daily = get_daily(df_merged)

    dow = du.dow_profile(daily)
    best_day = du.DOW[int(dow.idxmax())]
    gap = (dow.max() / dow.mean() - 1) * 100
    st.info(f"**{best_day}** is the busiest weekday, about {gap:.0f}% above the weekly average.")

    left, right = st.columns(2)
    with left:
        show_chart(du.fig_dow(dow))
    with right:
        show_chart(du.fig_dom(du.dom_profile(daily)))

    show_chart(du.fig_heatmap(du.month_heatmap(daily)))

    left, right = st.columns(2)
    with left:
        show_chart(du.fig_promo(du.promo_effect(df_merged)))
        st.caption("Promotion data starts in 2014. This shows an association, not proof that promotions "
                   "cause the difference (stores usually promote items that already sell well).")
    with right:
        show_chart(du.fig_holiday(du.holiday_index(daily)))
        st.caption("Each day is compared with the average of the surrounding 29 days.")


# ==========================================
# PAGE 3: FORECAST (single prediction)
# ==========================================
def page_forecast():
    st.subheader("🔮 Sales Forecast")
    st.sidebar.header("🔧 Prediction Parameters")
    selected_date = st.sidebar.date_input("Select Forecast Date", value=pd.to_datetime("2017-08-15"))
    selected_store = st.sidebar.selectbox("Select Store Number", sorted(store_metadata["store_nbr"].unique()))
    selected_family = st.sidebar.selectbox("Select Product Family", sorted(df_merged["family"].unique()))
    onpromotion = st.sidebar.number_input("On Promotion Items Count", min_value=0, value=0, step=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Date", str(selected_date))
    c2.metric("Store Number", selected_store)
    c3.metric("Product Family", selected_family)
    c4.metric("Promotion Count", onpromotion)
    st.markdown("---")

    if not st.button("🚀 Run Sales Forecast", type="primary"):
        return

    with st.spinner("Running model..."):
        dt = pd.Timestamp(selected_date)
        match = df_merged[
            (df_merged["date"] == dt)
            & (df_merged["store_nbr"] == selected_store)
            & (df_merged["family"] == selected_family)
        ]
        is_historical = not match.empty

        if is_historical:
            r = match.iloc[0]
            city, state, store_type, cluster = r["city"], r["state"], r["store_type"], r["cluster"]
            dcoilwtico = r["dcoilwtico"]
            holiday_type, locale, locale_name = r["holiday_type"], r["locale"], r["locale_name"]
            actual_sales, actual_promo = float(r["sales"]), int(r["onpromotion"])
        else:
            s = store_metadata[store_metadata["store_nbr"] == selected_store].iloc[0]
            city, state, store_type, cluster = s["city"], s["state"], s["store_type"], s["cluster"]
            dcoilwtico = float(df_merged.loc[df_merged["date"] == LAST_DATE, "dcoilwtico"].iloc[0])
            holiday_type, locale, locale_name = "WorkDay", "None", "None"
            actual_sales, actual_promo = None, None

        raw_row = pd.DataFrame([{
            "date": dt, "store_nbr": int(selected_store), "family": selected_family,
            "onpromotion": int(onpromotion), "city": city, "state": state, "store_type": store_type,
            "cluster": int(cluster), "dcoilwtico": float(dcoilwtico),
            "holiday_type": holiday_type, "locale": locale, "locale_name": locale_name,
        }])
        X = build_model_input(raw_row, art["onehot_enc"], art["scaler"],
                              art["feature_columns"], art["model_columns"])
        predicted_sales = max(0.0, float(np.expm1(model.predict(X)[0])))

    st.success("🎉 Prediction Completed Successfully!")
    left, right = st.columns(2)

    with left:
        st.metric("🔮 Predicted Sales (Units)", f"{predicted_sales:,.2f}")
        st.info(f"📦 Buffer Stock (+10% margin): **{predicted_sales * 1.10:,.2f} units**")
        st.caption("The 10% margin is a fixed rule of thumb, not a statistically derived safety stock.")

    with right:
        if is_historical:
            st.write("### 📌 Historical Ground Truth Match")
            if dt >= TEST_START:
                st.caption("✅ This date is in the **test period** (not seen during training).")
            else:
                st.caption("⚠️ This date is in the **training period**, so the error looks better "
                           "than it would on new data.")
            err = abs(predicted_sales - actual_sales)
            st.write(f"**Actual Sales (Recorded):** {actual_sales:,.2f}")
            st.write(f"**Absolute Error Gap:** {err:,.2f}")
            if actual_sales > 0:
                st.write(f"**Percentage Error:** {err / actual_sales * 100:.1f}%")
            if actual_promo != int(onpromotion):
                st.warning(f"You entered {int(onpromotion)} promo items, but the recorded value was "
                           f"{actual_promo}. The comparison is not like-for-like.")
        else:
            st.write("### 🔮 Future Forecast Mode")
            st.write("No record for this date. Used store info, the last known oil price, "
                     "and no holiday information.")

    with st.expander("🔍 View Processed Features & Inputs"):
        st.dataframe(raw_row)
        st.write(f"Model features used: {len(art['model_columns'])}")
        st.dataframe(pd.DataFrame(X, columns=art["model_columns"]))


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
def page_model():
    st.subheader("🧪 Model Performance")
    t = get_test_results(df_merged, art, str(TEST_START))
    n_days = t["date"].nunique()
    st.caption(f"Scored on the held-out test period: {t['date'].min():%Y-%m-%d} to {t['date'].max():%Y-%m-%d} "
               f"({n_days} days, {len(t):,} store-family-day rows). The model never saw these days in training.")

    m = du.metrics(t["sales"], t["pred"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m['r2']:.4f}")
    c2.metric("MAE (units)", f"{m['mae']:.2f}")
    c3.metric("RMSE (units)", f"{m['rmse']:.2f}")
    c4.metric("RMSLE", f"{m['rmsle']:.4f}")

    tb = t.dropna(subset=["baseline"])
    mb = du.metrics(tb["sales"], tb["baseline"])
    mm = du.metrics(tb["sales"], tb["pred"])
    gain = (1 - mm["mae"] / mb["mae"]) * 100
    st.markdown("#### Model vs a simple benchmark")
    b1, b2, b3 = st.columns(3)
    b1.metric("Benchmark MAE (same store & product, 7 days earlier)", f"{mb['mae']:.2f}")
    b2.metric("StockPulse model MAE", f"{mm['mae']:.2f}")
    b3.metric("Error reduction vs benchmark", f"{gain:.1f}%")
    st.caption("R² is dominated by high-volume families (e.g. GROCERY I), so also check the "
               "family-level error below.")

    show_chart(du.fig_actual_vs_pred(t))

    left, right = st.columns(2)
    with left:
        show_chart(du.fig_scatter(t))
    with right:
        show_chart(du.fig_importance(du.feature_importance(model, art["model_columns"])))

    fam = du.family_errors(t)
    show_chart(du.fig_family_wape(fam))
    with st.expander("Family error table"):
        st.dataframe(fam.rename(columns={"actual": "Actual units", "abs_err": "Abs error",
                                         "mae": "MAE", "avg_sales": "Avg sales", "wape": "WAPE %"}).round(2))


# ==========================================
# PAGE 5: INVENTORY REPLENISHMENT
# ==========================================
def page_replenishment():
    st.subheader("📦 Inventory Replenishment")
    st.sidebar.header("📦 Replenishment Inputs")
    store = st.sidebar.selectbox("Store", sorted(store_metadata["store_nbr"].unique()), key="rp_store")
    family = st.sidebar.selectbox("Product family", sorted(df_merged["family"].unique()), key="rp_family")

    recent = df_merged[(df_merged["date"] > LAST_DATE - pd.Timedelta(days=14))
                       & (df_merged["store_nbr"] == store) & (df_merged["family"] == family)]
    avg_recent = float(recent["sales"].mean()) if len(recent) else 0.0
    promo_default = int(round(recent["onpromotion"].mean())) if len(recent) else 0

    stock = st.sidebar.number_input("Current stock (units)", min_value=0.0, step=10.0,
                                    value=float(round(avg_recent * 5)), key=f"rp_stock_{store}_{family}")
    lead = st.sidebar.number_input("Supplier lead time (days)", min_value=1, max_value=30, value=7, step=1)
    horizon = st.sidebar.selectbox("Planning horizon (days)", [7, 14, 30], index=1)
    service = st.sidebar.selectbox("Service level", [0.90, 0.95, 0.99], index=1,
                                   format_func=lambda v: f"{v:.0%}")
    promo = st.sidebar.number_input("Assumed items on promotion per day", min_value=0,
                                    value=promo_default, step=1, key=f"rp_promo_{store}_{family}")

    horizon = max(horizon, int(lead))
    meta = store_metadata[store_metadata["store_nbr"] == store].iloc[0]
    oil = float(df_merged.loc[df_merged["date"] == LAST_DATE, "dcoilwtico"].iloc[0])
    start = LAST_DATE + pd.Timedelta(days=1)

    frame = du.future_frame(meta, family, start, horizon, promo, oil)
    fc = du.forecast_horizon(frame, art)
    t = get_test_results(df_merged, art, str(TEST_START))
    sigma, sigma_src = du.daily_error_sigma(t, int(store), family)
    plan = du.replenishment_plan(fc, stock, lead, sigma, service)

    st.caption(f"Planning starts on {start:%Y-%m-%d}, the day after the last date in the data. "
               f"Store {store} · {family}")

    if plan["order_now"]:
        when = ""
        if plan["stockout_idx"] is not None:
            so = start + pd.Timedelta(days=plan["stockout_idx"])
            when = f" Stock is projected to run out on **{so:%Y-%m-%d}**."
        st.error(f"🔴 **Order now.** Current stock ({stock:,.0f}) is at or below the reorder point "
                 f"({plan['rop']:,.0f}).{when}")
    else:
        st.success(f"🟢 **Stock is sufficient for now.** Current stock ({stock:,.0f}) is above the "
                   f"reorder point ({plan['rop']:,.0f}).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Forecast demand ({horizon} days)", f"{plan['horizon_demand']:,.0f}")
    c2.metric(f"Demand during lead time ({plan['lead']} days)", f"{plan['lt_demand']:,.0f}")
    c3.metric(f"Safety stock ({service:.0%} service)", f"{plan['safety']:,.0f}")
    c4.metric("Reorder point", f"{plan['rop']:,.0f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Current stock", f"{stock:,.0f}")
    cover = f"{plan['stockout_idx']} days" if plan["stockout_idx"] is not None else f"> {horizon} days"
    d2.metric("Stock covers", cover)
    d3.metric(f"Suggested order (cover {horizon} days)", f"{plan['order_qty']:,.0f}")

    left, right = st.columns(2)
    with left:
        show_chart(du.fig_forecast_band(frame["date"], fc, sigma, plan["z"]))
    with right:
        show_chart(du.fig_projected_stock(frame["date"], plan["projected"], stock, plan["safety"], plan["lead"]))

    st.info(f"**Old fixed rule vs statistical rule:** a flat +10% buffer on lead-time demand would hold "
            f"{plan['old_rule'] - plan['lt_demand']:,.0f} extra units. The statistical safety stock for a "
            f"{service:.0%} service level is {plan['safety']:,.0f} units, because it follows how uncertain "
            f"this product really is.")

    with st.expander("Day-by-day forecast"):
        st.dataframe(pd.DataFrame({"date": frame["date"].dt.strftime("%Y-%m-%d"),
                                   "forecast units": fc.round(1),
                                   "projected stock": plan["projected"].round(1)}))

    with st.expander("How this is calculated, and its limits"):
        st.markdown(
            f"- **Reorder point** = demand during the lead time + safety stock.\n"
            f"- **Safety stock** = z × σ × √(lead time). With a {service:.0%} service level z = {plan['z']:.2f}. "
            f"σ = {sigma:,.1f} units/day, the model's typical daily error measured on the held-out test period "
            f"({sigma_src}).\n"
            f"- **Suggested order** = forecast demand over the horizon + safety stock − current stock.\n"
            f"- σ comes from only 16 test days, so treat it as a rough estimate.\n"
            f"- The model does not know future holidays, promotions or oil prices. The forecast assumes "
            f"normal working days, the promotion level you entered, and the last known oil price.\n"
            f"- Stock is assumed to be the full amount on hand on the first planning day, with no "
            f"orders already on the way."
        )


# ==========================================
# NAVIGATION
# ==========================================
st.title("📦 StockPulse")
st.caption("Real-Time Demand Forecasting & Inventory Replenishment Engine")

PAGES = {
    "🏠 Company Overview": page_overview,
    "📊 Demand Patterns": page_patterns,
    "🔮 Sales Forecast": page_forecast,
    "📦 Inventory Replenishment": page_replenishment,
    "🧪 Model Performance": page_model,
}
page = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.markdown("---")
PAGES[page]()
