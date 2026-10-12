"""Data preparation and Plotly charts for the StockPulse dashboard (no Streamlit code here)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_model_input

BLUE, ORANGE, GREY = "#2E6FDB", "#E4572E", "#9AA5B1"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style(fig, title, height=380):
    fig.update_layout(
        title=title, template="plotly_white", height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _not_new_year(d):
    """Drop Jan 1 (stores are closed, sales ~ 0)."""
    return d[~((d.index.month == 1) & (d.index.day == 1))]


# ================= Overview =================
def daily_totals(df):
    g = df.groupby("date", observed=True).agg(sales=("sales", "sum"), holiday_type=("holiday_type", "first"))
    g["holiday_type"] = g["holiday_type"].astype(str)
    return g


def kpi_summary(df, daily):
    last, first = daily.index.max(), daily.index.min()
    start = pd.Timestamp(year=last.year, month=1, day=1)
    cur = daily.loc[start:last, "sales"].sum()
    prev = daily.loc[start - pd.DateOffset(years=1): last - pd.DateOffset(years=1), "sales"].sum()
    return dict(
        total=float(daily["sales"].sum()), stores=int(df["store_nbr"].nunique()),
        families=int(df["family"].nunique()), avg_daily=float(daily["sales"].mean()),
        yoy=float((cur / prev - 1) * 100) if prev > 0 else float("nan"), first=first, last=last,
    )


def top_n(df, col, n=10):
    return df.groupby(col, observed=True)["sales"].sum().nlargest(n).sort_values()


def fig_monthly(monthly):
    fig = go.Figure(go.Scatter(
        x=monthly.index, y=monthly.values, mode="lines", name="Avg daily sales",
        line=dict(color=BLUE, width=2.5), fill="tozeroy", fillcolor="rgba(46,111,219,0.10)"))
    fig.add_shape(type="rect", xref="x", yref="paper", x0="2016-04-16", x1="2016-05-20",
                  y0=0, y1=1, fillcolor="rgba(228,87,46,0.18)", line_width=0)
    fig.add_annotation(x="2016-04-16", y=1, xref="x", yref="paper", text="Apr 2016 earthquake",
                       showarrow=False, xanchor="left", yanchor="top", font=dict(color=ORANGE, size=11))
    fig.update_yaxes(title="Units per day (all stores)")
    return _style(fig, "Average daily sales by month")


def fig_bar(series, title, label_prefix=""):
    labels = [f"{label_prefix}{i}" for i in series.index]
    fig = go.Figure(go.Bar(x=series.values / 1e6, y=labels, orientation="h", marker_color=BLUE,
                           texttemplate="%{x:.1f}M", textposition="outside"))
    fig.update_xaxes(title="Total units sold (millions)")
    return _style(fig, title, height=420)


def fig_yearly(daily):
    s = daily["sales"].groupby(daily.index.year).sum()
    labels = [str(y) for y in s.index]
    labels[-1] += "*"
    fig = go.Figure(go.Bar(x=labels, y=s.values / 1e6, marker_color=BLUE,
                           texttemplate="%{y:.0f}M", textposition="outside"))
    fig.update_yaxes(title="Units (millions)")
    fig = _style(fig, "Total sales by year")
    fig.add_annotation(text="* partial year", xref="paper", yref="paper", x=1, y=1.12, showarrow=False)
    return fig


def fig_donut(series, title):
    fig = go.Figure(go.Pie(labels=[str(i) for i in series.index], values=series.values, hole=0.55,
                           textinfo="label+percent"))
    return _style(fig, title, height=380)


# ================= Demand patterns =================
def dow_profile(daily):
    d = _not_new_year(daily)
    return d["sales"].groupby(d.index.dayofweek).mean().reindex(range(7))


def dom_profile(daily):
    d = _not_new_year(daily)
    return d["sales"].groupby(d.index.day).mean()


def month_heatmap(daily):
    d = _not_new_year(daily)
    return d["sales"].groupby([d.index.year, d.index.month]).mean().unstack().reindex(columns=range(1, 13))


def promo_effect(df, n=10):
    d = df[df["date"] >= "2014-04-01"]            # promotion data starts in 2014
    top = d.groupby("family", observed=True)["sales"].sum().nlargest(n).index
    d = d[d["family"].isin(top)]
    g = d.groupby(["family", d["onpromotion"] > 0], observed=True)["sales"].mean().unstack()
    g = g.rename(columns={False: "No promotion", True: "On promotion"}).dropna()
    return g.sort_values("On promotion")


def holiday_index(daily):
    """Daily sales relative to the local 29-day average (1.00 = a normal day)."""
    d = daily.copy()
    d["idx"] = d["sales"] / d["sales"].rolling(29, center=True, min_periods=15).mean()
    d = _not_new_year(d)
    return d.groupby("holiday_type")["idx"].agg(["mean", "count"]).sort_values("mean")


def fig_dow(prof):
    colors = [ORANGE if v == prof.max() else BLUE for v in prof.values]
    fig = go.Figure(go.Bar(x=DOW, y=prof.values, marker_color=colors))
    fig.update_yaxes(title="Avg units per day")
    return _style(fig, "Average sales by day of week")


def fig_dom(prof):
    fig = go.Figure(go.Scatter(x=prof.index, y=prof.values, mode="lines+markers", line=dict(color=BLUE)))
    fig.update_xaxes(title="Day of month")
    fig.update_yaxes(title="Avg units per day")
    return _style(fig, "Average sales by day of month")


def fig_heatmap(pivot):
    fig = go.Figure(go.Heatmap(z=pivot.values, x=MONTHS, y=[str(y) for y in pivot.index],
                               colorscale="Blues", colorbar=dict(title="Avg/day")))
    return _style(fig, "Seasonality: average daily sales by year and month", height=360)


def fig_promo(g):
    fig = go.Figure()
    fig.add_bar(y=g.index.astype(str), x=g["No promotion"], name="No promotion", orientation="h", marker_color=GREY)
    fig.add_bar(y=g.index.astype(str), x=g["On promotion"], name="On promotion", orientation="h", marker_color=BLUE)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Avg units per store-day")
    return _style(fig, "Average sales per store-day: promotion vs no promotion", height=460)


def fig_holiday(h):
    pct = (h["mean"] - 1) * 100
    fig = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h",
                           marker_color=[ORANGE if v < 0 else BLUE for v in pct.values],
                           texttemplate="%{x:+.1f}%", textposition="outside"))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_xaxes(title="% vs normal days (local 29-day average)")
    return _style(fig, "Holiday effect on total daily sales", height=340)


# ================= Model performance =================
def test_predictions(df, art, test_start):
    """Predict the test period with the SAME pipeline as the app, plus a 7-day-ago baseline."""
    test_start = pd.Timestamp(test_start)
    t = df[df["date"] >= test_start].copy()
    for c in list(art["onehot_enc"].feature_names_in_):
        t[c] = t[c].astype(str)
    X = build_model_input(t, art["onehot_enc"], art["scaler"], art["feature_columns"], art["model_columns"])
    t["pred"] = np.maximum(0, np.expm1(art["model"].predict(X)))

    last = df["date"].max()
    prev = df.loc[(df["date"] >= test_start - pd.Timedelta(days=7)) & (df["date"] <= last - pd.Timedelta(days=7)),
                  ["date", "store_nbr", "family", "sales"]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=7)
    prev["family"] = prev["family"].astype(str)
    prev = prev.rename(columns={"sales": "baseline"})
    t["family"] = t["family"].astype(str)
    return t.merge(prev, on=["date", "store_nbr", "family"], how="left")


def metrics(y, p):
    return dict(
        r2=r2_score(y, p), mae=mean_absolute_error(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
        rmsle=float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
    )


def family_errors(t):
    e = t.assign(abs_err=(t["sales"] - t["pred"]).abs())
    g = e.groupby("family").agg(actual=("sales", "sum"), abs_err=("abs_err", "sum"),
                                mae=("abs_err", "mean"), avg_sales=("sales", "mean"))
    g["wape"] = g["abs_err"] / g["actual"] * 100
    return g.replace([np.inf, -np.inf], np.nan).dropna(subset=["wape"]).sort_values("wape")


def feature_importance(model, names, n=15):
    try:
        imp = model.booster_.feature_importance(importance_type="gain")
    except Exception:
        imp = model.feature_importances_
    return pd.Series(imp, index=names).nlargest(n).sort_values()


def fig_actual_vs_pred(t):
    d = t.groupby("date")[["sales", "pred"]].sum()
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d["sales"], name="Actual", mode="lines+markers", line=dict(color=BLUE))
    fig.add_scatter(x=d.index, y=d["pred"], name="Predicted", mode="lines+markers",
                    line=dict(color=ORANGE, dash="dash"))
    fig.update_yaxes(title="Total units (all stores, all families)")
    return _style(fig, "Test period: actual vs predicted total daily sales")


def fig_scatter(t, n=4000):
    s = t.sample(min(n, len(t)), random_state=42)
    x, y = np.log1p(s["sales"]), np.log1p(s["pred"])
    m = float(max(x.max(), y.max()))
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", marker=dict(color=BLUE, size=4, opacity=0.35), name="Rows")
    fig.add_scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=ORANGE, dash="dash"), name="Perfect")
    fig.update_xaxes(title="Actual, log(1 + units)")
    fig.update_yaxes(title="Predicted, log(1 + units)")
    return _style(fig, "Actual vs predicted per store-family-day (sample)")


def fig_family_wape(g):
    fig = go.Figure(go.Bar(x=g["wape"], y=g.index, orientation="h", marker_color=BLUE,
                           texttemplate="%{x:.0f}%", textposition="outside"))
    fig.update_xaxes(title="WAPE = total absolute error / total actual sales (%)")
    return _style(fig, "Error by product family (lower is better)", height=760)


def fig_importance(imp):
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE))
    fig.update_xaxes(title="Importance (gain)")
    return _style(fig, "Top features used by the model", height=460)


# ================= Inventory replenishment =================
from scipy.stats import norm  # noqa: E402  (scipy ships with scikit-learn)


def future_frame(meta_row, family, start, days, promo, oil):
    """Raw rows for the next `days` days. No holiday information is known for the future."""
    return pd.DataFrame({
        "date": pd.date_range(start, periods=days),
        "store_nbr": int(meta_row["store_nbr"]), "family": family, "onpromotion": int(promo),
        "city": meta_row["city"], "state": meta_row["state"], "store_type": meta_row["store_type"],
        "cluster": int(meta_row["cluster"]), "dcoilwtico": float(oil),
        "holiday_type": "WorkDay", "locale": "None", "locale_name": "None",
    })


def forecast_horizon(frame, art):
    X = build_model_input(frame, art["onehot_enc"], art["scaler"], art["feature_columns"], art["model_columns"])
    return np.maximum(0, np.expm1(art["model"].predict(X)))


def daily_error_sigma(t, store, family):
    """Typical daily forecast error (RMSE) for this store+family on the held-out test period."""
    s = t[(t["store_nbr"] == store) & (t["family"] == family)]
    source = "this store and product"
    if len(s) < 10:
        s, source = t[t["family"] == family], "this product family (all stores)"
    if s.empty:
        return 0.0, "no data"
    return float(np.sqrt(np.mean((s["sales"] - s["pred"]) ** 2))), source


def replenishment_plan(forecast, stock, lead_days, sigma, service_level):
    """Reorder point = demand during lead time + safety stock (z * sigma * sqrt(lead time))."""
    fc = np.asarray(forecast, dtype=float)
    z = float(norm.ppf(service_level))
    lead = int(min(lead_days, len(fc)))
    lt_demand = float(fc[:lead].sum())
    safety = z * sigma * np.sqrt(lead)
    rop = lt_demand + safety
    cum = fc.cumsum()
    projected = stock - cum
    out = np.where(projected < 0)[0]
    return dict(
        z=z, lead=lead, lt_demand=lt_demand, safety=safety, rop=rop,
        horizon_demand=float(fc.sum()), projected=projected,
        order_qty=max(0.0, float(fc.sum()) + safety - stock),
        stockout_idx=int(out[0]) if len(out) else None,
        order_now=bool(rop > 0 and stock <= rop),
        old_rule=lt_demand * 1.10,
    )


def fig_forecast_band(dates, fc, sigma, z):
    upper = fc + z * sigma
    lower = np.maximum(0, fc - z * sigma)
    fig = go.Figure()
    fig.add_scatter(x=dates, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=dates, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(46,111,219,0.15)", name="Likely range")
    fig.add_scatter(x=dates, y=fc, mode="lines+markers", name="Forecast", line=dict(color=BLUE, width=2.5))
    fig.update_yaxes(title="Units per day")
    return _style(fig, "Forecast demand with uncertainty band")


def fig_projected_stock(dates, projected, stock, safety, lead):
    x = [dates[0] - pd.Timedelta(days=1)] + list(dates)
    y = [stock] + list(projected)
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines+markers", name="Projected stock", line=dict(color=BLUE, width=2.5))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_hline(y=safety, line_dash="dash", line_color=ORANGE, annotation_text="Safety stock level",
                  annotation_position="top left")
    end = str((dates[0] + pd.Timedelta(days=lead - 1)).date())
    fig.add_shape(type="line", xref="x", yref="paper", x0=end, x1=end, y0=0, y1=1,
                  line=dict(color=GREY, dash="dot"))
    fig.add_annotation(x=end, y=1, xref="x", yref="paper", text="Lead time ends", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=GREY, size=11))
    fig.update_yaxes(title="Units in stock if nothing is ordered")
    return _style(fig, "Projected stock level without a new order")
