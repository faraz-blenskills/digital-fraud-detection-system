"""The nine pages of the Fraud Intelligence dashboard.  (ADDED)"""

import datetime as dt
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from common import (
    ACCENT, ACCENT2, BORDER, MUTED, RISK_COLORS, RISK_ORDER, TYPES,
    build_row, explain_row, kpi, load_all, load_model, load_scored, money, note, pill, risk_level,
)

REGION_XY = {  # centroid of each (simulated) region, for the bubble map
    "North America": (40, -100), "Latin America": (-15, -60), "Western Europe": (48, 8),
    "Eastern Europe": (52, 30), "Middle East": (29, 45), "Africa": (2, 20),
    "South Asia": (22, 79), "East Asia": (35, 112), "Oceania": (-25, 135),
}
SCORED_COLS = ["tx_id", "day", "hour", "type", "amount", "risk_score", "risk_level", "anomaly_score", "isFraud"]


# ---------------- cached aggregates ----------------
@st.cache_data
def daily_summary():
    df = load_all()
    d = df.assign(fa=df["amount"].astype("float64") * df["isFraud"])
    g = d.groupby("day", observed=True).agg(tx=("amount", "size"), fraud=("isFraud", "sum"), fraud_amt=("fa", "sum"))
    return g.reset_index()


@st.cache_data
def type_summary():
    df = load_all()
    g = df.groupby("type", observed=True).agg(tx=("amount", "size"), fraud=("isFraud", "sum"),
                                              avg_amount=("amount", "mean"))
    g["fraud_rate_%"] = g["fraud"] / g["tx"] * 100
    return g.reset_index()


@st.cache_data
def hour_summary():
    df = load_all()
    g = df.groupby("hour", observed=True).agg(tx=("amount", "size"), fraud=("isFraud", "sum")).reset_index()
    g["fraud_rate_%"] = g["fraud"] / g["tx"] * 100
    return g


@st.cache_data
def amount_sorted_by_type():
    df = load_all()
    return {t: np.sort(df.loc[df["type"] == t, "amount"].to_numpy()) for t in TYPES}


@st.cache_data
def pr_curve():
    sc = load_scored()
    p, r, t = precision_recall_curve(sc["isFraud"], sc["risk_score"])
    idx = np.unique(np.linspace(0, len(p) - 1, 1500).astype(int))
    return p[idx], r[idx], average_precision_score(sc["isFraud"], sc["risk_score"]), roc_auc_score(sc["isFraud"], sc["risk_score"])


def _table(df, **kw):
    st.dataframe(df, use_container_width=True, hide_index=True, **kw)


# =====================================================================
# 1. Executive Overview
# =====================================================================
def executive_overview():
    st.title("Executive Overview")
    st.caption("PaySim mobile-money transactions, 744 simulated hours (31 days).")
    df, sc = load_all(), load_scored()
    tot, fr = len(df), int(df["isFraud"].sum())
    loss = float(df.loc[df["isFraud"] == 1, "amount"].sum())

    c = st.columns(4)
    kpi(c[0], "Transactions", f"{tot:,}", f"value {money(float(df['amount'].sum()))}")
    kpi(c[1], "Fraud cases", f"{fr:,}", f"rate {fr / tot * 100:.3f}%  (about 1 in {tot // fr:,})")
    kpi(c[2], "Financial loss", money(loss), f"avg {money(loss / fr)} per fraud")
    p, r, ap, auc = pr_curve()
    kpi(c[3], "Model PR-AUC", f"{ap:.2f}", f"ROC-AUC {auc:.2f} on last 7 days")

    st.write("")
    a, b = st.columns([3, 2])
    d = daily_summary()
    with a:
        fig = go.Figure()
        fig.add_bar(x=d["day"], y=d["tx"], name="Transactions", marker_color=BORDER, yaxis="y")
        fig.add_scatter(x=d["day"], y=d["fraud"], name="Fraud cases", line=dict(color=RISK_COLORS["Critical"], width=2.5), yaxis="y2")
        fig.update_layout(title="Daily volume vs fraud cases", xaxis_title="Day", height=360,
                          yaxis=dict(title="Transactions"), yaxis2=dict(title="Fraud cases", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        t = type_summary()
        fig = px.bar(t, x="type", y="fraud", color="type", title="Fraud cases by transaction type", text="fraud")
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud loss over time")
    fig = px.area(d, x="day", y="fraud_amt", labels={"fraud_amt": "Fraud value", "day": "Day"})
    fig.update_traces(line_color=RISK_COLORS["Critical"], fillcolor="rgba(244,63,94,.25)")
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)

    hi = sc[sc["risk_level"].isin(["High", "Critical"])]
    note(f"<b>Last 7 days (model holdout):</b> {len(sc):,} transactions, {int(sc['isFraud'].sum()):,} actual frauds. "
         f"High + Critical alerts: <b>{len(hi):,}</b>, capturing <b>{int(hi['isFraud'].sum()):,}</b> "
         f"({hi['isFraud'].sum() / sc['isFraud'].sum() * 100:.0f}%) of the frauds.")


# =====================================================================
# 2. Transaction Analytics
# =====================================================================
def transaction_analytics():
    st.title("Transaction Analytics")
    df = load_all()
    f1, f2 = st.columns([2, 2])
    sel = f1.multiselect("Transaction types", TYPES, default=TYPES)
    d0, d1 = f2.slider("Day range", 0, 30, (0, 30))
    m = df[df["type"].isin(sel) & df["day"].between(d0, d1)]
    if m.empty:
        st.warning("No transactions match the filters.")
        return

    c = st.columns(4)
    kpi(c[0], "Transactions", f"{len(m):,}")
    kpi(c[1], "Total value", money(float(m["amount"].sum())))
    kpi(c[2], "Median amount", f"{m['amount'].median():,.0f}", f"mean {m['amount'].mean():,.0f}")
    kpi(c[3], "99th percentile", money(float(m["amount"].quantile(0.99))))
    st.write("")

    a, b = st.columns(2)
    with a:
        h = m.groupby("hour", observed=True).size().reset_index(name="tx")
        fig = px.bar(h, x="hour", y="tx", title="Volume by hour of day")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        dt_ = m.groupby(["day", "type"], observed=True).size().reset_index(name="tx")
        fig = px.area(dt_, x="day", y="tx", color="type", title="Daily volume by type")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        s = m.sample(min(len(m), 300_000), random_state=1)
        s = s.assign(log10_amount=np.log10(s["amount"].clip(lower=1)), label=np.where(s["isFraud"] == 1, "Fraud", "Legitimate"))
        fig = px.histogram(s, x="log10_amount", color="label", nbins=60, barmode="overlay", histnorm="probability",
                           title="Amount distribution (log10), normalised per class",
                           color_discrete_map={"Fraud": RISK_COLORS["Critical"], "Legitimate": ACCENT})
        fig.update_traces(opacity=0.65)
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        t = m.groupby("type", observed=True).agg(tx=("amount", "size"), value=("amount", "sum"), median=("amount", "median"),
                                                 fraud=("isFraud", "sum")).reset_index()
        fig = px.pie(t, names="type", values="value", hole=0.55, title="Share of transaction value")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    _table(t.rename(columns={"tx": "transactions", "value": "total value", "median": "median amount", "fraud": "fraud cases"}))


# =====================================================================
# 3. Fraud Detection
# =====================================================================
def fraud_detection():
    st.title("Fraud Detection")
    note("Scored on the <b>last 7 days</b> only (the model never trained on them), so these numbers are honest "
         "out-of-time results. The model deliberately excludes account-balance columns (see README).")
    sc = load_scored()
    thr = st.slider("Decision threshold (flag if risk score >=)", 0.01, 0.99, 0.50, 0.01)
    y, pred = sc["isFraud"].to_numpy(), (sc["risk_score"].to_numpy() >= thr)
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    c = st.columns(4)
    kpi(c[0], "Flagged", f"{tp + fp:,}", f"{(tp + fp) / len(sc) * 100:.1f}% of transactions")
    kpi(c[1], "Precision", f"{prec * 100:.1f}%", "flagged that are fraud")
    kpi(c[2], "Recall", f"{rec * 100:.1f}%", f"{tp:,} of {tp + fn:,} frauds caught")
    kpi(c[3], "F1", f"{f1:.3f}")
    st.write("")

    a, b = st.columns(2)
    with a:
        p, r, ap, _ = pr_curve()
        fig = go.Figure()
        fig.add_scatter(x=r, y=p, mode="lines", line=dict(color=ACCENT, width=2.5), name="PR curve")
        fig.add_scatter(x=[rec], y=[prec], mode="markers", marker=dict(size=13, color=RISK_COLORS["Critical"]), name="Your threshold")
        fig.add_hline(y=y.mean(), line_dash="dot", line_color=MUTED, annotation_text="random classifier")
        fig.update_layout(title=f"Precision-recall curve (PR-AUC {ap:.2f})", xaxis_title="Recall", yaxis_title="Precision", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(cm, x=["Predicted legit", "Predicted fraud"], y=["Actually legit", "Actually fraud"],
                        text_auto=",", color_continuous_scale=[[0, "#121a30"], [1, ACCENT2]], title="Confusion matrix")
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged transactions (highest risk first)")
    flagged = sc[sc["risk_score"] >= thr].sort_values("risk_score", ascending=False).head(300)
    _table(flagged[SCORED_COLS].rename(columns={"isFraud": "actual_fraud"}),
           column_config={"risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=1, format="%.3f"),
                          "anomaly_score": st.column_config.NumberColumn(format="%.2f"),
                          "amount": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Showing up to 300 rows. Use the Investigation Assistant page to see why a transaction was flagged.")


# =====================================================================
# 4. Risk Intelligence
# =====================================================================
def risk_intelligence():
    st.title("Risk Intelligence")
    sc = load_scored()
    note("Risk score = model fraud probability. Levels: <b>Low</b> &lt;5% · <b>Medium</b> 5-20% · <b>High</b> 20-50% · <b>Critical</b> &ge;50%.")
    g = sc.groupby("risk_level").agg(tx=("tx_id", "size"), fraud=("isFraud", "sum"), value=("amount", "sum")).reindex(RISK_ORDER)
    g["fraud_rate_%"] = g["fraud"] / g["tx"] * 100
    c = st.columns(4)
    for col, lv in zip(c, RISK_ORDER):
        kpi(col, f"{lv} risk", f"{int(g.loc[lv, 'tx']):,}", f"{int(g.loc[lv, 'fraud'])} frauds · {g.loc[lv, 'fraud_rate_%']:.2f}% rate")
    st.write("")

    a, b = st.columns(2)
    with a:
        fig = px.bar(g.reset_index(), x="risk_level", y="tx", color="risk_level", color_discrete_map=RISK_COLORS,
                     title="Transactions per risk level", text="tx", category_orders={"risk_level": RISK_ORDER})
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(g.reset_index(), x="risk_level", y="fraud_rate_%", color="risk_level", color_discrete_map=RISK_COLORS,
                     title="Actual fraud rate per risk level (%)", category_orders={"risk_level": RISK_ORDER})
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    hm = sc.pivot_table(index="type", columns="hour", values="risk_score", aggfunc="mean")
    fig = px.imshow(hm, aspect="auto", color_continuous_scale=[[0, "#121a30"], [0.5, ACCENT2], [1, RISK_COLORS["Critical"]]],
                    labels=dict(color="Avg risk"), title="Average risk score by transaction type and hour")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk transactions")
    _table(sc.sort_values("risk_score", ascending=False).head(50)[SCORED_COLS].rename(columns={"isFraud": "actual_fraud"}),
           column_config={"risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=1, format="%.3f")})


# =====================================================================
# 5. Geographic Intelligence
# =====================================================================
def geographic_intelligence():
    st.title("Geographic Intelligence")
    note("<b>PaySim has no location data.</b> Regions here are <b>simulated</b>: each sender account id is deterministically "
         "mapped to one of 9 regions. Fraud is not geographically concentrated by construction, so differences between "
         "regions are random noise. With a real dataset, replace this mapping with a real location field.", warn=True)
    sc = load_scored()
    g = sc.groupby("region").agg(tx=("tx_id", "size"), fraud=("isFraud", "sum"), avg_risk=("risk_score", "mean"),
                                 value=("amount", "sum")).reset_index()
    g["flagged"] = g["region"].map(sc[sc["risk_level"].isin(["High", "Critical"])].groupby("region").size()).fillna(0).astype(int)
    g["fraud_rate_%"] = g["fraud"] / g["tx"] * 100
    g["lat"] = g["region"].map(lambda r: REGION_XY[r][0])
    g["lon"] = g["region"].map(lambda r: REGION_XY[r][1])

    metric = st.radio("Bubble size", ["flagged", "fraud", "tx"], horizontal=True,
                      format_func=lambda m: {"flagged": "High+Critical alerts", "fraud": "Actual frauds", "tx": "Transactions"}[m])
    fig = px.scatter_geo(g, lat="lat", lon="lon", size=metric, color="avg_risk", hover_name="region",
                         hover_data={"lat": False, "lon": False, "tx": ":,", "fraud": True, "avg_risk": ":.3f"},
                         color_continuous_scale=[[0, ACCENT], [1, RISK_COLORS["Critical"]]], size_max=55, projection="natural earth")
    fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#16213d", oceancolor="#0b1020", showocean=True, showcountries=True,
                    countrycolor=BORDER, coastlinecolor=BORDER, showframe=False)
    fig.update_layout(height=460, coloraxis_colorbar=dict(title="Avg risk"))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(g.sort_values("fraud_rate_%", ascending=False), x="region", y="fraud_rate_%", title="Fraud rate by region (%) - simulated regions")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    _table(g.drop(columns=["lat", "lon"]).sort_values("flagged", ascending=False))


# =====================================================================
# 6. Anomaly Detection
# =====================================================================
def anomaly_detection():
    st.title("Anomaly Detection")
    note("An <b>Isolation Forest</b> is fitted on normal-looking historical transactions (amount, hour, type; <b>no fraud labels</b>) "
         "and scores how unusual each holdout transaction is (0 = normal, 1 = very unusual). It is independent of the supervised model.")
    sc = load_scored()
    thr = st.slider("Anomaly score threshold", 0.30, 0.99, 0.70, 0.01)
    an = sc[sc["anomaly_score"] >= thr]
    base = sc["isFraud"].mean()
    c = st.columns(3)
    kpi(c[0], "Anomalous transactions", f"{len(an):,}", f"{len(an) / len(sc) * 100:.2f}% of holdout")
    rate = an["isFraud"].mean() if len(an) else 0
    kpi(c[1], "Fraud rate among anomalies", f"{rate * 100:.2f}%", f"baseline {base * 100:.2f}%")
    kpi(c[2], "Lift", f"{rate / base:.1f}x" if base else "-", "vs picking at random")
    st.write("")

    a, b = st.columns(2)
    with a:
        s = sc.sample(min(len(sc), 25_000), random_state=3).assign(label=lambda d: np.where(d["isFraud"] == 1, "Fraud", "Legitimate"))
        s = pd.concat([s, sc[sc["isFraud"] == 1].sample(min(600, int(sc["isFraud"].sum())), random_state=3).assign(label="Fraud")])
        fig = px.scatter(s, x="amount", y="anomaly_score", color="label", log_x=True, opacity=0.5,
                         color_discrete_map={"Fraud": RISK_COLORS["Critical"], "Legitimate": ACCENT}, title="Amount vs anomaly score")
        fig.add_hline(y=thr, line_dash="dot", line_color=MUTED)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.histogram(sc, x="anomaly_score", nbins=50, color=sc["type"], title="Anomaly score distribution by type")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most unusual transactions")
    _table(sc.sort_values("anomaly_score", ascending=False).head(100)[SCORED_COLS].rename(columns={"isFraud": "actual_fraud"}),
           column_config={"anomaly_score": st.column_config.ProgressColumn("anomaly_score", min_value=0, max_value=1, format="%.3f")})


# =====================================================================
# 7. Reports & Insights
# =====================================================================
def _insights():
    df, sc = load_all(), load_scored()
    t, h = type_summary(), hour_summary()
    fraud_types = t[t["fraud"] > 0]
    hi = sc[sc["risk_level"].isin(["High", "Critical"])]
    worst_hour = h.sort_values("fraud_rate_%", ascending=False).iloc[0]
    p, r, ap, auc = pr_curve()
    out = [
        f"Fraud occurs only in {' and '.join(fraud_types['type'].astype(str))}; PAYMENT, CASH_IN and DEBIT had zero fraud.",
        f"Overall fraud rate is {df['isFraud'].mean() * 100:.3f}%, a severe class imbalance (about 1 in {int(1 / df['isFraud'].mean()):,}).",
        f"Fraud transactions are larger: median fraud amount {df.loc[df['isFraud'] == 1, 'amount'].median():,.0f} "
        f"vs {df.loc[df['isFraud'] == 0, 'amount'].median():,.0f} for legitimate.",
        f"Hour {int(worst_hour['hour']):02d}:00 has the highest fraud rate ({worst_hour['fraud_rate_%']:.2f}%). "
        "In PaySim this partly reflects low overall volume at night, not just more fraud.",
        f"On the 7-day holdout, High + Critical alerts ({len(hi):,} transactions, {len(hi) / len(sc) * 100:.1f}%) capture "
        f"{hi['isFraud'].sum() / sc['isFraud'].sum() * 100:.0f}% of frauds; precision is low ({hi['isFraud'].mean() * 100:.1f}%), "
        "so alerts should be triaged, not auto-blocked.",
        f"Model quality: ROC-AUC {auc:.2f}, PR-AUC {ap:.2f} (random would be about {sc['isFraud'].mean():.3f}).",
    ]
    return out, hi


def build_report_html() -> str:
    df, sc = load_all(), load_scored()
    ins, hi = _insights()
    t = type_summary()
    esc = html.escape
    rows = "".join(f"<tr><td>{esc(str(r.type))}</td><td>{r.tx:,}</td><td>{r.fraud:,}</td><td>{r.fraud_rate:.3f}%</td></tr>"
                   for r in t.rename(columns={"fraud_rate_%": "fraud_rate"}).itertuples())
    lv = sc.groupby("risk_level").agg(tx=("tx_id", "size"), fraud=("isFraud", "sum")).reindex(RISK_ORDER)
    lv_rows = "".join(f"<tr><td>{k}</td><td>{int(v.tx):,}</td><td>{int(v.fraud):,}</td><td>{v.fraud / v.tx * 100:.2f}%</td></tr>"
                      for k, v in lv.iterrows())
    top = sc.sort_values("risk_score", ascending=False).head(25)
    top_rows = "".join(
        f"<tr><td>{r.tx_id}</td><td>{r.type}</td><td>{r.amount:,.0f}</td><td>{r.day:.0f} / {r.hour:02.0f}:00</td>"
        f"<td>{r.risk_score:.3f}</td><td>{r.risk_level}</td><td>{'Yes' if r.isFraud else 'No'}</td></tr>" for r in top.itertuples())
    fig = px.bar(type_summary(), x="type", y="fraud", title="Fraud cases by type")
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Fraud Intelligence Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:32px auto;color:#1b2236;padding:0 16px}}
h1{{margin-bottom:0}} .sub{{color:#667}} table{{border-collapse:collapse;width:100%;margin:10px 0 24px}}
th,td{{border:1px solid #d9deea;padding:6px 10px;text-align:left;font-size:14px}} th{{background:#eef1f8}}
.k{{display:inline-block;margin:0 24px 12px 0}} .k b{{font-size:22px;display:block}}</style></head><body>
<h1>Digital Fraud Intelligence Report</h1><div class="sub">Generated {dt.datetime.now():%Y-%m-%d %H:%M} · PaySim dataset (synthetic)</div>
<h2>Headline numbers</h2>
<div class="k"><b>{len(df):,}</b>transactions</div><div class="k"><b>{int(df['isFraud'].sum()):,}</b>fraud cases</div>
<div class="k"><b>{df['isFraud'].mean() * 100:.3f}%</b>fraud rate</div>
<div class="k"><b>{money(float(df.loc[df['isFraud'] == 1, 'amount'].sum()))}</b>fraud value</div>
<h2>Key findings</h2><ul>{''.join(f'<li>{esc(i)}</li>' for i in ins)}</ul>
<h2>By transaction type</h2><table><tr><th>Type</th><th>Transactions</th><th>Fraud</th><th>Fraud rate</th></tr>{rows}</table>
{chart}
<h2>Risk levels (7-day holdout)</h2><table><tr><th>Level</th><th>Transactions</th><th>Actual fraud</th><th>Fraud rate</th></tr>{lv_rows}</table>
<h2>25 highest-risk transactions</h2><table><tr><th>ID</th><th>Type</th><th>Amount</th><th>Day / hour</th><th>Risk</th><th>Level</th><th>Actual fraud</th></tr>{top_rows}</table>
<h2>Limitations</h2><ul><li>PaySim is synthetic; results do not transfer directly to real banks.</li>
<li>Account-balance columns are excluded on purpose because they leak the label in this simulator.</li>
<li>Regions in the geographic module are simulated.</li></ul>
<p class="sub">Base repository: rayenlatrech/fraud-detection-paysim (MIT). Dashboard modules, scoring, anomaly detection and this report are additions.</p>
</body></html>"""


def reports_insights():
    st.title("Reports & Insights")
    ins, hi = _insights()
    st.subheader("Key findings")
    for i in ins:
        st.markdown(f"- {i}")

    sc = load_scored()
    st.subheader("Download")
    a, b = st.columns(2)
    a.download_button("Download Fraud Intelligence Report (HTML)", build_report_html(), "fraud_intelligence_report.html",
                      "text/html", use_container_width=True)
    csv = hi.sort_values("risk_score", ascending=False)[SCORED_COLS + ["nameOrig", "nameDest", "region"]].to_csv(index=False)
    b.download_button(f"Download {len(hi):,} High/Critical alerts (CSV)", csv, "high_critical_alerts.csv", "text/csv", use_container_width=True)
    note("The HTML report opens in any browser and can be printed to PDF (Ctrl+P). It includes headline numbers, findings, "
         "risk-level table, top-25 risky transactions and limitations.")


# =====================================================================
# 8. Fraud Investigation Assistant
# =====================================================================
def _reasons(row, X, contrib):
    """Plain-English reasons built from SHAP + simple statistics. Rule-based, not an LLM."""
    sorted_amt = amount_sorted_by_type()
    hs = hour_summary().set_index("hour")
    overall = load_all()["isFraud"].mean() * 100
    out = []
    for r in contrib.head(5).itertuples():
        if abs(r.shap) < 0.05:
            continue
        up = r.shap > 0
        verb = "raises" if up else "lowers"
        if r.feature.startswith("type_") and r.value == 1:
            ty = r.feature[5:]
            extra = ("fraud in this dataset only ever occurs in TRANSFER and CASH_OUT" if ty in ("TRANSFER", "CASH_OUT")
                     else "this type has no recorded fraud in the data")
            out.append((up, f"Type **{ty}** {verb} risk: {extra}."))
        elif r.feature in ("amount", "log_amount"):
            if any("percentile" in t for _, t in out):
                continue  # amount and log_amount describe the same thing; explain once
            arr = sorted_amt[row["type"]]
            pct = np.searchsorted(arr, row["amount"]) / len(arr) * 100
            out.append((up, f"Amount **{row['amount']:,.0f}** is at the **{pct:.0f}th percentile** of {row['type']} transactions, which {verb} risk."))
        elif r.feature == "hour":
            out.append((up, f"Hour **{int(row['hour']):02d}:00** {verb} risk (fraud rate at this hour {hs.loc[int(row['hour']), 'fraud_rate_%']:.2f}% vs {overall:.2f}% overall)."))
        elif r.feature in ("step", "day"):
            out.append((up, f"Timing in the simulation (**{r.feature} = {r.value:.0f}**) {verb} risk. This is a weak, simulator-specific signal."))
    if row["anomaly_score"] >= 0.7:
        out.append((True, f"Unusual pattern: anomaly score **{row['anomaly_score']:.2f}** (independent Isolation Forest)."))
    return out


ACTIONS = {
    "Low": "No action. Routine monitoring only.",
    "Medium": "Passive monitoring. Review if the sender shows further activity.",
    "High": "Queue for analyst review within the day; consider step-up authentication.",
    "Critical": "Hold the transaction and escalate for immediate analyst review.",
}


def investigation_assistant():
    st.title("Fraud Investigation Assistant")
    note("Explains <b>why</b> a transaction was flagged using SHAP contributions plus simple statistics. "
         "The text is generated by rules, not by a language model.")
    sc = load_scored()
    model, _ = load_model()

    src = st.radio("Choose transaction", ["Top alerts", "By transaction ID"], horizontal=True)
    if src == "Top alerts":
        pool = sc.sort_values("risk_score", ascending=False).head(200)
        labels = {int(r.tx_id): f"#{int(r.tx_id)} · {r.type} · {r.amount:,.0f} · risk {r.risk_score:.2f}" for r in pool.itertuples()}
        tx_id = st.selectbox("Alert", list(labels), format_func=labels.get)
    else:
        tx_id = int(st.number_input("Transaction ID", 0, len(sc) - 1, int(sc.sort_values("risk_score").index[-1])))
    row = sc.iloc[int(tx_id)]

    X = build_row(row["type"], row["amount"], int(row["hour"]), int(row["day"]))
    contrib = explain_row(X)
    lvl = row["risk_level"]

    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"### Transaction #{int(row['tx_id'])}  " + pill(lvl), unsafe_allow_html=True)
        st.markdown(f"**Type** {row['type']}  \n**Amount** {row['amount']:,.2f}  \n**When** day {int(row['day'])}, {int(row['hour']):02d}:00  \n"
                    f"**Risk score** {row['risk_score']:.4f}  \n**Anomaly score** {row['anomaly_score']:.2f}  \n"
                    f"**Sender** `{row['nameOrig']}` → **Receiver** `{row['nameDest']}`")
        same = int((sc["nameOrig"] == row["nameOrig"]).sum())
        st.caption(f"Sender appears in {same} holdout transaction(s). Ground truth (demo only): "
                   f"{'FRAUD' if row['isFraud'] else 'legitimate'}.")
        st.markdown(f"**Suggested action:** {ACTIONS[lvl]}")
    with b:
        c = contrib.sort_values("shap")
        fig = go.Figure(go.Bar(x=c["shap"], y=c["feature"], orientation="h",
                               marker_color=np.where(c["shap"] > 0, RISK_COLORS["Critical"], ACCENT)))
        fig.update_layout(title="What pushed the score (red = toward fraud)", height=340, xaxis_title="SHAP value (log-odds)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why it was flagged" if row["risk_score"] >= 0.2 else "Why the score is what it is")
    reasons = _reasons(row, X, contrib)
    if not reasons:
        st.write("No single attribute stands out; the score is driven by many small contributions.")
    for up, text in reasons:
        st.markdown(f"{'🔴' if up else '🟢'} {text}")


# =====================================================================
# 9. What-if Simulator
# =====================================================================
def whatif_simulator():
    st.title("Transaction What-if Simulator")
    note("Change a transaction's type, amount or time and watch the risk score react. Remember the model only learned "
         "from transaction attributes (balances were excluded), so it reacts mainly to type and amount.")
    model, _ = load_model()
    all_ = load_all()
    max_amt = float(all_["amount"].quantile(0.999))

    c = st.columns(4)
    ty = c[0].selectbox("Type", TYPES, index=1)
    amt = c[1].number_input("Amount", 1.0, max_amt, 250_000.0, step=10_000.0)
    hr = c[2].slider("Hour", 0, 23, 3)
    day = c[3].slider("Day", 0, 30, 28)

    X = build_row(ty, amt, hr, day)
    p = float(model.predict_proba(X)[0, 1])
    lvl = risk_level(p)

    a, b = st.columns([2, 3])
    with a:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number=dict(suffix="%", valueformat=".1f"),
            title=dict(text=f"Fraud probability · {lvl}"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=RISK_COLORS[lvl]),
                       steps=[dict(range=[0, 5], color="#12332f"), dict(range=[5, 20], color="#3a3213"),
                              dict(range=[20, 50], color="#3b2414"), dict(range=[50, 100], color="#3b1420")])))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Suggested action:** {ACTIONS[lvl]}")
    with b:
        contrib = explain_row(X).sort_values("shap")
        fig = go.Figure(go.Bar(x=contrib["shap"], y=contrib["feature"], orientation="h",
                               marker_color=np.where(contrib["shap"] > 0, RISK_COLORS["Critical"], ACCENT)))
        fig.update_layout(title="Feature contributions", height=300, xaxis_title="SHAP value")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensitivity")
    a, b = st.columns(2)
    amounts = np.unique(np.geomspace(10, max_amt, 40))
    sweep = pd.concat([pd.concat([build_row(t, v, hr, day) for v in amounts]).assign(_t=t, _a=amounts) for t in TYPES])
    sweep["risk"] = model.predict_proba(sweep.drop(columns=["_t", "_a"]))[:, 1]
    with a:
        fig = px.line(sweep, x="_a", y="risk", color="_t", log_x=True, labels={"_a": "Amount (log scale)", "risk": "Fraud probability", "_t": "Type"},
                      title=f"Risk vs amount, by type (hour {hr}, day {day})")
        fig.add_vline(x=amt, line_dash="dot", line_color=MUTED)
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    hours = np.arange(24)
    hs = pd.concat([build_row(ty, amt, int(h), day) for h in hours])
    hs["risk"] = model.predict_proba(hs)[:, 1]
    with b:
        fig = px.line(x=hours, y=hs["risk"].to_numpy(), markers=True, labels={"x": "Hour of day", "y": "Fraud probability"},
                      title=f"Risk vs hour ({ty}, amount {amt:,.0f})")
        fig.update_traces(line_color=ACCENT2)
        fig.add_vline(x=hr, line_dash="dot", line_color=MUTED)
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
