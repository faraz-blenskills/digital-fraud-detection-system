"""Shared loaders, theme and helpers for the Fraud Intelligence dashboard.  (ADDED)"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shap
import streamlit as st
from xgboost import XGBClassifier

from src.config import DATA_DIR, MODEL_DIR

ALL_PATH = DATA_DIR / "processed" / "intel_all.parquet"
SCORED_PATH = DATA_DIR / "processed" / "intel_scored.parquet"
FEATS_PATH = DATA_DIR / "processed" / "paysim_features.parquet"
MODEL_PATH = MODEL_DIR / "xgboost_gpu_model.json"

TYPES = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {"Low": "#2dd4bf", "Medium": "#fbbf24", "High": "#fb923c", "Critical": "#f43f5e"}

# ---- palette ----
BG = "#0b1020"
PANEL = "#121a30"
BORDER = "#1f2a48"
TEXT = "#e6ebf5"
MUTED = "#8b97b5"
ACCENT = "#38bdf8"
ACCENT2 = "#a78bfa"

CSS = f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; }}
.kpi .l {{ color: {MUTED}; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
.kpi .v {{ font-size: 28px; font-weight: 650; margin-top: 4px; }}
.kpi .s {{ color: {MUTED}; font-size: 12px; margin-top: 2px; }}
.note {{ background: {PANEL}; border-left: 3px solid {ACCENT}; padding: 10px 14px; border-radius: 6px;
        color: {MUTED}; font-size: 13px; }}
.warn {{ border-left-color: #fbbf24; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; color:#0b1020; }}
</style>
"""

# plotly dark template that matches the page
pio.templates["fraud_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        colorway=[ACCENT, ACCENT2, "#f43f5e", "#fbbf24", "#2dd4bf", "#fb923c"],
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)
pio.templates.default = "fraud_dark"


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def note(text, warn=False):
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def pill(level: str) -> str:
    return f'<span class="pill" style="background:{RISK_COLORS[level]}">{level}</span>'


def money(x: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:,.2f}{unit}"
    return f"{x:,.0f}"


def risk_level(p: float) -> str:
    return "Low" if p < 0.05 else "Medium" if p < 0.20 else "High" if p < 0.50 else "Critical"


# ---------------- loaders ----------------
@st.cache_data(show_spinner="Loading transactions...")
def load_all() -> pd.DataFrame:
    return pd.read_parquet(ALL_PATH)


@st.cache_data(show_spinner="Loading scored transactions...")
def load_scored() -> pd.DataFrame:
    return pd.read_parquet(SCORED_PATH)


@st.cache_data
def feature_columns() -> list:
    import pyarrow.parquet as pq
    return [c for c in pq.ParquetFile(FEATS_PATH).schema.names if c != "isFraud"]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, shap.TreeExplainer(model)


def build_row(tx_type: str, amount: float, hour: int, day: int) -> pd.DataFrame:
    """Feature vector in the exact column order used for training."""
    cols = feature_columns()
    row = {c: 0.0 for c in cols}
    row.update(step=float(day * 24 + hour), amount=float(amount), hour=float(hour),
               day=float(day), log_amount=float(np.log1p(amount)))
    row[f"type_{tx_type}"] = 1.0
    return pd.DataFrame([row])[cols].astype(float)


def explain_row(X: pd.DataFrame) -> pd.DataFrame:
    _, explainer = load_model()
    sv = explainer.shap_values(X)[0]
    return pd.DataFrame({"feature": X.columns, "value": X.iloc[0].values, "shap": sv}
                        ).assign(abs=lambda d: d["shap"].abs()).sort_values("abs", ascending=False)
