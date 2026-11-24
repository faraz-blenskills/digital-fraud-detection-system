"""
Digital Fraud Intelligence & Transaction Risk Analytics  (ADDED - not in the original repo)

Run from project root:

    python -m src.analytics.build_scores          # once
    streamlit run app/intel/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import common` / `import views` work

import streamlit as st

st.set_page_config(page_title="Fraud Intelligence", page_icon="🛡️", layout="wide")

import common  # noqa: E402
import views  # noqa: E402

common.setup_page()

missing = [p.name for p in (common.ALL_PATH, common.SCORED_PATH, common.MODEL_PATH) if not p.exists()]
if missing:
    st.error(f"Missing files: {missing}. Run the pipeline and `python -m src.analytics.build_scores` first (see README).")
    st.stop()

pages = [
    st.Page(views.executive_overview, title="Executive Overview", icon="📊", default=True),
    st.Page(views.transaction_analytics, title="Transaction Analytics", icon="💳"),
    st.Page(views.fraud_detection, title="Fraud Detection", icon="🚨"),
    st.Page(views.risk_intelligence, title="Risk Intelligence", icon="🎯"),
    st.Page(views.geographic_intelligence, title="Geographic Intelligence", icon="🌍"),
    st.Page(views.anomaly_detection, title="Anomaly Detection", icon="🔎"),
    st.Page(views.investigation_assistant, title="Investigation Assistant", icon="🕵️"),
    st.Page(views.whatif_simulator, title="What-if Simulator", icon="🧪"),
    st.Page(views.reports_insights, title="Reports & Insights", icon="📄"),
]
st.sidebar.markdown("### 🛡️ Fraud Intelligence")
st.sidebar.caption("PaySim · XGBoost · SHAP · Isolation Forest")
st.navigation(pages).run()
