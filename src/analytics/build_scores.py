"""
Build analytics tables for the Fraud Intelligence dashboard.   (ADDED - not in the original repo)

Run from project root:

    python -m src.analytics.build_scores

Outputs (data/processed/):
  - intel_all.parquet     compact copy of ALL transactions (descriptive BI, uses labels)
  - intel_scored.parquet  last-7-days holdout window with model risk score,
                          risk level, anomaly score and a simulated region
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from src.config import DATA_DIR, MODEL_DIR

CLEAN = DATA_DIR / "processed" / "paysim_cleaned.parquet"
FEATS = DATA_DIR / "processed" / "paysim_features.parquet"
MODEL = MODEL_DIR / "xgboost_gpu_model.json"
OUT_ALL = DATA_DIR / "processed" / "intel_all.parquet"
OUT_SCORED = DATA_DIR / "processed" / "intel_scored.parquet"

TEST_HOURS = 24 * 7

# PaySim has NO location data. Regions are assigned deterministically from the
# sender account id so the Geographic module has something to show. SIMULATED.
REGIONS = [
    "North America", "Latin America", "Western Europe", "Eastern Europe",
    "Middle East", "Africa", "South Asia", "East Asia", "Oceania",
]

RISK_BINS = [-0.01, 0.05, 0.20, 0.50, 1.01]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]


def simulated_region(names: pd.Series) -> pd.Series:
    # stable hash: digits of the account id, mod number of regions
    num = names.str.slice(1).astype("int64")
    return pd.Series(np.array(REGIONS)[(num % len(REGIONS)).to_numpy()], index=names.index)


def main():
    print("Loading data...")
    clean = pd.read_parquet(CLEAN, columns=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"])
    feats = pd.read_parquet(FEATS)
    assert len(clean) == len(feats) and (clean["step"].values == feats["step"].values).all()

    # ---- compact table of everything (no account ids) ----
    allx = pd.DataFrame({
        "step": clean["step"].astype("int16"),
        "type": clean["type"].astype("category"),
        "amount": clean["amount"].astype("float32"),
        "isFraud": clean["isFraud"].astype("int8"),
    })
    allx["day"] = (allx["step"] // 24).astype("int8")
    allx["hour"] = (allx["step"] % 24).astype("int8")
    allx.to_parquet(OUT_ALL)
    print("Saved", OUT_ALL, allx.shape)

    # ---- time-based split (same rule as the repo's training script) ----
    max_step = feats["step"].max()
    is_test = (feats["step"] > max_step - TEST_HOURS).to_numpy()
    test_feats = feats[is_test]
    train_feats = feats[~is_test]
    feature_cols = [c for c in feats.columns if c != "isFraud"]

    model = XGBClassifier()
    model.load_model(MODEL)
    risk = model.predict_proba(test_feats[feature_cols].astype(float))[:, 1]

    # ---- anomaly detection: unsupervised, fitted on a sample of TRAIN rows ----
    # fitted on normal behaviour (no labels used) -> scores deviation from it
    an_cols = ["log_amount", "hour"] + [c for c in feature_cols if c.startswith("type_")]
    sample = train_feats.sample(200_000, random_state=42)
    iso = IsolationForest(n_estimators=150, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(sample[an_cols].astype(float))
    raw = -iso.score_samples(test_feats[an_cols].astype(float))   # higher = more anomalous
    lo, hi = np.percentile(raw, [1, 99.9])
    anomaly = np.clip((raw - lo) / (hi - lo), 0, 1)

    meta = clean[is_test]
    scored = pd.DataFrame({
        "tx_id": np.arange(len(meta)),
        "step": meta["step"].to_numpy(),
        "day": (meta["step"].to_numpy() // 24),
        "hour": (meta["step"].to_numpy() % 24),
        "type": meta["type"].to_numpy(),
        "amount": meta["amount"].to_numpy(),
        "nameOrig": meta["nameOrig"].to_numpy(),
        "nameDest": meta["nameDest"].to_numpy(),
        "isFraud": meta["isFraud"].to_numpy(),
        "risk_score": risk.astype("float32"),
        "anomaly_score": anomaly.astype("float32"),
    })
    scored["risk_level"] = pd.cut(scored["risk_score"], RISK_BINS, labels=RISK_LABELS).astype(str)
    scored["region"] = simulated_region(scored["nameOrig"]).to_numpy()
    scored.to_parquet(OUT_SCORED)
    print("Saved", OUT_SCORED, scored.shape)
    print(scored["risk_level"].value_counts().to_dict())
    print(scored.groupby("risk_level")["isFraud"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
