# test_predict.py
# Purpose: Jenkins smoke-test for model.joblib (single artifact pipeline = preprocessing + model)

import sys
import os
import joblib
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")

REQUIRED_COLS = [
    "event_ts",
    "baseline_queue_min",
    "shortage_flag",
    "replenishment_eta_min",
    "machine_state",
    "queue_time_min",
    "down_minutes_last_60",
]

DROP_IF_PRESENT = [
    "eta_minutes_remaining",  # target
    "delay_flag",
    "delay_label_dev",
    "work_order_id",
]

def make_min_sample() -> pd.DataFrame:
    # Minimal valid row that matches training schema
    return pd.DataFrame([{
        "event_ts": "2026-01-24 10:00:00",
        "baseline_queue_min": 12.0,
        "shortage_flag": 0,
        "replenishment_eta_min": 0.0,
        "machine_state": "RUN",
        "queue_time_min": 10.0,
        "down_minutes_last_60": 0.0,
    }])

def main():
    m = joblib.load(MODEL_PATH)

    # Build X for prediction (DataFrame with required raw columns)
    X = make_min_sample()

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in X.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns for predict: {missing}")

    # Drop targets/ids if accidentally included
    X = X.drop(columns=[c for c in DROP_IF_PRESENT if c in X.columns], errors="ignore")

    # Keep only the columns the preprocessor expects (avoid extra cols breaking older setups)
    X = X[REQUIRED_COLS].copy()

    # Predict
    pred = m.predict(X)

    print("OK: predict() ran")
    print("Input columns:", list(X.columns))
    print("Prediction:", pred.tolist())

if __name__ == "__main__":
    main()
