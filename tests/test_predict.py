import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("src") / "model.joblib"   # change to "mode.joblib" if that is your real filename

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
    "eta_minutes_remaining",
    "delay_flag",
    "delay_label_dev",
    "work_order_id",
]

def make_min_sample() -> pd.DataFrame:
    return pd.DataFrame([{
        "event_ts": "2026-01-24 10:00:00",
        "baseline_queue_min": 12.0,
        "shortage_flag": 0,
        "replenishment_eta_min": 0.0,
        "machine_state": "RUN",
        "queue_time_min": 10.0,
        "down_minutes_last_60": 0.0,
    }])

def test_predict_smoke():
    assert MODEL_PATH.exists(), f"Missing model file at {MODEL_PATH}"

    m = joblib.load(MODEL_PATH)

    X = make_min_sample()

    # make event_ts robust if your pipeline expects datetime
    X["event_ts"] = pd.to_datetime(X["event_ts"], errors="coerce")

    # drop accidental cols
    X = X.drop(columns=[c for c in DROP_IF_PRESENT if c in X.columns], errors="ignore")

    # keep schema order
    X = X[REQUIRED_COLS].copy()

    pred = m.predict(X)

    assert pred is not None
    assert len(pred) == 1
