# src/app.py
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent  # src/
MODEL_PATH = BASE_DIR / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "event_ts",
    "shortage_flag",
    "replenishment_eta_min",
    "machine_state",
    "down_minutes_last_60",
    "queue_time_min",
    "baseline_queue_min",
]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "model_path": str(MODEL_PATH),
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    X = pd.DataFrame([{k: data[k] for k in FEATURE_ORDER}])

    try:
        yhat = model.predict(X)
        y = float(yhat[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")

    return {"prediction": y}
