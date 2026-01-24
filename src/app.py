# src/app.py
from pathlib import Path
import os
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = Path(
    os.getenv("ARTIFACT_DIR", REPO_ROOT / "models")
)

PREPROCESSOR_PATH = ARTIFACT_DIR / "model.joblib"
MODEL_PATH = ARTIFACT_DIR / "eta_xgb.json"

pre = None
model = None

@app.on_event("startup")
def load_artifacts():
    global pre, model

    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Missing model artifact: {PREPROCESSOR_PATH}")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model artifact: {MODEL_PATH}")

    pre = joblib.load(PREPROCESSOR_PATH)

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "preprocessor_loaded": pre is not None,
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: dict):
    if pre is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    FEATURE_ORDER = [
        "event_ts",
        "shortage_flag",
        "replenishment_eta_min",
        "machine_state",
        "down_minutes_last_60",
        "queue_time_min",
        "baseline_queue_min",
    ]

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    import pandas as pd

    X = pd.DataFrame([{k: data[k] for k in FEATURE_ORDER}])
    X_enc = pre.transform(X)

    yhat = float(model.predict(X_enc)[0])
    return {"prediction": yhat}
