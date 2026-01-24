from pathlib import Path
import os
import joblib
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parents[1]

# If you commit artifacts into repo: /models
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", REPO_ROOT / "models"))

# Expose these names because tests import them
PREPROCESSOR_PATH = ARTIFACT_DIR / "model.joblib"
MODEL_PATH = ARTIFACT_DIR / "eta_xgb.json"

pre = None
model = None

FEATURE_ORDER = [
    "event_ts",
    "shortage_flag",
    "replenishment_eta_min",
    "machine_state",
    "down_minutes_last_60",
    "queue_time_min",
    "baseline_queue_min",
]

@app.on_event("startup")
def load_artifacts():
    global pre, model

    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Missing preprocessor: {PREPROCESSOR_PATH}")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model: {MODEL_PATH}")

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
def predict(payload: dict):
    try:
        if pre is None or model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        missing = [k for k in FEATURE_ORDER if k not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

        X = pd.DataFrame([{k: payload[k] for k in FEATURE_ORDER}])
        X_enc = pre.transform(X)
        yhat = float(model.predict(X_enc)[0])
        return {"prediction": yhat}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")
