import os
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Literal

app = FastAPI()

# -------------------------
# Globals
# -------------------------
MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))

def _extract_model(obj):
    # Your saved artifact might be:
    # - a sklearn Pipeline
    # - a dict like {"pipeline": <Pipeline>, ...}
    if isinstance(obj, dict):
        for k in ("pipeline", "model", "estimator"):
            if k in obj:
                return obj[k]
        # fallback: first value that has predict
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
        return obj
    return obj

def _infer_preprocessor_loaded(obj) -> bool:
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                return True
        return False
    if isinstance(obj, ColumnTransformer) or hasattr(obj, "transform"):
        return True
    return False

def _as_float(x):
    if isinstance(x, (np.ndarray, list)):
        x = x[0]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    return float(x)

def _ensure_model_loaded():
    global MODEL, MODEL_LOADED, PREPROCESSOR_LOADED
    if MODEL_LOADED and MODEL is not None:
        return
    try:
        obj = joblib.load(MODEL_PATH)
        MODEL = _extract_model(obj)
        MODEL_LOADED = True
        PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)
    except Exception:
        MODEL = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False

@app.on_event("startup")
def load_artifact():
    _ensure_model_loaded()

@app.get("/health")
def health():
    _ensure_model_loaded()
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
    }

class PredictRequest(BaseModel):
    event_ts: str
    baseline_queue_min: float
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: Literal["RUN", "DOWN", "IDLE"]
    queue_time_min: float
    down_minutes_last_60: float

@app.post("/predict")
def predict(req: PredictRequest):
    _ensure_model_loaded()
    if not MODEL_LOADED or MODEL is None or not hasattr(MODEL, "predict"):
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        payload = req.model_dump()
        X = pd.DataFrame([payload])
        y = MODEL.predict(X)
        return {"prediction": _as_float(y)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
