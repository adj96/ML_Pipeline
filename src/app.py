import os
import joblib
import pandas as pd
from typing import Literal
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# -------------------------
# Globals (single source of truth)
# -------------------------
MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")


def _infer_preprocessor_loaded(obj) -> bool:
    if isinstance(obj, Pipeline):
        # Pipeline may contain preprocessor + estimator
        for _, step in obj.steps:
            if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                return True
        return False
    if isinstance(obj, ColumnTransformer) or hasattr(obj, "transform"):
        return True
    return False


@app.on_event("startup")
def load_artifact():
    global MODEL, MODEL_LOADED, PREPROCESSOR_LOADED
    try:
        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOADED = True
        PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)
        logger.info(f"Loaded model from {MODEL_PATH}. preprocessor_loaded={PREPROCESSOR_LOADED}")
    except Exception:
        logger.exception(f"Failed to load model from {MODEL_PATH}")
        MODEL = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "preprocessor_loaded": PREPROCESSOR_LOADED,
        "model_path": MODEL_PATH,
    }


class PredictRequest(BaseModel):
    event_ts: str
    baseline_queue_min: float
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: Literal["RUN"]
    queue_time_min: float
    down_minutes_last_60: float


@app.post("/predict")
def predict(payload: PredictRequest):
    if MODEL is None or not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # keep column order stable
        row = payload.model_dump()
        X = pd.DataFrame([row])

        # CRITICAL FIX: use MODEL (not undefined 'model')
        yhat = MODEL.predict(X)

        return {"prediction": float(yhat[0])}

    except Exception as e:
        logger.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(e))
