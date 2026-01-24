# src/app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "model.joblib")

app = FastAPI()

_pipeline = None
model_loaded = False
preprocessor_loaded = False  # will mirror model_loaded when pipeline includes preprocessing


class PredictRequest(BaseModel):
    event_ts: int
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: str
    down_minutes_last_60: float
    queue_time_min: float
    baseline_queue_min: float


@app.on_event("startup")
def _load_artifacts():
    global _pipeline, model_loaded, preprocessor_loaded
    if not os.path.exists(MODEL_PATH):
        _pipeline = None
        model_loaded = False
        preprocessor_loaded = False
        return

    _pipeline = joblib.load(MODEL_PATH)
    model_loaded = True
    # If model.joblib is a full sklearn Pipeline, preprocessing is inside it.
    preprocessor_loaded = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([req.model_dump()])
    yhat = _pipeline.predict(df)

    return {"prediction": float(yhat[0])}
