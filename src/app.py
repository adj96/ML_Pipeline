# src/app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Literal

app = FastAPI()

MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")

def _infer_preprocessor_loaded(obj) -> bool:
    # Case A: full Pipeline (preprocess + model)
    if isinstance(obj, Pipeline):
        # if any step is ColumnTransformer or has transform()
        for _, step in obj.steps:
            if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                return True
        return False

    # Case B: standalone preprocessor mistakenly saved as "model.joblib"
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
    except Exception:
        MODEL = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False

@app.get("/health")
def health():
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
    if not MODEL_LOADED or MODEL is None:
        return {"error": "model not loaded"}

    X = pd.DataFrame([req.model_dump()])

    # If MODEL is a sklearn Pipeline (preprocess+model), this will work directly
    y = MODEL.predict(X)

    return {"prediction": float(y[0])}
