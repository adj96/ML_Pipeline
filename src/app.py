import os
import joblib
import pandas as pd  # <-- ADD THIS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Literal

app = FastAPI()

MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))

def _infer_preprocessor_loaded(obj) -> bool:
    if isinstance(obj, Pipeline):
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
    priority_urgent: int = Field(ge=0, le=1)
    line_id: str
    product_family: str
    station: str
    shift: str
    remaining_units: int = Field(ge=0)
    queue_time_min: float = Field(ge=0)
    queue_length: int = Field(ge=0)
    machine_state: Literal["RUN", "DOWN", "IDLE"]
    down_minutes_last_60: float = Field(ge=0)
    alarm_rate_last_30: float = Field(ge=0)
    cycle_time_expected_sec: float = Field(ge=0)
    cycle_time_actual_sec: float = Field(ge=0)
    cycle_time_deviation: float
    shortage_flag: int = Field(ge=0, le=1)
    replenishment_eta_min: float = Field(ge=0)
    shortage_severity: float = Field(ge=0)
    operator_present: int = Field(ge=0, le=1)
    skill_level: str
    coverage_ratio: float = Field(ge=0)
    baseline_queue_min: float = Field(ge=0)
    station_backlog_ratio: float = Field(ge=0)
    delay_flag: int = Field(ge=0, le=1)

@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        X = pd.DataFrame([req.model_dump()])
        y = MODEL.predict(X)
        return {"prediction": float(y[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
