import os
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = FastAPI()

MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))

# These are the columns your model is complaining about.
# Defaults are neutral placeholders; replace with real domain defaults if you have them.
MISSING_DEFAULTS = {
    "line_speed_m_min": 0.0,
    "vibration_mm_s": 0.0,
    "inspection_interval_hrs": 0.0,
    "material_grade": "UNKNOWN",
    "operator_experience_yrs": 0.0,
    "humidity_pct": 0.0,
    "temperature_c": 0.0,
    "pressure_kpa": 0.0,
    "shift": "UNKNOWN",
    "machine_age_days": 0.0,
}

def _extract_model(obj):
    if isinstance(obj, Pipeline):
        return obj

    if isinstance(obj, dict):
        if "pipeline" in obj and hasattr(obj["pipeline"], "predict"):
            return obj["pipeline"]

        pre = obj.get("preprocessor") or obj.get("transformer")
        mdl = obj.get("model") or obj.get("estimator")

        if pre is not None and mdl is not None and hasattr(mdl, "predict"):
            return Pipeline([("preprocess", pre), ("model", mdl)])

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
    if isinstance(obj, ColumnTransformer) or hasattr(obj, "transform"):
        return True
    return False


def _as_float(x):
    if isinstance(x, (list, np.ndarray)):
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

        if not hasattr(MODEL, "predict"):
            raise RuntimeError(f"Loaded object has no predict(): {type(MODEL)}")

        MODEL_LOADED = True
        PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)

    except Exception as e:
        print(f"[MODEL LOAD ERROR] {type(e).__name__}: {e}", flush=True)
        MODEL = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False


@app.on_event("startup")
def startup():
    _ensure_model_loaded()


@app.get("/health")
def health():
    _ensure_model_loaded()
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "preprocessor_loaded": PREPROCESSOR_LOADED,
    }


class PredictRequest(BaseModel):
    event_ts: str
    baseline_queue_min: float
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: Literal["RUN", "DOWN", "IDLE"]
    queue_time_min: float
    down_minutes_last_60: float

    # Optional fields (if caller provides them, use them; else default)
    line_speed_m_min: Optional[float] = None
    vibration_mm_s: Optional[float] = None
    inspection_interval_hrs: Optional[float] = None
    material_grade: Optional[str] = None
    operator_experience_yrs: Optional[float] = None
    humidity_pct: Optional[float] = None
    temperature_c: Optional[float] = None
    pressure_kpa: Optional[float] = None
    shift: Optional[str] = None
    machine_age_days: Optional[float] = None


@app.post("/predict")
def predict(req: PredictRequest):
    _ensure_model_loaded()
    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        data = req.model_dump()

        # Fill any missing/None optional fields with defaults
        for k, v in MISSING_DEFAULTS.items():
            if k not in data or data[k] is None:
                data[k] = v

        X = pd.DataFrame([data])
        y = MODEL.predict(X)
        return {"prediction": _as_float(y)}

    except Exception as e:
        print(f"[PREDICT ERROR] {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
