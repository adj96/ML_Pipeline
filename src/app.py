import os
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ======================================================
# App
# ======================================================
app = FastAPI()

# ======================================================
# Globals
# ======================================================
MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))

# ======================================================
# Helpers
# ======================================================
def _extract_model(obj):
    """
    Accepts:
    - sklearn Pipeline
    - dict containing 'pipeline'
    - dict containing 'preprocessor' + 'model'
    """
    if isinstance(obj, Pipeline):
        return obj

    if isinstance(obj, dict):
        # Preferred explicit pipeline
        if "pipeline" in obj and hasattr(obj["pipeline"], "predict"):
            return obj["pipeline"]

        # Common training save pattern
        pre = obj.get("preprocessor") or obj.get("transformer")
        mdl = obj.get("model") or obj.get("estimator")

        if pre is not None and mdl is not None and hasattr(mdl, "predict"):
            return Pipeline([
                ("preprocess", pre),
                ("model", mdl),
            ])

        # Last-chance fallback
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


# ======================================================
# Startup
# ======================================================
@app.on_event("startup")
def startup():
    _ensure_model_loaded()


# ======================================================
# Endpoints
# ======================================================
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

    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        X = pd.DataFrame([req.model_dump()])
        y = MODEL.predict(X)
        return {"prediction": _as_float(y)}

    except Exception as e:
        print(f"[PREDICT ERROR] {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
