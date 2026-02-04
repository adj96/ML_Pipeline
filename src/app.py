import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Literal

app = FastAPI()

ARTIFACT = None           # can be dict or estimator
MODEL = None              # must be estimator/pipeline with .predict
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
    global ARTIFACT, MODEL, MODEL_LOADED, PREPROCESSOR_LOADED
    try:
        ARTIFACT = joblib.load(MODEL_PATH)

        # HARD LOGGING (startup proof)
        print("MODEL_PATH =", MODEL_PATH, flush=True)
        print("ARTIFACT_TYPE =", type(ARTIFACT), flush=True)
        print("ARTIFACT_IS_DICT =", isinstance(ARTIFACT, dict), flush=True)
        if isinstance(ARTIFACT, dict):
            print("ARTIFACT_KEYS =", list(ARTIFACT.keys()), flush=True)

        # dict artifact -> extract pipeline
        if isinstance(ARTIFACT, dict):
            if "pipeline" not in ARTIFACT:
                raise RuntimeError("model.joblib is dict but missing key: 'pipeline'")
            MODEL = ARTIFACT["pipeline"]
        else:
            MODEL = ARTIFACT

        # log final MODEL object used for prediction
        print("MODEL_TYPE =", type(MODEL), flush=True)
        print("MODEL_HAS_PREDICT =", hasattr(MODEL, "predict"), flush=True)

        if not hasattr(MODEL, "predict"):
            raise RuntimeError("Loaded MODEL does not have predict()")

        MODEL_LOADED = True
        PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)

    except Exception as e:
        print("MODEL_LOAD_FAILED =", str(e), flush=True)
        ARTIFACT = None
        MODEL = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False
        
@app.get("/health")
def health():
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

@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        X = pd.DataFrame([req.model_dump()])
        y = MODEL.predict(X)   # 반드시 pipeline/estimator에만 predict 호출
        return {"prediction": float(y[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
