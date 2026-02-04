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

        # 핵심 수정: dict artifact면 pipeline을 꺼내서 MODEL로 사용
        if isinstance(ARTIFACT, dict):
            if "pipeline" not in ARTIFACT:
                raise RuntimeError("model.joblib is dict but missing key: 'pipeline'")
            MODEL = ARTIFACT["pipeline"]
        else:
            MODEL = ARTIFACT

        # 보호: 반드시 predict 가능해야 함
        if not hasattr(MODEL, "predict"):
            raise RuntimeError("Loaded MODEL does not have predict()")

        MODEL_LOADED = True
        PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)

    except Exception as e:
        ARTIFACT = None
        MODEL = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False
        # 로그가 필요하면 print(e) 추가 가능 (컨테이너 로그로 확인)
        # print(f"Model load failed: {e}")

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
