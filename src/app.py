import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")
model = None  # sklearn Pipeline (preprocess + estimator)

class PredictRequest(BaseModel):
    event_ts: str
    baseline_queue_min: float
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: str
    queue_time_min: float
    down_minutes_last_60: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": model is not None,
        "model_path": MODEL_PATH,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = pd.DataFrame([{
        "event_ts": req.event_ts,
        "baseline_queue_min": req.baseline_queue_min,
        "shortage_flag": req.shortage_flag,
        "replenishment_eta_min": req.replenishment_eta_min,
        "machine_state": req.machine_state,
        "queue_time_min": req.queue_time_min,
        "down_minutes_last_60": req.down_minutes_last_60,
    }])

    yhat = model.predict(X)
    return {"prediction": float(yhat[0])}
