import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")

app = FastAPI()

model = None  # this will be the full sklearn Pipeline (preprocess + estimator)


class PredictRequest(BaseModel):
    event_ts: str
    baseline_queue_min: float
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: str
    queue_time_min: float
    down_minutes_last_60: float


@app.on_event("startup")
def load_artifacts():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
        raise RuntimeError(f"Failed to load model pipeline from {MODEL_PATH}: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": model is not None,  # pipeline includes preprocessing
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build 1-row DataFrame with the exact feature names expected by the pipeline
    X = pd.DataFrame([{
        "event_ts": req.event_ts,
        "baseline_queue_min": req.baseline_queue_min,
        "shortage_flag": req.shortage_flag,
        "replenishment_eta_min": req.replenishment_eta_min,
        "machine_state": req.machine_state,
        "queue_time_min": req.queue_time_min,
        "down_minutes_last_60": req.down_minutes_last_60,
    }])

    try:
        yhat = model.predict(X)
        pred = float(yhat[0])
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
