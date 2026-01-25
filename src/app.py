# src/app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# IMPORTANT: pickle is trying to import align_and_fix from uvicorn.__main__
import uvicorn.__main__ as uvicorn_main

MODEL = None
MODEL_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", "/app/src/model.joblib")  # container path

REQUIRED_COLS = [
    "event_ts",
    "baseline_queue_min",
    "shortage_flag",
    "replenishment_eta_min",
    "machine_state",
    "queue_time_min",
    "down_minutes_last_60",
]

def align_and_fix(X):
    df = pd.DataFrame(X).copy()

    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = 0

    df = df[REQUIRED_COLS]

    # keep as string unless your pipeline expects datetime;
    # if your trained pipeline expects datetime, keep this line.
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")

    return df

# Monkey-patch so joblib/pickle can resolve the symbol during load
uvicorn_main.align_and_fix = align_and_fix

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_LOADED
    try:
        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOADED = True
    except Exception:
        MODEL = None
        MODEL_LOADED = False
        raise
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(MODEL_LOADED),
    }

@app.post("/predict")
def predict(payload: dict):
    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([payload])
    y = MODEL.predict(X)
    return {"prediction": float(y[0])}
