# src/app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager

# IMPORTANT: this is the exact module pickle is trying to import from
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
    # Minimal safe implementation to satisfy unpickling + schema alignment
    df = pd.DataFrame(X).copy()

    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = 0

    # drop extras + enforce column order
    df = df[REQUIRED_COLS]

    # basic type hygiene (adjust to your training logic if needed)
    if "event_ts" in df.columns:
        df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")

    return df

# Monkey-patch so joblib/pickle can resolve the symbol during load
uvicorn_main.align_and_fix = align_and_fix

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_LOADED
    MODEL = joblib.load(MODEL_PATH)  # will raise if broken (fail fast)
    MODEL_LOADED = True
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(MODEL_LOADED)}

@app.post("/predict")
def predict(payload: dict):
    # adapt to your existing predict implementation
    X = pd.DataFrame([payload])
    y = MODEL.predict(X)
    return {"prediction": float(y[0])}
