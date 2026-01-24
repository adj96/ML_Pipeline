# src/app.py
import os
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

app = FastAPI()

HERE = Path(__file__).resolve().parent

# Default: artifacts are next to app.py (your current repo layout: src/model.joblib, src/eta_xgb.json)
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(HERE)))

PREPROCESSOR_PATH = ARTIFACT_DIR / "model.joblib"
MODEL_PATH = ARTIFACT_DIR / "eta_xgb.json"

pre = None
booster = None

def load_artifacts():
    global pre, booster
    if pre is None:
        pre = joblib.load(PREPROCESSOR_PATH)
    if booster is None:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))

@app.on_event("startup")
def _startup():
    # Try to load on startup; if missing, health will show False
    try:
        load_artifacts()
    except Exception:
        pass

@app.get("/health")
def health():
    return {
        "status": "ok",
        "preprocessor_loaded": pre is not None,
        "model_loaded": booster is not None,
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "model_path": str(MODEL_PATH),
    }
