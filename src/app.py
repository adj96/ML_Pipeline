# src/app.py
import os
import joblib
from fastapi import FastAPI

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = FastAPI()

MODEL = None
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

MODEL_PATH = os.getenv("MODEL_PATH", "src/model.joblib")  # keep your existing path if different

def _infer_preprocessor_loaded(obj) -> bool:
    # Case A: full Pipeline (preprocess + model)
    if isinstance(obj, Pipeline):
        # if any step is ColumnTransformer or has transform()
        for _, step in obj.steps:
            if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                return True
        return False

    # Case B: standalone preprocessor mistakenly saved as "model.joblib"
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
        "preprocessor_loaded": PREPROCESSOR_LOADED,
    }
