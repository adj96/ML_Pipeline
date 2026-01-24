from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

# Required for your current test_health.py
MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # default state
    app.state.model_loaded = False
    app.state.preprocessor_loaded = False
    app.state.model = None
    app.state.preprocessor = None

    # load preprocessor (must exist in Jenkins workspace)
    if PREPROCESSOR_PATH.exists():
        app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)
        app.state.preprocessor_loaded = True

    # load model (must exist in Jenkins workspace)
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        app.state.model_loaded = True

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "preprocessor_loaded": bool(app.state.preprocessor_loaded),
        "model_loaded": bool(app.state.model_loaded),
    }
