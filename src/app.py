import os
import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager

MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "/app/src/model.joblib")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = joblib.load(MODEL_PATH)   # raise if broken
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": (MODEL is not None and hasattr(MODEL, "predict")),
        "preprocessor_loaded": False
    }
