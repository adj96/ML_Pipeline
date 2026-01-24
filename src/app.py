from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Adjust these defaults to your repo structure
DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load artifacts ---
    model_path = DEFAULT_MODEL_PATH
    pre_path = DEFAULT_PREPROCESSOR_PATH

    # Initialize flags
    app.state.model_loaded = False
    app.state.preprocessor_loaded = False
    app.state.model = None
    app.state.preprocessor = None

    # Load preprocessor
    if not pre_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {pre_path.resolve()}")
    app.state.preprocessor = joblib.load(pre_path)
    app.state.preprocessor_loaded = True

    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    app.state.model = joblib.load(model_path)
    app.state.model_loaded = True

    yield

    # --- shutdown (optional) ---
    # cleanup if needed


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "preprocessor_loaded": bool(getattr(app.state, "preprocessor_loaded", False)),
            "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        },
    )
