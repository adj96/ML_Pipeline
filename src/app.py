import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")

app = FastAPI(title="ML Pipeline Service", version="1.0.0")

model = None
preprocessor = None
expected_input_columns = []


def _infer_expected_columns(obj) -> Optional[list]:
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    steps = getattr(obj, "steps", None)
    if steps:
        for _, step in steps:
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                return list(cols)

    return None


def _extract_objects(loaded: Any) -> Tuple[Any, Any]:
    """
    Robust extraction:
    - If loaded is dict:
        - preprocessor: try common keys
        - model: try common keys
        - else if dict has single value -> treat as model
        - else if dict has sklearn pipeline under unknown key -> pick first non-preprocessor-like object
    - Else: loaded is the model/pipeline
    """
    if not isinstance(loaded, dict):
        return loaded, None

    # preprocessor keys (optional)
    pre_keys = ["preprocessor", "transformer", "column_transformer", "ct", "prep"]
    p = None
    for k in pre_keys:
        if k in loaded:
            p = loaded.get(k)
            break

    # model keys (many possible)
    model_keys = ["model", "pipeline", "estimator", "clf", "classifier", "regressor", "rf", "xgb", "svc"]
    m = None
    for k in model_keys:
        if k in loaded:
            m = loaded.get(k)
            break

    # if still not found, and dict has exactly 1 value, use it as model
    if m is None and len(loaded) == 1:
        m = list(loaded.values())[0]
        return m, p

    # if still not found, pick first object that looks like it can predict
    if m is None:
        for v in loaded.values():
            if hasattr(v, "predict"):
                m = v
                break

    return m, p


@app.on_event("startup")
def startup_load():
    global model, preprocessor, expected_input_columns

    try:
        loaded = joblib.load(MODEL_PATH)
        model, preprocessor = _extract_objects(loaded)

        cols = None
        if preprocessor is not None:
            cols = _infer_expected_columns(preprocessor)
        if cols is None and model is not None:
            cols = _infer_expected_columns(model)

        expected_input_columns = cols if cols is not None else []

    except Exception as e:
        # Keep service alive so /health exposes the failure reason
        model = None
        preprocessor = None
        expected_input_columns = []
        app.state.startup_error = str(e)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": MODEL_PATH,
        "expected_columns_known": bool(expected_input_columns),
        "expected_columns_count": len(expected_input_columns),
        "startup_error": getattr(app.state, "startup_error", None),
    }


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    try:
        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model not loaded",
                    "startup_error": getattr(app.state, "startup_error", None),
                },
            )

        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": "Payload must be a JSON object."})

        cols = expected_input_columns if expected_input_columns else list(payload.keys())

        row: Dict[str, Any] = {}
        for c in cols:
            if c in payload:
                row[c] = payload[c]
            else:
                # safe defaults
                if c.lower().endswith(("_min", "_sec", "_ms", "_count", "_flag", "_id")):
                    row[c] = 0
                else:
                    row[c] = "UNKNOWN"

        X = pd.DataFrame([row])

        if preprocessor is not None:
            Xt = preprocessor.transform(X)
            y = model.predict(Xt)
        else:
            y = model.predict(X)

        pred = float(y[0]) if hasattr(y, "__len__") else float(y)
        return {"prediction": pred}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "detail": str(e),
                "expected_columns": expected_input_columns,
                "received_keys": list(payload.keys()) if isinstance(payload, dict) else None,
            },
        )
