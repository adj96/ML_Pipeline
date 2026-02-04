import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")

app = FastAPI(title="ML Pipeline Service", version="1.0.0")

model = None
preprocessor = None
expected_input_columns = None


def _infer_expected_columns(obj) -> Optional[list]:
    """
    Try to infer expected raw input columns from common sklearn objects.
    Works for:
    - Pipeline: uses feature_names_in_ if present
    - Estimator: feature_names_in_
    - ColumnTransformer: feature_names_in_
    """
    for candidate in [obj]:
        cols = getattr(candidate, "feature_names_in_", None)
        if cols is not None:
            return list(cols)

    # sklearn Pipeline: look for first step with feature_names_in_
    steps = getattr(obj, "steps", None)
    if steps:
        for _, step in steps:
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                return list(cols)

    return None


def _split_model_preprocessor(obj):
    """
    If the joblib contains a dict like {"model":..., "preprocessor":...}, use it.
    Else treat obj as a single pipeline model.
    """
    if isinstance(obj, dict):
        m = obj.get("model", None)
        p = obj.get("preprocessor", None)
        return m, p
    return obj, None


@app.on_event("startup")
def startup_load():
    global model, preprocessor, expected_input_columns

    loaded = joblib.load(MODEL_PATH)
    model, preprocessor = _split_model_preprocessor(loaded)

    # Prefer columns from preprocessor if available, else from model/pipeline
    expected_input_columns = None
    if preprocessor is not None:
        expected_input_columns = _infer_expected_columns(preprocessor)
    if expected_input_columns is None:
        expected_input_columns = _infer_expected_columns(model)

    # Fallback: unknown, we will use incoming keys
    if expected_input_columns is None:
        expected_input_columns = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": MODEL_PATH,
        "expected_columns_known": bool(expected_input_columns),
        "expected_columns_count": len(expected_input_columns) if expected_input_columns is not None else 0,
    }


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    """
    Accept ANY JSON object (no strict Pydantic schema) to prevent FastAPI 422.
    Then adapt it to the model’s expected input columns.

    Rules:
    - If expected columns are known, build a 1-row DataFrame with those columns.
      Missing numeric -> 0, missing string/cat -> "UNKNOWN".
    - If expected columns are unknown, use payload keys as columns.
    """
    try:
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "Payload must be a JSON object (key-value map)."},
            )

        # Determine columns
        cols = expected_input_columns if expected_input_columns else list(payload.keys())

        row: Dict[str, Any] = {}
        for c in cols:
            if c in payload:
                row[c] = payload[c]
            else:
                # Default fillers to avoid failure
                # Heuristic: common numeric-like fields -> 0, else "UNKNOWN"
                if c.lower().endswith(("_min", "_sec", "_ms", "_count", "_flag", "_id")) or c.lower() in (
                    "age", "qty", "quantity", "value", "score", "target"
                ):
                    row[c] = 0
                else:
                    row[c] = "UNKNOWN"

        X = pd.DataFrame([row])

        # If you have separate preprocessor + model
        if preprocessor is not None:
            Xt = preprocessor.transform(X)
            y = model.predict(Xt)
        else:
            # Assume model is a sklearn pipeline or estimator that can handle DF
            y = model.predict(X)

        # Normalize output
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
