import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Literal, Optional, Any

app = FastAPI()

ARTIFACT: Optional[Any] = None        # can be dict or estimator
MODEL: Optional[Any] = None           # must be estimator/pipeline with .predict
MODEL_LOADED = False
PREPROCESSOR_LOADED = False

CONTRACT: Optional[dict] = None
FEATURE_COLUMNS: Optional[list] = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))

def _infer_preprocessor_loaded(obj) -> bool:
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                return True
        return False
    if isinstance(obj, ColumnTransformer) or hasattr(obj, "transform"):
        return True
    return False

@app.on_event("startup")
def load_artifact():
    global ARTIFACT, MODEL, MODEL_LOADED, PREPROCESSOR_LOADED, CONTRACT, FEATURE_COLUMNS
    try:
        ARTIFACT = joblib.load(MODEL_PATH)

        print("MODEL_PATH =", MODEL_PATH)
        print("ARTIFACT_TYPE =", type(ARTIFACT))
        print("ARTIFACT_IS_DICT =", isinstance(ARTIFACT, dict))
        if isinstance(ARTIFACT, dict):
            print("ARTIFACT_KEYS =", list(ARTIFACT.keys()))

        # If dict artifact, model is inside ["pipeline"]
        if isinstance(ARTIFACT, dict):
            if "pipeline" not in ARTIFACT:
                raise RuntimeError("model.joblib is dict but missing key: 'pipeline'")
            MODEL = ARTIFACT["pipeline"]
        else:
            MODEL = ARTIFACT

        print("MODEL_TYPE =", type(MODEL))
        print("MODEL_HAS_PREDICT =", hasattr(MODEL, "predict"))

        if not hasattr(MODEL, "predict"):
            raise RuntimeError("Loaded MODEL does not have predict()")

        # Contract-driven schema (optional)
        CONTRACT = ARTIFACT.get("contract") if isinstance(ARTIFACT, dict) else None
        FEATURE_COLUMNS = None
        if isinstance(CONTRACT, dict):
            FEATURE_COLUMNS = (
                CONTRACT.get("feature_columns")
                or CONTRACT.get("features")
                or CONTRACT.get("columns")
            )

        print("CONTRACT_IS_DICT =", isinstance(CONTRACT, dict))
        print("FEATURE_COLUMNS_PRESENT =", bool(FEATURE_COLUMNS))
        if FEATURE_COLUMNS:
            print("FEATURE_COLUMNS_COUNT =", len(FEATURE_COLUMNS))
            print("FEATURE_COLUMNS_SAMPLE =", FEATURE_COLUMNS[:50])

        MODEL_LOADED = True
        PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)

    except Exception as e:
        print("Model load failed:", repr(e))
        ARTIFACT = None
        MODEL = None
        CONTRACT = None
        FEATURE_COLUMNS = None
        MODEL_LOADED = False
        PREPROCESSOR_LOADED = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "preprocessor_loaded": PREPROCESSOR_LOADED,
        "has_contract": isinstance(CONTRACT, dict),
        "feature_columns_count": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
    }

class PredictRequest(BaseModel):
    event_ts: str
    priority_urgent: int = Field(ge=0, le=1)
    line_id: str
    product_family: str
    station: str
    shift: str
    remaining_units: int = Field(ge=0)
    queue_time_min: float = Field(ge=0)
    queue_length: int = Field(ge=0)
    machine_state: Literal["RUN", "DOWN", "IDLE"]
    down_minutes_last_60: float = Field(ge=0)
    alarm_rate_last_30: float = Field(ge=0)
    cycle_time_expected_sec: float = Field(ge=0)
    cycle_time_actual_sec: float = Field(ge=0)
    cycle_time_deviation: float
    shortage_flag: int = Field(ge=0, le=1)
    replenishment_eta_min: float = Field(ge=0)
    shortage_severity: float = Field(ge=0)
    operator_present: int = Field(ge=0, le=1)
    skill_level: str
    coverage_ratio: float = Field(ge=0)
    baseline_queue_min: float = Field(ge=0)
    station_backlog_ratio: float = Field(ge=0)
    delay_flag: int = Field(ge=0, le=1)

@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        X = pd.DataFrame([req.model_dump()])

        # Align to training schema if available
        if FEATURE_COLUMNS:
            missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
            extra = [c for c in X.columns if c not in FEATURE_COLUMNS]

            if missing:
                raise HTTPException(status_code=422, detail=f"missing required fields: {missing}")

            if extra:
                X = X.drop(columns=extra)

            X = X[FEATURE_COLUMNS]

        y = MODEL.predict(X)
        return {"prediction": float(y[0])}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
