import os
import threading
import traceback
from typing import Any, Dict, Optional, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = FastAPI()

ARTIFACT: Optional[Dict[str, Any]] = None
MODEL = None

MODEL_LOADED = False
PREPROCESSOR_LOADED = False

CONTRACT: Optional[Dict[str, Any]] = None
CONFIG: Optional[Dict[str, Any]] = None
BEST_PARAMS: Optional[Dict[str, Any]] = None
METRICS: Optional[Dict[str, Any]] = None

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))

_load_lock = threading.Lock()
_load_attempted = False


def _infer_preprocessor_loaded(obj) -> bool:
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                return True
        return False
    if isinstance(obj, ColumnTransformer) or hasattr(obj, "transform"):
        return True
    return False


def _reset_state():
    global ARTIFACT, MODEL
    global MODEL_LOADED, PREPROCESSOR_LOADED
    global CONTRACT, CONFIG, BEST_PARAMS, METRICS

    ARTIFACT = None
    MODEL = None
    CONTRACT = None
    CONFIG = None
    BEST_PARAMS = None
    METRICS = None
    MODEL_LOADED = False
    PREPROCESSOR_LOADED = False


def load_artifact_or_raise() -> None:
    global ARTIFACT, MODEL
    global MODEL_LOADED, PREPROCESSOR_LOADED
    global CONTRACT, CONFIG, BEST_PARAMS, METRICS
    global _load_attempted

    with _load_lock:
        if MODEL_LOADED and MODEL is not None:
            return

        _load_attempted = True
        try:
            loaded = joblib.load(MODEL_PATH)

            if isinstance(loaded, dict):
                ARTIFACT = loaded
                MODEL = loaded.get("pipeline")
                CONTRACT = loaded.get("contract")
                CONFIG = loaded.get("config")
                BEST_PARAMS = loaded.get("best_params")
                METRICS = loaded.get("metrics")
            else:
                ARTIFACT = None
                MODEL = loaded
                CONTRACT = None
                CONFIG = None
                BEST_PARAMS = None
                METRICS = None

            if MODEL is None or not hasattr(MODEL, "predict"):
                raise RuntimeError("model.joblib missing valid 'pipeline' with predict().")

            MODEL_LOADED = True
            PREPROCESSOR_LOADED = _infer_preprocessor_loaded(MODEL)

        except Exception as e:
            _reset_state()
            print(f"[MODEL_LOAD_ERROR] path={MODEL_PATH} err={repr(e)}")
            traceback.print_exc()
            raise


@app.on_event("startup")
def startup_load():
    try:
        load_artifact_or_raise()
    except Exception:
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "preprocessor_loaded": PREPROCESSOR_LOADED,
        "has_contract": CONTRACT is not None,
        "feature_columns_count": len(CONTRACT.get("feature_columns", [])) if CONTRACT else None,
        "model_path": MODEL_PATH,
        "load_attempted": _load_attempted,
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


# Deterministic encoders to keep API compatible with a numeric-only training pipeline.
# Unknowns -> -1 (safe numeric placeholder)
MAP_LINE = {"L1": 1, "L2": 2, "L3": 3}
MAP_PF = {"PF1": 1, "PF2": 2, "PF3": 3}
MAP_STATION = {"ST01": 1, "ST02": 2, "ST03": 3}
MAP_SHIFT = {"A": 1, "B": 2, "C": 3}
MAP_STATE = {"RUN": 1, "DOWN": 2, "IDLE": 3}
MAP_SKILL = {"S1": 1, "S2": 2, "S3": 3}


def _encode_payload_to_numeric(data: Dict[str, Any]) -> Dict[str, Any]:
    # event_ts -> epoch seconds (float)
    ts = pd.to_datetime(data.get("event_ts"), errors="coerce", utc=True)
    if pd.isna(ts):
        raise HTTPException(status_code=422, detail="event_ts invalid datetime format")
    data["event_ts"] = float(ts.timestamp())

    # categorical -> numeric codes
    data["line_id"] = MAP_LINE.get(str(data.get("line_id", "")), -1)
    data["product_family"] = MAP_PF.get(str(data.get("product_family", "")), -1)
    data["station"] = MAP_STATION.get(str(data.get("station", "")), -1)
    data["shift"] = MAP_SHIFT.get(str(data.get("shift", "")), -1)
    data["machine_state"] = MAP_STATE.get(str(data.get("machine_state", "")), -1)
    data["skill_level"] = MAP_SKILL.get(str(data.get("skill_level", "")), -1)

    # hard-cast numeric fields (prevents dtype=object leaks)
    int_fields = [
        "priority_urgent",
        "remaining_units",
        "queue_length",
        "shortage_flag",
        "operator_present",
        "delay_flag",
    ]
    float_fields = [
        "queue_time_min",
        "down_minutes_last_60",
        "alarm_rate_last_30",
        "cycle_time_expected_sec",
        "cycle_time_actual_sec",
        "cycle_time_deviation",
        "replenishment_eta_min",
        "shortage_severity",
        "coverage_ratio",
        "baseline_queue_min",
        "station_backlog_ratio",
    ]

    for k in int_fields:
        if k in data and data[k] is not None:
            data[k] = int(data[k])

    for k in float_fields:
        if k in data and data[k] is not None:
            data[k] = float(data[k])

    return data


@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_LOADED or MODEL is None:
        try:
            load_artifact_or_raise()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"model not loaded: {repr(e)}")

    try:
        data = req.model_dump()
        data = _encode_payload_to_numeric(data)

        X = pd.DataFrame([data])

        # If contract defines a column order, enforce it
        if CONTRACT and isinstance(CONTRACT.get("feature_columns"), list) and CONTRACT["feature_columns"]:
            cols = CONTRACT["feature_columns"]
            for c in cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[cols]

        y = MODEL.predict(X)
        return {"prediction": float(y[0])}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
