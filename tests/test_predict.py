from pydantic import BaseModel

class PredictIn(BaseModel):
    event_ts: int
    shortage_flag: int
    replenishment_eta_min: float
    machine_state: str
    down_minutes_last_60: float
    queue_time_min: float
    baseline_queue_min: float

@app.post("/predict")
def predict(req: PredictIn):
    try:
        load_artifacts()

        df = pd.DataFrame([req.model_dump()])
        X = pre.transform(df)                 # ColumnTransformer output
        dmat = xgb.DMatrix(X)
        yhat = float(booster.predict(dmat)[0])

        return {"prediction": yhat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")
