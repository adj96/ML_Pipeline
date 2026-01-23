from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

bundle = joblib.load(BUNDLE_PATH)
pipe = bundle["pipeline"]
contract = bundle["contract"]
FEATURES = contract["numeric_features"]  # 10 features

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")

    X = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    y = pipe.predict(X)[0]
    return {"prediction": float(y)}
