# test_predict.py
# Purpose: Jenkins smoke-test for model.joblib (single artifact pipeline = preprocessing + model)
from fastapi.testclient import TestClient
from src.app import app


def test_predict_endpoint():
    with TestClient(app) as client:
        payload = {
        "event_ts": "2026-01-24 10:00:00",
        "baseline_queue_min": 12.0,
        "shortage_flag": 0,
        "replenishment_eta_min": 0.0,
        "machine_state": "RUN",
        "queue_time_min": 10.0,
        "down_minutes_last_60": 0.0,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert "prediction" in body
