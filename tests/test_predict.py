from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict():
    payload = {
        "event_ts": 1700000000,
        "shortage_flag": 0,
        "replenishment_eta_min": 15.0,
        "machine_state": "RUN",
        "down_minutes_last_60": 0.0,
        "queue_time_min": 5.0,
        "baseline_queue_min": 4.0
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
