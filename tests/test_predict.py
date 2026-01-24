from fastapi.testclient import TestClient
import src.app as appmod

client = TestClient(appmod.app)

def test_model_artifacts_exist():
    assert appmod.PREPROCESSOR_PATH.exists()
    assert appmod.MODEL_PATH.exists()

def test_predict():
    payload = {
        "event_ts": 1700000000,
        "shortage_flag": 0,
        "replenishment_eta_min": 0,
        "machine_state": "RUN",
        "down_minutes_last_60": 0,
        "queue_time_min": 5,
        "baseline_queue_min": 5,
    }

    r = client.post("/predict", json=payload)

    assert r.status_code == 200, r.text
    body = r.json()

    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
