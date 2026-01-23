# tests/test_predict.py

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_predict():
    payload = {
        "machine_age_days": 10,
        "temperature_c": 25,
        "pressure_kpa": 101,
        "vibration_mm_s": 1.2,
        "humidity_pct": 60,
        "operator_experience_yrs": 3,
        "shift": 2,
        "material_grade": 1,
        "line_speed_m_min": 120,
        "inspection_interval_hrs": 8
    }

    r = client.post("/predict", json=payload)

    assert r.status_code == 200
    body = r.json()

    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
