from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict_ok():
    payload = {
        "event_ts": "2026-02-03T00:00:00Z",
        "priority_urgent": 0,
        "line_id": "L1",
        "product_family": "PF1",
        "station": "ST01",
        "shift": "A",
        "remaining_units": 10,
        "queue_time_min": 2.0,
        "queue_length": 2,
        "machine_state": "RUN",
        "down_minutes_last_60": 0.0,
        "alarm_rate_last_30": 0.0,
        "cycle_time_expected_sec": 12.0,
        "cycle_time_actual_sec": 12.5,
        "cycle_time_deviation": 0.5,
        "shortage_flag": 0,
        "replenishment_eta_min": 5.0,
        "shortage_severity": 0.0,
        "operator_present": 1,
        "skill_level": "S1",
        "coverage_ratio": 1.0,
        "baseline_queue_min": 1.0,
        "station_backlog_ratio": 0.2,
        "delay_flag": 0,
    }

    res = client.post("/predict", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
