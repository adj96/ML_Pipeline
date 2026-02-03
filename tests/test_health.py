# tests/test_health.py
from fastapi.testclient import TestClient
from src.app import app


def test_health_ok_payload():
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert "status" in body
    assert body["status"] == "ok"

    # these keys must exist in the response
    assert "model_loaded" in body

    # optional: enforce booleans
    assert isinstance(body["model_loaded"], bool)

