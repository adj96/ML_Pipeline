from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "preprocessor_loaded" in body
    assert "model_loaded" in body
