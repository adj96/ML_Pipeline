from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = requests.get("http://localhost:8000/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
