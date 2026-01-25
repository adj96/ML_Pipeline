from fastapi.testclient import TestClient
import os
import src.app as appmod

# Force model path to the real artifact for test runtime
appmod.MODEL_PATH = os.path.abspath("src/model.joblib")

client = TestClient(appmod.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
