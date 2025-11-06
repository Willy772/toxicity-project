import os
os.environ["SKIP_MODEL_LOAD"] = "1"  # crucial pour ne rien charger de lourd

from fastapi.testclient import TestClient
from service.app import app

def test_health_skipped():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("status") == "skipped"
    assert isinstance(payload.get("labels"), list)
