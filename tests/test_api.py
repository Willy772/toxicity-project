from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_predict_endpoint():
    payload = {"texts": ["I love this product!", "This is awful."]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert isinstance(data["scores"], list)
    assert isinstance(data["scores"][0], dict)
