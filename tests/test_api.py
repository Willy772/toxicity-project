from fastapi.testclient import TestClient
from service.app import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ready"

def test_predict_endpoint():
    with TestClient(app) as client:
        payload = {"texts": ["I love this product!", "This is awful."]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert isinstance(data["scores"], list)
        assert isinstance(data["scores"][0], dict)
