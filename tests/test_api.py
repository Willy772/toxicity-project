import os
os.environ["APP_SKIP_STARTUP"] = "1"

from fastapi.testclient import TestClient
from service.app import app

class DummyTokenizer:
    def texts_to_sequences(self, texts):
        return [[1, 2, 3] for _ in texts]  # peu importe, c'est mocké

class DummyModel:
    def predict(self, pad, verbose=0):
        return [[0.1, 0.9] for _ in pad]

def test_api_predict_monkeypatch(monkeypatch):
    from service import app as mod
    mod.tokenizer = DummyTokenizer()
    mod.LABELS = ["nice", "toxic"]
    mod.model = DummyModel()

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"

    r = client.post("/predict", json={"texts": ["You are awesome", "I hate you"]})
    assert r.status_code == 200
    scores = r.json()["scores"]
    assert len(scores) == 2
    assert set(scores[0].keys()) == {"nice", "toxic"}
