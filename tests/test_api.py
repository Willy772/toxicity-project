import os
os.environ["APP_SKIP_STARTUP"] = "1"  # ne pas charger TF/model à l'import

from fastapi.testclient import TestClient
from service.app import app

# Injecter des mocks légers
class DummyTokenizer:
    def __init__(self):
        # mapping ultra simple pour test
        self.vocab = {"you":1, "are":2, "awesome":3, "stupid":4, "idiot":5, "i":6, "hate":7, "youre":8}
    def texts_to_sequences(self, texts):
        seqs = []
        for t in texts:
            toks = t.lower().split()
            seqs.append([self.vocab.get(w, 9) for w in toks])  # 9 = <unk>
        return seqs

class DummyModel:
    def predict(self, pad, verbose=0):
        # renvoie une proba constante pour 2 labels (exemple)
        import numpy as np
        n = len(pad)
        # Deux scores par échantillon
        return [[0.1, 0.9] for _ in range(n)]

def test_api_predict_monkeypatch(monkeypatch):
    # On installe le tokenizer, labels, model directement dans le module
    from service import app as mod
    mod.tokenizer = DummyTokenizer()
    mod.LABELS = ["nice", "toxic"]
    mod.model = DummyModel()

    client = TestClient(app)

    # /health doit refléter l'état "ready"
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["labels"] == ["nice", "toxic"]

    # /predict doit renvoyer 2 scores par texte
    payload = {"texts": ["You are awesome", "You are a stupid idiot"]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()["scores"]
    assert len(out) == 2
    assert set(out[0].keys()) == {"nice", "toxic"}
