from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import asyncio
import os

# import relatif vers preprocess.py
from .preprocess import clean_text

MAX_LEN = 120
app = FastAPI(title="Toxic Comment LSTM API", version="1.0")

BASE_DIR = Path(__file__).parent

# Globals initialisés à None, alimentés au startup
tokenizer = None  # doit fournir .texts_to_sequences(list[str]) -> List[List[int]]
LABELS = None     # list[str]
model = None      # doit fournir .predict(ndarray|list) -> list/ndarray shape (N, len(LABELS))

def _pad(seqs, maxlen=MAX_LEN):
    """Padding simple sans TensorFlow."""
    out = []
    for s in seqs:
        s = list(s)[:maxlen]
        if len(s) < maxlen:
            s = s + [0] * (maxlen - len(s))
        out.append(s)
    return out

@app.on_event("startup")
async def load_artifacts():
    # Permettre à la CI de skipper le chargement lourd
    if os.getenv("APP_SKIP_STARTUP", "0") == "1":
        return

    global tokenizer, LABELS, model

    # Charger tokenizer / labels
    tok_json = (BASE_DIR / "tokenizer.json").read_text(encoding="utf-8")
    from tensorflow.keras.preprocessing.text import tokenizer_from_json  # import tardif
    tokenizer = tokenizer_from_json(tok_json)

    LABELS = [l.strip() for l in (BASE_DIR / "labels.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    # Charger le modèle (import tardif + thread séparé)
    async def _load():
        import tensorflow as tf
        return tf.keras.models.load_model(str(BASE_DIR / "model.keras"))

    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, lambda: asyncio.run(_load()))

class PredictIn(BaseModel):
    texts: List[str]

class PredictOut(BaseModel):
    scores: List[Dict[str, float]]

@app.get("/health")
def health():
    status = "ready" if all([tokenizer, LABELS, model]) else "loading"
    return {"status": status, "labels": LABELS or []}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    assert tokenizer is not None and model is not None and LABELS is not None, "Model not ready yet"
    cleaned = [clean_text(t) for t in payload.texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    pad = _pad(seqs, maxlen=MAX_LEN)
    # Appel modèle (signature compat Keras: .predict(...))
    preds = model.predict(pad, verbose=0) if hasattr(model, "predict") else model(pad)  # support dummy
    out = []
    for row in preds:
        out.append({LABELS[i]: float(row[i]) for i in range(len(LABELS))})
    return PredictOut(scores=out)
