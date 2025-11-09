from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import asyncio
import os
import numpy as np  # ✅ pour garantir un ndarray au modèle

# === Import sécurisé du préprocesseur ===
# essaie d'importer secure_preprocess, sinon fallback sur clean_text
try:
    from .preprocess import secure_preprocess as _preprocess_fn
    _SECURE_MODE = True
except ImportError:
    from .preprocess import clean_text as _preprocess_fn
    _SECURE_MODE = False

MAX_LEN = 120
app = FastAPI(title="Social Score message", version="1.0")
BASE_DIR = Path(__file__).parent

# Globals initialisés à None, alimentés au startup
tokenizer = None   # doit exposer .texts_to_sequences(list[str]) -> List[List[int]]
LABELS = None      # list[str]
model = None       # doit exposer .predict(np.ndarray) -> np.ndarray shape (N, len(LABELS))


def _pad(seqs, maxlen=MAX_LEN):
    """Padding simple sans TensorFlow (right pad avec 0)."""
    out = []
    for s in seqs:
        s = list(s)[:maxlen]
        if len(s) < maxlen:
            s = s + [0] * (maxlen - len(s))
        out.append(s)
    return out


@app.on_event("startup")
async def load_artifacts():
    """Chargement lazy des artefacts. Skippable en CI via APP_SKIP_STARTUP=1."""
    if os.getenv("APP_SKIP_STARTUP", "0") == "1":
        return

    global tokenizer, LABELS, model

    # Charger tokenizer / labels (imports tardifs pour éviter de charger TF inutilement)
    tok_json = (BASE_DIR / "tokenizer.json").read_text(encoding="utf-8")
    from tensorflow.keras.preprocessing.text import tokenizer_from_json  # lazy import
    tokenizer = tokenizer_from_json(tok_json)

    LABELS = [
        l.strip() for l in (BASE_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    # Charger le modèle en thread (pas de asyncio.run ici)
    import tensorflow as tf  # lazy import
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(
        None,
        tf.keras.models.load_model,
        str(BASE_DIR / "model.keras"),
    )


class PredictIn(BaseModel):
    texts: List[str]


class PredictOut(BaseModel):
    # On renvoie uniquement la classe top-1 pour chaque texte
    labels: List[str]


@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "secure_mode": _SECURE_MODE
    }


@app.get("/health")
def health():
    status = "ready" if all([tokenizer, LABELS, model]) else "loading"
    return {
        "status": status,
        "labels": LABELS or [],
        "secure_mode": _SECURE_MODE
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    assert tokenizer is not None and model is not None and LABELS is not None, "Model not ready yet"

    # 1) preprocess (clean_text ou secure_preprocess selon ce qui est dispo)
    cleaned = [_preprocess_fn(t) for t in payload.texts]

    # 2) padding hors-TF
    pad = _pad(seqs=tokenizer.texts_to_sequences(cleaned), maxlen=MAX_LEN)

    # 3) garantir un ndarray int32 pour Keras
    arr = np.asarray(pad, dtype="int32")

    # 4) forward
    preds = model.predict(arr, verbose=0) if hasattr(model, "predict") else model(arr)
    preds = np.asarray(preds)

    # 5) formatage : renvoie uniquement le label top-1 (argmax)
    top_indices = np.argmax(preds, axis=1).tolist()
    top_labels = [LABELS[i] for i in top_indices]

    return PredictOut(labels=top_labels)
