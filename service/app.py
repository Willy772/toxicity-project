from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import asyncio
import os
import numpy as np  # ✅ garantir un ndarray pour le modèle

# === Import sécurisé du préprocesseur ===
try:
    from .preprocess import secure_preprocess as _preprocess_fn  # type: ignore
    _SECURE_MODE = True
except Exception:
    from .preprocess import clean_text as _preprocess_fn
    _SECURE_MODE = False

MAX_LEN = 120
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.5"))

app = FastAPI(title="social comment score", version="1.0")
BASE_DIR = Path(__file__).parent

# Globals initialisés à None, alimentés au startup
tokenizer = None   # .texts_to_sequences(list[str]) -> List[List[int]]
LABELS = None      # list[str]
model = None       # .predict(np.ndarray) -> np.ndarray shape (N, len(LABELS))


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
    # On renvoie "toxic" ou "non toxic" pour chaque texte (seuil sur le score toxic)
    labels: List[str]


@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "secure_mode": _SECURE_MODE,
        "toxic_threshold": TOXIC_THRESHOLD,
    }


@app.get("/health")
def health():
    status = "ready" if all([tokenizer, LABELS, model]) else "loading"
    return {
        "status": status,
        "labels": LABELS or [],
        "secure_mode": _SECURE_MODE,
        "toxic_threshold": TOXIC_THRESHOLD,
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    assert tokenizer is not None and model is not None and LABELS is not None, "Model not ready yet"

    # 0) trouver l'index du label "toxic"
    toxic_idx = next((i for i, lab in enumerate(LABELS) if lab.lower() == "toxic"), None)
    if toxic_idx is None:
        raise HTTPException(status_code=500, detail="Label 'toxic' introuvable dans LABELS.")

    # 1) preprocess (clean_text ou secure_preprocess selon ce qui est dispo)
    cleaned = [_preprocess_fn(t) for t in payload.texts]

    # 2) tokenisation + padding hors-TF
    seqs = tokenizer.texts_to_sequences(cleaned)
    pad = _pad(seqs=seqs, maxlen=MAX_LEN)

    # 3) ndarray int32 pour Keras
    arr = np.asarray(pad, dtype="int32")

    # 4) forward
    preds = model.predict(arr, verbose=0) if hasattr(model, "predict") else model(arr)
    preds = np.asarray(preds)  # shape (N, C)

    # 5) décision : si score toxic > seuil -> "toxic" sinon "non toxic"
    out_labels = []
    for row in preds:
        toxic_score = float(row[toxic_idx])
        out_labels.append("toxic" if toxic_score > TOXIC_THRESHOLD else "non toxic")

    return PredictOut(labels=out_labels)
