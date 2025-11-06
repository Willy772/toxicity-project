from typing import List, Dict
import os
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Import relatif : preprocess.py est dans le même package "service"
from .preprocess import clean_text

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
MAX_LEN = 120
app = FastAPI(title="Toxic Comment LSTM API", version="1.0")

# Dossier du fichier courant (service/)
BASE_DIR = Path(__file__).parent.resolve()

# En CI, on peut skipper le chargement lourd
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"

# Objets chargés au startup
tokenizer = None
LABELS: List[str] = []
model = None


# --------------------------------------------------------------------
# Schémas I/O
# --------------------------------------------------------------------
class PredictIn(BaseModel):
    texts: List[str]


class PredictOut(BaseModel):
    scores: List[Dict[str, float]]


# --------------------------------------------------------------------
# Chargement lazy au démarrage
# --------------------------------------------------------------------
@app.on_event("startup")
async def load_artifacts():
    """
    Charge tokenizer, labels, et modèle Keras au démarrage.
    En CI (SKIP_MODEL_LOAD=1), on ne charge rien de lourd.
    """
    global tokenizer, LABELS, model

    if SKIP_MODEL_LOAD:
        # Mode test/CI: on évite de charger TensorFlow/artefacts
        tokenizer = None
        LABELS = []
        model = None
        return

    # Charger tokenizer
    tok_path = BASE_DIR / "tokenizer.json"
    if not tok_path.exists():
        raise RuntimeError(f"tokenizer.json introuvable: {tok_path}")
    tok_json = tok_path.read_text(encoding="utf-8")
    tokenizer = tokenizer_from_json(tok_json)

    # Charger labels
    labels_path = BASE_DIR / "labels.txt"
    if not labels_path.exists():
        raise RuntimeError(f"labels.txt introuvable: {labels_path}")
    LABELS = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Charger modèle (hors thread principal pour ne pas bloquer)
    model_path = BASE_DIR / "model.keras"
    if not model_path.exists():
        raise RuntimeError(f"model.keras introuvable: {model_path}")

    loop = asyncio.get_running_loop()
    # tf.keras.models.load_model est bloquant : exécuter dans un executor
    model = await loop.run_in_executor(None, tf.keras.models.load_model, str(model_path))


# --------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------
@app.get("/health")
def health():
    """
    - "skipped" : mode CI, pas de modèle chargé
    - "loading" : démarrage/laod en cours
    - "ready"   : tout est prêt
    """
    if SKIP_MODEL_LOAD:
        return {"status": "skipped", "labels": []}
    status = "ready" if all([tokenizer is not None, LABELS, model is not None]) else "loading"
    return {"status": status, "labels": LABELS or []}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    """
    Prédit les scores multilabel pour chaque texte.
    """
    if SKIP_MODEL_LOAD:
        # En CI on ne sert pas /predict
        raise HTTPException(status_code=503, detail="Model loading skipped (CI mode)")

    if any(x is None for x in (tokenizer, model)) or not LABELS:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    cleaned = [clean_text(t) for t in payload.texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    pad = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    preds = model.predict(pad, verbose=0)
    out: List[Dict[str, float]] = []
    for row in preds.tolist():
        out.append({LABELS[i]: float(row[i]) for i in range(len(LABELS))})

    return PredictOut(scores=out)
