import argparse, json
from pathlib import Path
from .config import EXPORT_DIR
from .utils_text import clean_text

SERVICE = EXPORT_DIR  # Path("service")

def write_preprocess():
    (SERVICE).mkdir(parents=True, exist_ok=True)
    (SERVICE / "preprocess.py").write_text(
        "import re\n"
        "EMOJI_RE = re.compile(r\"[\\U00010000-\\U0010ffff]\", flags=re.UNICODE)\n"
        "URL_RE   = re.compile(r\"https?://\\S+|www\\.\\S+\")\n"
        "def clean_text(s: str) -> str:\n"
        "    s = str(s).lower()\n"
        "    s = URL_RE.sub(\" \", s)\n"
        "    s = EMOJI_RE.sub(\" \", s)\n"
        "    s = re.sub(r\"[^a-z0-9\\s']\", \" \", s)\n"
        "    s = re.sub(r\"\\s+\", \" \", s).strip()\n"
        "    return s\n",
        encoding="utf-8"
    )

def write_app():
    (SERVICE / "app.py").write_text(
        "from typing import List, Dict\n"
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n"
        "import tensorflow as tf\n"
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n"
        "from tensorflow.keras.preprocessing.text import tokenizer_from_json\n"
        "from preprocess import clean_text\n"
        "import json\n"
        "MAX_LEN = 120\n"
        "with open('tokenizer.json','r',encoding='utf-8') as f:\n"
        "    tok_json = f.read()\n"
        "tokenizer = tokenizer_from_json(tok_json)\n"
        "with open('labels.txt','r',encoding='utf-8') as f:\n"
        "    LABELS = [l.strip() for l in f if l.strip()]\n"
        "model = tf.keras.models.load_model('model.keras')\n"
        "app = FastAPI(title='Toxic Comment LSTM API', version='1.0')\n"
        "class PredictIn(BaseModel):\n"
        "    texts: List[str]\n"
        "class PredictOut(BaseModel):\n"
        "    scores: List[Dict[str, float]]\n"
        "@app.get('/health')\n"
        "def health():\n"
        "    return {'status':'ok','labels':LABELS}\n"
        "@app.post('/predict', response_model=PredictOut)\n"
        "def predict(payload: PredictIn):\n"
        "    cleaned = [clean_text(t) for t in payload.texts]\n"
        "    seqs = tokenizer.texts_to_sequences(cleaned)\n"
        "    pad = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')\n"
        "    preds = model.predict(pad, verbose=0)\n"
        "    out = []\n"
        "    for row in preds.tolist():\n"
        "        out.append({LABELS[i]: float(row[i]) for i in range(len(LABELS))})\n"
        "    return PredictOut(scores=out)\n",
        encoding="utf-8"
    )

def write_api_requirements_and_dockerfile():
    (SERVICE / "requirements.txt").write_text(
        "fastapi==0.115.0\nuvicorn[standard]==0.30.6\ntensorflow==2.16.1\n", encoding="utf-8"
    )
    (SERVICE / "Dockerfile").write_text(
        "FROM python:3.10-slim\n\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends build-essential "
        "&& rm -rf /var/lib/apt/lists/*\n\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/requirements.txt\n"
        "RUN pip install --no-cache-dir -r /app/requirements.txt\n\n"
        "COPY model.keras /app/model.keras\n"
        "COPY tokenizer.json /app/tokenizer.json\n"
        "COPY labels.txt /app/labels.txt\n"
        "COPY preprocess.py /app/preprocess.py\n"
        "COPY app.py /app/app.py\n\n"
        "EXPOSE 8080\n"
        "CMD [\"uvicorn\",\"app:app\",\"--host\",\"0.0.0.0\",\"--port\",\"8080\"]\n",
        encoding="utf-8"
    )

def main():
    SERVICE.mkdir(parents=True, exist_ok=True)
    # On suppose que model.keras, tokenizer.json, labels.txt existent déjà (Étape 2)
    assert (SERVICE / "model.keras").exists(),  "service/model.keras manquant (exécute step2_train)"
    assert (SERVICE / "tokenizer.json").exists(),"service/tokenizer.json manquant (exécute step2_train)"
    assert (SERVICE / "labels.txt").exists(),   "service/labels.txt manquant (exécute step2_train)"

    write_preprocess()
    write_app()
    write_api_requirements_and_dockerfile()
    print("API et fichiers d’export prêts dans ./service")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
