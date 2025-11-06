# --- skip total en CI : doit être AVANT tout import lourd ---
import os
if os.getenv("CI") == "true":
    import pytest
    pytest.skip("skip offline TF test on CI", allow_module_level=True)
# ------------------------------------------------------------



import json, re
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


SERVICE = Path("service")

def clean_text_local(s):
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

if __name__ == "__main__":
    tok = tokenizer_from_json((SERVICE/"tokenizer.json").read_text(encoding="utf-8"))
    labels = [l.strip() for l in (SERVICE/"labels.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    model = tf.keras.models.load_model(SERVICE/"model.keras")

    samples = [
        "You are awesome, thanks!",
        "You are a stupid idiot and I hate you.",
    ]
    cleaned = [clean_text_local(t) for t in samples]
    pad = pad_sequences(tok.texts_to_sequences(cleaned), maxlen=120, padding="post", truncating="post")
    preds = model.predict(pad, verbose=0)

    print("Sortie prédite (scores par label) :")
    for i, sc in enumerate(preds):
        print(f"- Ex{i+1}:")
        print({labels[j]: float(sc[j]) for j in range(len(labels))})
