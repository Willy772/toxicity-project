import argparse, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from .config import CSV_PATH, N_ROWS
from .dataio import load_df
from .utils_text import clean_text

LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def main(csv: str, n_rows: int, use_anonymized: bool = True):
    dfN = load_df(csv, n_rows)

    for c in LABEL_COLS:
        dfN[c] = pd.to_numeric(dfN[c], errors="coerce").fillna(0).astype(int)

    text_col = "comment_text_anonymized" if use_anonymized and "comment_text_anonymized" in dfN.columns else "comment_text"
    dfN["text_clean"] = dfN[text_col].apply(clean_text)
    X = dfN["text_clean"].astype(str).tolist()
    Y = dfN[LABEL_COLS].values

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    print("Taille train/val :", len(X_train), "/", len(X_val))

    MAX_VOCAB, MAX_LEN = 8000, 120
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)
    Xtr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post")
    Xva = pad_sequences(tokenizer.texts_to_sequences(X_val),   maxlen=MAX_LEN, padding="post", truncating="post")

    model = models.Sequential([
        layers.Embedding(input_dim=MAX_VOCAB, output_dim=64),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(LABEL_COLS), activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[])

    t0 = time.perf_counter()
    hist = model.fit(Xtr, Y_train, validation_data=(Xva, Y_val), epochs=8, batch_size=8, verbose=0)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    Yp  = model.predict(Xva, verbose=0)
    pred_time = time.perf_counter() - t1
    Yhat = (Yp >= 0.5).astype(int)

    print("=== Modèle (BiLSTM) ===")
    print(f"Temps entraînement : {train_time:.3f}s | Prédiction : {pred_time:.3f}s")
    for avg in ("micro","macro"):
        p = precision_score(Y_val, Yhat, average=avg, zero_division=0)
        r = recall_score   (Y_val, Yhat, average=avg, zero_division=0)
        f = f1_score       (Y_val, Yhat, average=avg, zero_division=0)
        print(f"{avg.title():<5} P={p:.3f} R={r:.3f} F1={f:.3f}")

    # Sauvegarde “checkpoint” pour l’étape 3
    model.save("service/model.keras")
    with open("service/tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    with open("service/labels.txt", "w", encoding="utf-8") as f:
        for lab in LABEL_COLS:
            f.write(lab + "\n")

    print("Artefacts sauvegardés dans ./service")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    ap.add_argument("--use-anonymized", action="store_true")
    args = ap.parse_args()
    main(args.csv, args.n_rows, use_anonymized=args.use_anonymized)
