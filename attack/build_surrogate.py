# attack/eval_surrogate.py
import joblib, requests, json
from sklearn.metrics import accuracy_score
from pathlib import Path

vec, clf = joblib.load("attack/results/surrogate.pkl")
test_texts = ["You are wonderful", "You are stupid"]
prod_preds, local_preds = [], []
API = "http://104.155.67.205/predict"

r = requests.post(API, json={"texts": test_texts})
prod_res = r.json()["scores"]
# normaliser prod labels
prod_labels = [max(d.items(), key=lambda x: x[1])[0] for d in prod_res]
local_labels = clf.predict(vec.transform(test_texts)).tolist()

print("prod:", prod_labels)
print("local:", local_labels)
print("accuracy:", accuracy_score(prod_labels, local_labels))
