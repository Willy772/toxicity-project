# attack/extractor.py
import requests
import json
import time
from pathlib import Path
from itertools import chain

API = "http://104.155.67.205/predict"
OUT = Path("attack/results/queries_responses.jsonl")
OUT.parent.mkdir(exist_ok=True)

# Générateur simple de textes variés (tu peux remplacer par un corpus réel)
def gen_texts(n=1000):
    base = [
        # Non toxiques
        "You are great",
        "I love this",
        "Thanks for your help",
        "You did a wonderful job",
        "I appreciate your support",
        "This product is excellent",
        "Have a nice day",
        "That’s a great idea",
        "I totally agree with you",
        "This is very helpful",
        "Congratulations on your success",
        "I enjoyed reading your comment",
        "Well done!",
        "You’re very kind",
        "This looks promising",

        # Toxiques (insultes/hostilité sans menaces)
        "I hate you",
        "This product is terrible",
        "You are an idiot",
        "You’re so stupid",
        "What a dumb comment",
        "Shut up, loser",
        "You’re worthless",
        "Get lost",
        "You don’t know anything",
        "This is garbage",
        "You’re a joke",
        "Such an annoying person",
        "You sound clueless",
        "Pathetic response",

        # Mélange supplémentaire (neutre / critique / poli / rude)
        "I don’t like this at all",
        "Please consider improving the design",
        "I disagree, but thanks for sharing",
        "Could you clarify your point?",
        "Your tone feels rude",
        "This makes no sense to me",
        "Great explanation, very clear",
        "Interesting perspective",
        "That was unnecessary",
        "Let’s keep this civil, please"
    ]

    for i in range(n):
        s = base[i % len(base)]
        # variations simples (dup, typos, synonyms) -> utile pour extraction
        if i % 5 == 0:
            s = s + " " + ("!" * ((i % 3)+1))
        if i % 7 == 0:
            s = s.replace(" ", "  ")
        yield s

def chunked(iterable, size=16):
    it = iter(iterable)
    while True:
        chunk = list()
        for _ in range(size):
            try:
                chunk.append(next(it))
            except StopIteration:
                break
        if not chunk:
            break
        yield chunk

def main():
    qgen = gen_texts(2000)  # adapte le volume
    for batch in chunked(qgen, size=8):
        payload = {"texts": batch}
        r = requests.post(API, json=payload, timeout=10)
        if r.status_code != 200:
            print("err", r.status_code, r.text)
            time.sleep(1)
            continue
        res = r.json()
        for t, out in zip(batch, res["scores"]):
            OUT.write_text(json.dumps({"text": t, "scores": out}) + "\n", encoding="utf-8", append=False)
        time.sleep(0.2)  # throttle to avoid immediate blocking

if __name__ == "__main__":
    main()
