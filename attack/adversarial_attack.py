import requests

API_URL = "http://104.155.67.205//predict"

samples = ["You are stupid", "You are nice"]
adv_samples = ["You are stuppid", "You are niiice"]

for text, adv in zip(samples, adv_samples):
    r1 = requests.post(API_URL, json={"texts": [text]})
    r2 = requests.post(API_URL, json={"texts": [adv]})
    print(f"Avant : {r1.json()} | Après (adversarial) : {r2.json()}")
