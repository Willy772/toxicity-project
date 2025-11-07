📘 README.md
# 🧠 Toxicity Project — FastAPI + LSTM

Ce projet met en place une API **FastAPI** qui expose un modèle **LSTM** de classification de toxicité de commentaires.  
L’intégration continue est assurée via **GitHub Actions** et le déploiement continu (CD) pourra être effectué sur **Google Kubernetes Engine (GKE)** via **Cloud Build**.

---

## 📂 Structure du projet

```bash
toxicity-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Workflow GitHub Actions : tests légers
│
├── service/                    # Service FastAPI exporté
│   ├── app.py                  # API principale FastAPI (.keras compatible)
│   ├── preprocess.py           # Nettoyage de texte (utilisé dans l'API)
│   ├── model.keras             # Modèle LSTM sauvegardé
│   ├── tokenizer.json          # Tokenizer Keras
│   ├── labels.txt              # Liste des labels multilabel
│   ├── requirements.txt        # Dépendances du service
│   └── Dockerfile              # Image Docker pour déploiement sur GKE
│
├── tests/                      # Tests unitaires légers (CI)
│   ├── test_preprocess_clean_text.py
│   └── test_api_files_present.py
│
├── .gitignore
├── pytest.ini                  # Restreint pytest à tests/
├── README.md                   # Ce document
└── requirements.txt (optionnel si besoin racine)

⚙️ Pré-requis

Python 3.10+

Git

VS Code / Terminal

(Optionnel) Docker si tu veux lancer l’image

🚀 Lancer le projet localement
1️⃣ Cloner le dépôt
git clone https://github.com/Willy772/toxicity-project.git
cd toxicity-project

2️⃣ Créer et activer un environnement virtuel
🪟 Sous Windows PowerShell :
python -m venv .venv
. .venv\Scripts\Activate.ps1

🐧 Sous Linux / macOS :
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Installer les dépendances du service
pip install -U pip
pip install -r service/requirements.txt


(les dépendances incluent fastapi, uvicorn, tensorflow, etc.)

4️⃣ Lancer l’API FastAPI

Depuis le dossier service/ :

cd service
python -m uvicorn app:app --port 8080


L’API démarre sur :
👉 http://127.0.0.1:8080

🔍 Vérification rapide

Endpoint de santé :
→ http://127.0.0.1:8080/health

Exemple de requête POST /predict :

curl -X POST http://127.0.0.1:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"texts": ["You are awesome!", "You are a stupid idiot."]}'

🧪 Lancer les tests localement

Les tests vérifient :

Le comportement de clean_text()

La présence et la structure des fichiers clés de service/

pytest


(pas besoin de TensorFlow ou du modèle pour ces tests — ils sont légers et rapides)

🧰 Intégration Continue (CI)

Le workflow GitHub Actions (.github/workflows/ci.yml) exécute automatiquement les tests à chaque push ou pull request sur main.

Badge à ajouter dans ton README (une fois le pipeline vert) :

![CI](https://github.com/Willy772/toxicity-project/actions/workflows/ci.yml/badge.svg)

🐳 Lancer avec Docker

Depuis la racine du projet :

cd service
docker build -t toxicity-api .
docker run -p 8080:8080 toxicity-api


Puis ouvre http://127.0.0.1:8080

☁️ Étapes futures — Déploiement continu (CD)

Le pipeline CD (prochaine étape) consistera à :

Cloud Build → Build & push image vers Artifact Registry

GKE (Kubernetes) → Déploiement automatisé via kubectl apply

GitHub Actions → Déclenchement de Cloud Build à chaque push sur main

(sera ajouté dans .github/workflows/deploy.yml et cloudbuild.yaml)

📄 Licence

Projet académique — libre d’utilisation à des fins éducatives.

✨ Auteur

Willy772
Projet réalisé dans le cadre de l’ESIGELEC — 2025.


---

### ✅ Tu peux coller ce texte directement dans ton `README.md` à la racine du projet.

# Souhaites-tu que je t’ajoute **le badge CI prêt à l’emploi** (avec ton lien GitHub A