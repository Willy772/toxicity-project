# Toxicity Project — FastAPI + LSTM (Digital Social Score)

[![CI](https://github.com/Willy772/toxicity-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Willy772/toxicity-project/actions/workflows/ci.yml)

API de détection de **toxicité de commentaires** (binaire `toxic` / `non toxic`) basée sur un **BiLSTM Keras**, servie via **FastAPI**.  
CI via **GitHub Actions** ; CD prêt pour **GCP** (Cloud Build → **GKE**).  
Conforme RGPD par **anonymisation amont** et **minimisation** (voir *Model Card*).

---

## 🗂️ Structure

```
toxicity-project/
├─ .github/workflows/ci.yml        # Tests unitaires (CI)
├─ service/                        # API FastAPI + artefacts modèle
│  ├─ app.py                       # Endpoints /health, /predict (binaire)
│  ├─ preprocess.py                # Nettoyage/normalisation des textes
│  ├─ model.keras                  # Modèle BiLSTM sauvegardé
│  ├─ tokenizer.json               # Tokenizer Keras
│  ├─ labels.txt                   # Labels d'entraînement (6 catégories)
│  ├─ requirements.txt             # Dépendances API
│  └─ Dockerfile                   # Image API
├─ src/                            # Pipeline entraînement & anonymisation
│  ├─ step1_anonymize.py           # Anonymisation (spaCy + regex)
│  ├─ step2_train.py               # Entraînement BiLSTM
│  └─ step3_export.py              # Export artefacts vers /service
├─ k8s/                            # Manifests GKE (Deployment/Service/HPA)
├─ tests/                          # Tests unitaires légers
├─ cloudbuild.yaml                 # CD Cloud Build (build/push/deploy)
└─ README.md
.
.
.
```

---

## 🚀 Démarrage rapide (local)

### 1) Prérequis
- Python **3.10+**
- (Optionnel) Docker 24+
- (Optionnel) spaCy `en_core_web_sm` si tu lances l’entraînement

### 2) Installation & run API
```bash
git clone https://github.com/Willy772/toxicity-project.git
cd toxicity-project
python -m venv .venv
# Windows: . .venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Dépendances API uniquement
pip install -U pip
pip install -r service/requirements.txt

# Lancer l’API
python -m uvicorn service.app:app --port 8080
# ➜ http://127.0.0.1:8080  (docs Swagger: /docs)
```

---

## 🧪 Tests (CI)

```bash
pytest -q
```
Les tests valident :
- la présence des fichiers clés de l’API,
- le comportement de `clean_text()`.

Le pipeline **GitHub Actions** s’exécute automatiquement sur `main`.

---

## 🧠 Entraînement du modèle (optionnel)

> Si tu souhaites régénérer `model.keras`, `tokenizer.json`, `labels.txt`.

1) Installer les dépendances “full” (entraînement + anonymisation) :
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2) Anonymisation (remplace DCP par tokens) :
```bash
python -m src.step1_anonymize --csv data/train.csv --n-rows 3000 --mask-labels
```

3) Entraînement BiLSTM + export API :
```bash
python -m src.step2_train --csv data/train.csv --n-rows 3000 --use-anonymized
python -m src.step3_export
# ➜ artefacts dans /service
```

---

## 🐳 Docker

```bash
cd service
docker build -t toxicity-api:local .
docker run -p 8080:8080 toxicity-api:local
# ➜ http://127.0.0.1:8080
```

---

## ☁️ Déploiement Cloud

### Option A — GKE (Kubernetes)

1) **Build & push** via Cloud Build (déclenché par commit)  
   Le fichier `cloudbuild.yaml` :
- construit l’image depuis `service/`,
- pousse dans **Artifact Registry**,
- `kubectl apply -f k8s/`,
- `kubectl set image` sur le Deployment.

2) **Manifests** (`k8s/`)
- `deployment.yaml` : `toxicity-api` (probes, ressources)
- `service.yaml` : `type: LoadBalancer` (IP publique)
- `hpa.yaml` (optionnel) : auto-scale sur CPU

> **Coûts faibles** : 1 seul nœud, HPA désactivé, `requests`/`limits` modestes.

### Option B — Cloud Run (conseillée si trafic faible)
```bash
gcloud run deploy toxicity-api   --image=europe-west1-docker.pkg.dev/PROJECT_ID/toxicity/toxicity-api:latest   --region=europe-west1 --memory=2Gi --cpu=1   --allow-unauthenticated
```
**Scale-to-zero** → 0 € sans trafic.

---

## 🔐 Sécurité & Conformité

- **Entrées** : nettoyage strict + option `secure_preprocess`  
- **Surface limitée** : l’API **ne renvoie pas** les scores bruts → sortie **binaire** `toxic/non toxic` (mitige *model extraction*).  
- **RGPD** : anonymisation amont (DCP masquées), pas de stockage des payloads d’inférence.  
- **Cloud** : TLS, IAM, isolation par conteneurs, Artifact Registry

📄 **Model Card RGPD** : voir `Model_Card_RGPD.md` .

---

## 📚 API (OpenAPI)

- **Docs interactives** : `GET /docs`
- **Santé** : `GET /health`  
  Renvoie `status`, `labels`, `secure_mode`.
- **Prédiction** : `POST /predict`
```json
// Input
{"texts":["hello world"]}

// Output binaire
{"labels":["non toxic"]}
```

---

## 🧩 Roadmap

- quotas
- Observabilité (traces, métriques custom)
- Cloud Run as a Service (coûts optimisés)

---

## 👤 Auteur / Licence

- Auteur : **Willy772**, **YannickNino**   
- Projet académique — usage éducatif.
