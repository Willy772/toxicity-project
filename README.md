## Lancer l’API en local (Windows / VS Code)

```powershell
# Cloner
git clone https://github.com/Willy772/toxicity-project.git
cd toxicity-project

# Créer / activer venv
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Installer deps
pip install -U pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# (si les artefacts ne sont pas versionnés)
# 1) mettre data\train.csv
# 2) entraîner et exporter
python -m src.step2_train --csv data\train.csv --n-rows 3000 --use-anonymized
python -m src.step3_export

# Lancer l’API
cd service
python -m uvicorn app:app --port 8080
