from pathlib import Path

# Par défaut – peuvent être écrasés en CLI
CSV_PATH   = Path("data/train.csv")
N_ROWS     = 3000

# Dossier d’export des artefacts (étape 3)
EXPORT_DIR = Path("service")

# spaCy model
SPACY_MODEL = "en_core_web_sm"
