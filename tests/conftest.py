from pathlib import Path
import sys
import os

# Ajouter la racine du repo au PYTHONPATH pour "from src..." / "from service..."
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Par sécurité, on s'assure que l'app ne charge rien de lourd au démarrage des tests
os.environ.setdefault("APP_SKIP_STARTUP", "1")
