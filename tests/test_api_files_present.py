from pathlib import Path
import re

def test_service_core_files_exist():
    root = Path("service")
    assert (root / "app.py").exists(), "service/app.py manquant"
    assert (root / "requirements.txt").exists(), "service/requirements.txt manquant"
    assert (root / "Dockerfile").exists(), "service/Dockerfile manquant"
    assert (root / "preprocess.py").exists(), "service/preprocess.py manquant"

def test_requirements_contains_fastapi_and_uvicorn():
    txt = (Path("service") / "requirements.txt").read_text(encoding="utf-8")
    assert "fastapi" in txt.lower()
    assert "uvicorn" in txt.lower()

def test_app_imports_preprocess_relatively():
    code = (Path("service") / "app.py").read_text(encoding="utf-8")
    assert re.search(r"from\s+\.\s*preprocess\s+import\s+clean_text", code), \
        "app.py devrait faire un import relatif: from .preprocess import clean_text"
