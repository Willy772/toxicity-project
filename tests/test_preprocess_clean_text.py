import importlib.util
from pathlib import Path

def load_preprocess_module():
    """Charge service/preprocess.py sans l'importer comme package (évite TensorFlow/app.py)."""
    mod_path = Path("service") / "preprocess.py"
    assert mod_path.exists(), "service/preprocess.py manquant"
    spec = importlib.util.spec_from_file_location("preprocess", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def test_clean_text_basic():
    m = load_preprocess_module()
    s = "Hello WORLD!!!  "
    assert m.clean_text(s) == "hello world"

def test_clean_text_url_and_emoji_removed():
    m = load_preprocess_module()
    s = "Visit https://example.com 😃 NOW!"
    out = m.clean_text(s)
    # URL et emoji supprimés, minuscules, espaces normalisés
    assert out == "visit now"

def test_clean_text_punctuation_and_spaces():
    m = load_preprocess_module()
    s = "I can't... believe\tthis!!"
    out = m.clean_text(s)
    # on garde l'apostrophe, ponctuation retirée, espaces normalisés
    assert out == "i can't believe this"
