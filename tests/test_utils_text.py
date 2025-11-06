from src.utils_text import clean_text

def test_clean_text_basic():
    s = "Hello!!!"
    out = clean_text(s)
    # URLs/emoji retirés, minuscules, caractères non alnum enlevés, espaces normalisés
    assert out == "hello visit now"
