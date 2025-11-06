from src.utils_text import clean_text

def test_clean_text_basic():
    s = "Hello!!! Visit https://example.com 🤖 NOW."
    out = clean_text(s)
    assert out == "hello visit now"
