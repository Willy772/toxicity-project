from service.preprocess import clean_text

def test_clean_text_basic():
    raw = "Hello!!! Visit https://example.com 😊  "
    out = clean_text(raw)
    # Doit être en minuscules, sans URL/emoji/ponctuation double
    assert out == "hello visit", f"Unexpected clean_text output: {out}"

def test_clean_text_apostrophes_and_spaces():
    raw = "I'm  HAPPY...  Aren't you??"
    out = clean_text(raw)
    # Apostrophes conservées, espaces normalisés
    assert out == "i'm happy aren't you", f"Unexpected: {out}"
