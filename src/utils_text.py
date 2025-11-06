import re

EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = URL_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def short(s, n=140):
    s = str(s).replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s
