# service/preprocess.py
import re
import unicodedata

EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

def _normalize_unicode(s: str) -> str:
    # Homogénéise accents/ligatures (é→e si composé, etc.)
    return unicodedata.normalize("NFKC", s)

def _reduce_elongation_keep_doubles(s: str) -> str:
    """
    Réduit uniquement les répétitions de 3+ occurrences à 2.
    Ex: niiiice -> niice, loooool -> lool, soooo -> so
    (préserve les doubles valides: hello, pizza, effet)
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            add = prev[j] + 1
            delete = cur[j-1] + 1
            change = prev[j-1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, change)
        prev, cur = cur, prev
    return prev[lb]

def _lev_ratio(a: str, b: str) -> float:
    max_len = max(len(a), len(b), 1)
    return 1.0 - (_levenshtein_distance(a, b) / max_len)

def clean_text(s: str, adversarial_threshold: float = 0.98) -> str:
    """
    Nettoyage + petite défense contre orthographe adversariale:
      - retire URLs, emojis
      - normalise unicode
      - lowercase
      - garde [a-z0-9 ']
      - réduit 3+ répétitions à 2 (préserve les doubles)
      - si la normalisation change beaucoup le texte (ratio < seuil), on garde la version normalisée
    """
    if s is None:
        return s

    # 1) Normalisation unicode
    s0 = _normalize_unicode(str(s))

    # 2) URLs & emojis
    s1 = URL_RE.sub(" ", s0)
    s1 = EMOJI_RE.sub(" ", s1)

    # 3) lowercase
    s1 = s1.lower()

    # 4) filtrage caractères (on garde lettres/chiffres/espace/apostrophe)
    s1 = re.sub(r"[^a-z0-9\s']", " ", s1)

    # 5) normalisation espaces
    s1 = re.sub(r"\s+", " ", s1).strip()

    # 6) réduction des allongements (3+ -> 2)
    s_norm = _reduce_elongation_keep_doubles(s1)

    # 7) si la réduction a réellement “corrigé” un texte très étiré, on préfère s_norm
    if s_norm != s1:
        if _lev_ratio(s1, s_norm) < adversarial_threshold:
            return s_norm

    return s1
