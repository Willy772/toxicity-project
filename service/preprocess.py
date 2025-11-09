# service/preprocess.py
"""
Prétraitement + correction "lightweight" basée sur le vocabulaire du tokenizer.json.
Comportement :
 - nettoie URL / emojis / caractères indésirables
 - lowercase, normalisation unicode
 - réduit allongements (3+ -> 2) en pré-traitement
 - si tokenizer.json présent, charge le vocabulaire (word_counts / word_index)
   et essaye de corriger les tokens inconnus en utilisant la distance de Levenshtein
   en choisissant le candidat le plus fréquent parmi ceux ayant une similarité acceptable.
 - fallback : si pas de tokenizer.json, on applique seulement les nettoyages légers.
"""
from pathlib import Path
import json
import unicodedata
import re
from functools import lru_cache

EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

# ------------ utilitaires ------------
def _normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _reduce_elongation_keep_doubles(s: str) -> str:
    # réduit répétitions de 3+ caractères sur la même lettre en 2 occurrences
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

# ------------ chargement vocabulaire tokenizer.json (si présent) ------------
_TOKENIZER_VOCAB = None
_WORD_COUNTS = None
_WORDS_BY_FIRST = None
_TOP_WORDS = None

def _load_tokenizer_vocab():
    global _TOKENIZER_VOCAB, _WORD_COUNTS, _WORDS_BY_FIRST, _TOP_WORDS
    if _TOKENIZER_VOCAB is not None:
        return
    try:
        p = Path(__file__).parent / "tokenizer.json"
        if not p.exists():
            _TOKENIZER_VOCAB = None
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        # standard Keras tokenizer fields: 'word_counts' and/or 'word_index'
        word_counts = {}
        if "word_counts" in data and isinstance(data["word_counts"], dict):
            # word_counts values may be strings; convert to int if needed
            for w, c in data["word_counts"].items():
                try:
                    word_counts[w] = int(c)
                except Exception:
                    try:
                        word_counts[w] = int(float(c))
                    except Exception:
                        word_counts[w] = 1
        elif "word_index" in data and isinstance(data["word_index"], dict):
            # fallback: build uniform counts from word_index
            for w in data["word_index"].keys():
                word_counts[w] = 1
        else:
            _TOKENIZER_VOCAB = None
            return

        _TOKENIZER_VOCAB = set(word_counts.keys())
        _WORD_COUNTS = word_counts

        # buckets by first letter for faster candidate search
        buckets = {}
        for w in _TOKENIZER_VOCAB:
            if not w:
                continue
            first = w[0]
            buckets.setdefault(first, []).append(w)
        _WORDS_BY_FIRST = buckets

        # precompute top words list (sorted by count desc)
        _TOP_WORDS = sorted(_TOKENIZER_VOCAB, key=lambda x: -_WORD_COUNTS.get(x, 0))

    except Exception:
        _TOKENIZER_VOCAB = None
        _WORD_COUNTS = None
        _WORDS_BY_FIRST = None
        _TOP_WORDS = None

# ------------ correction token -> mot du vocab le plus proche ------------
# paramètres ajustables
_MAX_CANDIDATES = 200        # nb max de candidats à tester (par bucket)
_MIN_RATIO = 0.70           # ratio min accepté pour remplacement (0..1)
_MAX_DISTANCE = 3           # distance de Levenshtein maximale autorisée

@lru_cache(maxsize=20000)
def _correct_token_cached(token: str):
    return _correct_token(token)

def _correct_token(token: str):
    """
    Si tokenizer vocab disponible :
      - si token connu -> retourne token
      - sinon recherche candidats dans bucket (même première lettre) + top words fallback
      - calcule ratio de similarité ; garde candidats avec ratio >= _MIN_RATIO
      - retourne meilleur candidat selon (ratio, fréquence)
    Sinon : retourne token inchangé.
    """
    if not token:
        return token

    if _TOKENIZER_VOCAB is None:
        return token

    # si déjà connu
    if token in _TOKENIZER_VOCAB:
        return token

    first = token[0]
    candidates = []
    if _WORDS_BY_FIRST and first in _WORDS_BY_FIRST:
        candidates = _WORDS_BY_FIRST[first][:_MAX_CANDIDATES]
    # fallback : top words (rare)
    if not candidates:
        candidates = _TOP_WORDS[:_MAX_CANDIDATES]

    best = None
    best_score = -1.0
    best_freq = 0
    for c in candidates:
        # quick length filter
        if abs(len(c) - len(token)) > 4:
            continue
        dist = _levenshtein_distance(token, c)
        if dist > _MAX_DISTANCE:
            continue
        ratio = 1.0 - (dist / max(len(c), len(token), 1))
        if ratio < _MIN_RATIO:
            continue
        freq = _WORD_COUNTS.get(c, 0)
        # score improvement: prefer higher ratio, then higher freq
        score = ratio + (freq / (freq + 1000)) * 0.001
        if score > best_score:
            best_score = score
            best = c
            best_freq = freq

    if best is not None:
        return best
    return token

# ------------ fonction publique clean_text ------------
def clean_text(s: str, enable_spellcorrect: bool = True) -> str:
    """
    Nettoyage + correction légère :
    - normalisation Unicode
    - suppression URL/emoji
    - lowercase
    - garde a-z0-9 et apostrophe
    - réduit allongements (3+ -> 2)
    - si enable_spellcorrect et tokenizer.json présent, corrige tokens inconnus via vocab
    """
    if s is None:
        return s

    # lazy load vocab (une seule fois)
    _load_tokenizer_vocab()

    # 1) normalise unicode
    s0 = _normalize_unicode(str(s))

    # 2) retire URLs & emojis
    s1 = URL_RE.sub(" ", s0)
    s1 = EMOJI_RE.sub(" ", s1)

    # 3) lowercase
    s1 = s1.lower()

    # 4) filtre caractères indésirables (garde letters/numbers/space/apostrophe)
    s1 = re.sub(r"[^a-z0-9\s']", " ", s1)

    # 5) collapse espaces
    s1 = re.sub(r"\s+", " ", s1).strip()

    # 6) réduction allongements (3+ -> 2)
    s_reduced = _reduce_elongation_keep_doubles(s1)

    if not enable_spellcorrect or _TOKENIZER_VOCAB is None:
        return s_reduced

    # 7) tokenisation simple (by space) + correction token par token
    tokens = s_reduced.split()
    corrected = []
    for t in tokens:
        # conserve apostrophe forms intact (e.g. "i'm") but treat token for correction
        t_clean = t.strip()
        # attempt correction
        corr = _correct_token_cached(t_clean)
        corrected.append(corr)

    return " ".join(corrected)
