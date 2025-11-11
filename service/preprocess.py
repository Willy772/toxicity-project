# service/preprocess.py
"""
Prétraitement + correction ultra-légère pour petits CPU et textes courts (<= ~50 chars).
API identique (clean_text, enable_spellcorrect).

Stratégie :
 - Nettoyage léger + réduction d’allongements (3+ -> 2) en amont
 - Correction étage A : bucket + Levenshtein (borné)
 - Correction étage B : fallback "Norvig-like" (edits1/edits2) très borné,
   priorisé par fréquence du tokenizer.json (si présent)

Réglages optimisés CPU:
 - _MIN_RATIO élevé (0.78) pour limiter les remplacements hasardeux
 - _MAX_DISTANCE réduit (2)
 - _MAX_CANDIDATES réduit (120)
 - Génération d’edits très limitée, alphabétisée par le vocab
 - caches LRU + coupes agressives
"""
from pathlib import Path
import json
import unicodedata
import re
from functools import lru_cache

EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
RE_SPACES = re.compile(r"\s+")
RE_NONALNUM_APOS = re.compile(r"[^a-z0-9\s']")

# ------------ utilitaires ------------
def _normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _reduce_elongation_keep_doubles(s: str) -> str:
    # réduit répétitions de 3+ caractères identiques en 2 occurrences (cooool -> coool)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def _collapse_all_elongations_to_single(s: str) -> str:
    # variante agressive utilisée uniquement pour la génération de candidats (cooool -> col)
    return re.sub(r"(.)\1{1,}", r"\1", s)

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
_ALPHABET = None  # lettres observées dans le vocab

def _load_tokenizer_vocab():
    global _TOKENIZER_VOCAB, _WORD_COUNTS, _WORDS_BY_FIRST, _TOP_WORDS, _ALPHABET
    if _TOKENIZER_VOCAB is not None:
        return
    try:
        p = Path(__file__).parent / "tokenizer.json"
        if not p.exists():
            _TOKENIZER_VOCAB = None
            return
        data = json.loads(p.read_text(encoding="utf-8"))

        word_counts = {}
        if "word_counts" in data and isinstance(data["word_counts"], dict):
            for w, c in data["word_counts"].items():
                if not isinstance(w, str): continue
                w = w.lower()
                try:
                    word_counts[w] = int(c)
                except Exception:
                    try:
                        word_counts[w] = int(float(c))
                    except Exception:
                        word_counts[w] = 1
        elif "word_index" in data and isinstance(data["word_index"], dict):
            for w in data["word_index"].keys():
                if isinstance(w, str):
                    word_counts[w.lower()] = 1
        else:
            _TOKENIZER_VOCAB = None
            return

        _TOKENIZER_VOCAB = set(word_counts.keys())
        _WORD_COUNTS = word_counts

        # buckets par première lettre
        buckets = {}
        for w in _TOKENIZER_VOCAB:
            if not w: continue
            buckets.setdefault(w[0], []).append(w)
        _WORDS_BY_FIRST = buckets

        # top words par fréquence
        _TOP_WORDS = sorted(_TOKENIZER_VOCAB, key=lambda x: -_WORD_COUNTS.get(x, 0))

        # alphabet observé (limite la génération de candidats)
        letters = set()
        for w in _TOKENIZER_VOCAB:
            for ch in w:
                if "a" <= ch <= "z" or "0" <= ch <= "9" or ch == "'":
                    letters.add(ch)
        if not letters:
            letters = set("abcdefghijklmnopqrstuvwxyz0123456789'")
        _ALPHABET = "".join(sorted(letters))

    except Exception:
        _TOKENIZER_VOCAB = None
        _WORD_COUNTS = None
        _WORDS_BY_FIRST = None
        _TOP_WORDS = None
        _ALPHABET = None

# ------------ correction token -> mot du vocab le plus proche ------------
# paramètres CPU-light (ajustables)
_MAX_CANDIDATES = 120      # nb max candidats par bucket (réduit)
_MIN_RATIO = 0.78          # ratio min accepté (plus strict)
_MAX_DISTANCE = 2          # distance Levenshtein max (réduit)

@lru_cache(maxsize=8000)   # cache plus petit (mémoire réduite)
def _correct_token_cached(token: str):
    return _correct_token(token)

def _best_by_ratio_and_freq(token: str, candidates):
    """Sélectionne le meilleur candidat en combinant similarité et fréquence."""
    best = None
    best_score = -1.0
    for c in candidates:
        if abs(len(c) - len(token)) > 3:  # filtre longueur plus strict
            continue
        dist = _levenshtein_distance(token, c)
        if dist > _MAX_DISTANCE:
            continue
        ratio = 1.0 - (dist / max(len(c), len(token), 1))
        if ratio < _MIN_RATIO:
            continue
        freq = _WORD_COUNTS.get(c, 0)
        # biais fréquence minimal pour départager les ex-aequo
        score = ratio + (freq / (freq + 800.0)) * 0.0008
        if score > best_score:
            best_score = score
            best = c
    return best

# ---- Tiny model Norvig-like (fallback, très borné) ----
@lru_cache(maxsize=8000)
def _edits1(word: str):
    letters = _ALPHABET or "abcdefghijklmnopqrstuvwxyz0123456789'"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

    # remplacements/insertions limités : on échantillonne les lettres
    # pour rester très compact sur CPU faible
    limited_letters = letters[:18]  # tronque l'alphabet (≈ 18 symboles suffisent en pratique)
    replaces   = [L + c + (R[1:] if R else "") for L, R in splits if R for c in limited_letters]
    inserts    = [L + c + R for L, R in splits for c in limited_letters]

    # limite dure globale
    pool = deletes + transposes
    pool += replaces[:220] + inserts[:220]  # bornes strictes
    return set(pool[:700])  # borne finale edits1

@lru_cache(maxsize=4000)
def _edits2(word: str):
    # compose partiellement edits1(edits1(word)) en coupant très tôt
    s = set()
    for e1 in list(_edits1(word))[:150]:
        s.update(list(_edits1(e1))[:80])
        if len(s) > 1200:  # borne stricte
            break
    return s

def _known(words):
    if _TOKENIZER_VOCAB is None:
        return set()
    return {w for w in words if w in _TOKENIZER_VOCAB}

def _tiny_model_fallback(token: str):
    """
    Fallback de correction "Norvig-like" très borné avec prior de fréquence.
    candidates = known(token) ∪ known(edits1) ∪ known(collapse, edits1(collapse)) ∪ known(edits2)
    Choix = max(freq, tie-break par similarité).
    """
    if not token:
        return token

    collapsed = _collapse_all_elongations_to_single(token)

    cand_sets = []
    cand_sets.append(_known([token]))
    cand_sets.append(_known(_edits1(token)))
    if collapsed != token:
        cand_sets.append(_known([collapsed]))
        cand_sets.append(_known(_edits1(collapsed)))
    # edits2 est coûteux : on ne l’emploie que si rien trouvé
    candidates = set()
    for s in cand_sets:
        if s:
            candidates |= s
        if len(candidates) >= 400:  # borne
            break

    if not candidates:
        candidates = _known(_edits2(token))

    if not candidates:
        return None

    best = None
    best_key = (-1, -1.0)
    for c in candidates:
        freq = _WORD_COUNTS.get(c, 0)
        sim = _lev_ratio(token, c)
        key = (freq, sim)
        if key > best_key:
            best_key = key
            best = c
    return best

def _correct_token(token: str):
    """
    Étape A : bucket/Levenshtein (borné).
    Étape B : tiny model fallback (ultra-borné).
    """
    if not token:
        return token
    if _TOKENIZER_VOCAB is None:
        return token
    if token in _TOKENIZER_VOCAB:
        return token

    # -------- Étape A : bucket + Levenshtein --------
    first = token[0]
    candidates = []
    if _WORDS_BY_FIRST and first in _WORDS_BY_FIRST:
        candidates = _WORDS_BY_FIRST[first][:_MAX_CANDIDATES]
    if not candidates:
        candidates = _TOP_WORDS[:_MAX_CANDIDATES]

    best = _best_by_ratio_and_freq(token, candidates)
    if best is not None:
        return best

    # -------- Étape B : tiny model fallback --------
    tm_best = _tiny_model_fallback(token)
    if tm_best:
        return tm_best

    return token

# ------------ fonction publique clean_text ------------
def clean_text(s: str, enable_spellcorrect: bool = True) -> str:
    """
    Nettoyage + correction légère (API inchangée) :
    - normalisation Unicode
    - suppression URL/emoji
    - lowercase
    - garde a-z0-9 et apostrophe
    - réduit allongements (3+ -> 2)
    - si enable_spellcorrect & tokenizer.json présent:
         A) correction bucket+Levenshtein (bornée)
         B) fallback tiny model (ultra-borné)
    """
    if s is None:
        return s

    _load_tokenizer_vocab()

    # 1) normalise unicode
    s0 = _normalize_unicode(str(s))

    # 2) retire URLs & emojis
    s1 = URL_RE.sub(" ", s0)
    s1 = EMOJI_RE.sub(" ", s1)

    # 3) lowercase
    s1 = s1.lower()

    # 4) filtre caractères indésirables (garde letters/numbers/space/apostrophe)
    s1 = RE_NONALNUM_APOS.sub(" ", s1)

    # 5) collapse espaces
    s1 = RE_SPACES.sub(" ", s1).strip()

    # 6) réduction allongements (3+ -> 2)
    s_reduced = _reduce_elongation_keep_doubles(s1)

    if not enable_spellcorrect or _TOKENIZER_VOCAB is None:
        return s_reduced

    # 7) tokenisation simple + correction
    tokens = s_reduced.split()
    corrected = []
    for t in tokens:
        t_clean = t.strip()
        corr = _correct_token_cached(t_clean)
        corrected.append(corr)

    return " ".join(corrected)
