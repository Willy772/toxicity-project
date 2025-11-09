import re
import unicodedata

# ----------------------------
# (1) Nettoyage d'origine (inchangé)
# ----------------------------

EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

def clean_text(s: str) -> str:
    """
    Nettoyage de base (conservé tel quel pour compatibilité tests) :
    - minuscules
    - suppression URL / emoji
    - garder lettres/chiffres/espace/apostrophe
    - normalisation des espaces
    """
    s = str(s).lower()
    s = URL_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------
# (2) Défense GENERIQUE (sans liste de mots)
# ----------------------------

# Caractères invisibles / zero-width & espaces exotiques
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")  # ZWSP, ZWJ, ZWNJ, WJ, BOM
MULTISPACES_RE = re.compile(r"\s+")

# Mapping leet/symboles courant -> ASCII (générique, pas spécifique à des mots)
LEET_TABLE = str.maketrans({
    "0": "o", "1": "i", "2": "z", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g",
    "@": "a", "$": "s", "+": "t", "!": "i"
})

# Apostrophes/tirets typographiques -> ASCII
PUNCT_SWAP_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201B": "'", "´": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-", "–": "-", "—": "-",
})

# Répétitions et ponctuation
REPEAT_3PLUS = re.compile(r"(.)\1{2,}")            # aaa -> aa
REPEAT_PUNCT = re.compile(r"([!?.,'\"-])\1{1,}")   # ??!! -> ? !

def _strip_zero_width(s: str) -> str:
    return ZERO_WIDTH_RE.sub("", s)

def _nfkc_fold_ascii(s: str) -> str:
    # Normalisation NFKC (homoglyphes), puis suppression des accents
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if unicodedata.category(ch) != "Mn")
    return s

def _swap_typo_punct(s: str) -> str:
    return s.translate(PUNCT_SWAP_TABLE)

def _leet_normalize(s: str) -> str:
    return s.translate(LEET_TABLE)

def _collapse_repetitions(s: str) -> str:
    s = REPEAT_3PLUS.sub(r"\1\1", s)     # limite 3+ identiques à 2 (niiiice -> niice)
    s = REPEAT_PUNCT.sub(r"\1", s)       # ponctuation doublée -> simple
    return s

def normalize_text(s: str) -> str:
    """
    Normalisation défensive GENERIQUE (sans liste de mots) :
    1) supprime zero-width et BOM
    2) NFKC + désaccentuation
    3) normalise apostrophes/tirets typographiques
    4) convertit leet/symboles courants vers ASCII
    5) borne les répétitions (3+ -> 2) et ponctuation doublée
    6) normalise les espaces
    """
    s = str(s)
    s = _strip_zero_width(s)
    s = _nfkc_fold_ascii(s).lower()
    s = _swap_typo_punct(s)
    s = _leet_normalize(s)
    s = _collapse_repetitions(s)
    s = MULTISPACES_RE.sub(" ", s).strip()
    return s

def secure_preprocess(s: str) -> str:
    """
    Pipeline recommandé côté API avant tokenization :
    secure_preprocess = normalize_text(clean_text(s))
    """
    return normalize_text(clean_text(s))
