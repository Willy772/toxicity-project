# service/preprocess.py
import re
import unicodedata

# emoji range
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

def _normalize_unicode(s: str) -> str:
    """Normalise les formes Unicode (NFKC) pour homogénéiser accents / ligatures."""
    return unicodedata.normalize("NFKC", s)

def _reduce_elongation(s: str) -> str:
    """
    Réduit les répétitions de caractères exagérées.
    Exemples :
      niiiice -> nice
      sooo -> so
      loooool -> lol
    On réduit toute séquence de la même lettre (2+) à une seule occurrence.
    """
    # Remplace les répétitions de caractère (lettres et chiffres) par un seul exemplaire
    return re.sub(r"(.)\1+", r"\1", s)

def _levenshtein_distance(a: str, b: str) -> int:
    """Distance de Levenshtein (itérative, O(len(a)*len(b)))."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # optimisation mémoire : deux lignes
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            add = prev[j] + 1
            delete = cur[j-1] + 1
            change = prev[j-1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, change)
        prev, cur = cur, prev
    return prev[lb]

def _levenshtein_ratio(a: str, b: str) -> float:
    """Ratio de similarité : 1 - distance/max_len. Valeur entre 0 et 1."""
    max_len = max(len(a), len(b), 1)
    dist = _levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)

def clean_text(s: str, adversarial_threshold: float = 0.98) -> str:
    """
    Nettoyage + défense simple contre attaques par perturbation orthographique.
    - supprime URLs et emojis
    - normalize unicode
    - lowercase
    - supprime caractères non a-z0-9 et apostrophe
    - réduit les allongements (e.g. niiiice -> nice)
    - compare versions via Levenshtein ratio ; si la normalisation change trop
      le texte (ratio < adversarial_threshold) on renvoie la version normalisée
      (plus robuste).
    Paramètre :
      adversarial_threshold : seuil de similarité (entre 0 et 1). Plus élevé =
      plus d'agressivité dans la détection d'attaque (ex: 0.98).
    """
    if s is None:
        return s

    # 1) Normalisation unicode initiale
    s0 = _normalize_unicode(str(s))

    # 2) Retirer URLs et emojis
    s1 = URL_RE.sub(" ", s0)
    s1 = EMOJI_RE.sub(" ", s1)

    # 3) lowercase
    s1 = s1.lower()

    # 4) suppression caracteres indésirables (garde a-z0-9 et apostrophe)
    s1 = re.sub(r"[^a-z0-9\s']", " ", s1)

    # 5) collapse espace
    s1 = re.sub(r"\s+", " ", s1).strip()

    # 6) version normalisée : réduction des allongements
    s_norm = _reduce_elongation(s1)

    # 7) si la normalisation modifie beaucoup le texte, on choisit la version normalisée
    #    (protection contre 'niiiice', 'stuuupid', 'loooool', etc.)
    if s_norm != s1:
        ratio = _levenshtein_ratio(s1, s_norm)
        # seuil par défaut : 0.98 (ajuste si trop agressif)
        if ratio < adversarial_threshold:
            return s_norm

    # sinon, on renvoie la version propre habituelle
    return s1
