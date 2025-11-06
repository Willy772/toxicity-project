import re
from typing import Dict
import spacy
from .pii_patterns import REGEX_PATTERNS, ADDRESS_RE, CREDIT_CARD_RE, USERNAME_RE, PII_LABELS
from .config import SPACY_MODEL

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    # permet "python -m spacy download en_core_web_sm" avant, sinon lève une erreur claire
    raise RuntimeError(f"spaCy model '{SPACY_MODEL}' introuvable. Installe-le: python -m spacy download {SPACY_MODEL}")

def anonymize_text(text: str, counts: Dict[str, int], use_label_tokens: bool = True) -> str:
    def _mask(label: str) -> str:
        return label if use_label_tokens else "****"

    if not text:
        return text
    tmp = text

    # 1) Regex DCP
    for label in ["EMAIL","IP","PHONE","URL"]:
        rx = REGEX_PATTERNS[label]
        def repl(m, L=label):
            counts[L] = counts.get(L, 0) + 1
            return _mask(L)
        tmp = rx.sub(repl, tmp)

    # Adresse postale
    tmp = ADDRESS_RE.sub(lambda m: (counts.__setitem__("ADDRESS", counts.get("ADDRESS",0)+1)) or _mask("ADDRESS"), tmp)

    # Cartes bancaires (filtrage)
    def cc_repl(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19:
            counts["CREDIT_CARD"] = counts.get("CREDIT_CARD", 0) + 1
            return _mask("CREDIT_CARD")
        return m.group(0)
    tmp = CREDIT_CARD_RE.sub(cc_repl, tmp)

    # @username
    tmp = USERNAME_RE.sub(lambda m: (counts.__setitem__("USERNAME", counts.get("USERNAME",0)+1)) or _mask("USERNAME"), tmp)

    # 2) spaCy PERSON uniquement
    doc = nlp(tmp)
    out, last = [], 0
    for ent in doc.ents:
        if ent.start_char > last:
            out.append(tmp[last:ent.start_char])
        if ent.label_ == "PERSON":
            counts["PERSON"] = counts.get("PERSON", 0) + 1
            out.append(_mask("PERSON"))
            last = ent.end_char
    out.append(tmp[last:])
    anonymized = "".join(out)

    # Nettoyage PERSON PERSON
    anonymized = re.sub(r"(PERSON)(?:\s*,?\s*PERSON)+", "PERSON", anonymized)
    return anonymized
