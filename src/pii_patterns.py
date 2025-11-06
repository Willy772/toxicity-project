import re

REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "IP":    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "URL":   re.compile(r"https?://\S+|www\.\S+"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"),
    "TIME":  re.compile(r"\b\d{1,2}:\d{2}\b"),
    "DATE":  re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?)"
        r"[\w\s,]*\d{4}\b", re.IGNORECASE
    ),
}

ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:rue|avenue|av\.?|bd|boulevard|impasse|allée|route|chemin|che\.?|place|pl\.?|quai|square|sq\.?|"
    r"street|st\.?|ave\.?|road|rd\.?|blvd\.?|lane|ln\.?|drive|dr\.?|court|ct\.?)\s+[A-Za-zÀ-ÖØ-öø-ÿ'’\-\. ]+\b",
    re.IGNORECASE
)
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
USERNAME_RE    = re.compile(r"(?<!\w)@[\w._\-]{2,32}")

PII_LABELS = ["EMAIL","IP","PHONE","URL","ADDRESS","CREDIT_CARD","USERNAME","PERSON"]
