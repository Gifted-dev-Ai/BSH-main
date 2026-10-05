"""Text normalization for robust pattern matching.

German shop-floor notes are full of accents, ``ß``, inconsistent casing, line
breaks and typos. We match lexicon patterns against an **accent-folded, lower-cased**
form so patterns can be written in plain ASCII (e.g. ``duse`` matches ``Düse``) and a
few well-known typos collapse onto their canonical spelling.
"""

from __future__ import annotations

import re
import unicodedata

# Known typos / spelling variants -> canonical fragment (applied on folded text).
# Keep this small; prefer flexible regex in the lexicon over piling up entries here.
_TYPO_FIXES: list[tuple[str, str]] = [
    (r"tackt", "takt"),            # Tackt zeit -> takt
    (r"verkann?tet", "verkantet"), # verkanntet -> verkantet
    (r"blickiert", "blockiert"),   # blickiert -> blockiert
    (r"vereinzel(?:n|ner|er|r)", "vereinzeler"),  # Vereinzler/Vereinzelner -> vereinzeler
]


def fold(text: str) -> str:
    """Lower-case and strip accents (ä->a, ö->o, ü->u, ß->ss); collapse whitespace."""
    if text is None:
        return ""
    s = str(text).replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize(text: str) -> str:
    """Folded text with known typos repaired — the string lexicon patterns match on."""
    s = fold(text)
    for pat, repl in _TYPO_FIXES:
        s = re.sub(pat, repl, s)
    return s
