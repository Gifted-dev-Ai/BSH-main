"""Structured information extraction from free-text maintenance columns.

Mines the German shop-floor free-text fields of the incident log
(``Problemdes`` + ``Moduldescr`` + ``Errdescr``) into structured, bilingual
columns: **component**, **phenomenon**, **action** and **reason**.

The core is an extensible, deterministic **lexicon / rule-based** extractor
(``lexicon.py`` holds the curated taxonomy — extend it by adding one ``Term``).
All extractors implement the same :class:`~src.eda.extraction.extractor.Extractor`
interface, so an LLM- or spaCy-backed extractor can be dropped in later without
changing callers.

Run:  ``python -m src.eda.extraction``  -> data/processed/incident_extracted.csv
"""

from .extractor import Extractor, LexiconExtractor, ExtractionResult
from .lexicon import LEXICON, Term, CATEGORIES

__all__ = ["Extractor", "LexiconExtractor", "ExtractionResult",
           "LEXICON", "Term", "CATEGORIES"]
