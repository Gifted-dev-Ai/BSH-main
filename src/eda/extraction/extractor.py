"""Extractor interface + the lexicon-based implementation.

``Extractor`` is the stable seam: any extractor (lexicon, spaCy, LLM) takes the
free-text fields of one incident and returns an :class:`ExtractionResult`. Callers
(the ``run`` CLI) depend only on this interface, so a future LLM extractor can be
swapped in — or chained as a fallback — without touching the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .lexicon import CATEGORIES, LEXICON, Term
from .normalize import normalize


@dataclass
class ExtractionResult:
    """Structured output for one incident."""

    labels: dict[str, list[str]] = field(default_factory=dict)  # category -> ["de (en)", ...]

    def joined(self, category: str) -> str:
        return "; ".join(self.labels.get(category, []))

    @property
    def n_matched_categories(self) -> int:
        return sum(1 for c in CATEGORIES if self.labels.get(c))


@runtime_checkable
class Extractor(Protocol):
    """Anything that turns the free-text fields of one incident into structure."""

    def extract(self, texts: dict[str, str]) -> ExtractionResult:
        ...


class LexiconExtractor:
    """Deterministic regex/keyword extractor driven by :data:`LEXICON`.

    The configured source columns are concatenated, normalized once, then every
    lexicon term is tested. All matching canonical labels are kept per category.
    """

    def __init__(self, lexicon: dict[str, list[Term]] = LEXICON,
                 source_cols: tuple[str, ...] = ("Problemdes", "Moduldescr",
                                                 "Errdescr")):
        self.lexicon = lexicon
        self.source_cols = source_cols
        # Pre-compile every pattern once, grouped per (category, term).
        self._compiled: dict[str, list[tuple[Term, list[re.Pattern]]]] = {
            cat: [(t, [re.compile(p) for p in t.patterns]) for t in terms]
            for cat, terms in lexicon.items()
        }

    def _consolidate(self, texts: dict[str, str]) -> str:
        parts = [str(texts.get(c, "")) for c in self.source_cols
                 if texts.get(c) not in (None, "")]
        return normalize(" . ".join(parts))

    def extract(self, texts: dict[str, str]) -> ExtractionResult:
        norm = self._consolidate(texts)
        labels: dict[str, list[str]] = {}
        if not norm:
            return ExtractionResult(labels)
        for cat, terms in self._compiled.items():
            hits: list[str] = []
            for term, regexes in terms:
                if any(rx.search(norm) for rx in regexes):
                    hits.append(term.label)
            if hits:
                labels[cat] = hits
        return ExtractionResult(labels)


class ChainExtractor:
    """Run a primary extractor, then fill only the *empty* categories from fallbacks.

    Enables `lexicon -> LLM` without changing the pipeline: pass
    ``ChainExtractor([LexiconExtractor(), LlmExtractor()])`` once the latter exists.
    """

    def __init__(self, extractors: list[Extractor]):
        self.extractors = extractors

    def extract(self, texts: dict[str, str]) -> ExtractionResult:
        merged: dict[str, list[str]] = {}
        for ex in self.extractors:
            res = ex.extract(texts)
            for cat, vals in res.labels.items():
                if not merged.get(cat):
                    merged[cat] = vals
        return ExtractionResult(merged)
