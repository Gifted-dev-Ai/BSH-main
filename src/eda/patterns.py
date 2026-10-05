"""Lightweight, dependency-free pattern mining for the incident deep-dive.

Two families, kept generic so they can be reused / extended:

* **Association mining (Apriori)** — frequent itemsets + association rules
  (support / confidence / lift) over a list of transactions (sets of items).
  Used for *dependence* among component / phenomenon / action / reason within a
  single incident.

* **Recency-aware sequence mining** — frequent n-grams over a time-ordered event
  stream, where two events are only chained if the gap between them is within a
  recency window. Used for *frequent sub-sequences* of consecutive incidents.

Everything is plain Python (small data), so there are no external dependencies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations


# --------------------------------------------------------------------------- #
# Apriori association rules
# --------------------------------------------------------------------------- #
def frequent_itemsets(transactions: list[set], min_count: int = 3,
                      max_len: int = 3) -> dict[frozenset, int]:
    """Return {itemset: support_count} for itemsets occurring >= ``min_count``."""
    items: Counter = Counter()
    for t in transactions:
        items.update(t)
    freq: dict[frozenset, int] = {frozenset([i]): c for i, c in items.items()
                                  if c >= min_count}
    current = list(freq)
    k = 2
    while current and k <= max_len:
        candidates: set[frozenset] = set()
        for a, b in combinations(current, 2):
            u = a | b
            if len(u) == k:
                candidates.add(u)
        current = []
        for c in candidates:
            cnt = sum(1 for t in transactions if c <= t)
            if cnt >= min_count:
                freq[c] = cnt
                current.append(c)
        k += 1
    return freq


@dataclass
class Rule:
    antecedent: frozenset
    consequent: frozenset
    count: int
    support: float          # P(A ∪ B)
    confidence: float       # P(B | A)
    lift: float             # confidence / P(B)


def association_rules(transactions: list[set], min_count: int = 3,
                      min_confidence: float = 0.6, max_len: int = 3) -> list[Rule]:
    """Mine association rules from transactions, ranked by lift then confidence."""
    n = len(transactions)
    if n == 0:
        return []
    freq = frequent_itemsets(transactions, min_count, max_len)

    def count(items: frozenset) -> int:
        return sum(1 for t in transactions if items <= t)

    rules: list[Rule] = []
    for itemset, cnt in freq.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante_t in combinations(sorted(itemset), r):
                ante = frozenset(ante_t)
                cons = itemset - ante
                ante_cnt = count(ante)
                cons_cnt = count(cons)
                if ante_cnt == 0:
                    continue
                conf = cnt / ante_cnt
                lift = conf / (cons_cnt / n) if cons_cnt else float("nan")
                if conf >= min_confidence:
                    rules.append(Rule(ante, cons, cnt, cnt / n, conf, lift))
    rules.sort(key=lambda x: (x.lift if x.lift == x.lift else 0.0,
                              x.confidence, x.count), reverse=True)
    return rules


# --------------------------------------------------------------------------- #
# Recency-aware sequence mining
# --------------------------------------------------------------------------- #
def recency_ngrams(events: list[tuple], max_gap_days: float, max_n: int = 3
                   ) -> tuple[Counter, dict]:
    """Frequent consecutive n-grams over a time-ordered event stream.

    ``events`` is a list of ``(timestamp, label)`` sorted by timestamp. Two events
    are only chained when their gap is <= ``max_gap_days`` (recency), so unrelated
    events far apart in time are never joined into a sub-sequence.

    Returns ``(grams, spans)`` where ``grams[tuple] = count`` and
    ``spans[tuple] = [total_span_days, ...]`` for n-grams with n >= 2.
    """
    grams: Counter = Counter()
    spans: dict = defaultdict(list)
    m = len(events)
    for i in range(m):
        seq = [events[i][1]]
        span = 0.0
        for j in range(i + 1, min(i + max_n, m)):
            gap = (events[j][0] - events[j - 1][0]).total_seconds() / 86400.0
            if gap > max_gap_days:
                break
            seq.append(events[j][1])
            span += gap
            key = tuple(seq)
            grams[key] += 1
            spans[key].append(span)
    return grams, spans
