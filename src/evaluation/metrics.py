from __future__ import annotations

import math
from typing import Iterable


def dcg_at_k(relevances: Iterable[float], k: int) -> float:
    rels = list(relevances)[:k]
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(rels))


def ndcg_at_k(relevances: Iterable[float], k: int) -> float:
    rels = list(relevances)
    best = sorted(rels, reverse=True)
    ideal = dcg_at_k(best, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(rels, k) / ideal


def precision_at_k(relevances: Iterable[float], k: int, threshold: float = 1.0) -> float:
    rels = list(relevances)[:k]
    if not rels:
        return 0.0
    hits = sum(1 for r in rels if r >= threshold)
    return hits / len(rels)


def recall_at_k(
    relevances: Iterable[float],
    total_relevant: int,
    k: int,
    threshold: float = 1.0,
) -> float:
    if total_relevant <= 0:
        return 0.0
    rels = list(relevances)[:k]
    hits = sum(1 for r in rels if r >= threshold)
    return hits / total_relevant