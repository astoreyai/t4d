"""Simple TF-IDF Memory Baseline for comparison benchmarks.

A naive memory implementation using TF-IDF similarity. Demonstrates the
value T4DM adds over simple approaches. No neural embeddings.
"""
from __future__ import annotations

import math
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryEntry:
    """A single memory entry with content and metadata."""
    id: str
    content: str
    tokens: list[str]
    timestamp: float
    tf: dict[str, float] = field(default_factory=dict)


class SimpleMemory:
    """Simple TF-IDF memory baseline with optional LRU recency weighting.

    Args:
        max_size: Maximum entries (LRU eviction when exceeded).
        recency_weight: Weight for recency vs similarity (0=TF-IDF, 1=recency).
    """

    def __init__(self, max_size: int = 1000, recency_weight: float = 0.0) -> None:
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._max_size = max_size
        self._recency_weight = max(0.0, min(1.0, recency_weight))
        self._df: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]+\b", text.lower())

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        return {tok: c / len(tokens) for tok, c in counts.items()}

    def _update_df(self, tokens: list[str], add: bool = True) -> None:
        delta = 1 if add else -1
        for tok in set(tokens):
            self._df[tok] = self._df.get(tok, 0) + delta
            if self._df[tok] <= 0:
                del self._df[tok]

    def _get_idf(self, term: str) -> float:
        n = len(self._entries)
        if n == 0:
            return 0.0
        df = self._df.get(term, 0)
        return math.log((n + 1) / (df + 1)) + 1 if df else 0.0

    def _tfidf_similarity(self, tf_query: dict[str, float], entry: MemoryEntry) -> float:
        if not tf_query or not entry.tf:
            return 0.0
        terms = set(tf_query.keys()) | set(entry.tf.keys())
        dot, norm_q, norm_e = 0.0, 0.0, 0.0
        for term in terms:
            idf = self._get_idf(term)
            tq, te = tf_query.get(term, 0.0) * idf, entry.tf.get(term, 0.0) * idf
            dot += tq * te
            norm_q += tq * tq
            norm_e += te * te
        return dot / (math.sqrt(norm_q) * math.sqrt(norm_e)) if norm_q and norm_e else 0.0

    def store(self, content: str, id: Optional[str] = None) -> str:
        """Store content and return its ID."""
        entry_id = id or uuid.uuid4().hex
        tokens = self._tokenize(content)
        tf = self._compute_tf(tokens)
        # Evict oldest if at capacity (LRU)
        if len(self._entries) >= self._max_size and entry_id not in self._entries:
            _, oldest = self._entries.popitem(last=False)
            self._update_df(oldest.tokens, add=False)
        # Update or insert
        if entry_id in self._entries:
            old = self._entries.pop(entry_id)
            self._update_df(old.tokens, add=False)
        entry = MemoryEntry(id=entry_id, content=content, tokens=tokens,
                           timestamp=time.time(), tf=tf)
        self._entries[entry_id] = entry
        self._update_df(tokens, add=True)
        return entry_id

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for similar memories. Returns list of {id, content, score}."""
        if not self._entries:
            return []
        tf_query = self._compute_tf(self._tokenize(query))
        timestamps = [e.timestamp for e in self._entries.values()]
        min_ts, max_ts = min(timestamps), max(timestamps)
        ts_range = max_ts - min_ts if max_ts > min_ts else 1.0
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            sim = self._tfidf_similarity(tf_query, entry)
            recency = (entry.timestamp - min_ts) / ts_range
            score = (1 - self._recency_weight) * sim + self._recency_weight * recency
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": e.id, "content": e.content, "score": s} for s, e in scored[:k]]

    def delete(self, id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted, False if not found."""
        if id not in self._entries:
            return False
        entry = self._entries.pop(id)
        self._update_df(entry.tokens, add=False)
        return True

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()
        self._df.clear()
