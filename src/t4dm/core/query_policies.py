"""
κ-based query policies for routing queries to appropriate memory strata.

Each policy generates QueryFilters that constrain retrieval based on
the κ consolidation level, time range, and item type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class QueryFilters:
    """Filters applied during memory retrieval."""

    kappa_min: float | None = None
    kappa_max: float | None = None
    time_min: datetime | None = None
    time_max: datetime | None = None
    item_type: str | None = None
    session_id: str | None = None
    min_importance: float | None = None


class EpisodicPolicy:
    """Recent, low-κ episodic memories (κ < 0.3, last 24h default)."""

    def __init__(self, hours: float = 24.0, kappa_max: float = 0.3):
        self.hours = hours
        self.kappa_max = kappa_max

    def filters(self, now: datetime | None = None) -> QueryFilters:
        now = now or datetime.now()
        return QueryFilters(
            kappa_min=0.0,
            kappa_max=self.kappa_max,
            time_min=now - timedelta(hours=self.hours),
            time_max=now,
            item_type="episodic",
        )


class SemanticPolicy:
    """Consolidated semantic knowledge (κ > 0.7, all time)."""

    def __init__(self, kappa_min: float = 0.7):
        self.kappa_min = kappa_min

    def filters(self) -> QueryFilters:
        return QueryFilters(
            kappa_min=self.kappa_min,
            kappa_max=1.0,
            item_type="semantic",
        )


class ProceduralPolicy:
    """Procedural memories ranked by success_rate in metadata."""

    def __init__(self, min_importance: float = 0.0):
        self.min_importance = min_importance

    def filters(self) -> QueryFilters:
        return QueryFilters(
            item_type="procedural",
            min_importance=self.min_importance,
        )


def select_policy(
    query_type: str = "episodic",
) -> EpisodicPolicy | SemanticPolicy | ProceduralPolicy:
    """Select a query policy by name."""
    policies = {
        "episodic": EpisodicPolicy,
        "semantic": SemanticPolicy,
        "procedural": ProceduralPolicy,
    }
    cls = policies.get(query_type)
    if cls is None:
        raise ValueError(f"Unknown query type: {query_type}")
    return cls()
