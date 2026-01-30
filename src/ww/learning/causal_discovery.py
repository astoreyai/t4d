"""
Causal Discovery and Attribution.

P4-2: Learn causal relationships from memory-outcome pairs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships."""
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    MODULATES = "modulates"
    CORRELATES = "correlates"


@dataclass
class CausalEdge:
    """A directed causal relationship."""
    source_id: UUID
    target_id: UUID
    relation: CausalRelationType
    strength: float = 0.5
    evidence_count: int = 1
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source_id),
            "target": str(self.target_id),
            "relation": self.relation.value,
            "strength": self.strength,
            "evidence": self.evidence_count,
        }


@dataclass
class CausalAttribution:
    """Attribution of outcome to causal factors."""
    outcome_id: UUID
    attributions: list[tuple[UUID, float]]
    total_explained: float
    residual: float

    def get_top_causes(self, k: int = 5) -> list[tuple[UUID, float]]:
        sorted_attr = sorted(self.attributions, key=lambda x: x[1], reverse=True)
        return sorted_attr[:k]


@dataclass
class CausalDiscoveryConfig:
    """Configuration for causal discovery."""
    learning_rate: float = 0.1
    decay_rate: float = 0.01
    min_evidence: int = 3
    max_context_depth: int = 10
    attribution_threshold: float = 0.05
    max_edges_per_node: int = 20
    prune_threshold: float = 0.1


class CausalGraph:
    """Directed acyclic graph of causal relationships."""

    def __init__(self, config: CausalDiscoveryConfig | None = None):
        self.config = config or CausalDiscoveryConfig()
        self._outgoing: dict[UUID, list[CausalEdge]] = defaultdict(list)
        self._incoming: dict[UUID, list[CausalEdge]] = defaultdict(list)
        self._total_edges = 0
        self._updates = 0

    def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        relation: CausalRelationType = CausalRelationType.CAUSES,
        initial_strength: float = 0.5,
    ) -> CausalEdge:
        for edge in self._outgoing[source_id]:
            if edge.target_id == target_id:
                edge.evidence_count += 1
                edge.strength = min(1.0, edge.strength + self.config.learning_rate)
                edge.last_updated = datetime.now()
                self._updates += 1
                return edge

        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            strength=initial_strength,
        )
        self._outgoing[source_id].append(edge)
        self._incoming[target_id].append(edge)
        self._total_edges += 1

        if len(self._outgoing[source_id]) > self.config.max_edges_per_node:
            self._prune_weak_edges(source_id)

        return edge

    def weaken_edge(self, source_id: UUID, target_id: UUID, amount: float | None = None) -> None:
        amount = amount or self.config.learning_rate
        for edge in self._outgoing[source_id]:
            if edge.target_id == target_id:
                edge.strength = max(0.0, edge.strength - amount)
                edge.last_updated = datetime.now()
                self._updates += 1
                return

    def get_causes(self, target_id: UUID) -> list[CausalEdge]:
        return list(self._incoming[target_id])

    def get_effects(self, source_id: UUID) -> list[CausalEdge]:
        return list(self._outgoing[source_id])

    def _prune_weak_edges(self, node_id: UUID) -> None:
        self._outgoing[node_id] = [
            e for e in self._outgoing[node_id] if e.strength >= self.config.prune_threshold
        ]

    def decay_all(self) -> None:
        for edges in self._outgoing.values():
            for edge in edges:
                edge.strength *= (1.0 - self.config.decay_rate)

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_edges": self._total_edges,
            "total_updates": self._updates,
            "nodes_with_outgoing": len(self._outgoing),
            "nodes_with_incoming": len(self._incoming),
        }


class CausalAttributor:
    """Attribute outcomes to causal factors."""

    def __init__(self, graph: CausalGraph, config: CausalDiscoveryConfig | None = None):
        self.graph = graph
        self.config = config or CausalDiscoveryConfig()

    def attribute(self, outcome_id: UUID, context_ids: list[UUID] | None = None) -> CausalAttribution:
        attributions: dict[UUID, float] = defaultdict(float)
        direct_causes = self.graph.get_causes(outcome_id)

        if not direct_causes:
            return CausalAttribution(outcome_id=outcome_id, attributions=[], total_explained=0.0, residual=1.0)

        total_strength = sum(e.strength for e in direct_causes)
        if total_strength > 0:
            for edge in direct_causes:
                if context_ids is None or edge.source_id in context_ids:
                    contribution = edge.strength / total_strength
                    attributions[edge.source_id] += contribution

        filtered = [
            (cause_id, contrib)
            for cause_id, contrib in attributions.items()
            if contrib >= self.config.attribution_threshold
        ]
        total_explained = sum(c for _, c in filtered)

        return CausalAttribution(
            outcome_id=outcome_id,
            attributions=filtered,
            total_explained=min(1.0, total_explained),
            residual=max(0.0, 1.0 - total_explained),
        )


class CausalLearner:
    """Learn causal structure from experience."""

    def __init__(self, config: CausalDiscoveryConfig | None = None):
        self.config = config or CausalDiscoveryConfig()
        self.graph = CausalGraph(self.config)
        self.attributor = CausalAttributor(self.graph, self.config)
        self._observation_buffer: list[tuple[list[UUID], UUID]] = []
        self._max_buffer = 1000
        self._total_observations = 0
        self._causal_edges_learned = 0

    def observe(self, context_ids: list[UUID], outcome_id: UUID, outcome_value: float = 1.0) -> None:
        self._observation_buffer.append((context_ids, outcome_id))
        if len(self._observation_buffer) > self._max_buffer:
            self._observation_buffer = self._observation_buffer[-self._max_buffer:]

        self._total_observations += 1
        recent_context = context_ids[-self.config.max_context_depth:]

        for i, ctx_id in enumerate(recent_context):
            recency = (i + 1) / len(recent_context)
            strength = 0.3 + 0.4 * recency

            relation = CausalRelationType.CAUSES if outcome_value > 0 else CausalRelationType.PREVENTS

            edge = self.graph.add_edge(
                source_id=ctx_id,
                target_id=outcome_id,
                relation=relation,
                initial_strength=strength,
            )

            if edge.evidence_count == 1:
                self._causal_edges_learned += 1

    def observe_counterfactual(self, expected_cause_id: UUID, actual_outcome_id: UUID, outcome_occurred: bool) -> None:
        if not outcome_occurred:
            self.graph.weaken_edge(expected_cause_id, actual_outcome_id)
        else:
            self.graph.add_edge(expected_cause_id, actual_outcome_id, CausalRelationType.CAUSES)

    def attribute_outcome(self, outcome_id: UUID, context_ids: list[UUID] | None = None) -> CausalAttribution:
        return self.attributor.attribute(outcome_id, context_ids)

    def get_predictive_causes(self, effect_id: UUID, k: int = 5) -> list[tuple[UUID, float]]:
        causes = self.graph.get_causes(effect_id)
        sorted_causes = sorted([(e.source_id, e.strength) for e in causes], key=lambda x: x[1], reverse=True)
        return sorted_causes[:k]

    def get_likely_effects(self, cause_id: UUID, k: int = 5) -> list[tuple[UUID, float]]:
        effects = self.graph.get_effects(cause_id)
        sorted_effects = sorted([(e.target_id, e.strength) for e in effects], key=lambda x: x[1], reverse=True)
        return sorted_effects[:k]

    def decay(self) -> None:
        self.graph.decay_all()

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_observations": self._total_observations,
            "causal_edges_learned": self._causal_edges_learned,
            "buffer_size": len(self._observation_buffer),
            "graph_stats": self.graph.get_statistics(),
        }

    def save_state(self) -> dict[str, Any]:
        edges = []
        for source_id, edge_list in self.graph._outgoing.items():
            for edge in edge_list:
                edges.append(edge.to_dict())
        return {"edges": edges, "statistics": self.get_statistics()}

    def load_state(self, state: dict[str, Any]) -> None:
        for edge_data in state.get("edges", []):
            self.graph.add_edge(
                source_id=UUID(edge_data["source"]),
                target_id=UUID(edge_data["target"]),
                relation=CausalRelationType(edge_data["relation"]),
                initial_strength=edge_data["strength"],
            )
