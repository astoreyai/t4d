"""
Neuro-Symbolic Representation for World Weaver.

Combines neural embeddings with symbolic graph structures for:
1. Interpretable memory retrieval (why was this retrieved?)
2. Symbolic reasoning over learned associations
3. Efficient graph-based queries alongside vector search
4. Explainable credit assignment

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Neuro-Symbolic Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Neural Side              │  Symbolic Side                       │
│  (Qdrant)                 │  (Neo4j)                             │
│                           │                                       │
│  ┌─────────────┐         │  ┌─────────────────────────────┐     │
│  │ Embedding   │←───────→│  │ (Memory)─[RELATES_TO]→     │     │
│  │ 1024-dim    │         │  │          (Memory)          │     │
│  └─────────────┘         │  │                             │     │
│                           │  │ (Memory)─[CAUSED]→         │     │
│  ┌─────────────┐         │  │          (Outcome)         │     │
│  │ Learned     │←───────→│  │                             │     │
│  │ Weights     │         │  │ (Memory)─[HAS_TYPE]→       │     │
│  └─────────────┘         │  │          (MemoryType)      │     │
│                           │  └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘

Triple Format: (Subject, Predicate, Object)
- Subject: Memory ID or Entity ID
- Predicate: Relationship type with learned weight
- Object: Target Memory, Entity, or Literal value

Learning Integration:
- Predicate weights are learned from retrieval outcomes
- Graph structure informs attention mechanism
- Symbolic paths provide credit attribution explanation
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)


# =============================================================================
# Core Triple Types
# =============================================================================

class PredicateType(str, Enum):
    """
    Predicate types with semantic meaning.

    Each predicate has:
    - A learned weight (initialized to 1.0)
    - A directional bias (affects credit flow)
    - A decay rate (how fast associations weaken)
    """

    # Temporal Relationships
    PRECEDED_BY = "preceded_by"      # Memory A came before B
    FOLLOWED_BY = "followed_by"      # Memory A came after B
    CONCURRENT_WITH = "concurrent"   # Same session/context

    # Semantic Relationships
    SIMILAR_TO = "similar_to"        # Vector similarity above threshold
    CONTRASTS_WITH = "contrasts"     # Semantic opposition
    ELABORATES = "elaborates"        # B provides more detail on A
    SUMMARIZES = "summarizes"        # A is summary of B

    # Causal Relationships
    CAUSED = "caused"                # Memory retrieval led to outcome
    CONTRIBUTED_TO = "contributed"   # Partial causal role
    BLOCKED = "blocked"              # Negative influence

    # Structural Relationships
    HAS_TYPE = "has_type"            # Memory type classification
    BELONGS_TO = "belongs_to"        # Project/domain membership
    DERIVED_FROM = "derived_from"    # Consolidation origin
    INSTANCE_OF = "instance_of"      # Entity type

    # Learning Relationships
    CO_RETRIEVED = "co_retrieved"    # Retrieved together frequently
    STRENGTHENS = "strengthens"      # Hebbian co-activation
    INHIBITS = "inhibits"            # Competitive inhibition

    # Citation Relationships
    CITED_BY = "cited_by"            # Explicitly referenced by user
    USED_IN = "used_in"              # Used in successful task


@dataclass
class Triple:
    """
    A single neuro-symbolic triple.

    Represents a relationship between two entities with
    learnable weight and confidence score.
    """

    subject: str           # Source node ID
    predicate: PredicateType
    object: str           # Target node ID or literal value

    # Learned properties
    weight: float = 1.0    # Learned importance (updated via Hebbian)
    confidence: float = 1.0  # Certainty of relationship
    count: int = 1         # Times this triple was observed

    # Metadata
    created: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "inferred"  # "explicit", "inferred", "learned"

    def to_cypher_create(self) -> str:
        """Generate Cypher CREATE statement."""
        props = {
            "weight": self.weight,
            "confidence": self.confidence,
            "count": self.count,
            "created": self.created.isoformat(),
            "source": self.source,
        }
        props_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
        return (
            f"MATCH (a {{id: '{self.subject}'}}), (b {{id: '{self.object}'}}) "
            f"MERGE (a)-[r:{self.predicate.value.upper()} {{{props_str}}}]->(b)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "s": self.subject,
            "p": self.predicate.value,
            "o": self.object,
            "w": self.weight,
            "c": self.confidence,
            "n": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Triple:
        """Deserialize from dictionary."""
        return cls(
            subject=data["s"],
            predicate=PredicateType(data["p"]),
            object=data["o"],
            weight=data.get("w", 1.0),
            confidence=data.get("c", 1.0),
            count=data.get("n", 1),
        )

    def __hash__(self) -> int:
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return False
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
        )


@dataclass
class TripleSet:
    """
    A collection of triples representing a memory's symbolic context.

    Supports:
    - Efficient lookup by subject/predicate/object
    - Weight normalization for attention
    - Path finding for credit attribution
    """

    triples: list[Triple] = field(default_factory=list)
    _by_subject: dict[str, list[Triple]] = field(default_factory=dict, repr=False)
    _by_object: dict[str, list[Triple]] = field(default_factory=dict, repr=False)

    def add(self, triple: Triple) -> None:
        """Add a triple, merging if duplicate."""
        for existing in self.triples:
            if existing == triple:
                # Merge: update count and weight
                existing.count += 1
                existing.weight = (existing.weight + triple.weight) / 2
                existing.last_updated = datetime.now()
                return

        self.triples.append(triple)
        self._index_triple(triple)

    def _index_triple(self, triple: Triple) -> None:
        """Add triple to indices."""
        if triple.subject not in self._by_subject:
            self._by_subject[triple.subject] = []
        self._by_subject[triple.subject].append(triple)

        if triple.object not in self._by_object:
            self._by_object[triple.object] = []
        self._by_object[triple.object].append(triple)

    def get_outgoing(self, subject: str) -> list[Triple]:
        """Get all triples with given subject."""
        return self._by_subject.get(subject, [])

    def get_incoming(self, obj: str) -> list[Triple]:
        """Get all triples with given object."""
        return self._by_object.get(obj, [])

    def get_by_predicate(self, predicate: PredicateType) -> list[Triple]:
        """Get all triples with given predicate."""
        return [t for t in self.triples if t.predicate == predicate]

    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3
    ) -> list[list[Triple]]:
        """
        Find all paths from source to target.

        Used for credit attribution - explains why a memory
        was relevant to an outcome.

        Args:
            source: Starting node ID
            target: Ending node ID
            max_depth: Maximum path length

        Returns:
            List of paths, each path is a list of triples
        """
        paths = []
        self._dfs_paths(source, target, [], paths, max_depth, set())
        return paths

    def _dfs_paths(
        self,
        current: str,
        target: str,
        path: list[Triple],
        all_paths: list[list[Triple]],
        remaining_depth: int,
        visited: set[str]
    ) -> None:
        """DFS helper for path finding."""
        if remaining_depth <= 0:
            return

        if current in visited:
            return

        visited.add(current)

        for triple in self.get_outgoing(current):
            new_path = path + [triple]

            if triple.object == target:
                all_paths.append(new_path)
            else:
                self._dfs_paths(
                    triple.object, target, new_path,
                    all_paths, remaining_depth - 1, visited.copy()
                )

    def compute_path_weight(self, path: list[Triple]) -> float:
        """
        Compute total weight of a path.

        Product of edge weights, used for credit attribution.
        """
        if not path:
            return 0.0
        weight = 1.0
        for triple in path:
            weight *= triple.weight
        return weight

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"triples": [t.to_dict() for t in self.triples]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TripleSet:
        """Deserialize from dictionary."""
        ts = cls()
        for t_data in data.get("triples", []):
            ts.add(Triple.from_dict(t_data))
        return ts

    def to_text(self) -> str:
        """Convert to human-readable text format."""
        lines = []
        for t in self.triples:
            lines.append(f"{t.subject[:8]}|{t.predicate.value}|{t.object[:8]} (w={t.weight:.2f})")
        return "\n".join(lines)


# =============================================================================
# Neuro-Symbolic Memory Wrapper
# =============================================================================

@dataclass
class NeuroSymbolicMemory:
    """
    A memory with both neural and symbolic representations.

    Combines:
    - Neural: Embedding vector for similarity search
    - Symbolic: Triple set for graph reasoning
    """

    memory_id: UUID = field(default_factory=uuid4)
    memory_type: str = "episodic"

    # Neural representation
    embedding: list[float] | None = None  # 1024-dim vector
    learned_features: list[float] | None = None  # Learned scoring features

    # Symbolic representation
    triples: TripleSet = field(default_factory=TripleSet)

    # Content
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Learning state
    retrieval_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_reward: float = 0.0
    last_retrieved: datetime | None = None

    def add_relationship(
        self,
        predicate: PredicateType,
        target: str,
        weight: float = 1.0,
        source: str = "inferred"
    ) -> Triple:
        """Add a symbolic relationship."""
        triple = Triple(
            subject=str(self.memory_id),
            predicate=predicate,
            object=target,
            weight=weight,
            source=source,
        )
        self.triples.add(triple)
        return triple

    def update_learning_stats(self, reward: float) -> None:
        """Update learning statistics after retrieval outcome."""
        self.retrieval_count += 1
        if reward > 0:
            self.success_count += 1
        elif reward < 0:
            self.failure_count += 1

        # Update running average reward
        alpha = 1.0 / (self.retrieval_count + 1)
        self.avg_reward = (1 - alpha) * self.avg_reward + alpha * reward
        self.last_retrieved = datetime.now()

    def get_success_rate(self) -> float:
        """Get success rate for this memory."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Prior
        return self.success_count / total

    def to_compact(self) -> str:
        """
        Convert to token-efficient representation for LLM context.

        Format: [id|type|success_rate|top_relationships]
        """
        sr = self.get_success_rate()
        top_rels = []
        for t in sorted(self.triples.triples, key=lambda x: x.weight, reverse=True)[:3]:
            top_rels.append(f"{t.predicate.value[:3]}→{t.object[:6]}")

        rels_str = ",".join(top_rels) if top_rels else "∅"
        return f"[{str(self.memory_id)[:8]}|{self.memory_type[0].upper()}|{sr:.0%}|{rels_str}]"

    def explain_relevance(self, outcome_id: str) -> str:
        """
        Generate explanation for why this memory was relevant.

        Uses symbolic paths to explain credit attribution.
        """
        paths = self.triples.find_paths(str(self.memory_id), outcome_id)

        if not paths:
            return f"Memory {str(self.memory_id)[:8]} - no direct path to outcome"

        explanations = []
        for path in paths[:3]:  # Top 3 paths
            weight = self.triples.compute_path_weight(path)
            steps = " → ".join(
                f"{t.predicate.value}({t.weight:.2f})"
                for t in path
            )
            explanations.append(f"  Path (w={weight:.3f}): {steps}")

        return f"Memory {str(self.memory_id)[:8]} relevance:\n" + "\n".join(explanations)


# =============================================================================
# Triple Extractors
# =============================================================================

class TripleExtractor(ABC):
    """Base class for extracting triples from memories."""

    @abstractmethod
    def extract(self, memory: NeuroSymbolicMemory, context: dict[str, Any]) -> list[Triple]:
        """Extract triples from memory and context."""


class CoRetrievalExtractor(TripleExtractor):
    """Extract CO_RETRIEVED relationships from retrieval events."""

    def __init__(self, min_co_occurrences: int = 2):
        self.co_occurrence_counts: dict[tuple[str, str], int] = {}
        self.min_co_occurrences = min_co_occurrences

    def extract(self, memory: NeuroSymbolicMemory, context: dict[str, Any]) -> list[Triple]:
        """Extract co-retrieval relationships."""
        triples = []
        retrieved_ids = context.get("retrieved_ids", [])
        mem_id = str(memory.memory_id)

        for other_id in retrieved_ids:
            other_str = str(other_id)
            if other_str == mem_id:
                continue

            # Update co-occurrence count
            pair = tuple(sorted([mem_id, other_str]))
            self.co_occurrence_counts[pair] = self.co_occurrence_counts.get(pair, 0) + 1

            if self.co_occurrence_counts[pair] >= self.min_co_occurrences:
                triples.append(Triple(
                    subject=mem_id,
                    predicate=PredicateType.CO_RETRIEVED,
                    object=other_str,
                    weight=min(self.co_occurrence_counts[pair] / 10.0, 1.0),
                    source="learned"
                ))

        return triples


class CausalExtractor(TripleExtractor):
    """Extract CAUSED/CONTRIBUTED relationships from outcomes."""

    def extract(self, memory: NeuroSymbolicMemory, context: dict[str, Any]) -> list[Triple]:
        """Extract causal relationships from outcome."""
        triples = []
        outcome = context.get("outcome")
        reward = context.get("reward", 0.0)

        if outcome is None:
            return triples

        outcome_id = str(outcome.get("id", "unknown"))
        mem_id = str(memory.memory_id)

        if reward > 0.5:
            triples.append(Triple(
                subject=mem_id,
                predicate=PredicateType.CAUSED,
                object=outcome_id,
                weight=reward,
                source="learned"
            ))
        elif reward > 0:
            triples.append(Triple(
                subject=mem_id,
                predicate=PredicateType.CONTRIBUTED_TO,
                object=outcome_id,
                weight=reward,
                source="learned"
            ))
        elif reward < 0:
            triples.append(Triple(
                subject=mem_id,
                predicate=PredicateType.BLOCKED,
                object=outcome_id,
                weight=abs(reward),
                source="learned"
            ))

        return triples


class SimilarityExtractor(TripleExtractor):
    """Extract SIMILAR_TO relationships from embedding similarity."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def extract(self, memory: NeuroSymbolicMemory, context: dict[str, Any]) -> list[Triple]:
        """Extract similarity relationships."""
        triples = []
        similar_memories = context.get("similar_memories", [])

        for other_id, similarity in similar_memories:
            if similarity >= self.threshold:
                triples.append(Triple(
                    subject=str(memory.memory_id),
                    predicate=PredicateType.SIMILAR_TO,
                    object=str(other_id),
                    weight=similarity,
                    source="inferred"
                ))

        return triples


# =============================================================================
# Learned Fusion Weights
# =============================================================================

class LearnedFusion(nn.Module):
    """
    Query-dependent fusion weights for neural-symbolic integration.

    Addresses Hinton critique: Fixed 60/40 neural/symbolic split contradicts
    learning goals. The optimal balance should be query-dependent.

    Architecture:
    - Input: Query embedding (1024-dim from sentence transformer)
    - Hidden: 64-dim ReLU layer for query characterization
    - Output: 4-component softmax weights

    Components:
    1. Neural similarity weight (vector space relevance)
    2. Symbolic graph weight (relationship-based relevance)
    3. Recency weight (temporal relevance)
    4. Outcome weight (success history relevance)

    Training:
    - Supervised by retrieval outcomes (was retrieval helpful?)
    - Gradient flows through fusion to weight network
    - Uses same ListMLE loss as scorer
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        n_components: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_components = n_components

        # Query characterization network
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_components),
            nn.Softmax(dim=-1)
        )

        # Initialize to roughly uniform (25% each)
        # But allow neural to start slightly higher
        self._init_weights()

        # Component names for interpretability
        self.component_names = ["neural", "symbolic", "recency", "outcome"]

    def _init_weights(self):
        """Initialize weights for stable starting point."""
        for module in self.weight_net.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for ReLU
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Bias final layer to start near uniform
                    nn.init.zeros_(module.bias)

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute query-dependent fusion weights.

        Args:
            query_embedding: Query vector [batch_size, embed_dim] or [embed_dim]

        Returns:
            Fusion weights [batch_size, 4] or [4] for
            [neural, symbolic, recency, outcome]
        """
        # Handle single query (no batch dimension)
        squeeze = False
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
            squeeze = True

        weights = self.weight_net(query_embedding)

        if squeeze:
            weights = weights.squeeze(0)

        return weights

    def get_weights_dict(self, query_embedding: torch.Tensor) -> dict[str, float]:
        """
        Get interpretable weight dictionary.

        Args:
            query_embedding: Query vector

        Returns:
            Dict mapping component name to weight
        """
        weights = self.forward(query_embedding)
        if weights.dim() > 1:
            weights = weights.squeeze(0)
        return {
            name: float(w)
            for name, w in zip(self.component_names, weights.tolist())
        }

    def fuse_scores(
        self,
        query_embedding: torch.Tensor,
        neural_scores: torch.Tensor,
        symbolic_scores: torch.Tensor,
        recency_scores: torch.Tensor,
        outcome_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multiple score sources with learned weights.

        Args:
            query_embedding: Query vector [embed_dim]
            neural_scores: Vector similarity scores [n_candidates]
            symbolic_scores: Graph-based scores [n_candidates]
            recency_scores: Time-based scores [n_candidates]
            outcome_scores: Historical success scores [n_candidates]

        Returns:
            Fused scores [n_candidates]
        """
        weights = self.forward(query_embedding)  # [4]

        # Stack all score sources [4, n_candidates]
        all_scores = torch.stack([
            neural_scores,
            symbolic_scores,
            recency_scores,
            outcome_scores
        ], dim=0)

        # Weighted sum: [4] @ [4, n_candidates] -> [n_candidates]
        fused = torch.einsum("c,cn->n", weights, all_scores)

        return fused


# =============================================================================
# Neuro-Symbolic Reasoner
# =============================================================================

class NeuroSymbolicReasoner:
    """
    Combines neural retrieval with symbolic reasoning.

    Query flow:
    1. Neural: Vector search for candidate memories
    2. Symbolic: Graph traversal to expand/filter candidates
    3. Fusion: Combine neural scores with symbolic path weights

    Supports both fixed weights (legacy) and learned weights (recommended).
    """

    def __init__(
        self,
        neural_weight: float = 0.6,
        symbolic_weight: float = 0.4,
        use_learned_fusion: bool = True,  # Per Hinton: should be query-dependent
        embed_dim: int = 1024,
        fusion_lr: float = 1e-4
    ):
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.use_learned_fusion = use_learned_fusion
        self.extractors: list[TripleExtractor] = [
            CoRetrievalExtractor(),
            CausalExtractor(),
            SimilarityExtractor(),
        ]

        # Initialize learned fusion if requested
        if use_learned_fusion:
            self.learned_fusion = LearnedFusion(embed_dim=embed_dim)
            self.fusion_optimizer = Adam(
                self.learned_fusion.parameters(),
                lr=fusion_lr
            )
            # Training statistics
            self._fusion_train_steps = 0
            self._fusion_total_loss = 0.0
        else:
            self.learned_fusion = None
            self.fusion_optimizer = None

    def fuse_scores(
        self,
        neural_scores: dict[str, float],
        symbolic_scores: dict[str, float],
        query_embedding: np.ndarray | None = None,
        recency_scores: dict[str, float] | None = None,
        outcome_scores: dict[str, float] | None = None
    ) -> dict[str, float]:
        """
        Fuse neural and symbolic scores.

        If learned fusion is enabled and query_embedding is provided,
        uses 4-component learned weights. Otherwise falls back to
        fixed 2-component weights.

        Args:
            neural_scores: Memory ID -> neural similarity score
            symbolic_scores: Memory ID -> symbolic relevance score
            query_embedding: Optional query embedding for learned fusion
            recency_scores: Optional recency scores for learned fusion
            outcome_scores: Optional outcome history scores for learned fusion

        Returns:
            Fused scores
        """
        all_ids = set(neural_scores.keys()) | set(symbolic_scores.keys())

        # Use learned fusion if available and we have query embedding
        if (self.use_learned_fusion and
            self.learned_fusion is not None and
            query_embedding is not None):

            return self._learned_fuse(
                all_ids, neural_scores, symbolic_scores,
                query_embedding, recency_scores, outcome_scores
            )

        # Fall back to fixed weights
        fused = {}
        for mem_id in all_ids:
            n_score = neural_scores.get(mem_id, 0.0)
            s_score = symbolic_scores.get(mem_id, 0.0)
            fused[mem_id] = (
                self.neural_weight * n_score +
                self.symbolic_weight * s_score
            )

        return fused

    def _learned_fuse(
        self,
        all_ids: set[str],
        neural_scores: dict[str, float],
        symbolic_scores: dict[str, float],
        query_embedding: np.ndarray,
        recency_scores: dict[str, float] | None,
        outcome_scores: dict[str, float] | None
    ) -> dict[str, float]:
        """Apply learned 4-component fusion."""
        # Convert to tensors
        ids_list = list(all_ids)
        len(ids_list)

        query_tensor = torch.from_numpy(query_embedding).float()

        neural_tensor = torch.tensor([
            neural_scores.get(mid, 0.0) for mid in ids_list
        ], dtype=torch.float32)

        symbolic_tensor = torch.tensor([
            symbolic_scores.get(mid, 0.0) for mid in ids_list
        ], dtype=torch.float32)

        # Default to 0.5 for missing scores
        recency_tensor = torch.tensor([
            (recency_scores or {}).get(mid, 0.5) for mid in ids_list
        ], dtype=torch.float32)

        outcome_tensor = torch.tensor([
            (outcome_scores or {}).get(mid, 0.5) for mid in ids_list
        ], dtype=torch.float32)

        # Get learned fusion
        with torch.no_grad():
            fused_tensor = self.learned_fusion.fuse_scores(
                query_tensor,
                neural_tensor,
                symbolic_tensor,
                recency_tensor,
                outcome_tensor
            )

        return {
            mid: float(score)
            for mid, score in zip(ids_list, fused_tensor.tolist())
        }

    def get_fusion_weights(
        self,
        query_embedding: np.ndarray
    ) -> dict[str, float]:
        """
        Get current fusion weights for a query.

        Args:
            query_embedding: Query vector

        Returns:
            Dict of component weights
        """
        if self.use_learned_fusion and self.learned_fusion is not None:
            query_tensor = torch.from_numpy(query_embedding).float()
            with torch.no_grad():
                return self.learned_fusion.get_weights_dict(query_tensor)
        else:
            return {
                "neural": self.neural_weight,
                "symbolic": self.symbolic_weight,
                "recency": 0.0,
                "outcome": 0.0
            }

    def compute_symbolic_score(
        self,
        memory_triples: TripleSet,
        query_context: dict[str, Any]
    ) -> float:
        """
        Compute symbolic relevance score for a memory.

        Considers:
        - Relationship weights to entities in query
        - Path weights to recent successful outcomes
        - Co-retrieval patterns
        """
        score = 0.0

        # Query entity matching
        query_entities = query_context.get("entities", [])
        for entity in query_entities:
            for triple in memory_triples.get_outgoing(entity):
                score += triple.weight * 0.3

        # Recent outcome connection
        recent_outcomes = query_context.get("recent_outcomes", [])
        for outcome_id in recent_outcomes:
            paths = memory_triples.find_paths(
                list(memory_triples._by_subject.keys())[0] if memory_triples._by_subject else "",
                outcome_id,
                max_depth=2
            )
            for path in paths:
                score += memory_triples.compute_path_weight(path) * 0.5

        # Co-retrieval bonus
        co_retrieved = memory_triples.get_by_predicate(PredicateType.CO_RETRIEVED)
        score += len(co_retrieved) * 0.1

        return min(score, 1.0)  # Clamp to [0, 1]

    def update_from_outcome(
        self,
        memories: list[NeuroSymbolicMemory],
        outcome: dict[str, Any],
        rewards: dict[str, float]
    ) -> int:
        """
        Update symbolic graph based on outcome.

        Args:
            memories: Memories involved in retrieval
            outcome: Outcome event data
            rewards: Per-memory rewards

        Returns:
            Number of triples updated
        """
        count = 0
        context = {"outcome": outcome}

        for memory in memories:
            reward = rewards.get(str(memory.memory_id), 0.0)
            context["reward"] = reward

            for extractor in self.extractors:
                new_triples = extractor.extract(memory, context)
                for triple in new_triples:
                    memory.triples.add(triple)
                    count += 1

            memory.update_learning_stats(reward)

        return count

    def train_fusion_step(
        self,
        query_embedding: np.ndarray,
        neural_scores: dict[str, float],
        symbolic_scores: dict[str, float],
        recency_scores: dict[str, float],
        outcome_scores: dict[str, float],
        target_rewards: dict[str, float]
    ) -> float:
        """
        Train the learned fusion weights from a single experience.

        Uses ListMLE loss to learn ranking from outcome-informed rewards.
        This enables end-to-end gradient flow per Hinton critique.

        Args:
            query_embedding: Query vector [embed_dim]
            neural_scores: Memory ID -> neural similarity
            symbolic_scores: Memory ID -> symbolic relevance
            recency_scores: Memory ID -> recency score
            outcome_scores: Memory ID -> outcome history
            target_rewards: Memory ID -> actual reward (ground truth)

        Returns:
            Loss value for this step
        """
        if not self.use_learned_fusion or self.learned_fusion is None:
            return 0.0

        # Must have at least 2 items for ranking loss
        ids_list = list(target_rewards.keys())
        if len(ids_list) < 2:
            return 0.0

        # Prepare tensors WITH gradients
        query_tensor = torch.from_numpy(query_embedding).float()

        neural_tensor = torch.tensor([
            neural_scores.get(mid, 0.0) for mid in ids_list
        ], dtype=torch.float32)

        symbolic_tensor = torch.tensor([
            symbolic_scores.get(mid, 0.0) for mid in ids_list
        ], dtype=torch.float32)

        recency_tensor = torch.tensor([
            recency_scores.get(mid, 0.5) for mid in ids_list
        ], dtype=torch.float32)

        outcome_tensor = torch.tensor([
            outcome_scores.get(mid, 0.5) for mid in ids_list
        ], dtype=torch.float32)

        # Target rewards as ground truth ranking
        target_tensor = torch.tensor([
            target_rewards[mid] for mid in ids_list
        ], dtype=torch.float32)

        # Forward pass WITH gradients (no torch.no_grad!)
        self.learned_fusion.train()
        fused = self.learned_fusion.fuse_scores(
            query_tensor,
            neural_tensor,
            symbolic_tensor,
            recency_tensor,
            outcome_tensor
        )

        # ListMLE loss: encourage fused scores to rank like target rewards
        # Higher reward items should have higher predicted scores
        loss = self._list_mle_loss(fused, target_tensor)

        # Backprop
        self.fusion_optimizer.zero_grad()
        loss.backward()
        self.fusion_optimizer.step()

        # Track stats
        self._fusion_train_steps += 1
        self._fusion_total_loss += loss.item()

        self.learned_fusion.eval()
        return loss.item()

    def _list_mle_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        ListMLE ranking loss.

        Encourages predicted scores to produce the same ranking as targets.
        """
        # Sort by target (descending) to get ideal ranking
        _, indices = torch.sort(target, descending=True)
        predicted_sorted = predicted[indices]

        # ListMLE: negative log-likelihood of correct ranking
        n = len(predicted_sorted)
        loss = 0.0
        for i in range(n - 1):
            # Probability that item i is ranked above items i+1...n
            remaining = predicted_sorted[i:]
            log_softmax = F.log_softmax(remaining, dim=0)
            loss -= log_softmax[0]

        return loss / max(n - 1, 1)

    def get_fusion_training_stats(self) -> dict[str, Any]:
        """Get training statistics for learned fusion."""
        if not self.use_learned_fusion:
            return {"enabled": False}

        return {
            "enabled": True,
            "train_steps": self._fusion_train_steps,
            "avg_loss": (
                self._fusion_total_loss / self._fusion_train_steps
                if self._fusion_train_steps > 0 else 0.0
            ),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CausalExtractor",
    "CoRetrievalExtractor",
    "LearnedFusion",
    "NeuroSymbolicMemory",
    "NeuroSymbolicReasoner",
    "PredicateType",
    "SimilarityExtractor",
    "Triple",
    "TripleExtractor",
    "TripleSet",
]
