"""
Learning Events for World Weaver.

Defines data structures for capturing retrieval and outcome events
that feed into the learning system. Includes multiple representation
formats for different use cases:

1. Full JSON - Complete fidelity for storage
2. ToonJSON - Token-optimized for LLM context injection
3. NeuroSymbolic - Triple-based for graph reasoning
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class OutcomeType(str, Enum):
    """Classification of task outcomes."""

    SUCCESS = "success"        # Task completed successfully
    PARTIAL = "partial"        # Partial success
    FAILURE = "failure"        # Task failed
    NEUTRAL = "neutral"        # Neither success nor failure
    UNKNOWN = "unknown"        # Outcome not yet determined


class FeedbackSignal(str, Enum):
    """Implicit feedback signals from user behavior."""

    ACCEPT = "accept"          # User accepted result
    REJECT = "reject"          # User rejected result
    MODIFY = "modify"          # User modified result
    REPEAT = "repeat"          # User repeated query (negative signal)
    EXPLICIT_POS = "explicit_positive"
    EXPLICIT_NEG = "explicit_negative"


class MemoryType(str, Enum):
    """Type of memory retrieved."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# =============================================================================
# Core Event Data Structures
# =============================================================================

@dataclass
class RetrievalEvent:
    """
    Records a single retrieval operation.

    Captures what was retrieved, how it scored, and context
    for later credit assignment.
    """

    retrieval_id: UUID = field(default_factory=uuid4)
    query: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC

    # What was retrieved (ordered by score)
    retrieved_ids: list[UUID] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)

    # Component scores for learning
    component_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    # e.g., {"mem_id": {"similarity": 0.8, "recency": 0.3, "importance": 0.5}}

    # Context for credit assignment
    context_hash: str = ""  # Hash of conversation context
    session_id: str = ""
    project: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def compute_context_hash(self, context: str) -> str:
        """Compute hash of context for matching with outcomes."""
        self.context_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        return self.context_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["retrieval_id"] = str(self.retrieval_id)
        d["retrieved_ids"] = [str(uid) for uid in self.retrieved_ids]
        d["memory_type"] = self.memory_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalEvent:
        """Reconstruct from dictionary."""
        data["retrieval_id"] = UUID(data["retrieval_id"])
        data["retrieved_ids"] = [UUID(uid) for uid in data["retrieved_ids"]]
        data["memory_type"] = MemoryType(data["memory_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class OutcomeEvent:
    """
    Records the outcome of a task/interaction.

    Used to compute rewards and attribute credit to retrievals.
    """

    outcome_id: UUID = field(default_factory=uuid4)
    outcome_type: OutcomeType = OutcomeType.UNKNOWN
    success_score: float = 0.5  # [0, 1] continuous success measure

    # Context matching
    context_hash: str = ""  # Matches to RetrievalEvent.context_hash
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Explicit citations (memories user explicitly referenced as helpful)
    explicit_citations: list[UUID] = field(default_factory=list)

    # Feedback signals detected
    feedback_signals: list[FeedbackSignal] = field(default_factory=list)

    # Metadata
    task_description: str = ""
    tool_results: dict[str, Any] = field(default_factory=dict)

    def compute_context_hash(self, context: str) -> str:
        """Compute hash of context for matching with retrievals."""
        self.context_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        return self.context_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["outcome_id"] = str(self.outcome_id)
        d["outcome_type"] = self.outcome_type.value
        d["explicit_citations"] = [str(uid) for uid in self.explicit_citations]
        d["feedback_signals"] = [f.value for f in self.feedback_signals]
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeEvent:
        """Reconstruct from dictionary."""
        data["outcome_id"] = UUID(data["outcome_id"])
        data["outcome_type"] = OutcomeType(data["outcome_type"])
        data["explicit_citations"] = [UUID(uid) for uid in data["explicit_citations"]]
        data["feedback_signals"] = [FeedbackSignal(f) for f in data["feedback_signals"]]
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Experience:
    """
    A complete learning experience combining retrieval and outcome.

    This is what goes into the replay buffer for training.
    """

    experience_id: UUID = field(default_factory=uuid4)

    # The retrieval
    query: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    retrieved_ids: list[UUID] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)

    # Component vectors for each memory (for learning scoring weights)
    component_vectors: list[list[float]] = field(default_factory=list)
    # Each inner list: [similarity, recency, importance, outcome_history, ...]

    # The outcome
    outcome_score: float = 0.5

    # Per-memory rewards (computed via credit assignment)
    per_memory_rewards: dict[str, float] = field(default_factory=dict)

    # Priority for replay sampling (based on TD error)
    priority: float = 1.0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["experience_id"] = str(self.experience_id)
        d["retrieved_ids"] = [str(uid) for uid in self.retrieved_ids]
        d["memory_type"] = self.memory_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


# =============================================================================
# Representation Formats
# =============================================================================

class RepresentationFormat(ABC):
    """Abstract base for memory representation formats."""

    @abstractmethod
    def encode(self, data: dict[str, Any]) -> str:
        """Encode data to string representation."""

    @abstractmethod
    def decode(self, encoded: str) -> dict[str, Any]:
        """Decode string back to data."""

    @property
    @abstractmethod
    def token_efficiency(self) -> float:
        """Estimated token reduction ratio vs full JSON."""


class FullJSON(RepresentationFormat):
    """
    Full JSON representation - complete fidelity.

    Use for: Storage, debugging, external APIs
    Token efficiency: 1.0 (baseline)
    """

    def encode(self, data: dict[str, Any]) -> str:
        return json.dumps(data, default=str)

    def decode(self, encoded: str) -> dict[str, Any]:
        return json.loads(encoded)

    @property
    def token_efficiency(self) -> float:
        return 1.0


class ToonJSON(RepresentationFormat):
    """
    Token-Optimized Nested JSON (ToonJSON).

    Reduces tokens by:
    1. Single-char keys (with legend)
    2. Omitting null/empty values
    3. Abbreviating common values
    4. Compact datetime format

    Use for: LLM context injection
    Token efficiency: ~0.4-0.6
    """

    # Key mappings (full -> short)
    KEY_MAP = {
        "retrieval_id": "ri",
        "query": "q",
        "memory_type": "mt",
        "retrieved_ids": "ids",
        "retrieval_scores": "sc",
        "component_scores": "cs",
        "context_hash": "ch",
        "session_id": "sid",
        "project": "p",
        "timestamp": "ts",
        "outcome_id": "oi",
        "outcome_type": "ot",
        "success_score": "ss",
        "explicit_citations": "ec",
        "feedback_signals": "fs",
        "similarity": "sim",
        "recency": "rec",
        "importance": "imp",
        "content": "c",
        "outcome": "o",
        # E1: Context injection keys
        "name": "n",
        "summary": "sum",
        "description": "d",
        "episodes": "eps",
        "entities": "ents",
        "skills": "sk",
        "personal_context": "pc",
        "history": "h",
        "known": "k",
        "status": "st",
    }

    # Value abbreviations
    VALUE_MAP = {
        "episodic": "E",
        "semantic": "S",
        "procedural": "P",
        "success": "+",
        "failure": "-",
        "partial": "~",
        "neutral": "0",
        "unknown": "?",
    }

    REVERSE_KEY_MAP = {v: k for k, v in KEY_MAP.items()}
    REVERSE_VALUE_MAP = {v: k for k, v in VALUE_MAP.items()}

    def encode(self, data: dict[str, Any]) -> str:
        """Encode to compact ToonJSON."""
        compact = self._compact_dict(data)
        return json.dumps(compact, separators=(",", ":"))

    def _compact_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively compact a dictionary."""
        result = {}
        for key, value in d.items():
            # Skip empty values
            if value is None or value == "" or value == [] or value == {}:
                continue

            # Shorten key
            short_key = self.KEY_MAP.get(key, key)

            # Process value
            if isinstance(value, dict):
                value = self._compact_dict(value)
            elif isinstance(value, list):
                value = [self._compact_dict(v) if isinstance(v, dict) else
                        self._compact_value(v) for v in value]
            else:
                value = self._compact_value(value)

            if value is not None:
                result[short_key] = value

        return result

    def _compact_value(self, value: Any) -> Any:
        """Compact a single value."""
        if isinstance(value, UUID):
            # Shorten UUID to first 8 chars
            return str(value)[:8]
        if isinstance(value, datetime):
            return value.strftime("%y%m%d%H%M")
        if isinstance(value, Enum):
            val = value.value.lower()
            return self.VALUE_MAP.get(val, val)
        if isinstance(value, str):
            # Check value abbreviations
            if value.lower() in self.VALUE_MAP:
                return self.VALUE_MAP[value.lower()]
            # Compact datetime
            if "T" in value and len(value) > 19:
                # ISO format -> compact
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return dt.strftime("%y%m%d%H%M")
                except ValueError:
                    pass
        elif isinstance(value, float):
            # Round to 2 decimal places
            return round(value, 2)
        return value

    def decode(self, encoded: str) -> dict[str, Any]:
        """Decode from ToonJSON back to full format."""
        compact = json.loads(encoded)
        return self._expand_dict(compact)

    def _expand_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively expand a dictionary."""
        result = {}
        for key, value in d.items():
            # Expand key
            full_key = self.REVERSE_KEY_MAP.get(key, key)

            # Process value
            if isinstance(value, dict):
                value = self._expand_dict(value)
            elif isinstance(value, list):
                value = [self._expand_dict(v) if isinstance(v, dict) else
                        self._expand_value(v) for v in value]
            else:
                value = self._expand_value(value)

            result[full_key] = value

        return result

    def _expand_value(self, value: Any) -> Any:
        """Expand a single value."""
        if isinstance(value, str):
            # Check abbreviated values
            if value in self.REVERSE_VALUE_MAP:
                return self.REVERSE_VALUE_MAP[value]
            # Expand compact datetime
            if len(value) == 10 and value.isdigit():
                try:
                    dt = datetime.strptime(value, "%y%m%d%H%M")
                    return dt.isoformat()
                except ValueError:
                    pass
        return value

    @property
    def token_efficiency(self) -> float:
        return 0.5  # ~50% token reduction


class NeuroSymbolicTriples(RepresentationFormat):
    """
    Neuro-Symbolic Triple Representation.

    Converts memories to (subject, predicate, object) triples
    for graph-based reasoning and symbolic manipulation.

    Format: subject|predicate|object per line

    Use for: Graph queries, symbolic reasoning, Neo4j integration
    Token efficiency: ~0.7 (depends on density)
    """

    def encode(self, data: dict[str, Any]) -> str:
        """Convert to triple format."""
        triples = []

        # Extract entity ID as subject
        subject = data.get("retrieval_id") or data.get("outcome_id") or data.get("id") or "unknown"
        if isinstance(subject, UUID):
            subject = str(subject)[:8]  # Short ID

        # Convert each field to a triple
        for key, value in data.items():
            if key in ("retrieval_id", "outcome_id", "id"):
                continue

            predicate = self._to_predicate(key)

            if isinstance(value, dict):
                # Nested dict becomes multiple triples
                for subkey, subval in value.items():
                    obj = self._to_object(subval)
                    if obj:
                        triples.append(f"{subject}|{predicate}:{subkey}|{obj}")
            elif isinstance(value, list):
                # List becomes multiple triples with same predicate
                for i, item in enumerate(value):
                    obj = self._to_object(item)
                    if obj:
                        triples.append(f"{subject}|{predicate}[{i}]|{obj}")
            else:
                obj = self._to_object(value)
                if obj:
                    triples.append(f"{subject}|{predicate}|{obj}")

        return "\n".join(triples)

    def _to_predicate(self, key: str) -> str:
        """Convert key to predicate format."""
        # CamelCase predicates
        parts = key.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])

    def _to_object(self, value: Any) -> str | None:
        """Convert value to object format."""
        if value is None or value == "" or value == [] or value == {}:
            return None
        if isinstance(value, UUID):
            return str(value)[:8]
        if isinstance(value, datetime):
            return value.strftime("%Y%m%d")
        if isinstance(value, float):
            return f"{value:.2f}"
        if isinstance(value, Enum):
            return value.value
        return str(value)

    def decode(self, encoded: str) -> dict[str, Any]:
        """Convert triples back to dict (lossy)."""
        result: dict[str, Any] = {}

        for line in encoded.strip().split("\n"):
            if not line or "|" not in line:
                continue

            parts = line.split("|")
            if len(parts) != 3:
                continue

            subject, predicate, obj = parts

            # Handle indexed predicates (lists)
            if "[" in predicate:
                base_pred = predicate.split("[")[0]
                if base_pred not in result:
                    result[base_pred] = []
                result[base_pred].append(obj)
            # Handle nested predicates
            elif ":" in predicate:
                base_pred, sub_key = predicate.split(":", 1)
                if base_pred not in result:
                    result[base_pred] = {}
                result[base_pred][sub_key] = obj
            else:
                result[predicate] = obj

        return result

    @property
    def token_efficiency(self) -> float:
        return 0.7


# =============================================================================
# Factory and Utilities
# =============================================================================

def get_representation(format_name: str = "full") -> RepresentationFormat:
    """Get representation format by name."""
    formats = {
        "full": FullJSON(),
        "toon": ToonJSON(),
        "triples": NeuroSymbolicTriples(),
    }
    return formats.get(format_name, FullJSON())


def estimate_tokens(text: str) -> int:
    """Rough token estimation (chars/4 for English)."""
    return len(text) // 4


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "Experience",
    "FeedbackSignal",
    "FullJSON",
    "MemoryType",
    "NeuroSymbolicTriples",
    "OutcomeEvent",
    "OutcomeType",
    "RepresentationFormat",
    "RetrievalEvent",
    "ToonJSON",
    "estimate_tokens",
    "get_representation",
]
