"""
Core data types for T4DM tripartite memory system.

Based on cognitive science foundations:
- Episodic: Autobiographical events (Tulving 1972)
- Semantic: Abstracted knowledge with Hebbian weighting
- Procedural: Learned skills with Memp lifecycle
"""

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# Type variables for generic types
T = TypeVar("T")


# Enums

class Outcome(str, Enum):
    """Episode outcome classification."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    NEUTRAL = "neutral"


class EntityType(str, Enum):
    """Semantic entity types."""
    CONCEPT = "CONCEPT"
    PERSON = "PERSON"
    PROJECT = "PROJECT"
    TOOL = "TOOL"
    TECHNIQUE = "TECHNIQUE"
    FACT = "FACT"


class RelationType(str, Enum):
    """Semantic relationship types."""
    USES = "USES"
    PRODUCES = "PRODUCES"
    REQUIRES = "REQUIRES"
    CAUSES = "CAUSES"
    PART_OF = "PART_OF"
    SIMILAR_TO = "SIMILAR_TO"
    IMPLEMENTS = "IMPLEMENTS"
    IMPROVES_ON = "IMPROVES_ON"
    CONSOLIDATED_INTO = "CONSOLIDATED_INTO"
    SOURCE_OF = "SOURCE_OF"


class Domain(str, Enum):
    """Procedural memory domains."""
    CODING = "coding"
    RESEARCH = "research"
    TRADING = "trading"
    DEVOPS = "devops"
    WRITING = "writing"


class ConsolidationType(str, Enum):
    """Memory consolidation event types."""
    EPISODIC_TO_SEMANTIC = "episodic_to_semantic"
    SKILL_MERGE = "skill_merge"
    PATTERN_EXTRACT = "pattern_extract"


class TemporalLinkType(str, Enum):
    """Temporal relationship types between episodes."""
    SEQUENCE = "sequence"  # A occurred before B in same session
    CAUSES = "causes"  # A causally led to B
    ENABLES = "enables"  # A made B possible
    INTERRUPTS = "interrupts"  # A interrupted ongoing B
    RESUMES = "resumes"  # A resumed previously interrupted B
    ELABORATES = "elaborates"  # A adds detail to B


# Context Models

class EpisodeContext(BaseModel):
    """Spatial context for an episode."""
    model_config = ConfigDict(extra="allow")

    project: str | None = None
    file: str | None = None
    tool: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    timestamp_local: str | None = None


class ProcedureStep(BaseModel):
    """Single step in a procedure."""
    order: int = Field(..., ge=1)
    action: str = Field(..., min_length=1)
    tool: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None


# Core Memory Types

class Episode(BaseModel):
    """
    Episodic memory: Autobiographical event with temporal-spatial context.

    Implements bi-temporal versioning:
    - timestamp (T_ref): When event occurred in real world
    - ingested_at (T_sys): When memory was created in system
    """
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4)
    session_id: str = Field(..., description="Instance namespace")
    content: str = Field(..., min_length=1, description="Full interaction text")
    embedding: list[float] | None = Field(default=None, description="1024-dim BGE-M3 vector")

    # Bi-temporal
    timestamp: datetime = Field(default_factory=datetime.now, description="Event time (T_ref)")
    ingested_at: datetime = Field(default_factory=datetime.now, description="System time (T_sys)")

    # Context
    context: EpisodeContext = Field(default_factory=EpisodeContext)
    outcome: Outcome = Field(default=Outcome.NEUTRAL)
    emotional_valence: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance [0,1]")

    # FSRS decay tracking
    access_count: int = Field(default=1, ge=1)
    last_accessed: datetime = Field(default_factory=datetime.now)
    stability: float = Field(default=1.0, gt=0, description="FSRS stability in days")

    # P5.2: Temporal structure
    previous_episode_id: UUID | None = Field(default=None, description="Link to previous episode in sequence")
    next_episode_id: UUID | None = Field(default=None, description="Link to next episode in sequence")
    sequence_position: int | None = Field(default=None, ge=0, description="Position within session (0-indexed)")
    duration_ms: int | None = Field(default=None, ge=0, description="Episode duration in milliseconds")
    end_timestamp: datetime | None = Field(default=None, description="When episode ended")

    # P1-1: Prediction error for prioritized replay
    prediction_error: float | None = Field(default=None, description="TD prediction error magnitude |δ| from dopamine system")
    prediction_error_timestamp: datetime | None = Field(default=None, description="When prediction error was computed")

    def retrievability(self, current_time: datetime | None = None) -> float:
        """
        Calculate FSRS retrievability.

        R(t, S) = (1 + 0.9 * t/S)^(-0.5)

        Where:
        - t = elapsed time in days since last access
        - S = stability (learned from access pattern)
        - 0.9 factor = slower decay than flashcard FSRS (1.0) for conversational context
        - -0.5 exponent = power law decay (cognitive forgetting curve)

        Returns value in [0,1]: 1=perfect recall, 0=completely forgotten
        """
        current_time = current_time or datetime.now()
        elapsed_days = (current_time - self.last_accessed).total_seconds() / 86400
        return (1 + 0.9 * elapsed_days / self.stability) ** (-0.5)


class Entity(BaseModel):
    """
    Semantic memory: Abstracted knowledge entity.

    Stores context-free knowledge with decay properties.
    """
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, description="Canonical entity name")
    entity_type: EntityType = Field(...)
    summary: str = Field(..., min_length=1, description="Short description")
    details: str | None = Field(default=None, description="Expanded context")
    embedding: list[float] | None = Field(default=None, description="1024-dim BGE-M3 vector")

    # Provenance
    source: str | None = Field(default=None, description="episode_id or 'user_provided'")

    # FSRS decay
    stability: float = Field(default=1.0, gt=0)
    access_count: int = Field(default=1, ge=1)
    last_accessed: datetime = Field(default_factory=datetime.now)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    # Bi-temporal versioning
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: datetime | None = Field(default=None, description="null = currently valid")

    def is_valid(self, at_time: datetime | None = None) -> bool:
        """Check if entity is valid at given time."""
        at_time = at_time or datetime.now()
        if at_time < self.valid_from:
            return False
        if self.valid_to and at_time >= self.valid_to:
            return False
        return True


class Relationship(BaseModel):
    """
    Hebbian-weighted relationship between semantic entities.

    Weight strengthens on co-retrieval: w' = w + lr * (1 - w)
    """
    model_config = ConfigDict(from_attributes=True)

    source_id: UUID
    target_id: UUID
    relation_type: RelationType
    weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Hebbian strength")
    co_access_count: int = Field(default=1, ge=1)
    last_co_access: datetime = Field(default_factory=datetime.now)

    def strengthen(self, learning_rate: float = 0.1) -> float:
        """
        Apply bounded Hebbian update.

        w' = w + lr * (1 - w)
        """
        self.weight = self.weight + learning_rate * (1.0 - self.weight)
        self.co_access_count += 1
        self.last_co_access = datetime.now()
        return self.weight


class TemporalLink(BaseModel):
    """
    P5.2: Temporal relationship between episodes.

    Models causal and sequential relationships in episodic memory,
    supporting temporal reasoning and narrative reconstruction.
    """
    model_config = ConfigDict(from_attributes=True)

    source_id: UUID = Field(..., description="Earlier/causing episode")
    target_id: UUID = Field(..., description="Later/caused episode")
    link_type: TemporalLinkType = Field(default=TemporalLinkType.SEQUENCE)
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Relationship confidence")
    temporal_gap_ms: int | None = Field(default=None, ge=0, description="Time gap between episodes")
    created_at: datetime = Field(default_factory=datetime.now)

    # For causal inference
    causal_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Inferred causality strength")
    evidence_count: int = Field(default=1, ge=1, description="Supporting co-occurrence count")

    def strengthen_causality(self, learning_rate: float = 0.1) -> float:
        """Strengthen causal confidence on repeated co-occurrence."""
        self.causal_confidence = self.causal_confidence + learning_rate * (1.0 - self.causal_confidence)
        self.evidence_count += 1
        return self.causal_confidence


class Procedure(BaseModel):
    """
    Procedural memory: Learned skill with Memp lifecycle.

    Stores dual format:
    - steps: Fine-grained action sequence
    - script: High-level abstraction
    """
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, description="Procedure identifier")
    domain: Domain = Field(...)
    trigger_pattern: str | None = Field(default=None, description="When to invoke")

    # Dual format storage
    steps: list[ProcedureStep] = Field(default_factory=list, description="Fine-grained actions")
    script: str | None = Field(default=None, description="High-level abstraction")

    embedding: list[float] | None = Field(default=None, description="1024-dim BGE-M3 vector")

    # Execution tracking
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_count: int = Field(default=1, ge=1)
    last_executed: datetime | None = Field(default=None)

    # Versioning
    version: int = Field(default=1, ge=1)
    deprecated: bool = Field(default=False)
    consolidated_into: UUID | None = Field(default=None)

    # Provenance
    created_at: datetime = Field(default_factory=datetime.now)
    created_from: str = Field(default="manual", description="trajectory, manual, consolidated")

    def update_success_rate(self, success: bool) -> float:
        """Update success rate after execution."""
        if success:
            self.success_rate = (
                (self.success_rate * self.execution_count + 1) /
                (self.execution_count + 1)
            )
        else:
            self.success_rate = (
                (self.success_rate * self.execution_count) /
                (self.execution_count + 1)
            )
        self.execution_count += 1
        self.last_executed = datetime.now()
        return self.success_rate

    def should_deprecate(self, min_executions: int = 10, min_success: float = 0.3) -> bool:
        """Check if procedure should be deprecated due to consistent failure."""
        return self.execution_count > min_executions and self.success_rate < min_success


class ConsolidationEvent(BaseModel):
    """Record of a memory consolidation operation."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    event_type: ConsolidationType
    source_ids: list[UUID] = Field(default_factory=list)
    target_id: UUID
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(..., ge=0.0, le=1.0)
    pattern_strength: int = Field(..., ge=1, description="Number of source instances")
    metadata: dict[str, Any] = Field(default_factory=dict)


# Query/Response Types

class EpisodeQuery(BaseModel):
    """Query parameters for episodic retrieval."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    session_filter: str | None = None
    time_start: datetime | None = None
    time_end: datetime | None = None


class EntityQuery(BaseModel):
    """Query parameters for semantic retrieval."""
    query: str = Field(..., min_length=1)
    context_entities: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    include_spreading: bool = Field(default=True)


class ProcedureQuery(BaseModel):
    """Query parameters for procedural retrieval."""
    task: str = Field(..., min_length=1)
    domain: Domain | None = None
    limit: int = Field(default=5, ge=1, le=20)


class ScoredResult(BaseModel, Generic[T]):
    """Generic scored result wrapper."""
    item: T
    score: float = Field(..., ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
