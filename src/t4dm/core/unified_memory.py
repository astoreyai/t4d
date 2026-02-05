"""
Unified Memory Schema with κ (kappa) consolidation gradient.

This module provides the unified MemoryItem schema and related utilities
for the SNN-integrated memory system. It extends the base memory_item.py
with additional fields for spike-based learning integration.

Key additions over base MemoryItem:
- spike_trace: Detailed spike timing information for STDP
- temporal_context: τ(t) gate state at encoding time
- consolidation_history: Track κ changes over time
- neuromod_context: Neuromodulator state at encoding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from t4dm.core.memory_item import MemoryItem
from t4dm.core.types import Entity, Episode, Outcome, Procedure

if TYPE_CHECKING:
    from t4dm.core.temporal_control import TemporalControlState
    from t4dm.learning.neuromodulators import NeuromodulatorState


@dataclass
class SpikeTrace:
    """
    Spike timing information for STDP learning.

    Records the spike pattern associated with encoding this memory,
    enabling accurate STDP weight updates during replay.
    """

    spike_times: list[float] = field(default_factory=list)  # Times in ms
    neuron_ids: list[int] = field(default_factory=list)  # Which neurons spiked
    num_spikes: int = 0  # Total spike count
    mean_rate: float = 0.0  # Mean firing rate (Hz)
    encoding_layer: int = 0  # Which cortical block encoded this
    temporal_pattern: str = "sparse"  # 'sparse', 'burst', 'tonic'

    def add_spike(self, time_ms: float, neuron_id: int) -> None:
        """Record a spike."""
        self.spike_times.append(time_ms)
        self.neuron_ids.append(neuron_id)
        self.num_spikes += 1

    def compute_rate(self, duration_ms: float) -> float:
        """Compute mean firing rate."""
        if duration_ms > 0:
            self.mean_rate = (self.num_spikes / duration_ms) * 1000.0  # Convert to Hz
        return self.mean_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "spike_times": self.spike_times,
            "neuron_ids": self.neuron_ids,
            "num_spikes": self.num_spikes,
            "mean_rate": self.mean_rate,
            "encoding_layer": self.encoding_layer,
            "temporal_pattern": self.temporal_pattern,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpikeTrace":
        """Create from dictionary."""
        return cls(
            spike_times=data.get("spike_times", []),
            neuron_ids=data.get("neuron_ids", []),
            num_spikes=data.get("num_spikes", 0),
            mean_rate=data.get("mean_rate", 0.0),
            encoding_layer=data.get("encoding_layer", 0),
            temporal_pattern=data.get("temporal_pattern", "sparse"),
        )


@dataclass
class ConsolidationEvent:
    """Record of a κ consolidation step."""

    timestamp: datetime
    old_kappa: float
    new_kappa: float
    phase: str  # 'nrem', 'rem', 'prune', 'access'
    delta: float = 0.0
    replay_count: int = 0

    def __post_init__(self) -> None:
        self.delta = self.new_kappa - self.old_kappa


@dataclass
class NeuromodContext:
    """Neuromodulator context at encoding time."""

    dopamine: float = 0.5
    norepinephrine: float = 0.5
    acetylcholine: float = 0.5
    serotonin: float = 0.5
    encoding_mode: str = "encoding"  # 'encoding' or 'retrieval'
    surprise_level: float = 0.0
    reward_signal: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dopamine": self.dopamine,
            "norepinephrine": self.norepinephrine,
            "acetylcholine": self.acetylcholine,
            "serotonin": self.serotonin,
            "encoding_mode": self.encoding_mode,
            "surprise_level": self.surprise_level,
            "reward_signal": self.reward_signal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuromodContext":
        """Create from dictionary."""
        return cls(
            dopamine=data.get("dopamine", 0.5),
            norepinephrine=data.get("norepinephrine", 0.5),
            acetylcholine=data.get("acetylcholine", 0.5),
            serotonin=data.get("serotonin", 0.5),
            encoding_mode=data.get("encoding_mode", "encoding"),
            surprise_level=data.get("surprise_level", 0.0),
            reward_signal=data.get("reward_signal", 0.0),
        )

    @classmethod
    def from_neuromod_state(cls, state: NeuromodulatorState) -> "NeuromodContext":
        """Create from NeuromodulatorState."""
        return cls(
            dopamine=state.dopamine_rpe,
            norepinephrine=state.norepinephrine_gain,
            acetylcholine=state.acetylcholine_mode,
            serotonin=state.serotonin_mood,
            encoding_mode="encoding" if state.acetylcholine_mode > 0.5 else "retrieval",
            surprise_level=state.norepinephrine_gain,  # NE tracks surprise
            reward_signal=state.dopamine_rpe,  # DA tracks reward
        )


class UnifiedMemoryItem(BaseModel):
    """
    Extended unified memory item with SNN integration fields.

    This extends the base MemoryItem with additional fields for:
    - Spike-based encoding information
    - Temporal control context
    - Consolidation history
    - Neuromodulator context at encoding
    """

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    # Core fields (same as MemoryItem)
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1)
    embedding: list[float] = Field(default_factory=list)
    event_time: datetime = Field(default_factory=datetime.now)
    record_time: datetime = Field(default_factory=datetime.now)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = None

    kappa: float = Field(default=0.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    item_type: Literal["episodic", "semantic", "procedural"] = "episodic"

    access_count: int = Field(default=0, ge=0)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # SNN integration fields
    spike_trace: dict[str, Any] | None = Field(
        default=None,
        description="Spike timing information for STDP (serialized SpikeTrace)",
    )
    tau_value: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="τ(t) gate value at encoding time",
    )
    neuromod_context: dict[str, Any] | None = Field(
        default=None,
        description="Neuromodulator state at encoding (serialized NeuromodContext)",
    )
    consolidation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of κ changes",
    )

    # Graph integration
    graph_delta: dict[str, Any] | None = Field(
        default=None,
        description="Pending graph updates (edges to create/strengthen)",
    )

    # Replay tracking
    replay_count: int = Field(default=0, ge=0)
    last_replay: datetime | None = None

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "UnifiedMemoryItem":
        """Convert a base MemoryItem to UnifiedMemoryItem."""
        return cls(
            id=item.id,
            content=item.content,
            embedding=item.embedding,
            event_time=item.event_time,
            record_time=item.record_time,
            valid_from=item.valid_from,
            valid_until=item.valid_until,
            kappa=item.kappa,
            importance=item.importance,
            item_type=item.item_type,
            access_count=item.access_count,
            session_id=item.session_id,
            metadata=item.metadata,
            spike_trace=item.spike_trace,
            graph_delta=item.graph_delta,
        )

    def to_memory_item(self) -> MemoryItem:
        """Convert back to base MemoryItem."""
        return MemoryItem(
            id=self.id,
            content=self.content,
            embedding=self.embedding,
            event_time=self.event_time,
            record_time=self.record_time,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            kappa=self.kappa,
            importance=self.importance,
            item_type=self.item_type,
            access_count=self.access_count,
            session_id=self.session_id,
            metadata=self.metadata,
            spike_trace=self.spike_trace,
            graph_delta=self.graph_delta,
        )

    @classmethod
    def from_episode(
        cls,
        ep: Episode,
        spike_trace: SpikeTrace | None = None,
        tau_value: float = 0.5,
        neuromod_context: NeuromodContext | None = None,
    ) -> "UnifiedMemoryItem":
        """
        Convert an Episode to a UnifiedMemoryItem.

        Args:
            ep: Episode to convert
            spike_trace: Spike timing from encoding
            tau_value: τ(t) gate value at encoding
            neuromod_context: Neuromodulator state at encoding

        Returns:
            UnifiedMemoryItem with κ=0 (raw episodic)
        """
        return cls(
            id=ep.id,
            content=ep.content,
            embedding=ep.embedding or [],
            event_time=ep.timestamp,
            record_time=ep.ingested_at,
            valid_from=ep.ingested_at,
            kappa=ep.kappa,  # Use episode's κ if set
            importance=ep.emotional_valence,
            item_type="episodic",
            access_count=ep.access_count,
            session_id=ep.session_id,
            metadata={
                "outcome": ep.outcome.value,
                "context": ep.context.model_dump() if ep.context else {},
                "prediction_error": ep.prediction_error,
            },
            spike_trace=spike_trace.to_dict() if spike_trace else None,
            tau_value=tau_value,
            neuromod_context=neuromod_context.to_dict() if neuromod_context else None,
        )

    @classmethod
    def from_entity(
        cls,
        ent: Entity,
        spike_trace: SpikeTrace | None = None,
    ) -> "UnifiedMemoryItem":
        """Convert an Entity to a UnifiedMemoryItem (κ near 1)."""
        return cls(
            id=ent.id,
            content=f"{ent.name}: {ent.summary}",
            embedding=ent.embedding or [],
            event_time=ent.created_at,
            record_time=ent.created_at,
            valid_from=ent.valid_from,
            valid_until=ent.valid_to,
            kappa=ent.kappa,  # Use entity's κ
            importance=0.7,
            item_type="semantic",
            access_count=ent.access_count,
            metadata={
                "entity_type": ent.entity_type.value,
                "name": ent.name,
                "details": ent.details,
                "source": ent.source,
            },
            spike_trace=spike_trace.to_dict() if spike_trace else None,
        )

    @classmethod
    def from_procedure(
        cls,
        proc: Procedure,
        spike_trace: SpikeTrace | None = None,
    ) -> "UnifiedMemoryItem":
        """Convert a Procedure to a UnifiedMemoryItem."""
        return cls(
            id=proc.id,
            content=f"{proc.name}: {proc.script or ''}",
            embedding=proc.embedding or [],
            event_time=proc.created_at,
            record_time=proc.created_at,
            valid_from=proc.created_at,
            kappa=proc.kappa,  # Use procedure's κ
            importance=proc.success_rate,
            item_type="procedural",
            access_count=proc.execution_count,
            metadata={
                "domain": proc.domain.value,
                "trigger_pattern": proc.trigger_pattern,
                "success_rate": proc.success_rate,
                "execution_count": proc.execution_count,
                "steps_count": len(proc.steps),
            },
            spike_trace=spike_trace.to_dict() if spike_trace else None,
        )

    def update_kappa(
        self,
        delta: float,
        phase: str = "access",
        replay_count: int = 0,
    ) -> None:
        """
        Update κ and record in consolidation history.

        Args:
            delta: Change in κ (can be negative for decay)
            phase: Consolidation phase ('nrem', 'rem', 'prune', 'access')
            replay_count: Number of replays in this phase
        """
        old_kappa = self.kappa
        self.kappa = max(0.0, min(1.0, self.kappa + delta))

        # Record history
        event = {
            "timestamp": datetime.now().isoformat(),
            "old_kappa": old_kappa,
            "new_kappa": self.kappa,
            "phase": phase,
            "delta": delta,
            "replay_count": replay_count,
        }
        self.consolidation_history.append(event)

    def record_replay(self) -> None:
        """Record that this memory was replayed."""
        self.replay_count += 1
        self.last_replay = datetime.now()

    def get_spike_trace(self) -> SpikeTrace | None:
        """Get spike trace as SpikeTrace object."""
        if self.spike_trace is None:
            return None
        return SpikeTrace.from_dict(self.spike_trace)

    def set_spike_trace(self, trace: SpikeTrace) -> None:
        """Set spike trace from SpikeTrace object."""
        self.spike_trace = trace.to_dict()

    def get_neuromod_context(self) -> NeuromodContext | None:
        """Get neuromodulator context as NeuromodContext object."""
        if self.neuromod_context is None:
            return None
        return NeuromodContext.from_dict(self.neuromod_context)

    def set_neuromod_context(self, context: NeuromodContext) -> None:
        """Set neuromodulator context from NeuromodContext object."""
        self.neuromod_context = context.to_dict()

    def is_episodic(self) -> bool:
        """Check if this is episodic (κ < 0.3)."""
        return self.kappa < 0.3

    def is_transitional(self) -> bool:
        """Check if this is transitional (0.3 <= κ < 0.7)."""
        return 0.3 <= self.kappa < 0.7

    def is_semantic(self) -> bool:
        """Check if this is semantic (κ >= 0.7)."""
        return self.kappa >= 0.7

    def should_replay(self, threshold: float = 0.3) -> bool:
        """
        Check if this memory should be replayed during consolidation.

        High importance + low κ = high replay priority.
        """
        return self.kappa < threshold and self.importance > 0.3


# Type alias for backward compatibility
EnhancedMemoryItem = UnifiedMemoryItem


def convert_to_unified(item: MemoryItem | Episode | Entity | Procedure) -> UnifiedMemoryItem:
    """
    Convert any memory type to UnifiedMemoryItem.

    Args:
        item: Memory item, Episode, Entity, or Procedure

    Returns:
        UnifiedMemoryItem
    """
    if isinstance(item, UnifiedMemoryItem):
        return item
    elif isinstance(item, MemoryItem):
        return UnifiedMemoryItem.from_memory_item(item)
    elif isinstance(item, Episode):
        return UnifiedMemoryItem.from_episode(item)
    elif isinstance(item, Entity):
        return UnifiedMemoryItem.from_entity(item)
    elif isinstance(item, Procedure):
        return UnifiedMemoryItem.from_procedure(item)
    else:
        raise TypeError(f"Cannot convert {type(item)} to UnifiedMemoryItem")


__all__ = [
    "UnifiedMemoryItem",
    "SpikeTrace",
    "ConsolidationEvent",
    "NeuromodContext",
    "EnhancedMemoryItem",
    "convert_to_unified",
]
