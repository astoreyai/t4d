"""Observation events emitted by T4DM components."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ObservationEvent(BaseModel):
    """Base event for the observation bus."""

    topic: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class SpikeEvent(ObservationEvent):
    """Emitted per cortical block forward pass."""

    topic: Literal["spike"] = "spike"
    block_index: int = 0
    firing_rate: float = 0.0
    prediction_error: float = 0.0
    goodness: float = 0.0
    mean_membrane: float = 0.0


class StorageEvent(ObservationEvent):
    """Emitted per T4DX storage operation."""

    topic: Literal["storage"] = "storage"
    operation: str = ""
    item_id: str = ""
    kappa: float | None = None
    duration_ms: float = 0.0
    segment_count: int = 0
    memtable_count: int = 0


class ConsolidationEvent(ObservationEvent):
    """Emitted during sleep-phase consolidation."""

    topic: Literal["consolidation"] = "consolidation"
    phase: Literal["nrem", "rem", "prune"] = "nrem"
    items_processed: int = 0
    kappa_delta_mean: float = 0.0
    segments_merged: int = 0
    items_pruned: int = 0


class NeuromodEvent(ObservationEvent):
    """Periodic neurotransmitter level snapshot."""

    topic: Literal["neuromod"] = "neuromod"
    da: float = 0.5
    ne: float = 0.5
    ach: float = 0.5
    serotonin: float = 0.5
