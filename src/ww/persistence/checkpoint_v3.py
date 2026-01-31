"""Checkpoint v3: extends CheckpointManager for T4DX + spiking state (P5-01).

Adds checkpointing of:
- T4DX MemTable state (items, edges, field_overlays, edge_deltas, tombstones)
- T4DX WAL position (LSN)
- Spiking cortical stack weights (state_dict)
- QLoRA adapter weights (state_dict)
- Membrane potentials (per-block state tensors)
- Projection layer weights
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ww.persistence.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointManager,
    CheckpointableComponent,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckpointV3Config(CheckpointConfig):
    """Extended config for v3 checkpoints."""
    include_spiking: bool = True
    include_qlora: bool = True
    include_projections: bool = True


class T4DXCheckpointable:
    """Wraps a T4DXEngine as a CheckpointableComponent for checkpointing."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Capture T4DX memtable state for checkpointing."""
        mt = self._engine._memtable

        # Serialize items
        items_data = {}
        for item_id, rec in mt._items.items():
            items_data[item_id.hex()] = rec.to_dict()

        # Serialize edges
        edges_data = [e.to_dict() for e in mt._edges]

        # Serialize field overlays
        overlays_data = [
            {"item_id": o.item_id.hex(), "fields": o.fields}
            for o in mt._field_overlays
        ]

        # Serialize edge deltas
        deltas_data = [
            {
                "source_id": d.source_id.hex(),
                "target_id": d.target_id.hex(),
                "edge_type": d.edge_type,
                "weight_delta": d.weight_delta,
            }
            for d in mt._edge_deltas
        ]

        # Tombstones
        tombstones = [t.hex() for t in mt._deleted_ids]

        return {
            "items": items_data,
            "edges": edges_data,
            "field_overlays": overlays_data,
            "edge_deltas": deltas_data,
            "tombstones": tombstones,
            "segment_count": self._engine.segment_count,
        }

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore T4DX memtable state from checkpoint."""
        from ww.storage.t4dx.memtable import EdgeDelta, FieldOverlay
        from ww.storage.t4dx.types import EdgeRecord, ItemRecord

        mt = self._engine._memtable

        # Clear current memtable
        mt._items = {}
        mt._edges = []
        mt._field_overlays = []
        mt._edge_deltas = []
        mt._deleted_ids = set()

        # Restore items
        for _hex_id, item_dict in state.get("items", {}).items():
            rec = ItemRecord.from_dict(item_dict)
            mt._items[rec.id] = rec

        # Restore edges
        for edge_dict in state.get("edges", []):
            mt._edges.append(EdgeRecord.from_dict(edge_dict))

        # Restore field overlays
        for ov in state.get("field_overlays", []):
            mt._field_overlays.append(
                FieldOverlay(
                    item_id=bytes.fromhex(ov["item_id"]),
                    fields=ov["fields"],
                )
            )

        # Restore edge deltas
        for d in state.get("edge_deltas", []):
            mt._edge_deltas.append(
                EdgeDelta(
                    source_id=bytes.fromhex(d["source_id"]),
                    target_id=bytes.fromhex(d["target_id"]),
                    edge_type=d["edge_type"],
                    weight_delta=d["weight_delta"],
                )
            )

        # Restore tombstones
        mt._deleted_ids = {bytes.fromhex(t) for t in state.get("tombstones", [])}


class SpikingCheckpointable:
    """Wraps spiking cortical stack + optional QLoRA + projections as checkpointable."""

    def __init__(
        self,
        cortical_stack: Any | None = None,
        qlora_model: Any | None = None,
        projection: Any | None = None,
        block_states: list[dict] | None = None,
    ) -> None:
        self._stack = cortical_stack
        self._qlora = qlora_model
        self._projection = projection
        self._block_states = block_states

    def set_block_states(self, states: list[dict]) -> None:
        """Update the block states (membrane potentials etc.)."""
        self._block_states = states

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Capture spiking state for checkpointing."""
        state: dict[str, Any] = {}

        if self._stack is not None and HAS_TORCH:
            buf = io.BytesIO()
            torch.save(self._stack.state_dict(), buf)
            state["cortical_stack"] = buf.getvalue()

        if self._qlora is not None and HAS_TORCH:
            buf = io.BytesIO()
            torch.save(self._qlora.state_dict(), buf)
            state["qlora_model"] = buf.getvalue()

        if self._projection is not None and HAS_TORCH:
            buf = io.BytesIO()
            torch.save(self._projection.state_dict(), buf)
            state["projection"] = buf.getvalue()

        if self._block_states is not None and HAS_TORCH:
            buf = io.BytesIO()
            torch.save(self._block_states, buf)
            state["block_states"] = buf.getvalue()

        return state

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore spiking state from checkpoint."""
        if not HAS_TORCH:
            return

        if "cortical_stack" in state and self._stack is not None:
            buf = io.BytesIO(state["cortical_stack"])
            sd = torch.load(buf, weights_only=True)
            self._stack.load_state_dict(sd)

        if "qlora_model" in state and self._qlora is not None:
            buf = io.BytesIO(state["qlora_model"])
            sd = torch.load(buf, weights_only=True)
            self._qlora.load_state_dict(sd)

        if "projection" in state and self._projection is not None:
            buf = io.BytesIO(state["projection"])
            sd = torch.load(buf, weights_only=True)
            self._projection.load_state_dict(sd)

        if "block_states" in state:
            buf = io.BytesIO(state["block_states"])
            self._block_states = torch.load(buf, weights_only=True)


class CheckpointManagerV3(CheckpointManager):
    """Extended checkpoint manager with convenience methods for T4DX + spiking."""

    def __init__(self, config: CheckpointV3Config | CheckpointConfig) -> None:
        super().__init__(config)
        self._t4dx: T4DXCheckpointable | None = None
        self._spiking: SpikingCheckpointable | None = None

    def register_t4dx(self, engine: Any) -> None:
        """Register T4DX engine for checkpointing."""
        self._t4dx = T4DXCheckpointable(engine)
        self.register_component("t4dx", self._t4dx)

    def register_spiking(
        self,
        cortical_stack: Any | None = None,
        qlora_model: Any | None = None,
        projection: Any | None = None,
    ) -> None:
        """Register spiking components for checkpointing."""
        self._spiking = SpikingCheckpointable(
            cortical_stack=cortical_stack,
            qlora_model=qlora_model,
            projection=projection,
        )
        self.register_component("spiking", self._spiking)

    def update_block_states(self, states: list[dict]) -> None:
        """Update membrane potentials / block states before checkpoint."""
        if self._spiking is not None:
            self._spiking.set_block_states(states)
