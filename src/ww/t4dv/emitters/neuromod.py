"""Neuromodulator emitter — periodic NT level sampling."""

from __future__ import annotations

from typing import Any

from ww.t4dv.bus import ObservationBus
from ww.t4dv.events import NeuromodEvent


def sample_neuromod(state: Any, bus: ObservationBus) -> None:
    """Emit a NeuromodEvent from a neuromodulator state object.

    *state* should have attributes ``da``, ``ne``, ``ach``, ``serotonin``
    (or ``sht``).  Missing attributes default to 0.5.
    """
    event = NeuromodEvent(
        da=_getf(state, "da"),
        ne=_getf(state, "ne"),
        ach=_getf(state, "ach"),
        serotonin=_getf(state, "serotonin", _getf(state, "sht")),
        source="neuromod_bus",
    )
    bus.emit_sync(event)


def _getf(obj: Any, attr: str, default: float = 0.5) -> float:
    val = getattr(obj, attr, default)
    return float(val) if val is not None else default
