"""Consolidation emitter — wraps NREM/REM/PRUNE compaction ops."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from t4dm.t4dv.bus import ObservationBus
from t4dm.t4dv.events import ConsolidationEvent


def attach_consolidation_hooks(engine: Any, bus: ObservationBus) -> None:
    """Wrap nrem_compact, rem_compact, prune on T4DXEngine."""
    for phase, method_name in [
        ("nrem", "nrem_compact"),
        ("rem", "rem_compact"),
        ("prune", "prune"),
    ]:
        original = getattr(engine, method_name, None)
        if original is None:
            continue
        wrapped = _wrap_consolidation(original, phase, engine, bus)
        setattr(engine, method_name, wrapped)


def _wrap_consolidation(
    fn: Callable,
    phase: str,
    engine: Any,
    bus: ObservationBus,
) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        seg_before = getattr(engine, "segment_count", 0)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        seg_after = getattr(engine, "segment_count", 0)

        items_processed = result if isinstance(result, int) else 0
        segments_merged = max(0, seg_before - seg_after)

        event = ConsolidationEvent(
            phase=phase,  # type: ignore[arg-type]
            items_processed=items_processed,
            segments_merged=segments_merged,
            source="t4dx_compactor",
            payload={"duration_ms": round(duration_ms, 3)},
        )
        bus.emit_sync(event)
        return result

    return wrapper
