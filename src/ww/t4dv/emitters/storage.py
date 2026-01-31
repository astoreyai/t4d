"""Storage emitter — decorator wrapping T4DX engine ops."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from ww.t4dv.bus import ObservationBus
from ww.t4dv.events import StorageEvent

_TRACKED_OPS = (
    "insert", "get", "search", "update_fields", "update_edge_weight",
    "traverse", "scan", "delete", "batch_scale_weights",
)


def attach_storage_hooks(engine: Any, bus: ObservationBus) -> None:
    """Monkey-patch T4DXEngine ops to emit StorageEvents.

    Each of the 9 public operations is wrapped so that after execution
    a ``StorageEvent`` is emitted with timing and engine stats.
    """
    for op_name in _TRACKED_OPS:
        original = getattr(engine, op_name, None)
        if original is None:
            continue
        wrapped = _wrap_op(original, op_name, engine, bus)
        setattr(engine, op_name, wrapped)


def _wrap_op(
    fn: Callable,
    op_name: str,
    engine: Any,
    bus: ObservationBus,
) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000

        item_id = ""
        if args and isinstance(args[0], bytes):
            item_id = args[0].hex()
        elif hasattr(args[0] if args else None, "id"):
            item_id = str(getattr(args[0], "id", ""))

        event = StorageEvent(
            operation=op_name,
            item_id=item_id,
            duration_ms=round(duration_ms, 3),
            segment_count=getattr(engine, "segment_count", 0),
            memtable_count=getattr(engine, "memtable_count", 0),
            source="t4dx_engine",
        )
        bus.emit_sync(event)
        return result

    return wrapper
