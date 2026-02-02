"""Bitemporal queries: 'What did we know when' over T4DX items."""

from __future__ import annotations

from typing import TYPE_CHECKING

from t4dm.storage.t4dx.types import ItemRecord

if TYPE_CHECKING:
    from t4dm.storage.t4dx.engine import T4DXEngine


class BitemporalQuery:
    """Builder for bitemporal queries combining system-time and event-time filters.

    System time (record_time): when T4DX recorded the item.
    Event time (event_time / valid_from / valid_until): when the event occurred in the world.
    """

    def __init__(self) -> None:
        self._system_time: float | None = None
        self._event_time: float | None = None
        self._event_start: float | None = None
        self._event_end: float | None = None

    def as_of(self, system_time: float) -> BitemporalQuery:
        """Filter to items with record_time <= system_time."""
        self._system_time = system_time
        return self

    def valid_at(self, event_time: float) -> BitemporalQuery:
        """Filter to items with valid_from <= event_time < valid_until."""
        self._event_time = event_time
        return self

    def between(self, event_start: float, event_end: float) -> BitemporalQuery:
        """Filter to items whose validity overlaps [event_start, event_end]."""
        self._event_start = event_start
        self._event_end = event_end
        return self

    def execute(self, engine: T4DXEngine) -> list[ItemRecord]:
        """Apply filters to engine.scan() results."""
        items = engine.scan()
        results: list[ItemRecord] = []
        for item in items:
            if not self._matches(item):
                continue
            results.append(item)
        return results

    def _matches(self, item: ItemRecord) -> bool:
        if self._system_time is not None:
            if item.record_time > self._system_time:
                return False

        if self._event_time is not None:
            if item.valid_from > self._event_time:
                return False
            if item.valid_until is not None and item.valid_until <= self._event_time:
                return False

        if self._event_start is not None and self._event_end is not None:
            # Item validity must overlap [event_start, event_end]
            if item.valid_from >= self._event_end:
                return False
            if item.valid_until is not None and item.valid_until <= self._event_start:
                return False

        return True
