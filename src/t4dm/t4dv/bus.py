"""ObservationBus — async in-process pub/sub with ring buffers."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
from collections import deque
from typing import Any, Callable

from t4dm.t4dv.events import ObservationEvent

logger = logging.getLogger(__name__)

Callback = Callable[[ObservationEvent], Any]

_bus_instance: ObservationBus | None = None
_bus_lock = threading.Lock()


class ObservationBus:
    """In-process observation bus with per-topic ring buffers.

    Subscribers register with topic patterns (glob-style).
    ``emit()`` is async; ``emit_sync()`` schedules onto the running loop
    from non-async (e.g. PyTorch forward) code via ``call_soon_threadsafe``.
    """

    def __init__(self, maxlen: int = 10_000) -> None:
        self._maxlen = maxlen
        self._buffers: dict[str, deque[ObservationEvent]] = {}
        self._subscribers: list[tuple[str, Callback]] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- publish --

    async def emit(self, event: ObservationEvent) -> None:
        """Emit an event (async context)."""
        self._store(event)
        await self._notify(event)

    def emit_sync(self, event: ObservationEvent) -> None:
        """Emit from synchronous / tensor-producing code.

        If an event loop is registered, schedules notification via
        ``call_soon_threadsafe``; otherwise stores silently.
        """
        self._store(event)
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(asyncio.ensure_future, self._notify(event))

    # -- subscribe --

    def subscribe(self, topic_pattern: str, callback: Callback) -> None:
        """Subscribe to events matching *topic_pattern* (fnmatch glob)."""
        with self._lock:
            self._subscribers.append((topic_pattern, callback))

    def unsubscribe(self, callback: Callback) -> None:
        with self._lock:
            self._subscribers = [
                (p, cb) for p, cb in self._subscribers if cb is not callback
            ]

    # -- query --

    def snapshot(self, topic: str) -> list[ObservationEvent]:
        """Return a copy of the ring buffer for *topic*."""
        with self._lock:
            buf = self._buffers.get(topic)
            return list(buf) if buf else []

    def topics(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())

    # -- lifecycle --

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()

    # -- internals --

    def _store(self, event: ObservationEvent) -> None:
        with self._lock:
            buf = self._buffers.get(event.topic)
            if buf is None:
                buf = deque(maxlen=self._maxlen)
                self._buffers[event.topic] = buf
            buf.append(event)

    async def _notify(self, event: ObservationEvent) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for pattern, cb in subs:
            if fnmatch.fnmatch(event.topic, pattern):
                try:
                    result = cb(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Subscriber error for topic=%s", event.topic)


def get_bus() -> ObservationBus:
    """Return the singleton ObservationBus."""
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = ObservationBus()
    return _bus_instance
