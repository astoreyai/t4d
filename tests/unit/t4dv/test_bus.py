"""Tests for ObservationBus."""

import asyncio

import pytest

from t4dm.t4dv.bus import ObservationBus
from t4dm.t4dv.events import ObservationEvent, SpikeEvent


class TestObservationBus:
    def test_emit_sync_stores(self):
        bus = ObservationBus(maxlen=100)
        e = SpikeEvent(block_index=0, firing_rate=0.5)
        bus.emit_sync(e)
        snap = bus.snapshot("spike")
        assert len(snap) == 1
        assert snap[0].firing_rate == 0.5

    def test_ring_buffer_overflow(self):
        bus = ObservationBus(maxlen=5)
        for i in range(10):
            bus.emit_sync(SpikeEvent(block_index=i))
        snap = bus.snapshot("spike")
        assert len(snap) == 5
        assert snap[0].block_index == 5

    def test_topics(self):
        bus = ObservationBus()
        bus.emit_sync(SpikeEvent())
        bus.emit_sync(ObservationEvent(topic="custom"))
        assert set(bus.topics()) == {"spike", "custom"}

    def test_clear(self):
        bus = ObservationBus()
        bus.emit_sync(SpikeEvent())
        bus.clear()
        assert bus.snapshot("spike") == []
        assert bus.topics() == []

    @pytest.mark.asyncio
    async def test_emit_async(self):
        bus = ObservationBus()
        e = SpikeEvent(block_index=2, firing_rate=0.9)
        await bus.emit(e)
        snap = bus.snapshot("spike")
        assert len(snap) == 1

    @pytest.mark.asyncio
    async def test_subscribe_receives_events(self):
        bus = ObservationBus()
        received = []
        bus.subscribe("spike", lambda e: received.append(e))
        await bus.emit(SpikeEvent(block_index=1))
        assert len(received) == 1
        assert received[0].block_index == 1

    @pytest.mark.asyncio
    async def test_subscribe_glob_pattern(self):
        bus = ObservationBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        await bus.emit(SpikeEvent())
        await bus.emit(ObservationEvent(topic="storage"))
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = ObservationBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe("spike", cb)
        bus.unsubscribe(cb)
        await bus.emit(SpikeEvent())
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_break(self):
        bus = ObservationBus()

        def bad_cb(e):
            raise ValueError("oops")

        received = []
        bus.subscribe("spike", bad_cb)
        bus.subscribe("spike", lambda e: received.append(e))
        await bus.emit(SpikeEvent())
        assert len(received) == 1
