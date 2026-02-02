"""Tests for VisualizationStreamer."""

import time

from t4dm.visualization.stream import VisualizationStreamer


class FakeVisualizer:
    """Fake visualizer that implements export_data()."""

    def __init__(self, data: dict | None = None):
        self._data = data or {"value": 42}

    def export_data(self) -> dict:
        return self._data


class BrokenVisualizer:
    """Visualizer that raises on export."""

    def export_data(self) -> dict:
        raise RuntimeError("broken")


class TestVisualizationStreamer:
    def test_register_and_list(self):
        s = VisualizationStreamer()
        assert s.registered_modules == []
        s.register_visualizer("foo", FakeVisualizer())
        assert "foo" in s.registered_modules

    def test_unregister(self):
        s = VisualizationStreamer()
        s.register_visualizer("foo", FakeVisualizer())
        s.unregister_visualizer("foo")
        assert s.registered_modules == []

    def test_snapshot_all(self):
        s = VisualizationStreamer()
        s.register_visualizer("a", FakeVisualizer({"x": 1}))
        s.register_visualizer("b", FakeVisualizer({"y": 2}))
        snap = s.get_snapshot()
        assert "timestamp" in snap
        assert snap["modules"]["a"] == {"x": 1}
        assert snap["modules"]["b"] == {"y": 2}

    def test_snapshot_filtered(self):
        s = VisualizationStreamer()
        s.register_visualizer("a", FakeVisualizer({"x": 1}))
        s.register_visualizer("b", FakeVisualizer({"y": 2}))
        snap = s.get_snapshot(modules=["a"])
        assert "a" in snap["modules"]
        assert "b" not in snap["modules"]

    def test_snapshot_missing_module(self):
        s = VisualizationStreamer()
        snap = s.get_snapshot(modules=["nonexistent"])
        assert snap["modules"] == {}

    def test_snapshot_broken_visualizer(self):
        s = VisualizationStreamer()
        s.register_visualizer("broken", BrokenVisualizer())
        snap = s.get_snapshot()
        assert "error" in snap["modules"]["broken"]

    def test_snapshot_timestamp_is_recent(self):
        s = VisualizationStreamer()
        before = time.time()
        snap = s.get_snapshot()
        after = time.time()
        assert before <= snap["timestamp"] <= after
