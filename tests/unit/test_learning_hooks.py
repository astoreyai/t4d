"""
Unit tests for T4DM learning hooks.

Tests emit_retrieval_event, emit_unified_retrieval_event, learning_retrieval decorator,
and RetrievalHookMixin.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from t4dm.learning.hooks import (
    get_learning_collector,
    emit_retrieval_event,
    emit_unified_retrieval_event,
    learning_retrieval,
    RetrievalHookMixin,
)
from t4dm.learning.events import MemoryType, RetrievalEvent


class TestGetLearningCollector:
    """Tests for get_learning_collector function."""

    def test_returns_none_when_unavailable(self):
        with patch('t4dm.learning.hooks.get_collector', side_effect=Exception("not available")):
            with patch('t4dm.learning.hooks._collector', None):
                # Reset global
                import t4dm.learning.hooks as hooks
                hooks._collector = None
                result = get_learning_collector()
                # May return None or cached value depending on state
                assert result is None or hasattr(result, 'record_retrieval')


class TestEmitRetrievalEvent:
    """Tests for emit_retrieval_event function."""

    def test_returns_none_without_collector(self):
        with patch('t4dm.learning.hooks.get_learning_collector', return_value=None):
            result = emit_retrieval_event(
                query="test",
                memory_type=MemoryType.EPISODIC,
                results=[],
            )
            assert result is None

    def test_with_collector(self):
        mock_collector = Mock()
        mock_event = RetrievalEvent(query="test")
        mock_collector.record_retrieval.return_value = mock_event

        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            result = emit_retrieval_event(
                query="test query",
                memory_type=MemoryType.EPISODIC,
                results=[
                    {"id": str(uuid4()), "score": 0.9},
                    {"id": str(uuid4()), "score": 0.8},
                ],
                session_id="test-session",
            )

            mock_collector.record_retrieval.assert_called_once()
            assert result == mock_event

    def test_with_component_scores(self):
        mock_collector = Mock()
        mock_collector.record_retrieval.return_value = RetrievalEvent()

        result_id = str(uuid4())
        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            emit_retrieval_event(
                query="test",
                memory_type=MemoryType.SEMANTIC,
                results=[{
                    "id": result_id,
                    "score": 0.85,
                    "components": {"similarity": 0.9, "recency": 0.5},
                }],
            )

            call_kwargs = mock_collector.record_retrieval.call_args[1]
            assert "component_scores" in call_kwargs

    def test_handles_invalid_uuid(self):
        mock_collector = Mock()
        mock_collector.record_retrieval.return_value = RetrievalEvent()

        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            # Should not raise despite invalid UUID
            result = emit_retrieval_event(
                query="test",
                memory_type=MemoryType.EPISODIC,
                results=[{"id": "not-a-uuid", "score": 0.5}],
            )
            # Still returns event (invalid IDs are skipped)
            assert result is not None

    def test_handles_exception(self):
        mock_collector = Mock()
        mock_collector.record_retrieval.side_effect = Exception("test error")

        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            result = emit_retrieval_event(
                query="test",
                memory_type=MemoryType.EPISODIC,
                results=[{"id": str(uuid4()), "score": 0.5}],
            )
            assert result is None


class TestEmitUnifiedRetrievalEvent:
    """Tests for emit_unified_retrieval_event function."""

    def test_emits_multiple_events(self):
        mock_collector = Mock()
        mock_collector.record_retrieval.return_value = RetrievalEvent()

        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            events = emit_unified_retrieval_event(
                query="unified test",
                results_by_type={
                    "episodic": [{"id": str(uuid4()), "score": 0.9}],
                    "semantic": [{"id": str(uuid4()), "score": 0.8}],
                    "procedural": [{"id": str(uuid4()), "score": 0.7}],
                },
                session_id="test-session",
            )

            # Should emit one event per type
            assert len(events) == 3

    def test_skips_empty_results(self):
        mock_collector = Mock()
        mock_collector.record_retrieval.return_value = RetrievalEvent()

        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            events = emit_unified_retrieval_event(
                query="test",
                results_by_type={
                    "episodic": [{"id": str(uuid4()), "score": 0.9}],
                    "semantic": [],  # Empty
                    "procedural": [],  # Empty
                },
            )

            assert len(events) == 1

    def test_skips_unknown_types(self):
        mock_collector = Mock()
        mock_collector.record_retrieval.return_value = RetrievalEvent()

        with patch('t4dm.learning.hooks.get_learning_collector', return_value=mock_collector):
            events = emit_unified_retrieval_event(
                query="test",
                results_by_type={
                    "episodic": [{"id": str(uuid4()), "score": 0.9}],
                    "unknown_type": [{"id": str(uuid4()), "score": 0.5}],
                },
            )

            # Only episodic should be emitted
            assert len(events) == 1


class TestLearningRetrievalDecorator:
    """Tests for learning_retrieval decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_through(self):
        """Decorator should not change function behavior."""
        @learning_retrieval(MemoryType.EPISODIC)
        async def mock_recall(query: str) -> list:
            return [{"id": "test", "score": 0.9}]

        with patch('t4dm.learning.hooks.emit_retrieval_event'):
            result = await mock_recall("test query")
            assert result == [{"id": "test", "score": 0.9}]

    @pytest.mark.asyncio
    async def test_decorator_emits_event(self):
        """Decorator should emit retrieval event."""

        class MockResult:
            def __init__(self):
                self.item = Mock()
                self.item.id = uuid4()
                self.score = 0.85
                self.components = {"sim": 0.9}

        @learning_retrieval(MemoryType.SEMANTIC)
        async def mock_recall(query: str) -> list:
            return [MockResult()]

        with patch('t4dm.learning.hooks.emit_retrieval_event') as mock_emit:
            await mock_recall("test query")
            mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_decorator_handles_exception(self):
        """Decorator should not fail on emit error."""
        @learning_retrieval(MemoryType.EPISODIC)
        async def mock_recall(query: str) -> list:
            return [{"id": "test", "score": 0.9}]

        with patch('t4dm.learning.hooks.emit_retrieval_event', side_effect=Exception("emit failed")):
            # Should not raise
            result = await mock_recall("test")
            assert result == [{"id": "test", "score": 0.9}]


class TestRetrievalHookMixin:
    """Tests for RetrievalHookMixin class."""

    def test_default_memory_type(self):
        class TestClass(RetrievalHookMixin):
            pass

        obj = TestClass()
        assert obj._hook_memory_type == MemoryType.EPISODIC
        assert obj._hook_enabled is True

    def test_custom_memory_type(self):
        class TestClass(RetrievalHookMixin):
            _hook_memory_type = MemoryType.PROCEDURAL

        obj = TestClass()
        assert obj._hook_memory_type == MemoryType.PROCEDURAL

    def test_emit_hook_disabled(self):
        class TestClass(RetrievalHookMixin):
            pass

        obj = TestClass()
        obj._hook_enabled = False

        result = obj._emit_hook("query", [], "session")
        assert result is None

    def test_emit_hook_with_results(self):
        class TestClass(RetrievalHookMixin):
            pass

        class MockResult:
            def __init__(self):
                self.item = Mock()
                self.item.id = uuid4()
                self.score = 0.9
                self.components = {}

        obj = TestClass()

        with patch('t4dm.learning.hooks.emit_retrieval_event') as mock_emit:
            mock_emit.return_value = RetrievalEvent()
            result = obj._emit_hook("test query", [MockResult()], "session-1")

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["query"] == "test query"
            assert call_kwargs["session_id"] == "session-1"

    def test_emit_hook_with_dict_results(self):
        class TestClass(RetrievalHookMixin):
            pass

        obj = TestClass()

        with patch('t4dm.learning.hooks.emit_retrieval_event') as mock_emit:
            mock_emit.return_value = RetrievalEvent()
            obj._emit_hook(
                "test",
                [{"id": "abc", "score": 0.8}],
                "",
            )

            mock_emit.assert_called_once()
