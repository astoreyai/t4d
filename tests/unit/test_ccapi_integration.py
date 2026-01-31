"""
Unit tests for World Weaver ccapi integration.

Tests WWMemory, WWObserver, and API routes.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from uuid import uuid4

from t4dm.integration.ccapi_memory import WWMemory, Message, create_ww_memory
from t4dm.integration.ccapi_observer import WWObserver, Event, Span, EventType, create_ww_observer


class TestMessage:
    """Tests for Message stub class."""

    def test_creation_minimal(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.metadata == {}
        assert isinstance(msg.timestamp, datetime)

    def test_creation_full(self):
        msg = Message(
            role="assistant",
            content="Response",
            name="TestBot",
            tool_call_id="call_123",
            metadata={"key": "value"},
        )
        assert msg.name == "TestBot"
        assert msg.tool_call_id == "call_123"
        assert msg.metadata["key"] == "value"


class TestWWMemory:
    """Tests for WWMemory adapter class."""

    def test_creation_default(self):
        memory = WWMemory()
        assert memory.session_id == "default"
        assert memory.project == ""
        assert memory.max_messages == 1000
        assert memory.enable_learning is True
        assert len(memory) == 0

    def test_creation_custom(self):
        memory = WWMemory(
            session_id="test-session",
            project="test-project",
            max_messages=500,
            enable_learning=False,
        )
        assert memory.session_id == "test-session"
        assert memory.project == "test-project"
        assert memory.max_messages == 500
        assert memory.enable_learning is False

    def test_add_message(self):
        memory = WWMemory()
        msg = Message(role="user", content="Test")
        memory.add(msg)
        assert len(memory) == 1

    def test_add_many_messages(self):
        memory = WWMemory()
        msgs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
        ]
        memory.add_many(msgs)
        assert len(memory) == 3

    def test_get_messages(self):
        memory = WWMemory()
        for i in range(5):
            memory.add(Message(role="user", content=f"Message {i}"))

        messages = memory.get_messages()
        assert len(messages) == 5

    def test_get_messages_with_limit(self):
        memory = WWMemory()
        for i in range(10):
            memory.add(Message(role="user", content=f"Message {i}"))

        messages = memory.get_messages(limit=3)
        assert len(messages) == 3
        # Should be last 3 messages
        assert "Message 7" in messages[0].content

    def test_simple_search(self):
        memory = WWMemory()
        memory.add(Message(role="user", content="Hello world"))
        memory.add(Message(role="assistant", content="Hi there"))
        memory.add(Message(role="user", content="Hello again"))

        results = memory._simple_search("Hello", limit=5)
        assert len(results) == 2

    def test_simple_search_ranking(self):
        memory = WWMemory()
        memory.add(Message(role="user", content="Hello Hello Hello"))
        memory.add(Message(role="user", content="Hello"))

        results = memory._simple_search("Hello", limit=5)
        # First should have more occurrences
        assert "Hello Hello Hello" in results[0].content

    def test_clear(self):
        memory = WWMemory()
        memory.add(Message(role="user", content="Test"))
        assert len(memory) == 1
        memory.clear()
        assert len(memory) == 0

    def test_save_load(self, tmp_path):
        memory = WWMemory(session_id="save-test")
        memory.add(Message(role="user", content="Message 1"))
        memory.add(Message(role="assistant", content="Response 1"))

        path = tmp_path / "memory.json"
        memory.save(str(path))

        new_memory = WWMemory()
        new_memory.load(str(path))

        assert len(new_memory) == 2
        assert new_memory.session_id == "save-test"

    def test_get_context(self):
        memory = WWMemory()
        for i in range(10):
            memory.add(Message(role="user", content=f"Message {i}"))

        context = memory.get_context(max_messages=5)
        assert len(context) == 5

    def test_get_context_token_limit(self):
        memory = WWMemory()
        memory.add(Message(role="user", content="Short"))
        memory.add(Message(role="user", content="A" * 1000))  # ~250 tokens

        # With low token limit, should only get short message
        context = memory.get_context(max_tokens=50)
        assert len(context) <= 2

    def test_get_context_exclude_system(self):
        memory = WWMemory()
        memory.add(Message(role="system", content="System prompt"))
        memory.add(Message(role="user", content="User message"))

        context = memory.get_context(include_system=False)
        assert len(context) == 1
        assert context[0].role == "user"

    def test_get_last_message(self):
        memory = WWMemory()
        assert memory.get_last_message() is None

        memory.add(Message(role="user", content="First"))
        memory.add(Message(role="assistant", content="Last"))

        last = memory.get_last_message()
        assert last.content == "Last"

    def test_get_messages_by_role(self):
        memory = WWMemory()
        memory.add(Message(role="user", content="U1"))
        memory.add(Message(role="assistant", content="A1"))
        memory.add(Message(role="user", content="U2"))

        user_msgs = memory.get_messages_by_role("user")
        assert len(user_msgs) == 2


class TestCreateWWMemory:
    """Tests for create_ww_memory factory function."""

    def test_creates_instance(self):
        memory = create_ww_memory(session_id="factory-test")
        assert isinstance(memory, WWMemory)
        assert memory.session_id == "factory-test"


class TestEvent:
    """Tests for Event stub class."""

    def test_creation_minimal(self):
        event = Event(type="agent.start", name="TestAgent")
        assert event.type == "agent.start"
        assert event.name == "TestAgent"
        assert event.data == {}
        assert isinstance(event.timestamp, datetime)

    def test_creation_full(self):
        event = Event(
            type="tool.end",
            name="shell",
            data={"result": "success"},
            duration_ms=150.5,
            tags={"env": "test"},
        )
        assert event.data["result"] == "success"
        assert event.duration_ms == 150.5


class TestSpan:
    """Tests for Span stub class."""

    def test_creation(self):
        span = Span(
            name="agent-run",
            trace_id="trace-123",
            span_id="span-456",
        )
        assert span.name == "agent-run"
        assert span.status == "ok"


class TestWWObserver:
    """Tests for WWObserver class."""

    def test_creation_default(self):
        observer = WWObserver()
        assert observer.session_id == "default"
        assert observer.enable_learning is True
        assert observer.buffer_size == 100
        assert len(observer._event_buffer) == 0

    def test_creation_custom(self):
        observer = WWObserver(
            session_id="test",
            project="test-proj",
            enable_learning=False,
            buffer_size=50,
        )
        assert observer.session_id == "test"
        assert observer.enable_learning is False

    def test_on_event_buffering(self):
        observer = WWObserver()
        event = Event(type="agent.start", name="test")
        observer.on_event(event)
        assert len(observer._event_buffer) == 1

    def test_on_span_start(self):
        observer = WWObserver()
        span = Span(name="test", trace_id="t1", span_id="s1")
        observer.on_span_start(span)
        assert "s1" in observer._active_spans

    def test_on_span_end(self):
        observer = WWObserver()
        span = Span(name="test", trace_id="t1", span_id="s1")
        observer.on_span_start(span)
        observer.on_span_end(span)
        assert "s1" not in observer._active_spans

    def test_handle_memory_retrieve(self):
        observer = WWObserver()
        event = Event(
            type="memory.retrieve",
            name="recall",
            data={
                "query": "test query",
                "results": [{"id": "1", "score": 0.9}],
                "memory_type": "episodic",
            },
        )
        observer._handle_memory_retrieve(event)
        assert len(observer._recent_retrievals) == 1

    def test_recent_retrievals_limit(self):
        observer = WWObserver()
        observer._max_retrievals = 5

        for i in range(10):
            event = Event(
                type="memory.retrieve",
                name="recall",
                data={"query": f"query {i}"},
            )
            observer._handle_memory_retrieve(event)

        assert len(observer._recent_retrievals) == 5

    def test_flush(self):
        observer = WWObserver()
        observer.on_event(Event(type="test", name="t1"))
        observer.on_event(Event(type="test", name="t2"))

        observer.flush()
        assert len(observer._event_buffer) == 0

    def test_close(self):
        observer = WWObserver()
        observer.on_event(Event(type="test", name="t"))
        observer._active_spans["s1"] = Mock()
        observer._recent_retrievals.append({})

        observer.close()
        assert len(observer._active_spans) == 0
        assert len(observer._recent_retrievals) == 0

    def test_get_recent_retrievals(self):
        observer = WWObserver()
        observer._handle_memory_retrieve(Event(
            type="memory.retrieve",
            name="test",
            data={"query": "q1"},
        ))

        retrievals = observer.get_recent_retrievals()
        assert len(retrievals) == 1
        assert retrievals[0]["query"] == "q1"

    def test_get_session_stats(self):
        observer = WWObserver(session_id="stats-test")
        observer.on_event(Event(type="test", name="t"))

        stats = observer.get_session_stats()
        assert stats["session_id"] == "stats-test"
        assert stats["buffered_events"] == 1
        assert stats["learning_enabled"] is True

    def test_extract_citations(self):
        observer = WWObserver()

        # With memory_references
        citations = observer._extract_citations({
            "memory_references": [str(uuid4()), str(uuid4())],
        })
        assert len(citations) == 2

        # With used_memories
        citations = observer._extract_citations({
            "used_memories": [str(uuid4())],
        })
        assert len(citations) == 1

        # With invalid UUIDs
        citations = observer._extract_citations({
            "memory_references": ["not-a-uuid"],
        })
        assert len(citations) == 0


class TestCreateWWObserver:
    """Tests for create_ww_observer factory function."""

    def test_creates_instance(self):
        observer = create_ww_observer(session_id="factory-test")
        assert isinstance(observer, WWObserver)
        assert observer.session_id == "factory-test"


class TestScheduleAsync:
    """Tests for _schedule_async and _run_pending_tasks."""

    def test_schedule_async_adds_to_pending(self):
        observer = WWObserver()

        async def dummy_coro():
            pass

        observer._schedule_async(dummy_coro())
        assert len(observer._pending_tasks) == 1

    def test_run_pending_tasks_clears(self):
        observer = WWObserver()

        async def dummy_coro():
            return "done"

        observer._pending_tasks.append(dummy_coro())
        observer._pending_tasks.append(dummy_coro())

        observer._run_pending_tasks()
        assert len(observer._pending_tasks) == 0


class TestWWObserverEventHandlers:
    """Tests for WWObserver event handler methods."""

    def test_handle_agent_end_success(self):
        """Test handling successful agent end event."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.end",
            name="TestAgent",
            data={"status": "ok", "result": "task completed"},
        )
        observer._handle_agent_end(event)
        # No exception means success handling

    def test_handle_agent_end_partial(self):
        """Test handling partial success agent end event."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.end",
            name="TestAgent",
            data={"status": "partial", "result": "partially done"},
        )
        observer._handle_agent_end(event)

    def test_handle_agent_end_failure(self):
        """Test handling failed agent end event."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.end",
            name="TestAgent",
            data={"status": "error", "error": "failed"},
        )
        observer._handle_agent_end(event)

    def test_handle_agent_end_with_positive_feedback(self):
        """Test handling agent end with positive user feedback."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.end",
            name="TestAgent",
            data={
                "status": "ok",
                "user_feedback": "positive",
            },
        )
        observer._handle_agent_end(event)

    def test_handle_agent_end_with_negative_feedback(self):
        """Test handling agent end with negative user feedback."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.end",
            name="TestAgent",
            data={
                "status": "ok",
                "user_feedback": "negative",
            },
        )
        observer._handle_agent_end(event)

    def test_handle_tool_end_success(self):
        """Test handling successful tool end event."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="tool.end",
            name="shell",
            data={"tool": "bash", "result": "output"},
            duration_ms=100.5,
        )
        observer._handle_tool_end(event)

    def test_handle_tool_end_no_duration(self):
        """Test handling tool end event without duration."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="tool.end",
            name="read_file",
            data={"result": "file content"},
        )
        observer._handle_tool_end(event)

    def test_handle_tool_error(self):
        """Test handling tool error event."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="tool.error",
            name="shell",
            data={"tool": "bash", "error": "command not found"},
        )
        observer._handle_tool_error(event)

    def test_handle_tool_error_with_exception(self):
        """Test handling tool error with exception details."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="tool.error",
            name="api_call",
            data={
                "tool": "http_request",
                "error": Exception("Connection timeout"),
            },
        )
        observer._handle_tool_error(event)

    def test_handle_agent_error(self):
        """Test handling agent error event."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.error",
            name="TestAgent",
            data={"error": "Agent crashed"},
        )
        observer._handle_agent_error(event)

    def test_handle_agent_error_unknown(self):
        """Test handling agent error with no error details."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="agent.error",
            name="TestAgent",
            data={},  # No error field
        )
        observer._handle_agent_error(event)


class TestWWObserverSpanProcessing:
    """Tests for WWObserver span processing."""

    def test_process_agent_outcome_success(self):
        """Test processing successful agent span."""
        observer = WWObserver(enable_learning=False)
        span = Span(
            name="agent-test",
            trace_id="trace-1",
            span_id="span-1",
            status="ok",
            attributes={"result": "completed"},
        )
        observer._process_agent_outcome(span)

    def test_process_agent_outcome_error(self):
        """Test processing failed agent span."""
        observer = WWObserver(enable_learning=False)
        span = Span(
            name="agent-test",
            trace_id="trace-1",
            span_id="span-1",
            status="error",
            attributes={"error": "failed"},
        )
        observer._process_agent_outcome(span)

    def test_process_agent_outcome_neutral(self):
        """Test processing neutral status agent span."""
        observer = WWObserver(enable_learning=False)
        span = Span(
            name="agent-test",
            trace_id="trace-1",
            span_id="span-1",
            status="cancelled",  # Not ok or error
            attributes={},
        )
        observer._process_agent_outcome(span)

    def test_on_span_end_triggers_outcome(self):
        """Test that agent span end triggers outcome processing."""
        observer = WWObserver(enable_learning=False)
        span = Span(name="agent-run", trace_id="t1", span_id="s1", status="ok")
        observer.on_span_start(span)
        observer.on_span_end(span)
        # Agent span should trigger outcome processing
        assert "s1" not in observer._active_spans

    def test_on_span_end_non_agent(self):
        """Test that non-agent spans don't trigger outcome."""
        observer = WWObserver(enable_learning=False)
        span = Span(name="tool-call", trace_id="t1", span_id="s1")
        observer.on_span_start(span)
        observer.on_span_end(span)
        # No exception means proper handling


class TestWWObserverEventRouting:
    """Tests for WWObserver event type routing."""

    def test_on_event_routes_agent_end(self):
        """Test on_event routes agent.end events."""
        observer = WWObserver(enable_learning=False)
        event = Event(type="agent.end", name="test", data={"status": "ok"})
        observer.on_event(event)
        assert len(observer._event_buffer) == 1

    def test_on_event_routes_tool_end(self):
        """Test on_event routes tool.end events."""
        observer = WWObserver(enable_learning=False)
        event = Event(type="tool.end", name="test", data={"result": "ok"})
        observer.on_event(event)
        assert len(observer._event_buffer) == 1

    def test_on_event_routes_tool_error(self):
        """Test on_event routes tool.error events."""
        observer = WWObserver(enable_learning=False)
        event = Event(type="tool.error", name="test", data={"error": "fail"})
        observer.on_event(event)
        assert len(observer._event_buffer) == 1

    def test_on_event_routes_agent_error(self):
        """Test on_event routes agent.error events."""
        observer = WWObserver(enable_learning=False)
        event = Event(type="agent.error", name="test", data={"error": "crash"})
        observer.on_event(event)
        assert len(observer._event_buffer) == 1

    def test_on_event_routes_memory_retrieve(self):
        """Test on_event routes memory.retrieve events."""
        observer = WWObserver(enable_learning=False)
        event = Event(
            type="memory.retrieve",
            name="recall",
            data={"query": "test", "results": []},
        )
        observer.on_event(event)
        assert len(observer._event_buffer) == 1
        assert len(observer._recent_retrievals) == 1

    def test_on_event_buffer_flush(self):
        """Test on_event flushes when buffer is full."""
        observer = WWObserver(buffer_size=3)
        for i in range(5):
            observer.on_event(Event(type="test", name=f"event-{i}"))
        # Buffer should have been flushed once (at 3) and have 2 remaining
        assert len(observer._event_buffer) == 2


class TestWWObserverCitations:
    """Tests for WWObserver citation extraction."""

    def test_extract_citations_from_memory_references(self):
        """Test extracting citations from memory_references field."""
        observer = WWObserver()
        uid1, uid2 = uuid4(), uuid4()
        citations = observer._extract_citations({
            "memory_references": [str(uid1), str(uid2)],
        })
        assert len(citations) == 2
        assert uid1 in citations
        assert uid2 in citations

    def test_extract_citations_from_used_memories(self):
        """Test extracting citations from used_memories field."""
        observer = WWObserver()
        uid = uuid4()
        citations = observer._extract_citations({
            "used_memories": [str(uid)],
        })
        assert len(citations) == 1
        assert uid in citations

    def test_extract_citations_with_uuid_objects(self):
        """Test extracting citations when UUIDs are already UUID objects."""
        observer = WWObserver()
        uid = uuid4()
        citations = observer._extract_citations({
            "memory_references": [uid],  # Already UUID
        })
        assert len(citations) == 1
        assert uid in citations

    def test_extract_citations_mixed_formats(self):
        """Test extracting citations with mixed formats."""
        observer = WWObserver()
        uid1, uid2 = uuid4(), uuid4()
        citations = observer._extract_citations({
            "memory_references": [str(uid1), "invalid-uuid"],
            "used_memories": [uid2],
        })
        assert len(citations) == 2

    def test_extract_citations_empty(self):
        """Test extracting citations from empty data."""
        observer = WWObserver()
        citations = observer._extract_citations({})
        assert len(citations) == 0

    def test_extract_citations_non_list(self):
        """Test extracting citations when fields are not lists."""
        observer = WWObserver()
        citations = observer._extract_citations({
            "memory_references": "not-a-list",
        })
        assert len(citations) == 0


class TestEventType:
    """Tests for EventType stub class."""

    def test_event_type_values(self):
        """Test EventType has expected values."""
        assert EventType.AGENT_START == "agent.start"
        assert EventType.AGENT_END == "agent.end"
        assert EventType.AGENT_ERROR == "agent.error"
        assert EventType.TOOL_START == "tool.start"
        assert EventType.TOOL_END == "tool.end"
        assert EventType.TOOL_ERROR == "tool.error"
        assert EventType.LLM_RESPONSE == "llm.response"
        assert EventType.MEMORY_RETRIEVE == "memory.retrieve"
