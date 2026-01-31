"""Tests for Kymera voice memory bridge."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from t4dm.integrations.kymera.bridge import (
    VoiceMemoryBridge,
    VoiceContext,
    MemoryContext,
    TemporalBatcher,
)
from t4dm.core.memory_gate import StorageDecision, GateResult


class TestVoiceContext:
    """Tests for VoiceContext dataclass."""

    def test_default_values(self):
        """VoiceContext has correct defaults."""
        ctx = VoiceContext(session_id="test")
        assert ctx.session_id == "test"
        assert ctx.is_voice is True
        assert ctx.turn_number == 0
        assert ctx.project is None

    def test_to_dict(self):
        """VoiceContext converts to dict."""
        ctx = VoiceContext(
            session_id="test",
            project="my-project",
            cwd="/home/user",
        )
        d = ctx.to_dict()
        assert d["session_id"] == "test"
        assert d["project"] == "my-project"
        assert d["is_voice"] is True


class TestMemoryContext:
    """Tests for MemoryContext dataclass."""

    def test_empty_prompt(self):
        """Empty memory context produces empty prompt."""
        ctx = MemoryContext()
        assert ctx.to_prompt() == ""

    def test_prompt_with_episodes(self):
        """Memory context includes episodes in prompt."""
        ctx = MemoryContext(
            episodes=[
                {"content": "First episode"},
                {"content": "Second episode"},
            ]
        )
        prompt = ctx.to_prompt()
        assert "Recent Relevant History" in prompt
        assert "First episode" in prompt

    def test_prompt_with_entities(self):
        """Memory context includes entities in prompt."""
        ctx = MemoryContext(
            entities=[
                {"name": "Python", "summary": "Programming language"},
            ]
        )
        prompt = ctx.to_prompt()
        assert "Relevant Knowledge" in prompt
        assert "Python" in prompt

    def test_prompt_with_skills(self):
        """Memory context includes skills in prompt."""
        ctx = MemoryContext(
            skills=[
                {"name": "git_commit", "description": "Commit changes to git"},
            ]
        )
        prompt = ctx.to_prompt()
        assert "Available Procedures" in prompt
        assert "git_commit" in prompt

    def test_prompt_with_personal_context(self):
        """Memory context includes personal context."""
        ctx = MemoryContext(personal_context="Today's meeting at 3pm")
        prompt = ctx.to_prompt()
        assert "Current Context" in prompt
        assert "meeting at 3pm" in prompt


class TestVoiceMemoryBridge:
    """Tests for VoiceMemoryBridge class."""

    @pytest.fixture
    def bridge(self):
        """Create bridge with mocked WW client."""
        ww_client = MagicMock()
        ww_client.call_tool = AsyncMock()
        return VoiceMemoryBridge(ww_client=ww_client)

    @pytest.fixture
    def voice_context(self):
        """Create test voice context."""
        return VoiceContext(
            session_id="test-session",
            project="test-project",
            conversation_id="conv-123",
        )

    def test_initialization(self, bridge):
        """Bridge initializes with components."""
        assert bridge.ww is not None
        assert bridge.gate is not None
        assert bridge.privacy is not None
        assert bridge.batcher is not None

    @pytest.mark.asyncio
    async def test_on_user_speech_blocked(self, bridge, voice_context):
        """Blocked speech returns None."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=True, content="")

            result = await bridge.on_user_speech("secret info", voice_context)
            assert result is None

    @pytest.mark.asyncio
    async def test_on_user_speech_skip(self, bridge, voice_context):
        """Low-importance speech is skipped."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=False, content="hi")

            with patch.object(bridge.gate, "evaluate") as mock_gate:
                mock_gate.return_value = GateResult(
                    decision=StorageDecision.SKIP,
                    score=0.1,
                    reasons=["Low importance"],
                    suggested_importance=0.1,
                )

                result = await bridge.on_user_speech("hi", voice_context)
                assert result is None

    @pytest.mark.asyncio
    async def test_on_user_speech_store(self, bridge, voice_context):
        """High-importance speech is stored."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=False, content="important info")

            with patch.object(bridge.gate, "evaluate") as mock_gate:
                mock_gate.return_value = GateResult(
                    decision=StorageDecision.STORE,
                    score=0.8,
                    reasons=["Important"],
                    suggested_importance=0.8,
                )

                bridge.ww.call_tool.return_value = {
                    "episode_id": "550e8400-e29b-41d4-a716-446655440000"
                }

                result = await bridge.on_user_speech("important info", voice_context)
                assert result is not None
                assert isinstance(result, UUID)

    @pytest.mark.asyncio
    async def test_on_user_speech_immediate(self, bridge, voice_context):
        """Immediate storage bypasses gate."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=False, content="save this")

            bridge.ww.call_tool.return_value = {
                "episode_id": "550e8400-e29b-41d4-a716-446655440000"
            }

            result = await bridge.on_user_speech(
                "save this",
                voice_context,
                store_immediately=True,
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_on_assistant_response_no_trigger(self, bridge, voice_context):
        """Response without triggers not stored."""
        result = await bridge.on_assistant_response(
            text="Here's the answer.",
            spoken_text="Here's the answer.",
            context=voice_context,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_on_assistant_response_with_trigger(self, bridge, voice_context):
        """Response with memory trigger is stored."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=False, content="I'll remember that")

            bridge.ww.call_tool.return_value = {
                "episode_id": "550e8400-e29b-41d4-a716-446655440000"
            }

            result = await bridge.on_assistant_response(
                text="I'll remember that for you.",
                spoken_text="I'll remember that.",
                context=voice_context,
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_on_assistant_response_action(self, bridge, voice_context):
        """Action response is stored."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=False, content="Done")

            bridge.ww.call_tool.return_value = {
                "episode_id": "550e8400-e29b-41d4-a716-446655440000"
            }

            result = await bridge.on_assistant_response(
                text="Done, I've sent the email.",
                spoken_text="Email sent.",
                context=voice_context,
                was_action=True,
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_on_conversation_end_flushes_batches(self, bridge, voice_context):
        """Conversation end flushes batched content."""
        # Add something to batch
        bridge.batcher._batches["test"] = [(datetime.now(), "batched content")]

        bridge.ww.call_tool.return_value = {
            "episode_id": "550e8400-e29b-41d4-a716-446655440000"
        }

        await bridge.on_conversation_end(voice_context)
        # Batch should be flushed
        assert len(bridge.batcher._batches) == 0

    @pytest.mark.asyncio
    async def test_on_conversation_end_with_summary(self, bridge, voice_context):
        """Conversation end stores summary."""
        bridge.ww.call_tool.return_value = {
            "episode_id": "550e8400-e29b-41d4-a716-446655440000"
        }

        result = await bridge.on_conversation_end(
            voice_context,
            summary="Discussed project plans",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, bridge, voice_context):
        """Relevant context retrieves from WW."""
        bridge.ww.call_tool.side_effect = [
            {"episodes": [{"content": "Previous discussion"}]},
            {"entities": [{"name": "Python"}]},
            {"skills": [{"name": "git_commit"}]},
        ]

        ctx = await bridge.get_relevant_context("tell me about Python", voice_context)

        assert len(ctx.episodes) == 1
        assert len(ctx.entities) == 1
        assert len(ctx.skills) == 1

    @pytest.mark.asyncio
    async def test_get_relevant_context_handles_error(self, bridge, voice_context):
        """Context retrieval handles errors gracefully."""
        bridge.ww.call_tool.side_effect = Exception("Connection error")

        ctx = await bridge.get_relevant_context("query", voice_context)
        # Should return empty context, not raise
        assert ctx.episodes == []
        assert ctx.entities == []

    @pytest.mark.asyncio
    async def test_store_explicit_memory(self, bridge, voice_context):
        """Explicit memory request stores with high importance."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=False, content="remember this")

            bridge.ww.call_tool.return_value = {
                "episode_id": "550e8400-e29b-41d4-a716-446655440000"
            }

            result = await bridge.store_explicit_memory("remember this", voice_context)
            assert result is not None

    @pytest.mark.asyncio
    async def test_store_explicit_memory_blocked(self, bridge, voice_context):
        """Blocked explicit memory raises error."""
        with patch.object(bridge.privacy, "filter") as mock_filter:
            mock_filter.return_value = MagicMock(blocked=True, content="")

            with pytest.raises(ValueError, match="blocked"):
                await bridge.store_explicit_memory("SSN 123-45-6789", voice_context)

    @pytest.mark.asyncio
    async def test_recall_explicit(self, bridge, voice_context):
        """Explicit recall retrieves memories."""
        bridge.ww.call_tool.return_value = {
            "episodes": [{"content": "You said you like pizza"}]
        }

        results = await bridge.recall_explicit("pizza", voice_context)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_forget(self, bridge, voice_context):
        """Forget request is logged."""
        count = await bridge.forget("embarrassing thing", voice_context)
        # Currently returns 0 as placeholder
        assert count == 0


class TestTemporalBatcher:
    """Tests for TemporalBatcher class."""

    @pytest.fixture
    def batcher(self):
        """Create batcher with short window."""
        return TemporalBatcher(batch_window_minutes=1, max_batch_size=3)

    def test_add_first_item(self, batcher):
        """First item doesn't complete batch."""
        result = batcher.add("key1", "first content")
        assert result is None
        assert len(batcher._batches["key1"]) == 1

    def test_add_multiple_items(self, batcher):
        """Multiple items accumulate in batch."""
        batcher.add("key1", "first")
        batcher.add("key1", "second")
        assert len(batcher._batches["key1"]) == 2

    def test_add_reaches_max_size(self, batcher):
        """Batch returns when max size reached."""
        batcher.add("key1", "first")
        batcher.add("key1", "second")
        batcher.add("key1", "third")  # Reach max_batch_size (3)
        result = batcher.add("key1", "fourth")  # Fourth triggers flush

        # Fourth item triggers flush of first three
        assert result == "first | second | third"
        assert len(batcher._batches["key1"]) == 1  # Only fourth remains

    def test_flush_all(self, batcher):
        """Flush all returns all batches."""
        batcher.add("key1", "content1")
        batcher.add("key2", "content2")

        results = batcher.flush_all()

        assert len(results) == 2
        assert ("key1", "content1") in results
        assert ("key2", "content2") in results
        assert len(batcher._batches) == 0

    def test_flush_empty(self, batcher):
        """Flush empty batches returns empty list."""
        results = batcher.flush_all()
        assert results == []

    def test_combine_removes_duplicates(self, batcher):
        """Combine batch removes duplicate content."""
        batch = [
            (datetime.now(), "content"),
            (datetime.now(), "content"),  # Duplicate
            (datetime.now(), "other"),
        ]
        result = batcher._combine_batch(batch)
        assert result == "content | other"

    def test_separate_keys(self, batcher):
        """Different keys maintain separate batches."""
        batcher.add("key1", "content1")
        batcher.add("key2", "content2")

        assert len(batcher._batches) == 2
        assert len(batcher._batches["key1"]) == 1
        assert len(batcher._batches["key2"]) == 1
