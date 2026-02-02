"""
Unit tests for T4DM learning events.

Tests RetrievalEvent, OutcomeEvent, Experience, and representation formats.
"""

import json
import pytest
from datetime import datetime
from uuid import UUID, uuid4

from t4dm.learning.events import (
    OutcomeType,
    FeedbackSignal,
    MemoryType,
    RetrievalEvent,
    OutcomeEvent,
    Experience,
    FullJSON,
    ToonJSON,
    NeuroSymbolicTriples,
    get_representation,
    estimate_tokens,
)


class TestOutcomeType:
    """Tests for OutcomeType enum."""

    def test_all_values(self):
        assert OutcomeType.SUCCESS.value == "success"
        assert OutcomeType.PARTIAL.value == "partial"
        assert OutcomeType.FAILURE.value == "failure"
        assert OutcomeType.NEUTRAL.value == "neutral"
        assert OutcomeType.UNKNOWN.value == "unknown"

    def test_from_string(self):
        assert OutcomeType("success") == OutcomeType.SUCCESS
        assert OutcomeType("failure") == OutcomeType.FAILURE


class TestFeedbackSignal:
    """Tests for FeedbackSignal enum."""

    def test_all_values(self):
        assert FeedbackSignal.ACCEPT.value == "accept"
        assert FeedbackSignal.REJECT.value == "reject"
        assert FeedbackSignal.MODIFY.value == "modify"
        assert FeedbackSignal.REPEAT.value == "repeat"
        assert FeedbackSignal.EXPLICIT_POS.value == "explicit_positive"
        assert FeedbackSignal.EXPLICIT_NEG.value == "explicit_negative"


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_all_values(self):
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"


class TestRetrievalEvent:
    """Tests for RetrievalEvent dataclass."""

    def test_default_creation(self):
        event = RetrievalEvent()
        assert isinstance(event.retrieval_id, UUID)
        assert event.query == ""
        assert event.memory_type == MemoryType.EPISODIC
        assert event.retrieved_ids == []
        assert event.retrieval_scores == {}
        assert event.component_scores == {}
        assert event.context_hash == ""
        assert isinstance(event.timestamp, datetime)

    def test_full_creation(self):
        mem_id = uuid4()
        event = RetrievalEvent(
            query="test query",
            memory_type=MemoryType.SEMANTIC,
            retrieved_ids=[mem_id],
            retrieval_scores={"abc12345": 0.95},
            component_scores={"abc12345": {"similarity": 0.9, "recency": 0.8}},
            session_id="test-session",
            project="test-project",
        )
        assert event.query == "test query"
        assert event.memory_type == MemoryType.SEMANTIC
        assert len(event.retrieved_ids) == 1
        assert event.retrieval_scores["abc12345"] == 0.95

    def test_compute_context_hash(self):
        event = RetrievalEvent()
        hash1 = event.compute_context_hash("context A")
        hash2 = event.compute_context_hash("context B")

        assert len(hash1) == 16
        assert hash1 != hash2
        assert event.context_hash == hash2  # Last computed

    def test_to_dict(self):
        mem_id = uuid4()
        event = RetrievalEvent(
            query="test",
            memory_type=MemoryType.PROCEDURAL,
            retrieved_ids=[mem_id],
        )
        d = event.to_dict()

        assert isinstance(d["retrieval_id"], str)
        assert d["query"] == "test"
        assert d["memory_type"] == "procedural"
        assert d["retrieved_ids"] == [str(mem_id)]
        assert "timestamp" in d

    def test_from_dict(self):
        original = RetrievalEvent(
            query="roundtrip test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[uuid4()],
            retrieval_scores={"test": 0.5},
        )
        d = original.to_dict()
        restored = RetrievalEvent.from_dict(d)

        assert restored.query == original.query
        assert restored.memory_type == original.memory_type
        assert len(restored.retrieved_ids) == 1


class TestOutcomeEvent:
    """Tests for OutcomeEvent dataclass."""

    def test_default_creation(self):
        event = OutcomeEvent()
        assert isinstance(event.outcome_id, UUID)
        assert event.outcome_type == OutcomeType.UNKNOWN
        assert event.success_score == 0.5
        assert event.explicit_citations == []
        assert event.feedback_signals == []

    def test_full_creation(self):
        citation_id = uuid4()
        event = OutcomeEvent(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.95,
            session_id="test-session",
            explicit_citations=[citation_id],
            feedback_signals=[FeedbackSignal.ACCEPT],
            task_description="Complete task",
            tool_results={"status": "ok"},
        )
        assert event.outcome_type == OutcomeType.SUCCESS
        assert event.success_score == 0.95
        assert len(event.explicit_citations) == 1
        assert FeedbackSignal.ACCEPT in event.feedback_signals

    def test_compute_context_hash(self):
        event = OutcomeEvent()
        hash_val = event.compute_context_hash("some context")
        assert len(hash_val) == 16
        assert event.context_hash == hash_val

    def test_to_dict(self):
        event = OutcomeEvent(
            outcome_type=OutcomeType.PARTIAL,
            success_score=0.6,
            feedback_signals=[FeedbackSignal.MODIFY],
        )
        d = event.to_dict()

        assert isinstance(d["outcome_id"], str)
        assert d["outcome_type"] == "partial"
        assert d["success_score"] == 0.6
        assert "modify" in d["feedback_signals"]

    def test_from_dict(self):
        original = OutcomeEvent(
            outcome_type=OutcomeType.FAILURE,
            success_score=0.1,
            explicit_citations=[uuid4()],
        )
        d = original.to_dict()
        restored = OutcomeEvent.from_dict(d)

        assert restored.outcome_type == OutcomeType.FAILURE
        assert restored.success_score == 0.1
        assert len(restored.explicit_citations) == 1


class TestExperience:
    """Tests for Experience dataclass."""

    def test_default_creation(self):
        exp = Experience()
        assert isinstance(exp.experience_id, UUID)
        assert exp.query == ""
        assert exp.memory_type == MemoryType.EPISODIC
        assert exp.component_vectors == []
        assert exp.outcome_score == 0.5
        assert exp.priority == 1.0

    def test_full_creation(self):
        exp = Experience(
            query="test query",
            memory_type=MemoryType.PROCEDURAL,
            retrieved_ids=[uuid4(), uuid4()],
            retrieval_scores=[0.9, 0.7],
            component_vectors=[[0.9, 0.3, 0.5, 0.2], [0.7, 0.4, 0.6, 0.3]],
            outcome_score=0.85,
            per_memory_rewards={"mem1": 0.8, "mem2": 0.6},
            priority=2.5,
        )
        assert exp.query == "test query"
        assert len(exp.component_vectors) == 2
        assert exp.outcome_score == 0.85
        assert exp.priority == 2.5

    def test_to_dict(self):
        exp = Experience(
            query="dict test",
            outcome_score=0.7,
        )
        d = exp.to_dict()

        assert isinstance(d["experience_id"], str)
        assert d["query"] == "dict test"
        assert d["outcome_score"] == 0.7


class TestFullJSON:
    """Tests for FullJSON representation."""

    def test_encode_decode_roundtrip(self):
        fmt = FullJSON()
        data = {
            "retrieval_id": str(uuid4()),
            "query": "test query",
            "scores": [0.9, 0.8, 0.7],
        }

        encoded = fmt.encode(data)
        decoded = fmt.decode(encoded)

        assert decoded["query"] == "test query"
        assert decoded["scores"] == [0.9, 0.8, 0.7]

    def test_token_efficiency(self):
        fmt = FullJSON()
        assert fmt.token_efficiency == 1.0


class TestToonJSON:
    """Tests for ToonJSON token-optimized representation."""

    def test_key_abbreviation(self):
        fmt = ToonJSON()
        data = {
            "retrieval_id": "abc123",
            "query": "test",
            "memory_type": "episodic",
        }

        encoded = fmt.encode(data)
        assert "ri" in encoded  # Abbreviated key
        assert "retrieval_id" not in encoded

    def test_value_abbreviation(self):
        fmt = ToonJSON()
        data = {
            "memory_type": "episodic",
            "outcome": "success",
        }

        encoded = fmt.encode(data)
        # Should contain abbreviated values
        assert "E" in encoded or "+" in encoded

    def test_empty_value_omission(self):
        fmt = ToonJSON()
        data = {
            "query": "test",
            "empty_list": [],
            "empty_dict": {},
            "empty_string": "",
            "null_value": None,
        }

        encoded = fmt.encode(data)
        decoded = json.loads(encoded)

        assert "q" in decoded
        assert "empty_list" not in encoded
        assert "empty_dict" not in encoded

    def test_float_rounding(self):
        fmt = ToonJSON()
        data = {"score": 0.123456789}

        encoded = fmt.encode(data)
        decoded = json.loads(encoded)

        assert decoded["score"] == 0.12

    def test_uuid_shortening(self):
        fmt = ToonJSON()
        full_uuid = uuid4()
        data = {"id": full_uuid}

        encoded = fmt.encode(data)
        decoded = json.loads(encoded)

        assert len(decoded["id"]) == 8

    def test_decode_roundtrip(self):
        fmt = ToonJSON()
        data = {
            "query": "test",
            "memory_type": "episodic",
            "score": 0.85,
        }

        encoded = fmt.encode(data)
        decoded = fmt.decode(encoded)

        assert decoded["query"] == "test"
        assert decoded["memory_type"] == "episodic"

    def test_token_efficiency(self):
        fmt = ToonJSON()
        assert fmt.token_efficiency == 0.5

    def test_nested_dict_compaction(self):
        fmt = ToonJSON()
        data = {
            "component_scores": {
                "mem1": {"similarity": 0.9, "recency": 0.5}
            }
        }

        encoded = fmt.encode(data)
        # Should have abbreviated keys
        assert "cs" in encoded or "sim" in encoded


class TestNeuroSymbolicTriples:
    """Tests for NeuroSymbolicTriples representation."""

    def test_basic_encoding(self):
        fmt = NeuroSymbolicTriples()
        data = {
            "retrieval_id": "abc12345",
            "query": "test query",
            "score": 0.85,
        }

        encoded = fmt.encode(data)
        lines = encoded.strip().split("\n")

        assert len(lines) >= 2
        assert "|" in lines[0]

    def test_triple_format(self):
        fmt = NeuroSymbolicTriples()
        data = {
            "id": "subj123",
            "name": "TestEntity",
        }

        encoded = fmt.encode(data)
        # Format: subject|predicate|object
        for line in encoded.split("\n"):
            parts = line.split("|")
            assert len(parts) == 3

    def test_list_handling(self):
        fmt = NeuroSymbolicTriples()
        data = {
            "id": "subj",
            "tags": ["tag1", "tag2", "tag3"],
        }

        encoded = fmt.encode(data)
        # Should have indexed predicates
        assert "[0]" in encoded
        assert "[1]" in encoded

    def test_nested_dict_handling(self):
        fmt = NeuroSymbolicTriples()
        data = {
            "id": "subj",
            "scores": {"sim": 0.9, "rec": 0.5},
        }

        encoded = fmt.encode(data)
        # Should have colon-separated predicates
        assert ":sim" in encoded or ":rec" in encoded

    def test_decode_basic(self):
        fmt = NeuroSymbolicTriples()
        triples = "subj|name|TestEntity\nsubj|score|0.85"

        decoded = fmt.decode(triples)
        assert decoded["name"] == "TestEntity"
        assert decoded["score"] == "0.85"

    def test_token_efficiency(self):
        fmt = NeuroSymbolicTriples()
        assert fmt.token_efficiency == 0.7


class TestGetRepresentation:
    """Tests for get_representation factory."""

    def test_full_format(self):
        fmt = get_representation("full")
        assert isinstance(fmt, FullJSON)

    def test_toon_format(self):
        fmt = get_representation("toon")
        assert isinstance(fmt, ToonJSON)

    def test_triples_format(self):
        fmt = get_representation("triples")
        assert isinstance(fmt, NeuroSymbolicTriples)

    def test_unknown_defaults_to_full(self):
        fmt = get_representation("unknown")
        assert isinstance(fmt, FullJSON)


class TestEstimateTokens:
    """Tests for estimate_tokens utility."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # 4 chars per token estimate
        assert estimate_tokens("test") == 1

    def test_longer_string(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25


class TestTokenReduction:
    """Integration tests for token reduction across formats."""

    def test_toon_reduces_tokens(self):
        full = FullJSON()
        toon = ToonJSON()

        data = {
            "retrieval_id": str(uuid4()),
            "query": "What is the capital of France?",
            "memory_type": "episodic",
            "retrieved_ids": [str(uuid4()) for _ in range(3)],
            "retrieval_scores": {"a": 0.9, "b": 0.8, "c": 0.7},
            "component_scores": {
                "a": {"similarity": 0.95, "recency": 0.3, "importance": 0.8}
            },
        }

        full_encoded = full.encode(data)
        toon_encoded = toon.encode(data)

        full_tokens = estimate_tokens(full_encoded)
        toon_tokens = estimate_tokens(toon_encoded)

        # ToonJSON should use fewer tokens
        assert toon_tokens < full_tokens
        # At least 20% reduction
        assert toon_tokens < full_tokens * 0.8
