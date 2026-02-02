"""Tests for MemoryItem unified schema."""

import pytest
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from t4dm.core.memory_item import MemoryItem
from t4dm.core.types import (
    Domain,
    Entity,
    EntityType,
    Episode,
    EpisodeContext,
    Outcome,
    Procedure,
    ProcedureStep,
)


class TestMemoryItem:
    def test_defaults(self):
        item = MemoryItem(content="hello")
        assert item.kappa == 0.0
        assert item.importance == 0.5
        assert item.item_type == "episodic"
        assert item.access_count == 0
        assert item.embedding == []
        assert item.valid_until is None

    def test_kappa_bounds(self):
        MemoryItem(content="ok", kappa=0.0)
        MemoryItem(content="ok", kappa=1.0)
        with pytest.raises(ValidationError):
            MemoryItem(content="ok", kappa=-0.1)
        with pytest.raises(ValidationError):
            MemoryItem(content="ok", kappa=1.1)

    def test_importance_bounds(self):
        with pytest.raises(ValidationError):
            MemoryItem(content="ok", importance=1.5)

    def test_content_required(self):
        with pytest.raises(ValidationError):
            MemoryItem(content="")

    def test_spike_trace_and_graph_delta_default_none(self):
        item = MemoryItem(content="hello")
        assert item.spike_trace is None
        assert item.graph_delta is None

    def test_spike_trace_and_graph_delta_round_trip(self):
        trace = {"neuron_ids": [1, 2, 3], "firing_rate": 0.7}
        delta = {"added_edges": [("a", "b")], "removed_edges": []}
        item = MemoryItem(content="x", spike_trace=trace, graph_delta=delta)
        dumped = item.model_dump()
        assert dumped["spike_trace"] == trace
        assert dumped["graph_delta"] == delta
        restored = MemoryItem.model_validate(dumped)
        assert restored.spike_trace == trace
        assert restored.graph_delta == delta


class TestFromEpisode:
    @pytest.fixture
    def episode(self):
        return Episode(
            session_id="test",
            content="test episode",
            embedding=[0.1] * 10,
            outcome=Outcome.SUCCESS,
            emotional_valence=0.8,
        )

    def test_round_trip(self, episode):
        item = MemoryItem.from_episode(episode)
        assert item.id == episode.id
        assert item.content == episode.content
        assert item.kappa == 0.0
        assert item.importance == 0.8
        assert item.item_type == "episodic"

        ep2 = item.to_episode()
        assert ep2.id == episode.id
        assert ep2.content == episode.content
        assert ep2.outcome == Outcome.SUCCESS

    def test_episode_no_embedding(self):
        ep = Episode(session_id="s", content="x")
        item = MemoryItem.from_episode(ep)
        assert item.embedding == []


class TestFromEntity:
    def test_entity_conversion(self):
        ent = Entity(
            name="Python",
            entity_type=EntityType.TOOL,
            summary="Programming language",
        )
        item = MemoryItem.from_entity(ent)
        assert item.kappa == 0.85
        assert item.item_type == "semantic"
        assert "Python" in item.content


class TestFromProcedure:
    def test_procedure_conversion(self):
        proc = Procedure(
            name="deploy",
            domain=Domain.DEVOPS,
            steps=[ProcedureStep(order=1, action="build")],
            success_rate=0.9,
        )
        item = MemoryItem.from_procedure(proc)
        assert item.kappa == 0.5
        assert item.item_type == "procedural"
        assert item.importance == 0.9
        assert item.metadata["success_rate"] == 0.9
