"""
Unit tests for World Weaver biological plasticity module.

Tests LTD, homeostatic scaling, metaplasticity, and synaptic tagging.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from t4dm.learning.plasticity import (
    PlasticityType,
    PlasticityEvent,
    SynapseState,
    LTDEngine,
    HomeostaticScaler,
    MetaplasticityController,
    SynapticTag,
    SynapticTagger,
    PlasticityManager,
)


class TestPlasticityEvent:
    """Tests for PlasticityEvent dataclass."""

    def test_creation(self):
        event = PlasticityEvent(
            event_type=PlasticityType.LTD,
            source_id="a",
            target_id="b",
            old_weight=0.8,
            new_weight=0.7
        )
        assert event.event_type == PlasticityType.LTD
        assert event.source_id == "a"
        assert event.target_id == "b"

    def test_delta_positive(self):
        event = PlasticityEvent(
            event_type=PlasticityType.LTP,
            source_id="a",
            target_id="b",
            old_weight=0.5,
            new_weight=0.7
        )
        assert abs(event.delta - 0.2) < 1e-10

    def test_delta_negative(self):
        event = PlasticityEvent(
            event_type=PlasticityType.LTD,
            source_id="a",
            target_id="b",
            old_weight=0.8,
            new_weight=0.6
        )
        assert abs(event.delta - (-0.2)) < 1e-10


class TestSynapseState:
    """Tests for SynapseState dataclass."""

    def test_creation(self):
        state = SynapseState(
            source_id="a",
            target_id="b",
            weight=0.5
        )
        assert state.weight == 0.5
        assert state.activation_count == 0
        assert not state.tagged_for_consolidation


class MockRelationshipStore:
    """Mock relationship store for testing."""

    def __init__(self, relationships: dict = None):
        self.relationships = relationships or {}
        self.updated_weights = {}

    async def get_relationships(self, node_id: str, direction: str = "out"):
        return self.relationships.get(node_id, [])

    async def update_relationship_weight(
        self, source_id: str, target_id: str, new_weight: float
    ):
        self.updated_weights[(source_id, target_id)] = new_weight


class TestLTDEngine:
    """Tests for LTDEngine."""

    @pytest.fixture
    def engine(self):
        return LTDEngine(ltd_rate=0.1, min_weight=0.01)

    @pytest.fixture
    def store_with_rels(self):
        store = MockRelationshipStore({
            "a": [
                {"other_id": "b", "properties": {"weight": 0.5}},
                {"other_id": "c", "properties": {"weight": 0.8}},
                {"other_id": "d", "properties": {"weight": 0.3}},
            ],
            "b": [
                {"other_id": "a", "properties": {"weight": 0.5}},
                {"other_id": "c", "properties": {"weight": 0.6}},
            ]
        })
        return store

    def test_creation_default(self):
        engine = LTDEngine()
        assert engine.ltd_rate == 0.05
        assert engine.min_weight == 0.01

    def test_creation_custom(self):
        engine = LTDEngine(ltd_rate=0.2, min_weight=0.05)
        assert engine.ltd_rate == 0.2
        assert engine.min_weight == 0.05

    @pytest.mark.asyncio
    async def test_apply_ltd_basic(self, engine, store_with_rels):
        # Activate a and b, so c and d should be weakened from a's perspective
        activated = {"a", "b"}

        events = await engine.apply_ltd(activated, store_with_rels)

        # a->c and a->d should be weakened (c and d not activated)
        # b->c should be weakened (c not activated)
        assert len(events) >= 2

        # Check weights were reduced
        for event in events:
            assert event.event_type == PlasticityType.LTD
            assert event.new_weight < event.old_weight

    @pytest.mark.asyncio
    async def test_apply_ltd_respects_min_weight(self, store_with_rels):
        engine = LTDEngine(ltd_rate=0.99, min_weight=0.1)

        # All neighbors should be clamped at min_weight
        events = await engine.apply_ltd({"a"}, store_with_rels)

        for event in events:
            assert event.new_weight >= 0.1

    @pytest.mark.asyncio
    async def test_apply_ltd_skips_activated(self, engine, store_with_rels):
        # If b is activated, a->b should NOT be weakened
        activated = {"a", "b"}

        events = await engine.apply_ltd(activated, store_with_rels)

        # Find events from a
        a_events = [e for e in events if e.source_id == "a"]

        # a->b should NOT be in events (b was activated)
        targets = [e.target_id for e in a_events]
        assert "b" not in targets

    @pytest.mark.asyncio
    async def test_apply_ltd_empty_activated(self, engine, store_with_rels):
        events = await engine.apply_ltd(set(), store_with_rels)
        assert events == []

    def test_get_history(self, engine):
        # Add some events manually
        event = PlasticityEvent(
            event_type=PlasticityType.LTD,
            source_id="x",
            target_id="y",
            old_weight=0.5,
            new_weight=0.4
        )
        engine._events.append(event)

        history = engine.get_history()
        assert len(history) == 1
        assert history[0].source_id == "x"


class TestHomeostaticScaler:
    """Tests for HomeostaticScaler."""

    @pytest.fixture
    def scaler(self):
        return HomeostaticScaler(
            target_total=1.0,
            tolerance=0.2,
            max_weight=1.0,
            min_weight=0.01
        )

    @pytest.fixture
    def store_high_weights(self):
        return MockRelationshipStore({
            "a": [
                {"other_id": "b", "properties": {"weight": 0.6}},
                {"other_id": "c", "properties": {"weight": 0.8}},
            ]  # Total = 1.4, above target of 1.0
        })

    @pytest.fixture
    def store_low_weights(self):
        return MockRelationshipStore({
            "a": [
                {"other_id": "b", "properties": {"weight": 0.2}},
                {"other_id": "c", "properties": {"weight": 0.3}},
            ]  # Total = 0.5, below target of 1.0
        })

    def test_creation_default(self):
        scaler = HomeostaticScaler()
        assert scaler.target_total == 10.0
        assert scaler.tolerance == 0.2

    @pytest.mark.asyncio
    async def test_scale_node_downward(self, scaler, store_high_weights):
        events = await scaler.scale_node("a", store_high_weights)

        assert len(events) == 2
        for event in events:
            assert event.event_type == PlasticityType.HOMEOSTATIC
            assert event.new_weight < event.old_weight

    @pytest.mark.asyncio
    async def test_scale_node_upward(self, scaler, store_low_weights):
        events = await scaler.scale_node("a", store_low_weights)

        assert len(events) == 2
        for event in events:
            assert event.new_weight > event.old_weight

    @pytest.mark.asyncio
    async def test_scale_node_within_tolerance(self, scaler):
        store = MockRelationshipStore({
            "a": [
                {"other_id": "b", "properties": {"weight": 0.5}},
                {"other_id": "c", "properties": {"weight": 0.55}},
            ]  # Total = 1.05, within 20% of 1.0
        })

        events = await scaler.scale_node("a", store)
        assert len(events) == 0  # No scaling needed

    @pytest.mark.asyncio
    async def test_scale_node_respects_max_weight(self, scaler, store_low_weights):
        events = await scaler.scale_node("a", store_low_weights)

        for event in events:
            assert event.new_weight <= scaler.max_weight

    @pytest.mark.asyncio
    async def test_scale_batch(self, scaler, store_high_weights):
        events = await scaler.scale_batch(["a"], store_high_weights)
        assert len(events) >= 1


class TestMetaplasticityController:
    """Tests for MetaplasticityController."""

    @pytest.fixture
    def controller(self):
        return MetaplasticityController(
            base_threshold=0.5,
            adaptation_rate=0.1,
            min_threshold=0.1,
            max_threshold=0.9
        )

    def test_creation(self):
        ctrl = MetaplasticityController()
        assert ctrl.base_threshold == 0.5
        assert ctrl.adaptation_rate == 0.1

    def test_get_threshold_default(self, controller):
        threshold = controller.get_threshold("unknown")
        assert threshold == 0.5  # base threshold

    def test_update_activity_increases_threshold(self, controller):
        # High activity should increase threshold
        controller.update_activity("a", 0.9)
        controller.update_activity("a", 0.9)
        controller.update_activity("a", 0.9)

        threshold = controller.get_threshold("a")
        assert threshold > 0.5

    def test_update_activity_decreases_threshold(self, controller):
        # First set high
        for _ in range(5):
            controller.update_activity("a", 0.9)

        high_threshold = controller.get_threshold("a")

        # Then set low - threshold should decrease
        for _ in range(10):
            controller.update_activity("a", 0.1)

        low_threshold = controller.get_threshold("a")
        assert low_threshold < high_threshold

    def test_should_potentiate(self, controller):
        # Signal above threshold
        assert controller.should_potentiate("x", 0.7) is True
        # Signal below threshold
        assert controller.should_potentiate("x", 0.3) is False

    def test_should_depress(self, controller):
        # Signal below half threshold (0.5 * 0.5 = 0.25)
        assert controller.should_depress("x", 0.2) is True
        # Signal above half threshold
        assert controller.should_depress("x", 0.4) is False

    def test_reset_specific(self, controller):
        controller.update_activity("a", 0.9)
        controller.update_activity("b", 0.9)

        controller.reset("a")

        # a should be reset
        assert controller.get_threshold("a") == 0.5
        # b should remain modified
        assert controller.get_threshold("b") > 0.5

    def test_reset_all(self, controller):
        controller.update_activity("a", 0.9)
        controller.update_activity("b", 0.9)

        controller.reset()

        assert controller.get_threshold("a") == 0.5
        assert controller.get_threshold("b") == 0.5


class TestSynapticTag:
    """Tests for SynapticTag dataclass."""

    def test_creation(self):
        tag = SynapticTag(
            source_id="a",
            target_id="b",
            tag_type="late",
            created_at=datetime.now(),
            strength=0.8
        )
        assert tag.tag_type == "late"
        assert not tag.captured


class TestSynapticTagger:
    """Tests for SynapticTagger."""

    @pytest.fixture
    def tagger(self):
        return SynapticTagger(
            early_threshold=0.3,
            late_threshold=0.7,
            tag_lifetime_hours=2.0
        )

    def test_creation(self):
        tagger = SynapticTagger()
        assert tagger.early_threshold == 0.3
        assert tagger.late_threshold == 0.7

    def test_tag_synapse_late(self, tagger):
        tag = tagger.tag_synapse("a", "b", 0.9)

        assert tag is not None
        assert tag.tag_type == "late"
        assert tag.strength == 0.9

    def test_tag_synapse_early(self, tagger):
        tag = tagger.tag_synapse("a", "b", 0.5)

        assert tag is not None
        assert tag.tag_type == "early"

    def test_tag_synapse_below_threshold(self, tagger):
        tag = tagger.tag_synapse("a", "b", 0.1)
        assert tag is None

    def test_tag_synapse_overwrites_weaker(self, tagger):
        tagger.tag_synapse("a", "b", 0.5)  # Early
        tagger.tag_synapse("a", "b", 0.9)  # Late (stronger)

        tags = tagger.get_tagged_synapses()
        assert len(tags) == 1
        assert tags[0].tag_type == "late"

    def test_get_tagged_synapses_filter_type(self, tagger):
        tagger.tag_synapse("a", "b", 0.9)  # Late
        tagger.tag_synapse("c", "d", 0.5)  # Early

        late_tags = tagger.get_tagged_synapses(tag_type="late")
        early_tags = tagger.get_tagged_synapses(tag_type="early")

        assert len(late_tags) == 1
        assert len(early_tags) == 1

    def test_capture_tags(self, tagger):
        tagger.tag_synapse("a", "b", 0.9)
        tagger.tag_synapse("c", "d", 0.5)

        captured = tagger.capture_tags()

        assert len(captured) == 2
        for tag in captured:
            assert tag.captured

    def test_capture_tags_idempotent(self, tagger):
        tagger.tag_synapse("a", "b", 0.9)

        captured1 = tagger.capture_tags()
        captured2 = tagger.capture_tags()

        assert len(captured1) == 1
        assert len(captured2) == 0  # Already captured

    def test_expired_tags_pruned(self, tagger):
        tag = tagger.tag_synapse("a", "b", 0.9)
        # Manually expire
        tag.created_at = datetime.now() - timedelta(hours=3)

        tags = tagger.get_tagged_synapses()
        assert len(tags) == 0  # Expired and pruned

    def test_clear(self, tagger):
        tagger.tag_synapse("a", "b", 0.9)
        tagger.clear()

        tags = tagger.get_tagged_synapses()
        assert len(tags) == 0


class TestPlasticityManager:
    """Tests for PlasticityManager."""

    @pytest.fixture
    def manager(self):
        return PlasticityManager()

    @pytest.fixture
    def store(self):
        return MockRelationshipStore({
            "a": [
                {"other_id": "b", "properties": {"weight": 0.5}},
                {"other_id": "c", "properties": {"weight": 0.5}},
            ],
            "b": [
                {"other_id": "a", "properties": {"weight": 0.5}},
            ]
        })

    def test_creation(self, manager):
        assert manager.ltd is not None
        assert manager.homeostatic is not None
        assert manager.metaplasticity is not None
        assert manager.tagger is not None

    @pytest.mark.asyncio
    async def test_on_retrieval(self, manager, store):
        result = await manager.on_retrieval({"a", "b"}, store)

        assert "ltd_events" in result
        assert "tags_created" in result
        assert result["tags_created"] >= 0

    @pytest.mark.asyncio
    async def test_on_consolidation(self, manager, store):
        # First create some tags
        manager.tagger.tag_synapse("a", "b", 0.9)

        result = await manager.on_consolidation(["a", "b"], store)

        assert "scaling_events" in result
        assert "tags_captured" in result

    def test_get_stats(self, manager):
        stats = manager.get_stats()

        assert "ltd_history_size" in stats
        assert "homeostatic_history_size" in stats
        assert "active_tags" in stats
        assert "captured_tags" in stats
