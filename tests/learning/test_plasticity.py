"""Tests for biological plasticity mechanisms."""

import pytest
import numpy as np
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


class TestPlasticityType:
    """Tests for PlasticityType enum."""

    def test_types_exist(self):
        """All plasticity types are defined."""
        assert PlasticityType.LTP.value == "ltp"
        assert PlasticityType.LTD.value == "ltd"
        assert PlasticityType.HOMEOSTATIC.value == "homeostatic"
        assert PlasticityType.METAPLASTIC.value == "metaplastic"


class TestPlasticityEvent:
    """Tests for PlasticityEvent dataclass."""

    def test_event_creation(self):
        """Create event with all fields."""
        event = PlasticityEvent(
            event_type=PlasticityType.LTD,
            source_id="src",
            target_id="tgt",
            old_weight=0.8,
            new_weight=0.6,
        )
        assert event.event_type == PlasticityType.LTD
        assert event.source_id == "src"
        assert event.target_id == "tgt"

    def test_event_has_timestamp(self):
        """Event has automatic timestamp."""
        event = PlasticityEvent(
            event_type=PlasticityType.LTP,
            source_id="a",
            target_id="b",
            old_weight=0.5,
            new_weight=0.7,
        )
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_delta_positive(self):
        """Delta is positive for potentiation."""
        event = PlasticityEvent(
            event_type=PlasticityType.LTP,
            source_id="a",
            target_id="b",
            old_weight=0.5,
            new_weight=0.7,
        )
        assert event.delta == pytest.approx(0.2)

    def test_delta_negative(self):
        """Delta is negative for depression."""
        event = PlasticityEvent(
            event_type=PlasticityType.LTD,
            source_id="a",
            target_id="b",
            old_weight=0.8,
            new_weight=0.6,
        )
        assert event.delta == pytest.approx(-0.2)


class TestSynapseState:
    """Tests for SynapseState dataclass."""

    def test_state_creation(self):
        """Create state with all fields."""
        state = SynapseState(
            source_id="src",
            target_id="tgt",
            weight=0.7,
        )
        assert state.source_id == "src"
        assert state.target_id == "tgt"
        assert state.weight == 0.7
        assert state.activation_count == 0
        assert state.tagged_for_consolidation is False


class TestLTDEngine:
    """Tests for LTDEngine class."""

    @pytest.fixture
    def ltd(self):
        """Create LTD engine."""
        return LTDEngine(
            ltd_rate=0.1,
            min_weight=0.01,
        )

    @pytest.fixture
    def mock_store(self):
        """Create mock relationship store."""
        store = MagicMock()
        store.get_relationships = AsyncMock(return_value=[])
        store.update_relationship_weight = AsyncMock()
        return store

    def test_initialization(self, ltd):
        """Test initialization."""
        assert ltd.ltd_rate == 0.1
        assert ltd.min_weight == 0.01

    def test_initialization_defaults(self):
        """Test default initialization."""
        ltd = LTDEngine()
        assert ltd.ltd_rate == 0.05
        assert ltd.min_weight == 0.01

    @pytest.mark.asyncio
    async def test_apply_ltd_empty(self, ltd, mock_store):
        """Apply LTD with no activated entities."""
        events = await ltd.apply_ltd(set(), mock_store)
        assert events == []

    @pytest.mark.asyncio
    async def test_apply_ltd_with_non_activated_neighbor(self, ltd, mock_store):
        """Apply LTD weakens non-co-activated neighbors."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "neighbor", "properties": {"weight": 0.8}}
        ])

        events = await ltd.apply_ltd({"entity1"}, mock_store)

        # Should have weakened the neighbor connection
        assert len(events) == 1
        assert events[0].event_type == PlasticityType.LTD
        assert events[0].old_weight == 0.8
        assert events[0].new_weight == pytest.approx(0.72)  # 0.8 * 0.9

    @pytest.mark.asyncio
    async def test_apply_ltd_skips_co_activated(self, ltd, mock_store):
        """Apply LTD skips co-activated entities."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "entity2", "properties": {"weight": 0.8}}
        ])

        # Both entity1 and entity2 are activated
        events = await ltd.apply_ltd({"entity1", "entity2"}, mock_store)

        # Should not have weakened (both activated)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_apply_ltd_respects_min_weight(self, ltd, mock_store):
        """Apply LTD respects minimum weight."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "neighbor", "properties": {"weight": 0.02}}
        ])

        events = await ltd.apply_ltd({"entity1"}, mock_store)

        # Weight should not go below min
        if events:
            assert events[0].new_weight >= ltd.min_weight

    @pytest.mark.asyncio
    async def test_apply_ltd_handles_error(self, ltd, mock_store):
        """Apply LTD handles errors gracefully."""
        mock_store.get_relationships = AsyncMock(side_effect=Exception("Test error"))

        events = await ltd.apply_ltd({"entity1"}, mock_store)
        assert events == []

    def test_get_history_empty(self, ltd):
        """Get history when empty."""
        history = ltd.get_history()
        assert history == []


class TestHomeostaticScaler:
    """Tests for HomeostaticScaler class."""

    @pytest.fixture
    def scaler(self):
        """Create homeostatic scaler."""
        return HomeostaticScaler(
            target_total=5.0,
            tolerance=0.2,
            max_weight=1.0,
            min_weight=0.01,
        )

    @pytest.fixture
    def mock_store(self):
        """Create mock relationship store."""
        store = MagicMock()
        store.get_relationships = AsyncMock(return_value=[])
        store.update_relationship_weight = AsyncMock()
        return store

    def test_initialization(self, scaler):
        """Test initialization."""
        assert scaler.target_total == 5.0
        assert scaler.tolerance == 0.2

    def test_initialization_defaults(self):
        """Test default initialization."""
        scaler = HomeostaticScaler()
        assert scaler.target_total == 10.0
        assert scaler.tolerance == 0.2

    @pytest.mark.asyncio
    async def test_scale_node_no_relationships(self, scaler, mock_store):
        """Scale node with no relationships."""
        events = await scaler.scale_node("entity1", mock_store)
        assert events == []

    @pytest.mark.asyncio
    async def test_scale_node_within_tolerance(self, scaler, mock_store):
        """Scale node within tolerance - no changes."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "a", "properties": {"weight": 2.5}},
            {"other_id": "b", "properties": {"weight": 2.5}},
        ])  # Total = 5.0, exactly at target

        events = await scaler.scale_node("entity1", mock_store)
        assert events == []

    @pytest.mark.asyncio
    async def test_scale_node_scales_down(self, scaler, mock_store):
        """Scale node scales down when above target."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "a", "properties": {"weight": 5.0}},
            {"other_id": "b", "properties": {"weight": 5.0}},
        ])  # Total = 10.0, double target

        events = await scaler.scale_node("entity1", mock_store)

        # Should have scaled down
        assert len(events) == 2
        assert events[0].new_weight < events[0].old_weight

    @pytest.mark.asyncio
    async def test_scale_node_scales_up(self, scaler, mock_store):
        """Scale node scales up when below target."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "a", "properties": {"weight": 0.5}},
            {"other_id": "b", "properties": {"weight": 0.5}},
        ])  # Total = 1.0, well below target of 5.0 (outside 20% tolerance)

        events = await scaler.scale_node("entity1", mock_store)

        # Should have scaled up since 1.0 < 5.0 * 0.8 = 4.0
        assert len(events) == 2
        assert events[0].new_weight > events[0].old_weight

    @pytest.mark.asyncio
    async def test_scale_batch(self, scaler, mock_store):
        """Scale multiple nodes."""
        mock_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "a", "properties": {"weight": 10.0}},
        ])

        events = await scaler.scale_batch(["e1", "e2", "e3"], mock_store)

        # Each node should have generated events
        assert len(events) >= 3


class TestMetaplasticityController:
    """Tests for MetaplasticityController class."""

    @pytest.fixture
    def controller(self):
        """Create metaplasticity controller."""
        return MetaplasticityController(
            base_threshold=0.5,
            adaptation_rate=0.2,
            min_threshold=0.1,
            max_threshold=0.9,
        )

    def test_initialization(self, controller):
        """Test initialization."""
        assert controller.base_threshold == 0.5
        assert controller.adaptation_rate == 0.2

    def test_initialization_defaults(self):
        """Test default initialization."""
        controller = MetaplasticityController()
        assert controller.base_threshold == 0.5
        assert controller.adaptation_rate == 0.1

    def test_get_threshold_default(self, controller):
        """Get threshold returns base for unknown entity."""
        threshold = controller.get_threshold("unknown")
        assert threshold == 0.5

    def test_update_activity_increases_threshold(self, controller):
        """High activity increases threshold."""
        initial = controller.get_threshold("entity1")

        # High activity
        new_threshold = controller.update_activity("entity1", 0.9)

        assert new_threshold > initial

    def test_update_activity_repeated(self, controller):
        """Repeated high activity keeps raising threshold."""
        controller.update_activity("entity1", 0.9)
        t1 = controller.get_threshold("entity1")

        controller.update_activity("entity1", 0.9)
        t2 = controller.get_threshold("entity1")

        assert t2 >= t1

    def test_threshold_bounded(self, controller):
        """Threshold stays within bounds."""
        # Many high activity updates
        for _ in range(100):
            controller.update_activity("entity1", 1.0)

        threshold = controller.get_threshold("entity1")
        assert controller.min_threshold <= threshold <= controller.max_threshold

    def test_should_potentiate(self, controller):
        """Should potentiate when signal above threshold."""
        assert controller.should_potentiate("entity1", 0.8) is True
        assert controller.should_potentiate("entity1", 0.3) is False

    def test_should_depress(self, controller):
        """Should depress when signal below half threshold."""
        # Default threshold is 0.5, so depression below 0.25
        assert controller.should_depress("entity1", 0.1) is True
        assert controller.should_depress("entity1", 0.4) is False

    def test_reset_single(self, controller):
        """Reset single entity."""
        controller.update_activity("entity1", 0.9)
        controller.reset("entity1")
        assert controller.get_threshold("entity1") == 0.5

    def test_reset_all(self, controller):
        """Reset all entities."""
        controller.update_activity("entity1", 0.9)
        controller.update_activity("entity2", 0.9)
        controller.reset()
        assert controller.get_threshold("entity1") == 0.5
        assert controller.get_threshold("entity2") == 0.5


class TestSynapticTag:
    """Tests for SynapticTag dataclass."""

    def test_tag_creation(self):
        """Create tag with all fields."""
        tag = SynapticTag(
            source_id="src",
            target_id="tgt",
            tag_type="late",
            created_at=datetime.now(),
            strength=0.8,
        )
        assert tag.source_id == "src"
        assert tag.target_id == "tgt"
        assert tag.tag_type == "late"
        assert tag.captured is False


class TestSynapticTagger:
    """Tests for SynapticTagger class."""

    @pytest.fixture
    def tagger(self):
        """Create synaptic tagger."""
        return SynapticTagger(
            early_threshold=0.3,
            late_threshold=0.7,
            tag_lifetime_hours=2.0,
        )

    def test_initialization(self, tagger):
        """Test initialization."""
        assert tagger.early_threshold == 0.3
        assert tagger.late_threshold == 0.7

    def test_initialization_defaults(self):
        """Test default initialization."""
        tagger = SynapticTagger()
        assert tagger.early_threshold == 0.3
        assert tagger.late_threshold == 0.7

    def test_tag_synapse_below_threshold(self, tagger):
        """No tag created below threshold."""
        tag = tagger.tag_synapse("src", "tgt", 0.1)
        assert tag is None

    def test_tag_synapse_early(self, tagger):
        """Early tag created for medium signal."""
        tag = tagger.tag_synapse("src", "tgt", 0.5)
        assert tag is not None
        assert tag.tag_type == "early"

    def test_tag_synapse_late(self, tagger):
        """Late tag created for strong signal."""
        tag = tagger.tag_synapse("src", "tgt", 0.8)
        assert tag is not None
        assert tag.tag_type == "late"

    def test_tag_synapse_overwrites_weaker(self, tagger):
        """Stronger tag overwrites weaker."""
        tagger.tag_synapse("src", "tgt", 0.4)  # Early
        tag = tagger.tag_synapse("src", "tgt", 0.9)  # Late
        assert tag.tag_type == "late"

    def test_get_tagged_synapses_empty(self, tagger):
        """Get tags when empty."""
        tags = tagger.get_tagged_synapses()
        assert tags == []

    def test_get_tagged_synapses_all(self, tagger):
        """Get all tags."""
        tagger.tag_synapse("a", "b", 0.5)
        tagger.tag_synapse("c", "d", 0.8)
        tags = tagger.get_tagged_synapses()
        assert len(tags) == 2

    def test_get_tagged_synapses_by_type(self, tagger):
        """Filter tags by type."""
        tagger.tag_synapse("a", "b", 0.5)  # Early
        tagger.tag_synapse("c", "d", 0.8)  # Late

        early = tagger.get_tagged_synapses(tag_type="early")
        late = tagger.get_tagged_synapses(tag_type="late")

        assert len(early) == 1
        assert len(late) == 1

    def test_capture_tags(self, tagger):
        """Capture all tags."""
        tagger.tag_synapse("a", "b", 0.5)
        tagger.tag_synapse("c", "d", 0.8)

        captured = tagger.capture_tags()

        assert len(captured) == 2
        for tag in tagger.get_tagged_synapses():
            assert tag.captured is True

    def test_capture_tags_only_uncaptured(self, tagger):
        """Capture only uncaptured tags."""
        tagger.tag_synapse("a", "b", 0.5)
        tagger.capture_tags()  # Capture first
        tagger.tag_synapse("c", "d", 0.8)  # New tag

        captured = tagger.capture_tags()
        assert len(captured) == 1
        assert captured[0].source_id == "c"

    def test_clear(self, tagger):
        """Clear all tags."""
        tagger.tag_synapse("a", "b", 0.5)
        tagger.clear()
        assert len(tagger.get_tagged_synapses()) == 0


class TestPlasticityManager:
    """Tests for PlasticityManager class."""

    @pytest.fixture
    def manager(self):
        """Create plasticity manager."""
        return PlasticityManager()

    @pytest.fixture
    def mock_store(self):
        """Create mock relationship store."""
        store = MagicMock()
        store.get_relationships = AsyncMock(return_value=[])
        store.update_relationship_weight = AsyncMock()
        return store

    def test_initialization(self, manager):
        """Test initialization."""
        assert manager.ltd is not None
        assert manager.homeostatic is not None
        assert manager.metaplasticity is not None
        assert manager.tagger is not None

    def test_initialization_custom(self):
        """Test custom initialization."""
        ltd = LTDEngine(ltd_rate=0.1)
        manager = PlasticityManager(ltd_engine=ltd)
        assert manager.ltd.ltd_rate == 0.1

    @pytest.mark.asyncio
    async def test_on_retrieval(self, manager, mock_store):
        """Process retrieval event."""
        result = await manager.on_retrieval({"e1", "e2"}, mock_store)

        assert "ltd_events" in result
        assert "tags_created" in result
        assert result["tags_created"] >= 0

    @pytest.mark.asyncio
    async def test_on_consolidation(self, manager, mock_store):
        """Process consolidation event."""
        # Add some tags first
        manager.tagger.tag_synapse("a", "b", 0.8)

        result = await manager.on_consolidation(["e1", "e2"], mock_store)

        assert "scaling_events" in result
        assert "tags_captured" in result
        assert result["tags_captured"] >= 1

    def test_get_stats(self, manager):
        """Get plasticity statistics."""
        stats = manager.get_stats()

        assert "ltd_history_size" in stats
        assert "homeostatic_history_size" in stats
        assert "active_tags" in stats
        assert "captured_tags" in stats


class TestPlasticityIntegration:
    """Integration tests for plasticity mechanisms."""

    @pytest.mark.asyncio
    async def test_full_plasticity_cycle(self):
        """Test full plasticity cycle: retrieval -> consolidation."""
        manager = PlasticityManager()

        # Mock store
        store = MagicMock()
        store.get_relationships = AsyncMock(return_value=[
            {"other_id": "neighbor", "properties": {"weight": 0.5}}
        ])
        store.update_relationship_weight = AsyncMock()

        # Retrieval activates entities
        activated = {"e1", "e2", "e3"}
        retrieval_result = await manager.on_retrieval(activated, store)

        # Check metaplasticity was updated
        for eid in activated:
            threshold = manager.metaplasticity.get_threshold(eid)
            assert threshold >= manager.metaplasticity.base_threshold

        # Consolidation captures tags
        all_entities = ["e1", "e2", "e3", "e4", "e5"]
        consolidation_result = await manager.on_consolidation(all_entities, store)

        stats = manager.get_stats()
        assert stats["captured_tags"] >= 0

    def test_metaplasticity_bcm_dynamics(self):
        """Test BCM-like dynamics in metaplasticity."""
        controller = MetaplasticityController(
            adaptation_rate=0.5  # Fast adaptation for test
        )

        # Simulate varying activity patterns
        # High activity should raise threshold
        for _ in range(10):
            controller.update_activity("high_activity", 0.9)

        # Low activity should lower threshold
        for _ in range(10):
            controller.update_activity("low_activity", 0.1)

        high_threshold = controller.get_threshold("high_activity")
        low_threshold = controller.get_threshold("low_activity")

        assert high_threshold > low_threshold

    def test_synaptic_tagging_consolidation(self):
        """Test tag-and-capture consolidation."""
        tagger = SynapticTagger()

        # Create tags of different strengths
        tagger.tag_synapse("strong1", "strong2", 0.9)  # Late LTP
        tagger.tag_synapse("weak1", "weak2", 0.4)  # Early LTP

        late_tags = tagger.get_tagged_synapses(tag_type="late")
        early_tags = tagger.get_tagged_synapses(tag_type="early")

        assert len(late_tags) == 1
        assert len(early_tags) == 1

        # Capture (simulate protein synthesis)
        captured = tagger.capture_tags()
        assert len(captured) == 2

        # Verify all captured
        for tag in tagger.get_tagged_synapses():
            assert tag.captured is True
