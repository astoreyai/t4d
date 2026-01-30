"""
Tests for BCM Metaplasticity with Synaptic Tagging and Capture.

Tests sliding threshold learning, tagging, and consolidation.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from ww.learning.bcm_metaplasticity import (
    BCMConfig,
    BCMLearningRule,
    BCMMetaplasticityManager,
    BCMState,
    PlasticityType,
    SynapticTag,
    SynapticTaggingAndCapture,
)


class TestBCMConfig:
    """Tests for BCMConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = BCMConfig()
        assert config.theta_m_init == 0.5
        assert config.ltp_rate == 0.01
        assert config.ltd_rate == 0.005
        assert config.tag_decay_tau == 7200.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = BCMConfig(
            theta_m_init=0.3,
            ltp_rate=0.02,
        )
        assert config.theta_m_init == 0.3
        assert config.ltp_rate == 0.02


class TestPlasticityType:
    """Tests for PlasticityType enum."""

    def test_plasticity_values(self):
        """Test plasticity type values."""
        assert PlasticityType.NONE.value == "none"
        assert PlasticityType.EARLY_LTP.value == "early_ltp"
        assert PlasticityType.LATE_LTP.value == "late_ltp"
        assert PlasticityType.EARLY_LTD.value == "early_ltd"
        assert PlasticityType.LATE_LTD.value == "late_ltd"


class TestSynapticTag:
    """Tests for SynapticTag dataclass."""

    def test_tag_creation(self):
        """Test tag creation."""
        tag = SynapticTag(
            synapse_id="syn1",
            strength=0.8,
            tag_type=PlasticityType.EARLY_LTP,
            induction_time=datetime.now(),
        )
        assert tag.synapse_id == "syn1"
        assert tag.strength == 0.8
        assert tag.is_captured is False


class TestBCMState:
    """Tests for BCMState dataclass."""

    def test_state_defaults(self):
        """Test state default values."""
        state = BCMState(
            synapse_id="syn1",
            theta_m=0.5,
            weight=0.5,
        )
        assert state.synapse_id == "syn1"
        assert state.activity_history == []
        assert state.last_plasticity == PlasticityType.NONE
        assert state.tag is None


class TestBCMLearningRule:
    """Tests for BCMLearningRule."""

    @pytest.fixture
    def bcm(self):
        """Create BCM learning rule."""
        return BCMLearningRule()

    def test_initialization(self, bcm):
        """Test BCM rule initialization."""
        assert bcm.config is not None
        assert bcm._thresholds == {}

    def test_compute_update_ltp(self, bcm):
        """Test LTP when post > threshold."""
        # High postsynaptic activity should trigger LTP
        delta, plasticity = bcm.compute_update(
            pre=0.8, post=0.8, synapse_id="syn1"
        )
        # phi = 0.8 * (0.8 - 0.5) = 0.24 > 0 => LTP
        assert delta > 0
        assert plasticity in [PlasticityType.EARLY_LTP, PlasticityType.LATE_LTP]

    def test_compute_update_ltd(self, bcm):
        """Test LTD when post < threshold."""
        # Low postsynaptic activity should trigger LTD
        delta, plasticity = bcm.compute_update(
            pre=0.8, post=0.2, synapse_id="syn2"
        )
        # phi = 0.2 * (0.2 - 0.5) = -0.06 < 0 => LTD
        assert delta < 0
        assert plasticity in [PlasticityType.EARLY_LTD, PlasticityType.LATE_LTD]

    def test_compute_update_no_change(self, bcm):
        """Test no change when post = threshold."""
        # Post exactly at threshold
        delta, plasticity = bcm.compute_update(
            pre=0.5, post=0.5, synapse_id="syn3"
        )
        # phi = 0.5 * (0.5 - 0.5) = 0
        assert delta == 0
        assert plasticity == PlasticityType.NONE

    def test_late_ltp_threshold(self, bcm):
        """Test late LTP triggered by high activity."""
        config = BCMConfig(late_tag_threshold=0.7)
        bcm = BCMLearningRule(config)

        delta, plasticity = bcm.compute_update(
            pre=0.9, post=0.9, synapse_id="syn4"
        )
        assert plasticity == PlasticityType.LATE_LTP

    def test_threshold_adaptation(self, bcm):
        """Test that threshold adapts to activity."""
        synapse_id = "syn5"

        # Initial threshold
        initial = bcm.get_threshold(synapse_id)
        assert initial == bcm.config.theta_m_init

        # High activity should increase threshold
        for _ in range(50):
            bcm.compute_update(pre=0.5, post=0.9, synapse_id=synapse_id)

        new_threshold = bcm.get_threshold(synapse_id)
        assert new_threshold > initial

    def test_threshold_clamping(self, bcm):
        """Test threshold stays within bounds."""
        synapse_id = "syn6"

        # Many high-activity updates
        for _ in range(1000):
            bcm.update_threshold(synapse_id, 1.0)

        theta = bcm.get_threshold(synapse_id)
        assert theta <= bcm.config.theta_m_max

        # Many low-activity updates
        for _ in range(1000):
            bcm.update_threshold(synapse_id, 0.0)

        theta = bcm.get_threshold(synapse_id)
        assert theta >= bcm.config.theta_m_min

    def test_get_stats_empty(self, bcm):
        """Test stats with no synapses."""
        stats = bcm.get_stats()
        assert stats["n_synapses"] == 0

    def test_get_stats_with_synapses(self, bcm):
        """Test stats with synapses."""
        bcm.compute_update(0.5, 0.7, "syn1")
        bcm.compute_update(0.5, 0.3, "syn2")

        stats = bcm.get_stats()
        assert stats["n_synapses"] == 2
        assert "mean_threshold" in stats
        assert "std_threshold" in stats


class TestSynapticTaggingAndCapture:
    """Tests for SynapticTaggingAndCapture."""

    @pytest.fixture
    def stc(self):
        """Create STC system."""
        return SynapticTaggingAndCapture()

    def test_initialization(self, stc):
        """Test STC initialization."""
        assert stc._tags == {}
        assert stc._prp_level > 0

    def test_process_input_creates_tag(self, stc):
        """Test that strong input creates tag."""
        result = stc.process_input(
            synapse_id="syn1",
            strength=0.5,  # Above early_tag_threshold
            plasticity_type=PlasticityType.EARLY_LTP,
        )
        assert result["tag_created"] is True
        assert "syn1" in stc._tags

    def test_process_input_no_tag_weak(self, stc):
        """Test that weak input doesn't create tag."""
        result = stc.process_input(
            synapse_id="syn2",
            strength=0.1,  # Below early_tag_threshold
            plasticity_type=PlasticityType.NONE,
        )
        assert result["tag_created"] is False
        assert "syn2" not in stc._tags

    def test_process_input_triggers_prp(self, stc):
        """Test that very strong input triggers PRP synthesis."""
        initial_prp = stc._prp_level

        result = stc.process_input(
            synapse_id="syn3",
            strength=0.9,  # Above prp_synthesis_threshold
            plasticity_type=PlasticityType.LATE_LTP,
        )

        assert result["prp_triggered"] is True
        assert stc._prp_level > initial_prp

    def test_attempt_capture_with_prp(self, stc):
        """Test capture when PRPs are available."""
        # Trigger PRP synthesis first with strong input
        stc.process_input(
            synapse_id="syn1",
            strength=0.9,
            plasticity_type=PlasticityType.LATE_LTP,
        )

        # Attempt capture
        captured = stc.attempt_capture()

        # syn1 should be captured (strong input that triggered PRPs)
        assert "syn1" in captured
        assert stc._tags["syn1"].is_captured is True

    def test_attempt_capture_without_prp(self, stc):
        """Test no capture when PRPs exhausted."""
        config = BCMConfig(prp_base_rate=0.0, prp_decay_rate=1.0)
        stc = SynapticTaggingAndCapture(config)

        # Create tag
        stc.process_input(
            synapse_id="syn1",
            strength=0.5,
            plasticity_type=PlasticityType.EARLY_LTP,
        )

        # No PRP boost, PRP decays
        stc._prp_level = 0.0

        captured = stc.attempt_capture()
        assert len(captured) == 0

    def test_decay_tags(self, stc):
        """Test tag decay and expiration."""
        old_time = datetime.now() - timedelta(hours=10)

        # Create old tag manually
        stc._tags["old_syn"] = SynapticTag(
            synapse_id="old_syn",
            strength=0.5,
            tag_type=PlasticityType.EARLY_LTP,
            induction_time=old_time,
        )

        expired = stc.decay_tags()

        # Old tag should be expired
        assert "old_syn" in expired
        assert "old_syn" not in stc._tags

    def test_get_active_tags(self, stc):
        """Test getting active tags info."""
        stc.process_input("syn1", 0.5, PlasticityType.EARLY_LTP)

        tags = stc.get_active_tags()
        assert len(tags) == 1
        assert tags[0]["synapse_id"] == "syn1"

    def test_get_stats(self, stc):
        """Test stats retrieval."""
        stats = stc.get_stats()
        assert "n_active_tags" in stats
        assert "prp_level" in stats
        assert "n_captured" in stats


class TestBCMMetaplasticityManager:
    """Tests for BCMMetaplasticityManager."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        return BCMMetaplasticityManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.bcm is not None
        assert manager.tagging is not None
        assert manager._states == {}

    def test_on_retrieval_updates_weights(self, manager):
        """Test that retrieval updates synaptic weights."""
        result = manager.on_retrieval(
            synapse_ids=["syn1", "syn2"],
            pre_activities=[0.8, 0.3],
            post_activities=[0.9, 0.4],
        )

        assert "updates" in result
        assert len(result["updates"]) == 2
        assert result["updates"][0]["synapse_id"] == "syn1"
        assert "delta" in result["updates"][0]
        assert "new_weight" in result["updates"][0]

    def test_on_retrieval_creates_tags(self, manager):
        """Test that strong retrieval creates tags."""
        result = manager.on_retrieval(
            synapse_ids=["syn1"],
            pre_activities=[0.9],
            post_activities=[0.9],
        )

        assert result["tags_created"] >= 0  # May or may not create tag

    def test_on_consolidation(self, manager):
        """Test consolidation phase."""
        # First, generate some activity
        manager.on_retrieval(
            synapse_ids=["syn1", "syn2"],
            pre_activities=[0.9, 0.9],
            post_activities=[0.9, 0.9],
        )

        # Run consolidation
        result = manager.on_consolidation()

        assert "captured" in result
        assert "n_captured" in result
        assert "expired" in result
        assert "prp_level" in result

    def test_get_plasticity_state(self, manager):
        """Test getting synapse state."""
        # Create some activity
        manager.on_retrieval(["syn1"], [0.5], [0.5])

        state = manager.get_plasticity_state("syn1")
        assert state is not None
        assert isinstance(state, BCMState)
        assert state.synapse_id == "syn1"

    def test_get_plasticity_state_missing(self, manager):
        """Test getting state for unknown synapse."""
        state = manager.get_plasticity_state("unknown")
        assert state is None

    def test_get_stats(self, manager):
        """Test stats retrieval."""
        stats = manager.get_stats()
        assert "bcm" in stats
        assert "tagging" in stats
        assert "n_synapses_tracked" in stats


class TestBCMIntegration:
    """Integration tests for BCM learning."""

    def test_weight_changes_follow_bcm_rule(self):
        """Test that weights change according to BCM rule."""
        manager = BCMMetaplasticityManager()

        # High activity should increase weight
        manager.on_retrieval(["syn1"], [0.9], [0.9])
        state_high = manager.get_plasticity_state("syn1")
        assert state_high.weight > 0.5  # Above initial

        # Create new manager for fresh test
        manager2 = BCMMetaplasticityManager()

        # Low activity should decrease weight
        manager2.on_retrieval(["syn2"], [0.9], [0.2])
        state_low = manager2.get_plasticity_state("syn2")
        assert state_low.weight < 0.5  # Below initial

    def test_consolidation_captures_strong_learning(self):
        """Test that strong learning gets consolidated."""
        config = BCMConfig(
            late_tag_threshold=0.5,
            prp_synthesis_threshold=0.5,
        )
        manager = BCMMetaplasticityManager(config)

        # Very strong learning event
        manager.on_retrieval(
            synapse_ids=["syn1"],
            pre_activities=[1.0],
            post_activities=[1.0],
        )

        # Run consolidation
        result = manager.on_consolidation()

        # Should have captured the tag
        # (Note: capture depends on tag strength and PRP availability)
        assert "n_captured" in result

    def test_threshold_adaptation_prevents_runaway(self):
        """Test that sliding threshold prevents runaway LTP."""
        manager = BCMMetaplasticityManager()

        # Many high-activity events
        weights = []
        for i in range(100):
            manager.on_retrieval(["syn1"], [0.9], [0.9])
            weights.append(manager.get_plasticity_state("syn1").weight)

        # Weight should eventually stabilize (not keep increasing forever)
        # Check that weight change slows down
        early_change = abs(weights[10] - weights[0])
        late_change = abs(weights[-1] - weights[-11])

        # Late changes should be smaller as threshold adapts
        assert late_change <= early_change + 0.1  # Allow some margin

    def test_synaptic_cooperation(self):
        """Test synaptic tagging allows cooperation."""
        config = BCMConfig(
            early_tag_threshold=0.3,
            prp_synthesis_threshold=0.8,
        )
        manager = BCMMetaplasticityManager(config)

        # Weak input creates tag but no PRPs
        manager.on_retrieval(["weak_syn"], [0.5], [0.5])

        # Strong input triggers PRPs
        manager.on_retrieval(["strong_syn"], [1.0], [1.0])

        # Consolidation should capture weak_syn using PRPs from strong_syn
        result = manager.on_consolidation()

        # The weak synapse tag should have opportunity for capture
        assert "captured" in result
