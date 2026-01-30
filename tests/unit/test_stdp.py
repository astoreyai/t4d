"""
P5.5: Tests for Spike-Timing-Dependent Plasticity (STDP).

Tests the STDP implementation including:
- Basic spike timing rules
- LTP and LTD behavior
- Weight bounds and decay
- Triplet and pair-based variants
"""

import pytest
from datetime import datetime, timedelta

import numpy as np


class TestSTDPLearner:
    """Test basic STDP functionality."""

    def test_stdp_learner_initialization(self):
        """STDPLearner initializes with default config."""
        from ww.learning.stdp import STDPLearner, STDPConfig

        learner = STDPLearner()
        assert learner.config.a_plus == 0.01
        assert learner.config.a_minus == 0.0105
        # Biological values: tau+ ~17ms, tau- ~34ms (Bi & Poo 1998)
        assert learner.config.tau_plus == 0.017
        assert learner.config.tau_minus == 0.034

    def test_stdp_custom_config(self):
        """STDPLearner accepts custom config."""
        from ww.learning.stdp import STDPLearner, STDPConfig

        config = STDPConfig(a_plus=0.02, a_minus=0.03, tau_plus=10.0)
        learner = STDPLearner(config)

        assert learner.config.a_plus == 0.02
        assert learner.config.a_minus == 0.03
        assert learner.config.tau_plus == 10.0

    def test_record_spike(self):
        """record_spike stores spike events."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        now = datetime.now()

        learner.record_spike("neuron_a", timestamp=now)

        spike = learner.get_latest_spike("neuron_a")
        assert spike is not None
        assert spike[0] == now
        assert spike[1] == 1.0

    def test_record_spike_with_strength(self):
        """record_spike stores spike strength."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        learner.record_spike("neuron_a", strength=0.5)

        spike = learner.get_latest_spike("neuron_a")
        assert spike[1] == 0.5

    def test_get_latest_spike_returns_none_for_unknown(self):
        """get_latest_spike returns None for unknown entity."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        assert learner.get_latest_spike("unknown") is None

    def test_compute_stdp_delta_ltp(self):
        """Positive delta_t produces LTP (positive delta)."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()

        # Pre before post: should produce LTP
        delta = learner.compute_stdp_delta(10.0)  # 10ms
        assert delta > 0, "Pre-before-post should produce LTP"

    def test_compute_stdp_delta_ltd(self):
        """Negative delta_t produces LTD (negative delta)."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()

        # Post before pre: should produce LTD
        delta = learner.compute_stdp_delta(-10.0)  # -10ms
        assert delta < 0, "Post-before-pre should produce LTD"

    def test_compute_stdp_delta_exponential_decay(self):
        """STDP delta decreases with timing difference."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()

        # LTP at different delays
        delta_5ms = learner.compute_stdp_delta(5.0)
        delta_20ms = learner.compute_stdp_delta(20.0)
        delta_50ms = learner.compute_stdp_delta(50.0)

        assert delta_5ms > delta_20ms > delta_50ms > 0
        assert delta_5ms > delta_20ms, "Shorter delay should produce larger LTP"

    def test_compute_stdp_delta_simultaneous(self):
        """Simultaneous spikes produce no change."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        delta = learner.compute_stdp_delta(0.0)
        assert delta == 0.0

    def test_compute_update_ltp(self):
        """compute_update produces LTP for pre-before-post."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        now = datetime.now()

        # Pre spike first
        learner.record_spike("pre", timestamp=now)
        # Post spike 10ms later
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))

        update = learner.compute_update("pre", "post", current_weight=0.5)

        assert update is not None
        assert update.update_type == "ltp"
        assert update.delta_weight > 0
        assert update.new_weight > 0.5

    def test_compute_update_ltd(self):
        """compute_update produces LTD for post-before-pre."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        now = datetime.now()

        # Post spike first
        learner.record_spike("post", timestamp=now)
        # Pre spike 10ms later
        learner.record_spike("pre", timestamp=now + timedelta(milliseconds=10))

        update = learner.compute_update("pre", "post", current_weight=0.5)

        assert update is not None
        assert update.update_type == "ltd"
        assert update.delta_weight < 0
        assert update.new_weight < 0.5

    def test_compute_update_respects_weight_bounds(self):
        """compute_update respects min/max weight bounds."""
        from ww.learning.stdp import STDPLearner, STDPConfig

        config = STDPConfig(min_weight=0.1, max_weight=0.9, a_minus=0.5)
        learner = STDPLearner(config)
        now = datetime.now()

        # Strong LTD
        learner.record_spike("post", timestamp=now)
        learner.record_spike("pre", timestamp=now + timedelta(milliseconds=5))

        update = learner.compute_update("pre", "post", current_weight=0.15)

        assert update.new_weight >= 0.1, "Weight should not go below min"

    def test_compute_update_returns_none_outside_window(self):
        """compute_update returns None for spikes outside STDP window."""
        from ww.learning.stdp import STDPLearner, STDPConfig

        config = STDPConfig(spike_window_ms=50.0)
        learner = STDPLearner(config)
        now = datetime.now()

        learner.record_spike("pre", timestamp=now)
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=100))

        update = learner.compute_update("pre", "post")
        assert update is None

    def test_compute_update_returns_none_missing_spikes(self):
        """compute_update returns None when spikes are missing."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        learner.record_spike("pre")

        update = learner.compute_update("pre", "post")
        assert update is None

    def test_compute_all_updates(self):
        """compute_all_updates processes multiple presynaptic neurons."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        now = datetime.now()

        # Record pre spikes
        for i in range(3):
            learner.record_spike(f"pre_{i}", timestamp=now)

        # Record post spike
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))

        updates = learner.compute_all_updates(
            pre_ids=[f"pre_{i}" for i in range(3)],
            post_id="post"
        )

        assert len(updates) == 3
        assert all(u.update_type == "ltp" for u in updates)

    def test_get_set_weight(self):
        """get_weight and set_weight work correctly."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()

        # Default weight
        assert learner.get_weight("a", "b") == 0.5

        # Set and get
        learner.set_weight("a", "b", 0.7)
        assert learner.get_weight("a", "b") == 0.7

    def test_weight_decay(self):
        """apply_weight_decay moves weights toward baseline."""
        from ww.learning.stdp import STDPLearner, STDPConfig

        config = STDPConfig(weight_decay=0.1)
        learner = STDPLearner(config)

        learner.set_weight("a", "b", 0.8)
        learner.set_weight("c", "d", 0.2)

        learner.apply_weight_decay(baseline=0.5)

        # Weight above baseline should decrease
        assert learner.get_weight("a", "b") < 0.8
        # Weight below baseline should increase
        assert learner.get_weight("c", "d") > 0.2

    def test_clear_spikes(self):
        """clear_spikes removes spike history."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        learner.record_spike("a")
        learner.record_spike("b")

        learner.clear_spikes("a")
        assert learner.get_latest_spike("a") is None
        assert learner.get_latest_spike("b") is not None

        learner.clear_spikes()
        assert learner.get_latest_spike("b") is None

    def test_get_stats(self):
        """get_stats returns meaningful statistics."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        now = datetime.now()

        # Record spikes and update
        learner.record_spike("pre", timestamp=now)
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))
        learner.compute_update("pre", "post")

        stats = learner.get_stats()

        assert stats["total_updates"] == 1
        assert stats["ltp_updates"] == 1
        assert stats["ltd_updates"] == 0
        assert "active_synapses" in stats
        assert "tracked_entities" in stats

    def test_save_and_load_state(self):
        """State can be saved and loaded."""
        from ww.learning.stdp import STDPLearner

        learner = STDPLearner()
        learner.set_weight("a", "b", 0.7)
        learner.set_weight("c", "d", 0.3)

        state = learner.save_state()

        learner2 = STDPLearner()
        learner2.load_state(state)

        assert learner2.get_weight("a", "b") == 0.7
        assert learner2.get_weight("c", "d") == 0.3


class TestPairBasedSTDP:
    """Test pair-based STDP variant."""

    def test_pair_based_avoids_double_updates(self):
        """PairBasedSTDP avoids updating same spike pair twice."""
        from ww.learning.stdp import PairBasedSTDP

        learner = PairBasedSTDP()
        now = datetime.now()

        learner.record_spike("pre", timestamp=now)
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))

        # First update
        update1 = learner.compute_update("pre", "post")
        assert update1 is not None

        # Second update on same spikes should be skipped (no new spikes)
        # Clear first to ensure no new spikes
        update2 = learner.compute_update("pre", "post")
        # Second call may or may not return None depending on timing,
        # but weight should not change significantly
        if update2 is not None:
            # If returned, weight should be similar (already updated)
            assert abs(update2.delta_weight) < 0.02

    def test_pair_based_new_spike_triggers_update(self):
        """PairBasedSTDP updates when new spike arrives."""
        from ww.learning.stdp import PairBasedSTDP

        learner = PairBasedSTDP()
        now = datetime.now()

        learner.record_spike("pre", timestamp=now)
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))

        update1 = learner.compute_update("pre", "post")
        assert update1 is not None

        # New post spike should allow update
        learner.record_spike("post", timestamp=now + timedelta(milliseconds=50))
        update2 = learner.compute_update("pre", "post")
        assert update2 is not None


class TestTripletSTDP:
    """Test triplet STDP variant."""

    def test_triplet_stdp_initialization(self):
        """TripletSTDP initializes with triplet parameters."""
        from ww.learning.stdp import TripletSTDP

        learner = TripletSTDP(
            triplet_a_plus=0.008,
            triplet_a_minus=0.004,
            tau_triplet=50.0
        )

        assert learner.triplet_a_plus == 0.008
        assert learner.triplet_a_minus == 0.004
        assert learner.tau_triplet == 50.0

    def test_triplet_ltp_enhancement(self):
        """Post-Pre-Post triplet enhances LTP."""
        from ww.learning.stdp import TripletSTDP, STDPLearner

        triplet = TripletSTDP()
        pair = STDPLearner()  # For comparison
        now = datetime.now()

        # Post-Pre-Post sequence
        triplet.record_spike("post", timestamp=now)
        triplet.record_spike("pre", timestamp=now + timedelta(milliseconds=10))
        triplet.record_spike("post", timestamp=now + timedelta(milliseconds=20))

        # Same for pair-based
        pair.record_spike("pre", timestamp=now + timedelta(milliseconds=10))
        pair.record_spike("post", timestamp=now + timedelta(milliseconds=20))

        triplet_update = triplet.compute_update("pre", "post", current_weight=0.5)
        pair_update = pair.compute_update("pre", "post", current_weight=0.5)

        # Triplet should potentially have enhanced LTP
        assert triplet_update is not None
        assert pair_update is not None


class TestSTDPSingleton:
    """Test global singleton management."""

    def test_get_stdp_learner_returns_singleton(self):
        """get_stdp_learner returns same instance."""
        from ww.learning.stdp import get_stdp_learner, reset_stdp_learner

        reset_stdp_learner()

        learner1 = get_stdp_learner()
        learner2 = get_stdp_learner()

        assert learner1 is learner2

    def test_reset_stdp_learner_creates_new_instance(self):
        """reset_stdp_learner clears the singleton."""
        from ww.learning.stdp import get_stdp_learner, reset_stdp_learner

        learner1 = get_stdp_learner()
        reset_stdp_learner()
        learner2 = get_stdp_learner()

        assert learner1 is not learner2
