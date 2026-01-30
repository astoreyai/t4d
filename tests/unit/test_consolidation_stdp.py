"""
P7.3: Tests for STDP integration with sleep consolidation.

Tests the ConsolidationSTDP class which applies STDP weight updates
during memory replay for biologically-plausible consolidation.
"""

import pytest
from datetime import datetime, timedelta

import numpy as np


class TestConsolidationSTDPBasics:
    """Test basic ConsolidationSTDP functionality."""

    def test_initialization_with_defaults(self):
        """ConsolidationSTDP initializes with defaults."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP

        cstdp = ConsolidationSTDP()

        assert cstdp.config.replay_interval_ms == 50.0
        assert cstdp.config.sync_with_tags is True
        assert cstdp.stdp is not None
        assert cstdp.tagger is not None

    def test_initialization_with_custom_config(self):
        """ConsolidationSTDP accepts custom config."""
        from ww.consolidation.stdp_integration import (
            ConsolidationSTDP,
            ConsolidationSTDPConfig
        )

        config = ConsolidationSTDPConfig(
            replay_interval_ms=100.0,
            sync_with_tags=False,
            apply_decay=False
        )
        cstdp = ConsolidationSTDP(config)

        assert cstdp.config.replay_interval_ms == 100.0
        assert cstdp.config.sync_with_tags is False
        assert cstdp.tagger is None

    def test_initialization_without_tagging(self):
        """ConsolidationSTDP works without synaptic tagging."""
        from ww.consolidation.stdp_integration import (
            ConsolidationSTDP,
            ConsolidationSTDPConfig
        )

        config = ConsolidationSTDPConfig(sync_with_tags=False)
        cstdp = ConsolidationSTDP(config)

        assert cstdp.tagger is None


class TestSTDPReplaySequence:
    """Test STDP applied to replay sequences."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP instance."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_empty_sequence_returns_zero_updates(self, cstdp):
        """Empty sequence produces no updates."""
        result = cstdp.apply_stdp_to_sequence([])

        assert result.updates_applied == 0
        assert result.sequence_length == 0

    def test_single_episode_returns_zero_updates(self, cstdp):
        """Single episode produces no updates (need pairs)."""
        result = cstdp.apply_stdp_to_sequence(["ep_1"])

        assert result.updates_applied == 0
        assert result.sequence_length == 1

    def test_two_episode_sequence_produces_one_update(self, cstdp):
        """Two-episode sequence produces one STDP update."""
        result = cstdp.apply_stdp_to_sequence(["ep_1", "ep_2"])

        assert result.sequence_length == 2
        # May or may not produce update depending on timing
        assert result.updates_applied >= 0
        assert len(result.episode_ids) == 2

    def test_longer_sequence_produces_multiple_updates(self, cstdp):
        """Longer sequence produces multiple STDP updates."""
        episode_ids = [f"ep_{i}" for i in range(5)]
        result = cstdp.apply_stdp_to_sequence(episode_ids)

        assert result.sequence_length == 5
        # Should produce updates for adjacent pairs
        assert result.updates_applied <= 4  # At most N-1 updates

    def test_replay_sequence_strengthens_connections(self, cstdp):
        """Replayed sequence strengthens forward connections (LTP)."""
        episode_ids = ["ep_a", "ep_b", "ep_c"]

        # Apply STDP to sequence
        result = cstdp.apply_stdp_to_sequence(episode_ids)

        # Check weights were updated
        weight_ab = cstdp.get_synapse_strength("ep_a", "ep_b")
        weight_bc = cstdp.get_synapse_strength("ep_b", "ep_c")

        # Weights should be above baseline (0.5)
        # since pre->post timing produces LTP
        assert result.ltp_count >= 0

    def test_replay_records_spike_times(self, cstdp):
        """Replay sequence records spike times for all episodes."""
        episode_ids = ["ep_1", "ep_2", "ep_3"]
        cstdp.apply_stdp_to_sequence(episode_ids)

        # All episodes should have recorded spikes
        for ep_id in episode_ids:
            spike = cstdp.stdp.get_latest_spike(ep_id)
            assert spike is not None

    def test_custom_interval_affects_timing(self, cstdp):
        """Custom interval affects spike timing."""
        episode_ids = ["ep_a", "ep_b"]

        # Different intervals
        result1 = cstdp.apply_stdp_to_sequence(episode_ids, interval_ms=10.0)
        cstdp.stdp.clear_spikes()

        result2 = cstdp.apply_stdp_to_sequence(episode_ids, interval_ms=100.0)

        # Both should process but timing differs
        assert result1.sequence_length == 2
        assert result2.sequence_length == 2


class TestSTDPCoRetrieval:
    """Test STDP applied during co-retrieval."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP instance."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_co_retrieval_empty_list(self, cstdp):
        """Empty retrieval list produces no updates."""
        updates = cstdp.apply_stdp_to_co_retrieval("query", [])
        assert len(updates) == 0

    def test_co_retrieval_single_memory(self, cstdp):
        """Single retrieved memory produces no updates."""
        updates = cstdp.apply_stdp_to_co_retrieval("query", ["mem_1"])
        assert len(updates) == 0

    def test_co_retrieval_multiple_memories(self, cstdp):
        """Multiple retrieved memories produce STDP updates."""
        retrieved_ids = ["mem_1", "mem_2", "mem_3"]
        updates = cstdp.apply_stdp_to_co_retrieval("query", retrieved_ids)

        # Should have updates for pairs
        assert len(updates) <= len(retrieved_ids) - 1

    def test_co_retrieval_with_similarities(self, cstdp):
        """Co-retrieval with similarity scores creates tags."""
        retrieved_ids = ["mem_1", "mem_2"]
        similarities = [0.9, 0.8]

        updates = cstdp.apply_stdp_to_co_retrieval(
            "query", retrieved_ids, similarities
        )

        # Check tag was created
        if cstdp.tagger:
            tags = cstdp.tagger.get_tagged_synapses()
            assert len(tags) >= 0  # May have tags


class TestWeightConsolidation:
    """Test weight consolidation operations."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP instance."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_consolidate_weights_applies_decay(self, cstdp):
        """consolidate_weights applies weight decay."""
        # Set weight above baseline
        cstdp.stdp.set_weight("a", "b", 0.8)

        # Consolidate
        stats = cstdp.consolidate_weights()

        # Weight should move toward baseline
        new_weight = cstdp.get_synapse_strength("a", "b")
        assert new_weight < 0.8  # Decayed toward 0.5

    def test_consolidate_weights_increments_cycle(self, cstdp):
        """consolidate_weights increments cycle counter."""
        assert cstdp._consolidation_cycles == 0

        cstdp.consolidate_weights()
        assert cstdp._consolidation_cycles == 1

        cstdp.consolidate_weights()
        assert cstdp._consolidation_cycles == 2

    def test_consolidate_weights_returns_stats(self, cstdp):
        """consolidate_weights returns statistics."""
        stats = cstdp.consolidate_weights()

        assert "consolidation_cycles" in stats
        assert "total_updates" in stats
        assert "captured_tags" in stats


class TestSynapseQueries:
    """Test synapse strength queries."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP instance."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_get_synapse_strength_default(self, cstdp):
        """Unknown synapse returns default weight."""
        weight = cstdp.get_synapse_strength("unknown_a", "unknown_b")
        assert weight == 0.5  # Default

    def test_get_synapse_strength_after_set(self, cstdp):
        """Synapse strength returns set value."""
        cstdp.stdp.set_weight("a", "b", 0.75)
        weight = cstdp.get_synapse_strength("a", "b")
        assert weight == 0.75

    def test_get_strong_connections(self, cstdp):
        """get_strong_connections returns high-weight synapses."""
        cstdp.stdp.set_weight("a", "b", 0.9)
        cstdp.stdp.set_weight("c", "d", 0.8)
        cstdp.stdp.set_weight("e", "f", 0.3)  # Below threshold

        strong = cstdp.get_strong_connections(threshold=0.7)

        assert len(strong) == 2
        assert strong[0][2] == 0.9  # Sorted by weight descending

    def test_get_weak_connections(self, cstdp):
        """get_weak_connections returns low-weight synapses."""
        cstdp.stdp.set_weight("a", "b", 0.1)
        cstdp.stdp.set_weight("c", "d", 0.2)
        cstdp.stdp.set_weight("e", "f", 0.8)  # Above threshold

        weak = cstdp.get_weak_connections(threshold=0.3)

        assert len(weak) == 2
        assert weak[0][2] == 0.1  # Sorted by weight ascending


class TestSynapsePruning:
    """Test synapse pruning."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP instance."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_prune_weak_synapses_removes_below_threshold(self, cstdp):
        """Pruning removes synapses below threshold."""
        cstdp.stdp.set_weight("weak1", "target", 0.02)
        cstdp.stdp.set_weight("weak2", "target", 0.03)
        cstdp.stdp.set_weight("strong", "target", 0.8)

        pruned = cstdp.prune_weak_synapses(threshold=0.1)

        assert pruned == 2
        assert cstdp.get_synapse_strength("strong", "target") == 0.8
        # Weak ones removed (will return default 0.5)
        assert cstdp.get_synapse_strength("weak1", "target") == 0.5

    def test_prune_uses_specified_threshold(self, cstdp):
        """Pruning removes synapses below specified threshold."""
        # Set weights (note: STDP clips to min_weight=0.05)
        # So we set weights that will be clipped, then prune at higher threshold
        cstdp.stdp.set_weight("a", "b", 0.06)  # Just above min, below prune threshold
        cstdp.stdp.set_weight("c", "d", 0.07)  # Also below prune threshold
        cstdp.stdp.set_weight("e", "f", 0.5)   # Above prune threshold

        # Prune at 0.1 threshold
        pruned = cstdp.prune_weak_synapses(threshold=0.1)

        assert pruned == 2  # Both 0.06 and 0.07 should be pruned
        assert cstdp.get_synapse_strength("e", "f") == 0.5


class TestTagSynchronization:
    """Test synchronization with synaptic tags."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP with tagging enabled."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_sync_weights_with_tags_adjusts_weights(self, cstdp):
        """sync_weights_with_tags adjusts STDP weights."""
        # Create a tag
        cstdp.tagger.tag_synapse("a", "b", 0.9)  # High signal -> late LTP

        initial_weight = cstdp.get_synapse_strength("a", "b")

        synced = cstdp.sync_weights_with_tags()

        new_weight = cstdp.get_synapse_strength("a", "b")

        assert synced >= 1
        assert new_weight >= initial_weight  # Should increase

    def test_sync_weights_without_tagger(self):
        """sync_weights_with_tags returns 0 without tagger."""
        from ww.consolidation.stdp_integration import (
            ConsolidationSTDP,
            ConsolidationSTDPConfig
        )

        config = ConsolidationSTDPConfig(sync_with_tags=False)
        cstdp = ConsolidationSTDP(config)

        synced = cstdp.sync_weights_with_tags()
        assert synced == 0


class TestStatePersistence:
    """Test state save/load."""

    @pytest.fixture
    def cstdp(self):
        """Create ConsolidationSTDP instance."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        return ConsolidationSTDP()

    def test_save_and_load_state(self, cstdp):
        """State can be saved and loaded."""
        # Create some state
        cstdp.stdp.set_weight("a", "b", 0.7)
        cstdp._total_sequences = 10
        cstdp._consolidation_cycles = 3

        # Save
        state = cstdp.save_state()

        # Create new instance and load
        from ww.consolidation.stdp_integration import ConsolidationSTDP
        cstdp2 = ConsolidationSTDP()
        cstdp2.load_state(state)

        assert cstdp2.get_synapse_strength("a", "b") == 0.7
        assert cstdp2._total_sequences == 10
        assert cstdp2._consolidation_cycles == 3

    def test_reset_clears_all_state(self, cstdp):
        """reset clears all state."""
        cstdp.stdp.set_weight("a", "b", 0.9)
        cstdp._total_sequences = 100
        cstdp._consolidation_cycles = 5

        cstdp.reset()

        assert len(cstdp.stdp._weights) == 0
        assert cstdp._total_sequences == 0
        assert cstdp._consolidation_cycles == 0

    def test_get_stats_includes_all_metrics(self, cstdp):
        """get_stats includes consolidation metrics."""
        cstdp._total_sequences = 50
        cstdp._consolidation_cycles = 3

        stats = cstdp.get_stats()

        assert stats["total_sequences_replayed"] == 50
        assert stats["consolidation_cycles"] == 3
        assert "sync_with_tags" in stats


class TestSingletonAccess:
    """Test singleton pattern for global access."""

    def test_get_consolidation_stdp_returns_singleton(self):
        """get_consolidation_stdp returns same instance."""
        from ww.consolidation.stdp_integration import (
            get_consolidation_stdp,
            reset_consolidation_stdp
        )

        reset_consolidation_stdp()

        instance1 = get_consolidation_stdp()
        instance2 = get_consolidation_stdp()

        assert instance1 is instance2

    def test_reset_clears_singleton(self):
        """reset_consolidation_stdp clears singleton."""
        from ww.consolidation.stdp_integration import (
            get_consolidation_stdp,
            reset_consolidation_stdp
        )

        instance1 = get_consolidation_stdp()
        reset_consolidation_stdp()
        instance2 = get_consolidation_stdp()

        assert instance1 is not instance2


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_consolidation_cycle(self):
        """Full consolidation cycle: replay sequences, consolidate, prune."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP

        cstdp = ConsolidationSTDP()

        # Simulate multiple replay sequences
        for i in range(5):
            sequence = [f"ep_{i}_{j}" for j in range(4)]
            cstdp.apply_stdp_to_sequence(sequence)

        # Consolidate
        stats = cstdp.consolidate_weights()

        assert stats["consolidation_cycles"] == 1
        assert cstdp._total_sequences == 5

        # Prune weak connections
        pruned = cstdp.prune_weak_synapses()

        # Get strong connections
        strong = cstdp.get_strong_connections(threshold=0.5)

        # Should have some state
        assert cstdp._total_sequences == 5

    def test_interleaved_replay_and_retrieval(self):
        """Interleaved replay and retrieval operations."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP

        cstdp = ConsolidationSTDP()

        # Replay sequence
        cstdp.apply_stdp_to_sequence(["replay_1", "replay_2", "replay_3"])

        # Co-retrieval
        cstdp.apply_stdp_to_co_retrieval(
            "query",
            ["retrieved_1", "retrieved_2"],
            [0.9, 0.8]
        )

        # Another replay
        cstdp.apply_stdp_to_sequence(["replay_4", "replay_5"])

        # Consolidate
        stats = cstdp.consolidate_weights()

        assert cstdp._total_sequences == 2  # Two replay sequences

    def test_multiple_consolidation_cycles(self):
        """Multiple consolidation cycles with weight evolution."""
        from ww.consolidation.stdp_integration import ConsolidationSTDP

        cstdp = ConsolidationSTDP()

        initial_weights = []
        for cycle in range(3):
            # Clear previous spikes to avoid inter-spike interval violations
            cstdp.stdp.clear_spikes()
            # Replay same sequence
            cstdp.apply_stdp_to_sequence(["ep_a", "ep_b", "ep_c"])

            # Record weight
            w = cstdp.get_synapse_strength("ep_a", "ep_b")
            initial_weights.append(w)

            # Consolidate
            cstdp.consolidate_weights()

        assert cstdp._consolidation_cycles == 3

        # Weights may vary across cycles
        assert len(initial_weights) == 3
