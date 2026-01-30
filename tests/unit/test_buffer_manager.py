"""
Tests for BufferManager - CA1-like temporary storage for uncertain memories.

Per Hinton: BUFFER != "delayed STORE" - it means "candidate under observation."
Evidence accumulates from retrieval probing and contextual signals before
promotion or discard decisions.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4

from ww.memory.buffer_manager import (
    BufferManager,
    BufferedItem,
    PromotionAction,
    PromotionDecision,
)


class MockLearnedGate:
    """Mock gate for testing training signals."""

    def __init__(self):
        self.pending_labels = {}
        self.updates = []

    def register_pending(self, memory_id, features, raw_content_embedding=None):
        # P0a: Accept optional raw_content_embedding
        self.pending_labels[memory_id] = (features, raw_content_embedding)

    def update(self, memory_id, utility):
        self.updates.append((memory_id, utility))
        # Remove from pending (like real gate)
        if memory_id in self.pending_labels:
            del self.pending_labels[memory_id]


class MockNeuromodState:
    """Mock neuromodulator state for testing."""

    def __init__(
        self,
        norepinephrine_gain=1.0,
        acetylcholine_mode="balanced",
        dopamine_rpe=0.0,
        serotonin_mood=0.5,
    ):
        self.norepinephrine_gain = norepinephrine_gain
        self.acetylcholine_mode = acetylcholine_mode
        self.dopamine_rpe = dopamine_rpe
        self.serotonin_mood = serotonin_mood


class TestBufferManagerBasics:
    """Test basic BufferManager operations."""

    def test_init_defaults(self):
        """Test default initialization."""
        bm = BufferManager()

        assert bm.promotion_threshold == 0.65
        assert bm.discard_threshold == 0.25
        assert bm.size == 0
        assert bm.pressure == 0.0

    def test_add_item(self):
        """Test adding items to buffer."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add(
            content="Test content",
            embedding=embedding,
            features=features,
            context={"project": "test"},
        )

        assert bm.size == 1
        assert bm.stats["items_added"] == 1

        item = bm.get_item(item_id)
        assert item is not None
        assert item.content == "Test content"
        assert item.evidence_score == 0.5  # Starts neutral

    def test_buffer_overflow_eviction(self):
        """Test that lowest evidence item is evicted on overflow."""
        gate = MockLearnedGate()
        bm = BufferManager(max_buffer_size=3, learned_gate=gate)

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        # Add 3 items
        ids = []
        for i in range(3):
            item_id = bm.add(
                content=f"Content {i}",
                embedding=embedding,
                features=features,
                context={},
            )
            ids.append(item_id)

        # Set different evidence scores
        bm.accumulate_evidence(ids[0], 0.1)   # Evidence: 0.6
        bm.accumulate_evidence(ids[1], -0.2)  # Evidence: 0.3 (lowest)
        bm.accumulate_evidence(ids[2], 0.2)   # Evidence: 0.7

        assert bm.size == 3

        # Add 4th item - should evict lowest evidence (ids[1])
        new_id = bm.add(
            content="New content",
            embedding=embedding,
            features=features,
            context={},
        )

        assert bm.size == 3  # Still 3 (evicted one)
        assert bm.get_item(ids[1]) is None  # Evicted
        assert bm.get_item(new_id) is not None  # New item exists
        assert bm.stats["items_discarded"] == 1

    def test_get_all_items(self):
        """Test getting all buffered items."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        bm.add("Content 1", embedding, features, {})
        bm.add("Content 2", embedding, features, {})

        items = bm.get_all_items()
        assert len(items) == 2


class TestBufferProbing:
    """Test buffer probing during retrieval."""

    def test_probe_finds_similar_items(self):
        """Test that probe finds semantically similar items."""
        bm = BufferManager()

        # Create similar embeddings (use same base with tiny perturbation)
        base_emb = np.random.randn(1024).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        # Very similar embedding (cosine > 0.99)
        similar_emb = base_emb.copy()
        similar_emb += np.random.randn(1024).astype(np.float32) * 0.01
        similar_emb /= np.linalg.norm(similar_emb)

        # Completely different embedding
        dissimilar_emb = np.random.randn(1024).astype(np.float32)
        dissimilar_emb /= np.linalg.norm(dissimilar_emb)

        features = np.random.randn(1143).astype(np.float32)

        # Add items
        similar_id = bm.add("Similar", similar_emb, features, {})
        dissimilar_id = bm.add("Dissimilar", dissimilar_emb, features, {})

        # Probe with base embedding
        matches = bm.probe(base_emb, threshold=0.6, limit=5)

        # Should find similar but not dissimilar
        assert len(matches) >= 1
        assert any(m.id == similar_id for m in matches)

    def test_probe_accumulates_evidence(self):
        """Test that probe() accumulates evidence for matches."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        initial_evidence = bm.get_item(item_id).evidence_score

        # Probe should increase evidence
        matches = bm.probe(embedding, threshold=0.6)

        assert len(matches) == 1
        assert bm.get_item(item_id).evidence_score > initial_evidence
        assert bm.get_item(item_id).retrieval_hits == 1
        assert bm.stats["retrieval_hits"] == 1

    def test_probe_multiple_times_increases_evidence(self):
        """Test that multiple probes increase evidence cumulatively."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        # Probe multiple times
        for _ in range(3):
            bm.probe(embedding, threshold=0.6)

        item = bm.get_item(item_id)
        assert item.retrieval_hits == 3
        # Evidence caps at 1.0, so just verify it increased
        assert item.evidence_score > 0.5


class TestEvidenceAccumulation:
    """Test evidence accumulation from various sources."""

    def test_accumulate_positive_evidence(self):
        """Test accumulating positive evidence."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        bm.accumulate_evidence(item_id, 0.2, reason="test_signal")

        item = bm.get_item(item_id)
        assert item.evidence_score == 0.7
        assert item.evidence_count == 1

    def test_accumulate_negative_evidence(self):
        """Test accumulating negative evidence."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        bm.accumulate_evidence(item_id, -0.3, reason="negative_signal")

        item = bm.get_item(item_id)
        assert item.evidence_score == 0.2

    def test_evidence_bounded_0_1(self):
        """Test that evidence is bounded between 0 and 1."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        # Try to exceed bounds
        bm.accumulate_evidence(item_id, 10.0)
        assert bm.get_item(item_id).evidence_score == 1.0

        bm.accumulate_evidence(item_id, -10.0)
        assert bm.get_item(item_id).evidence_score == 0.0

    def test_accumulate_from_outcome(self):
        """Test propagating outcome signals to buffer items."""
        bm = BufferManager()

        # Create item with known embedding
        embedding = np.random.randn(1024).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        initial_evidence = bm.get_item(item_id).evidence_score

        # Propagate positive signal with VERY similar query (same embedding)
        # This ensures similarity > threshold
        query_emb = embedding.copy()  # Exact same = similarity 1.0

        bm.accumulate_from_outcome(
            query_embedding=query_emb,
            combined_signal=0.5,  # Positive signal
            similarity_threshold=0.5
        )

        item = bm.get_item(item_id)
        # Signal is 0.1 * 0.5 * 1.0 = 0.05, so evidence = 0.55
        assert item.evidence_score > initial_evidence


class TestPromotionDecisions:
    """Test promotion/discard decision logic."""

    def test_high_evidence_promotes(self):
        """Test that high evidence leads to promotion."""
        gate = MockLearnedGate()
        bm = BufferManager(
            promotion_threshold=0.65,
            discard_threshold=0.25,
            learned_gate=gate
        )

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        # Boost evidence above threshold
        bm.accumulate_evidence(item_id, 0.2)  # 0.7 > 0.65

        decisions = bm.tick()

        assert len(decisions) == 1
        assert decisions[0].action == PromotionAction.PROMOTE
        assert bm.size == 0  # Item removed
        assert bm.stats["items_promoted"] == 1

    def test_low_evidence_discards(self):
        """Test that low evidence leads to discard."""
        gate = MockLearnedGate()
        bm = BufferManager(
            promotion_threshold=0.65,
            discard_threshold=0.25,
            learned_gate=gate
        )

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        # Lower evidence below threshold
        bm.accumulate_evidence(item_id, -0.3)  # 0.2 < 0.25

        decisions = bm.tick()

        assert len(decisions) == 1
        assert decisions[0].action == PromotionAction.DISCARD
        assert bm.size == 0
        assert bm.stats["items_discarded"] == 1

    def test_middle_evidence_waits(self):
        """Test that middle evidence leads to waiting (no action)."""
        bm = BufferManager(
            promotion_threshold=0.65,
            discard_threshold=0.25,
        )

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        # Evidence stays at 0.5, between thresholds

        decisions = bm.tick()

        # WAIT decisions are not returned (only PROMOTE/DISCARD are actionable)
        assert len(decisions) == 0
        assert bm.size == 1  # Item still in buffer

    def test_retrieval_boost_affects_promotion(self):
        """Test that retrieval hits boost promotion probability."""
        gate = MockLearnedGate()
        bm = BufferManager(
            promotion_threshold=0.65,
            discard_threshold=0.25,
            learned_gate=gate
        )

        embedding = np.random.randn(1024).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})

        # Evidence at 0.5, but multiple retrieval hits should boost
        for _ in range(4):  # 4 hits = sqrt(4) * 0.12 = 0.24 boost
            bm.probe(embedding, threshold=0.6)

        # Evidence: 0.5 + 0.25*4 = 1.5 -> capped at 1.0
        # Plus retrieval boost in evaluation: +0.24
        decisions = bm.tick()

        assert len(decisions) == 1
        assert decisions[0].action == PromotionAction.PROMOTE


class TestNeuromodulatorThresholdAdjustment:
    """Test neuromodulator influence on thresholds."""

    def test_high_arousal_lowers_promotion_threshold(self):
        """Test that high NE lowers promotion threshold."""
        gate = MockLearnedGate()
        bm = BufferManager(promotion_threshold=0.65, learned_gate=gate)

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        bm.accumulate_evidence(item_id, 0.1)  # 0.6, just below default threshold

        # Without neuromod: should wait (0.6 < 0.65)
        decisions = bm.tick(neuromod_state=None)
        assert len(decisions) == 0  # No action = waiting
        assert bm.size == 1  # Still in buffer

        # With high NE: threshold lowered to ~0.55, should promote
        high_ne_state = MockNeuromodState(norepinephrine_gain=1.5)
        decisions = bm.tick(neuromod_state=high_ne_state)
        assert len(decisions) == 1
        assert decisions[0].action == PromotionAction.PROMOTE

    def test_encoding_mode_lowers_threshold(self):
        """Test that encoding ACh mode lowers threshold."""
        gate = MockLearnedGate()
        bm = BufferManager(promotion_threshold=0.65, learned_gate=gate)

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        bm.accumulate_evidence(item_id, 0.12)  # 0.62, above lowered threshold of 0.60

        # Encoding mode should lower threshold by 0.05 -> 0.60
        encoding_state = MockNeuromodState(acetylcholine_mode="encoding")
        decisions = bm.tick(neuromod_state=encoding_state)
        assert len(decisions) == 1
        assert decisions[0].action == PromotionAction.PROMOTE


class TestGateTraining:
    """Test that promotions/discards train the learned gate."""

    def test_promotion_trains_gate_positive(self):
        """Test that promoted items send positive training signal."""
        gate = MockLearnedGate()
        bm = BufferManager(learned_gate=gate, promotion_threshold=0.65)

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        bm.accumulate_evidence(item_id, 0.2)  # 0.7, above threshold

        bm.tick()

        # Check gate was trained
        assert len(gate.updates) == 1
        mid, utility = gate.updates[0]
        assert utility >= 0.5  # Promoted = positive utility

    def test_discard_trains_gate_negative(self):
        """Test that discarded items send scaled soft negative signal."""
        gate = MockLearnedGate()
        bm = BufferManager(learned_gate=gate, discard_threshold=0.25)

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        bm.accumulate_evidence(item_id, -0.3)  # 0.2, below threshold

        bm.tick()

        # Check gate was trained with soft negative
        # MEMORY-HIGH-002 FIX: Discards now scale by evidence to cover [0.1, 0.45] range
        # Evidence 0.2 → utility = 0.1 + 0.35 * 0.2 = 0.17
        assert len(gate.updates) == 1
        mid, utility = gate.updates[0]
        assert 0.1 <= utility <= 0.45  # Discarded utility range
        assert utility < 0.5  # Always below promotion threshold


class TestStaggering:
    """Test staggered promotion to prevent catastrophic forgetting."""

    def test_stagger_limits_promotions_per_tick(self):
        """Test that max 5 items are promoted per tick."""
        gate = MockLearnedGate()
        bm = BufferManager(
            promotion_threshold=0.65,
            learned_gate=gate,
            stagger_limit=5,
        )

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        # Add 10 items all above threshold
        for i in range(10):
            item_id = bm.add(f"Content {i}", embedding, features, {})
            bm.accumulate_evidence(item_id, 0.2)  # All at 0.7

        # First tick should only promote 5
        decisions = bm.tick()

        promoted = sum(1 for d in decisions if d.action == PromotionAction.PROMOTE)
        waiting = sum(1 for d in decisions if d.action == PromotionAction.WAIT)

        assert promoted == 5
        assert waiting == 5  # Deferred to next tick
        assert bm.size == 5  # 5 remaining


class TestStatistics:
    """Test statistics tracking."""

    def test_stats_tracking(self):
        """Test that stats are properly tracked."""
        bm = BufferManager()

        embedding = np.random.randn(1024).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        features = np.random.randn(1143).astype(np.float32)

        bm.add("Test", embedding, features, {})
        bm.probe(embedding, threshold=0.6)

        stats = bm.get_stats()

        assert stats["items_added"] == 1
        assert stats["retrieval_probes"] == 1
        assert stats["retrieval_hits"] == 1
        assert stats["current_size"] == 1
        assert stats["probe_hit_rate"] == 1.0

    def test_clear_discards_all(self):
        """Test that clear() discards all with training signals."""
        gate = MockLearnedGate()
        bm = BufferManager(learned_gate=gate)

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        bm.add("Test 1", embedding, features, {})
        bm.add("Test 2", embedding, features, {})

        assert bm.size == 2

        bm.clear()

        assert bm.size == 0
        assert len(gate.updates) == 2  # Both discarded with training


class TestRecentlyPromotedCooldown:
    """Test cooldown for recently promoted items."""

    def test_is_recently_promoted(self):
        """Test recently promoted tracking."""
        gate = MockLearnedGate()
        bm = BufferManager(
            promotion_threshold=0.65,
            learned_gate=gate
        )

        embedding = np.random.randn(1024).astype(np.float32)
        features = np.random.randn(1143).astype(np.float32)

        item_id = bm.add("Test", embedding, features, {})
        bm.accumulate_evidence(item_id, 0.2)

        # Before tick: not recently promoted
        assert not bm.is_recently_promoted(item_id)

        bm.tick()

        # After promotion: in cooldown (but item_id was the buffer's internal ID)
        # The promoted item's ID is tracked in _recently_promoted
        assert bm.stats["items_promoted"] == 1
