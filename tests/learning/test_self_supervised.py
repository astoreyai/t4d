"""Tests for P1-2: Self-Supervised Credit Estimation."""

import numpy as np
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from t4dm.learning.self_supervised import (
    ImplicitCredit,
    SelfSupervisedCredit,
    RetrievalEvent,
)


class TestSelfSupervisedCredit:
    """Test SelfSupervisedCredit class."""

    def test_initialization(self):
        """Test default initialization."""
        ssc = SelfSupervisedCredit()
        assert ssc.contrastive_temperature == 0.1
        assert ssc.coactivation_boost == 0.1
        assert len(ssc._retrieval_history) == 0

    def test_record_single_retrieval(self):
        """Test recording a single retrieval event."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        mem_id = uuid4()
        embedding = np.random.randn(1024).astype(np.float32)

        credits = ssc.record_retrieval(
            query_embedding=query,
            retrieved_memories=[(mem_id, embedding)],
        )

        # Should get contrastive and frequency credits
        assert len(credits) >= 2
        assert any(c.credit_type == "contrastive" for c in credits)
        assert any(c.credit_type == "frequency" for c in credits)

    def test_contrastive_credit_similarity(self):
        """Test that contrastive credit correlates with similarity."""
        ssc = SelfSupervisedCredit()

        # Create query and two memories - one similar, one dissimilar
        query = np.array([1.0, 0.0, 0.0] + [0.0] * 1021, dtype=np.float32)
        similar_emb = np.array([0.9, 0.1, 0.0] + [0.0] * 1021, dtype=np.float32)
        dissimilar_emb = np.array([0.0, 0.0, 1.0] + [0.0] * 1021, dtype=np.float32)

        similar_id = uuid4()
        dissimilar_id = uuid4()

        # Record similar memory
        credits_sim = ssc.record_retrieval(
            query_embedding=query,
            retrieved_memories=[(similar_id, similar_emb)],
        )

        # Record dissimilar memory with new query
        credits_dissim = ssc.record_retrieval(
            query_embedding=query,
            retrieved_memories=[(dissimilar_id, dissimilar_emb)],
        )

        # Find contrastive credits
        sim_credit = next(c for c in credits_sim if c.credit_type == "contrastive")
        dissim_credit = next(c for c in credits_dissim if c.credit_type == "contrastive")

        # Similar should have higher credit
        assert sim_credit.credit_value > dissim_credit.credit_value

    def test_coactivation_credit(self):
        """Test co-activation credit for co-retrieved memories."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        id1, id2 = uuid4(), uuid4()
        emb1 = np.random.randn(1024).astype(np.float32)
        emb2 = np.random.randn(1024).astype(np.float32)

        credits = ssc.record_retrieval(
            query_embedding=query,
            retrieved_memories=[(id1, emb1), (id2, emb2)],
        )

        # Should have coactivation credits
        coact_credits = [c for c in credits if c.credit_type == "coactivation"]
        assert len(coact_credits) == 2  # One for each memory

        # Check coactivation matrix
        strength = ssc.get_coactivation_strength(id1, id2)
        assert strength > 0

    def test_frequency_credit_accumulates(self):
        """Test that frequency credit accumulates with retrievals."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        mem_id = uuid4()
        embedding = np.random.randn(1024).astype(np.float32)

        # First retrieval
        ssc.record_retrieval(query, [(mem_id, embedding)])
        count1 = ssc.get_retrieval_count(mem_id)
        credit1 = ssc.get_implicit_credit(mem_id)

        # Second retrieval
        ssc.record_retrieval(query, [(mem_id, embedding)])
        count2 = ssc.get_retrieval_count(mem_id)
        credit2 = ssc.get_implicit_credit(mem_id)

        assert count2 == count1 + 1
        assert credit2 > credit1

    def test_temporal_chain_credit(self):
        """Test temporal credit propagation from outcomes."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        id1, id2 = uuid4(), uuid4()
        emb1 = np.random.randn(1024).astype(np.float32)
        emb2 = np.random.randn(1024).astype(np.float32)

        # Record co-retrieval
        ssc.record_retrieval(query, [(id1, emb1), (id2, emb2)])

        # Get credit before propagation
        credit_before = ssc.get_implicit_credit(id1)

        # Propagate outcome credit from id2
        temporal_credits = ssc.propagate_outcome_credit(
            outcome_memory_id=id2,
            outcome_value=1.0,
            lookback_window=timedelta(hours=1),
        )

        # id1 should receive credit from id2's outcome
        assert len(temporal_credits) > 0
        credit_after = ssc.get_implicit_credit(id1)
        assert credit_after > credit_before

    def test_implicit_credit_retrieval(self):
        """Test get_implicit_credit method."""
        ssc = SelfSupervisedCredit()

        mem_id = uuid4()
        query = np.random.randn(1024).astype(np.float32)
        embedding = np.random.randn(1024).astype(np.float32)

        # Before any retrievals
        assert ssc.get_implicit_credit(mem_id) == 0.0

        # After retrieval
        ssc.record_retrieval(query, [(mem_id, embedding)])
        assert ssc.get_implicit_credit(mem_id) > 0.0

    def test_statistics(self):
        """Test get_statistics method."""
        ssc = SelfSupervisedCredit()

        stats = ssc.get_statistics()
        assert stats["num_tracked_memories"] == 0
        assert stats["total_retrievals"] == 0

        # After some activity
        query = np.random.randn(1024).astype(np.float32)
        for _ in range(5):
            mem_id = uuid4()
            emb = np.random.randn(1024).astype(np.float32)
            ssc.record_retrieval(query, [(mem_id, emb)])

        stats = ssc.get_statistics()
        assert stats["num_tracked_memories"] == 5
        assert stats["total_retrievals"] == 5
        assert stats["retrieval_history_size"] == 5

    def test_save_load_state(self):
        """Test state persistence."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        id1, id2 = uuid4(), uuid4()
        emb1 = np.random.randn(1024).astype(np.float32)
        emb2 = np.random.randn(1024).astype(np.float32)

        ssc.record_retrieval(query, [(id1, emb1), (id2, emb2)])

        # Save state
        state = ssc.save_state()

        # Create new instance and load
        ssc2 = SelfSupervisedCredit()
        ssc2.load_state(state)

        # Verify state restored
        assert ssc2.get_implicit_credit(id1) == ssc.get_implicit_credit(id1)
        assert ssc2.get_retrieval_count(id1) == ssc.get_retrieval_count(id1)

    def test_empty_retrieval(self):
        """Test handling of empty retrieval."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        credits = ssc.record_retrieval(query, [])

        assert len(credits) == 0

    def test_credit_value_bounds(self):
        """Test that credit values are bounded."""
        ssc = SelfSupervisedCredit()

        query = np.random.randn(1024).astype(np.float32)
        mem_id = uuid4()
        embedding = np.random.randn(1024).astype(np.float32)

        credits = ssc.record_retrieval(query, [(mem_id, embedding)])

        for credit in credits:
            assert -1.0 <= credit.credit_value <= 1.0
            assert 0.0 <= credit.confidence <= 1.0


class TestImplicitCredit:
    """Test ImplicitCredit dataclass."""

    def test_creation(self):
        """Test ImplicitCredit creation."""
        credit = ImplicitCredit(
            memory_id=uuid4(),
            credit_type="contrastive",
            credit_value=0.5,
            confidence=0.8,
        )
        assert credit.credit_type == "contrastive"
        assert credit.credit_value == 0.5
        assert credit.confidence == 0.8

    def test_with_source_ids(self):
        """Test creation with source IDs."""
        id1, id2 = uuid4(), uuid4()
        credit = ImplicitCredit(
            memory_id=id1,
            credit_type="temporal",
            credit_value=0.3,
            confidence=0.7,
            source_ids=[id2],
        )
        assert id2 in credit.source_ids


class TestRetrievalEvent:
    """Test RetrievalEvent dataclass."""

    def test_creation(self):
        """Test RetrievalEvent creation."""
        query = np.random.randn(1024).astype(np.float32)
        mem_id = uuid4()
        emb = np.random.randn(1024).astype(np.float32)

        event = RetrievalEvent(
            query_embedding=query,
            retrieved_ids=[mem_id],
            retrieved_embeddings=[emb],
        )
        assert len(event.retrieved_ids) == 1
        assert event.session_id == "default"
        assert isinstance(event.timestamp, datetime)
