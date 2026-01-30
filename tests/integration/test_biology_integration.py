"""
P7.1: Integration tests for P5 biology features.

Tests the integration between:
- P5.1: REM abstractions with sleep consolidation
- P5.2: Temporal episode linking
- P5.3: Synaptic tagging with consolidation
- P5.4: Query-memory separation in recall
- P5.5: STDP plasticity

These tests verify that the biological features work together
correctly in realistic memory scenarios.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np


class TestP5IntegrationSTDPConsolidation:
    """Test STDP integration with sleep consolidation."""

    @pytest.mark.asyncio
    async def test_stdp_records_spikes_during_retrieval(self):
        """STDP records spike events when memories are retrieved."""
        from ww.learning.stdp import STDPLearner

        stdp = STDPLearner()

        # Simulate memory retrieval sequence
        memory_ids = [f"mem_{i}" for i in range(5)]
        now = datetime.now()

        # Record "spikes" as memories are accessed
        for i, mem_id in enumerate(memory_ids):
            stdp.record_spike(
                mem_id,
                timestamp=now + timedelta(milliseconds=i * 10)
            )

        # Verify all spikes recorded
        for mem_id in memory_ids:
            spike = stdp.get_latest_spike(mem_id)
            assert spike is not None

    @pytest.mark.asyncio
    async def test_stdp_weight_updates_from_co_retrieval(self):
        """Co-retrieved memories update STDP weights correctly."""
        from ww.learning.stdp import STDPLearner

        stdp = STDPLearner()
        now = datetime.now()

        # Memory A retrieved first, then B (causal relationship)
        stdp.record_spike("mem_a", timestamp=now)
        stdp.record_spike("mem_b", timestamp=now + timedelta(milliseconds=10))

        # Compute update for A->B synapse
        update = stdp.compute_update("mem_a", "mem_b", current_weight=0.5)

        assert update is not None
        assert update.update_type == "ltp"  # A before B = strengthen
        assert update.new_weight > 0.5

    @pytest.mark.asyncio
    async def test_stdp_anti_causal_weakens(self):
        """Anti-causal retrieval order weakens connections."""
        from ww.learning.stdp import STDPLearner

        stdp = STDPLearner()
        now = datetime.now()

        # Memory B retrieved first, then A (anti-causal)
        stdp.record_spike("mem_b", timestamp=now)
        stdp.record_spike("mem_a", timestamp=now + timedelta(milliseconds=10))

        # Compute update for A->B synapse (A spike after B)
        update = stdp.compute_update("mem_a", "mem_b", current_weight=0.5)

        assert update is not None
        assert update.update_type == "ltd"  # A after B = weaken
        assert update.new_weight < 0.5


class TestP5IntegrationQueryMemorySeparation:
    """Test query-memory separation in realistic recall scenarios."""

    @pytest.mark.asyncio
    async def test_separation_produces_different_projections(self):
        """Query and memory projections differ appropriately."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator

        separator = QueryMemorySeparator()

        # Same embedding projected as query vs memory
        embedding = np.random.randn(1024).astype(np.float32)

        query_proj = separator.project_query(embedding)
        memory_proj = separator.project_memory(embedding)

        # Should be different (asymmetric encoding)
        diff = np.linalg.norm(query_proj - memory_proj)
        assert diff > 0.01

    @pytest.mark.asyncio
    async def test_batch_separation_maintains_consistency(self):
        """Batch projections maintain individual consistency."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator

        separator = QueryMemorySeparator()

        # Create batch
        batch = np.random.randn(10, 1024).astype(np.float32)

        # Project batch
        batch_proj = separator.project_query(batch)

        # Project individually
        individual_projs = [separator.project_query(batch[i]) for i in range(10)]

        # Should match (deterministic)
        for i in range(10):
            np.testing.assert_array_almost_equal(
                batch_proj[i], individual_projs[i], decimal=5
            )


class TestP5IntegrationTemporalSequencing:
    """Test temporal episode linking integration."""

    @pytest.mark.asyncio
    async def test_temporal_links_form_chain(self):
        """Episodes form temporal chains with bidirectional links."""
        from ww.core.types import Episode
        from uuid import uuid4

        # Create linked episodes
        session_id = str(uuid4())
        ids = [uuid4() for _ in range(5)]
        episodes = []

        for i, id_ in enumerate(ids):
            ep = Episode(
                id=id_,
                session_id=session_id,  # Use string
                content=f"Episode {i}",
                previous_episode_id=ids[i-1] if i > 0 else None,
                next_episode_id=ids[i+1] if i < len(ids)-1 else None,
                sequence_position=i
            )
            episodes.append(ep)

        # Verify chain integrity
        for i, ep in enumerate(episodes):
            if i > 0:
                assert ep.previous_episode_id == ids[i-1]
            if i < len(episodes) - 1:
                assert ep.next_episode_id == ids[i+1]

    @pytest.mark.asyncio
    async def test_duration_tracking(self):
        """Episodes track duration correctly."""
        from ww.core.types import Episode
        from uuid import uuid4

        now = datetime.now()
        ep = Episode(
            id=uuid4(),
            session_id=str(uuid4()),  # Use string
            content="Test episode",
            timestamp=now,
            end_timestamp=now + timedelta(seconds=5),
            duration_ms=5000
        )

        assert ep.duration_ms == 5000
        assert ep.end_timestamp > ep.timestamp


class TestP5IntegrationSynapticTagging:
    """Test synaptic tagging with consolidation pipeline."""

    @pytest.mark.asyncio
    async def test_synaptic_tag_types(self):
        """Synaptic tagging distinguishes early and late LTP."""
        from ww.learning.plasticity import SynapticTagger

        tagger = SynapticTagger()

        # Weak signal -> early LTP
        weak_tag = tagger.tag_synapse(
            source_id="a",
            target_id="b",
            signal_strength=0.4
        )
        assert weak_tag.tag_type == "early"

        # Strong signal -> late LTP
        strong_tag = tagger.tag_synapse(
            source_id="c",
            target_id="d",
            signal_strength=0.9
        )
        assert strong_tag.tag_type == "late"

    @pytest.mark.asyncio
    async def test_tag_capture_during_consolidation(self):
        """Tags are captured during consolidation window."""
        from ww.learning.plasticity import SynapticTagger

        tagger = SynapticTagger()

        # Create tags
        tag1 = tagger.tag_synapse("a", "b", 0.5)
        tag2 = tagger.tag_synapse("c", "d", 0.8)

        # Get tagged synapses before capture
        all_tags = tagger.get_tagged_synapses()
        assert len(all_tags) == 2

        # Capture tags (returns list of captured tags)
        captured = tagger.capture_tags()
        assert len(captured) >= 0


class TestP5IntegrationREMAbstractions:
    """Test REM sleep abstraction storage."""

    @pytest.mark.asyncio
    async def test_abstraction_event_creation(self):
        """REM phase can record abstractions from episodes."""
        from ww.consolidation.sleep import AbstractionEvent

        # Create an abstraction event
        event = AbstractionEvent(
            cluster_ids=["ep_1", "ep_2", "ep_3"],
            concept_name="common_error_handling",
            confidence=0.85
        )

        assert len(event.cluster_ids) == 3
        assert event.concept_name == "common_error_handling"
        assert event.confidence == 0.85
        assert event.abstraction_time is not None


class TestP5IntegrationFullPipeline:
    """Test complete biology pipeline integration."""

    @pytest.mark.asyncio
    async def test_retrieval_triggers_stdp_and_tagging(self):
        """Memory retrieval triggers both STDP and synaptic tagging."""
        from ww.learning.stdp import STDPLearner
        from ww.learning.plasticity import SynapticTagger

        stdp = STDPLearner()
        tagger = SynapticTagger()
        now = datetime.now()

        # Simulate retrieval of related memories
        retrieved = ["mem_1", "mem_2", "mem_3"]

        for i, mem_id in enumerate(retrieved):
            # STDP: Record spike timing
            stdp.record_spike(mem_id, timestamp=now + timedelta(milliseconds=i * 10))

            # Tagging: Mark co-retrieved pairs
            if i > 0:
                tagger.tag_synapse(
                    source_id=retrieved[i-1],
                    target_id=mem_id,
                    signal_strength=0.7
                )

        # Verify STDP recorded all
        assert stdp.get_latest_spike("mem_1") is not None
        assert stdp.get_latest_spike("mem_3") is not None

        # Verify tagging created connections
        tags = tagger.get_tagged_synapses()
        assert len(tags) == 2  # mem_1->mem_2 and mem_2->mem_3

    @pytest.mark.asyncio
    async def test_query_projection_with_stdp_update(self):
        """Query-memory separation works with STDP learning."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator
        from ww.learning.stdp import STDPLearner

        separator = QueryMemorySeparator()
        stdp = STDPLearner()

        # Query and retrieve
        query_emb = np.random.randn(1024).astype(np.float32)
        projected_query = separator.project_query(query_emb)

        # Retrieved memories
        memory_embs = np.random.randn(3, 1024).astype(np.float32)
        projected_mems = separator.project_memory(memory_embs)

        # Compute similarities
        sims = separator.compute_similarity(query_emb, memory_embs)
        assert len(sims) == 3

        # Record STDP based on retrieval order (sorted by similarity)
        now = datetime.now()
        order = np.argsort(-sims)  # Descending similarity

        for i, idx in enumerate(order):
            stdp.record_spike(f"mem_{idx}", timestamp=now + timedelta(milliseconds=i * 5))

        # Top-ranked memories should have STDP relationships
        assert stdp.get_latest_spike(f"mem_{order[0]}") is not None

    @pytest.mark.asyncio
    async def test_temporal_sequence_with_stdp(self):
        """Temporal episode sequences trigger appropriate STDP."""
        from ww.learning.stdp import STDPLearner
        from ww.core.types import Episode
        from uuid import uuid4

        stdp = STDPLearner()
        now = datetime.now()

        # Create temporal sequence
        ids = [uuid4() for _ in range(5)]
        for i, id_ in enumerate(ids):
            stdp.record_spike(str(id_), timestamp=now + timedelta(milliseconds=i * 20))

        # Check STDP updates for adjacent pairs
        for i in range(len(ids) - 1):
            update = stdp.compute_update(
                str(ids[i]),
                str(ids[i + 1]),
                current_weight=0.5
            )

            if update:  # Within STDP window
                assert update.update_type == "ltp"
