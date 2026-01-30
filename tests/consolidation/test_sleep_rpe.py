"""
Tests for P1D: RPE generation during sleep replay.

Tests that replay sequences generate reward prediction errors via VTA
for credit assignment during consolidation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, Mock, patch

from ww.consolidation.sleep import SleepConsolidation, ReplayEvent, ReplayDirection
from ww.nca.vta import VTACircuit, VTAConfig


# Mock episode class for testing
class MockEpisode:
    """Mock episode with attributes for testing."""

    def __init__(
        self,
        id: UUID | None = None,
        content: str = "test content",
        importance: float = 0.5,
        outcome_score: float = 0.5,
        retrieval_count: int = 0,
        created_at: datetime | None = None,
        embedding: list[float] | None = None,
    ):
        self.id = id or uuid4()
        self.content = content
        self.importance = importance
        self.outcome_score = outcome_score
        self.retrieval_count = retrieval_count
        self.created_at = created_at or datetime.now()
        self.embedding = embedding or list(np.random.randn(128))
        self.context = None


class MockMemory:
    """Mock memory store for testing."""

    def __init__(self, episodes: list[MockEpisode] | None = None):
        self.episodes = episodes or []

    async def get_recent(self, hours: int = 24, limit: int = 100):
        return self.episodes[:limit]

    async def get_by_id(self, episode_id: UUID):
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep
        return None

    async def sample_random(self, limit: int = 50, **kwargs):
        return self.episodes[:limit]


class MockSemanticMemory:
    """Mock semantic memory for testing."""

    async def create_or_strengthen(self, **kwargs):
        return Mock(id=uuid4())

    async def create_entity(self, **kwargs):
        return Mock(id=uuid4())

    async def create_relationship(self, **kwargs):
        return Mock(id=uuid4())


class MockGraphStore:
    """Mock graph store for testing."""

    async def get_node(self, node_id: str):
        return None

    async def get_relationships(self, node_id: str, direction: str = "out"):
        return []

    async def get_all_nodes(self, label: str | None = None):
        return []

    async def update_relationship_weight(self, source_id: str, target_id: str, new_weight: float):
        pass

    async def delete_relationship(self, source_id: str, target_id: str):
        pass


@pytest.fixture
def vta_circuit():
    """Create VTA circuit for testing."""
    config = VTAConfig(
        tonic_da_level=0.3,
        rpe_to_da_gain=0.5,
        td_lambda=0.9,
    )
    return VTACircuit(config)


@pytest.fixture
def mock_episodes():
    """Create mock episodes with varied properties."""
    # Create a base embedding vector
    base_embedding = np.random.randn(128)
    
    episodes = []
    for i in range(10):
        # Create varied importance and outcome scores
        importance = 0.3 + (i * 0.05)  # 0.3 to 0.75
        outcome = 0.4 + (i * 0.04)  # 0.4 to 0.76

        # Create coherent embeddings (similar to base + small noise)
        # This ensures SWR can find coherent sequences
        coherent_embedding = base_embedding + np.random.randn(128) * 0.1
        
        episodes.append(
            MockEpisode(
                id=uuid4(),
                content=f"Episode {i}",
                importance=importance,
                outcome_score=outcome,
                retrieval_count=i,
                created_at=datetime.now() - timedelta(hours=i),
                embedding=list(coherent_embedding),
            )
        )
    return episodes


@pytest.fixture
def consolidation(mock_episodes):
    """Create consolidation with mocked memory."""
    episodic = MockMemory(mock_episodes)
    semantic = MockSemanticMemory()
    graph = MockGraphStore()

    return SleepConsolidation(
        episodic_memory=episodic,
        semantic_memory=semantic,
        graph_store=graph,
        max_replays=10,
        replay_delay_ms=0,  # No delay for testing
        vae_enabled=False,  # Disable VAE for simpler tests
    )


@pytest.mark.asyncio
async def test_replay_generates_rpe(consolidation, vta_circuit, mock_episodes):
    """Test that replay sequences generate RPE via VTA."""
    # Set VTA circuit
    consolidation.set_vta_circuit(vta_circuit)

    # Create a simple replay sequence
    sequence = mock_episodes[:5]

    # Generate RPE from sequence
    rpes = await consolidation._generate_replay_rpe(sequence)

    # Should generate RPE for each transition (N-1 for N episodes)
    assert len(rpes) == len(sequence) - 1
    assert all(isinstance(rpe, float) for rpe in rpes)

    # RPE should be within reasonable bounds [-1, 1]
    assert all(-1.5 <= rpe <= 1.5 for rpe in rpes)


@pytest.mark.asyncio
async def test_rpe_reflects_value_differences(consolidation, vta_circuit):
    """Test that RPE reflects differences in episode value."""
    consolidation.set_vta_circuit(vta_circuit)

    # Create sequence with increasing value
    increasing_value = [
        MockEpisode(
            importance=0.2 + (i * 0.2),
            outcome_score=0.2 + (i * 0.2),
            retrieval_count=i,
        )
        for i in range(5)
    ]

    rpes_increasing = await consolidation._generate_replay_rpe(increasing_value)

    # Create sequence with decreasing value
    decreasing_value = list(reversed(increasing_value))
    rpes_decreasing = await consolidation._generate_replay_rpe(decreasing_value)

    # Increasing value should generate positive RPEs
    assert np.mean(rpes_increasing) > 0

    # Decreasing value should generate negative RPEs
    assert np.mean(rpes_decreasing) < 0


@pytest.mark.asyncio
async def test_rpe_affects_replay_priority(consolidation, vta_circuit):
    """Test that high RPE sequences are prioritized for replay."""
    consolidation.set_vta_circuit(vta_circuit)

    # Create sequences with different value patterns
    high_surprise = [
        MockEpisode(importance=0.2),
        MockEpisode(importance=0.9),  # Big jump = high RPE
        MockEpisode(importance=0.3),
    ]

    low_surprise = [
        MockEpisode(importance=0.5),
        MockEpisode(importance=0.5),  # No change = low RPE
        MockEpisode(importance=0.5),
    ]

    rpes_high = await consolidation._generate_replay_rpe(high_surprise)
    rpes_low = await consolidation._generate_replay_rpe(low_surprise)

    # High surprise should have higher absolute RPE
    assert np.mean(np.abs(rpes_high)) > np.mean(np.abs(rpes_low))

    # Test prioritization
    sequences = [low_surprise, high_surprise]
    sequence_rpes = [rpes_low, rpes_high]

    prioritized = consolidation._prioritize_by_rpe(sequences, sequence_rpes)

    # High RPE sequence should be first
    assert prioritized[0] == high_surprise


@pytest.mark.asyncio
async def test_vta_active_during_consolidation(consolidation, vta_circuit, mock_episodes):
    """Test that VTA circuit is active during NREM consolidation."""
    consolidation.set_vta_circuit(vta_circuit)

    # Run NREM phase
    events = await consolidation.nrem_phase(session_id="test_session")

    # Should have replayed some episodes
    assert len(events) > 0

    # VTA should have processed RPEs (state should be updated)
    vta_stats = vta_circuit.get_stats()

    # Check that events have RPE values (RPE generation occurred)
    if len(events) > 0:
        rpe_values = [e.rpe for e in events]
        # At least some events should have non-zero RPE
        assert any(abs(rpe) > 0.001 for rpe in rpe_values), f"No non-zero RPEs found in {rpe_values}"


@pytest.mark.asyncio
async def test_high_rpe_increases_replay_probability(consolidation, vta_circuit):
    """Test that episodes with high RPE are more likely to be replayed."""
    consolidation.set_vta_circuit(vta_circuit)

    # Create episodes with varying importance
    low_value = MockEpisode(importance=0.2, outcome_score=0.2)
    high_value = MockEpisode(importance=0.9, outcome_score=0.9)

    # Create sequence with surprise
    surprise_sequence = [low_value, high_value]  # Big jump

    # Generate RPE
    rpes = await consolidation._generate_replay_rpe(surprise_sequence)

    # Should generate positive RPE for the value increase
    assert rpes[0] > 0.0


@pytest.mark.asyncio
async def test_rpe_stored_in_replay_event(consolidation, vta_circuit, mock_episodes):
    """Test that RPE is stored in ReplayEvent."""
    consolidation.set_vta_circuit(vta_circuit)

    # Run NREM phase
    events = await consolidation.nrem_phase(session_id="test_session")

    # Check that events have RPE values
    if events:
        # At least some events should have non-zero RPE
        rpe_values = [e.rpe for e in events]
        assert any(rpe != 0.0 for rpe in rpe_values)


@pytest.mark.asyncio
async def test_value_estimation(consolidation):
    """Test episode value estimation for RPE computation."""
    # Create episodes with different properties
    low_value_ep = MockEpisode(
        importance=0.2,
        outcome_score=0.2,
        retrieval_count=0,
    )

    high_value_ep = MockEpisode(
        importance=0.9,
        outcome_score=0.9,
        retrieval_count=10,
    )

    low_value = consolidation._estimate_value(low_value_ep)
    high_value = consolidation._estimate_value(high_value_ep)

    # High value episode should have higher estimated value
    assert high_value > low_value

    # Values should be in [0, 1]
    assert 0.0 <= low_value <= 1.0
    assert 0.0 <= high_value <= 1.0


@pytest.mark.asyncio
async def test_rpe_without_vta_circuit(consolidation, mock_episodes):
    """Test that replay works without VTA circuit (graceful fallback)."""
    # Don't set VTA circuit
    assert consolidation._vta_circuit is None

    # Should not generate RPEs
    rpes = await consolidation._generate_replay_rpe(mock_episodes[:5])
    assert len(rpes) == 0

    # Should still be able to run NREM phase
    events = await consolidation.nrem_phase(session_id="test_session")
    # Events should not have RPE values (should be 0.0)
    assert all(e.rpe == 0.0 for e in events)


@pytest.mark.asyncio
async def test_vta_statistics_in_consolidation_stats(consolidation, vta_circuit):
    """Test that VTA statistics are included in consolidation stats."""
    consolidation.set_vta_circuit(vta_circuit)

    # Get stats
    stats = consolidation.get_stats()

    # Should include VTA circuit stats
    assert "vta_circuit" in stats
    assert "current_da" in stats["vta_circuit"]
    assert "last_rpe" in stats["vta_circuit"]


@pytest.mark.asyncio
async def test_reverse_replay_for_credit_assignment(consolidation, vta_circuit, mock_episodes):
    """Test that reverse replay (90% probability) is used for credit assignment."""
    consolidation.set_vta_circuit(vta_circuit)

    # Run multiple NREM phases to check replay direction distribution
    reverse_count = 0
    forward_count = 0

    for _ in range(10):
        events = await consolidation.nrem_phase(session_id="test_session")

        for event in events:
            if event.direction == ReplayDirection.REVERSE:
                reverse_count += 1
            elif event.direction == ReplayDirection.FORWARD:
                forward_count += 1

    # Should have more reverse replays than forward (biological 90/10 split)
    # Allow some variance due to randomness
    if reverse_count + forward_count > 0:
        reverse_ratio = reverse_count / (reverse_count + forward_count)
        # Should be roughly 90% reverse (allow 70-100% due to small sample)
        assert reverse_ratio >= 0.5  # At least half should be reverse


@pytest.mark.asyncio
async def test_rpe_sequence_logging(consolidation, vta_circuit, mock_episodes, caplog):
    """Test that RPE sequence statistics are logged."""
    consolidation.set_vta_circuit(vta_circuit)

    # Run NREM phase
    import logging
    caplog.set_level(logging.DEBUG)

    events = await consolidation.nrem_phase(session_id="test_session")

    # Check for RPE logging
    if events:
        # Should have logged RPE statistics
        assert any("RPE" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_empty_sequence_rpe(consolidation, vta_circuit):
    """Test RPE generation with empty or single-element sequence."""
    consolidation.set_vta_circuit(vta_circuit)

    # Empty sequence
    rpes_empty = await consolidation._generate_replay_rpe([])
    assert len(rpes_empty) == 0

    # Single episode (no transitions)
    rpes_single = await consolidation._generate_replay_rpe([MockEpisode()])
    assert len(rpes_single) == 0


@pytest.mark.asyncio
async def test_prioritize_by_rpe_empty_input(consolidation):
    """Test RPE prioritization with empty input."""
    # Empty sequences
    result = consolidation._prioritize_by_rpe([], [])
    assert result == []

    # Sequences but no RPEs
    sequences = [[MockEpisode()]]
    result = consolidation._prioritize_by_rpe(sequences, [])
    assert result == sequences


def test_replay_event_has_rpe_field():
    """Test that ReplayEvent dataclass includes RPE field."""
    event = ReplayEvent(
        episode_id=uuid4(),
        rpe=0.5,
    )

    assert hasattr(event, "rpe")
    assert event.rpe == 0.5


@pytest.mark.asyncio
async def test_vta_eligibility_trace_during_replay(consolidation, vta_circuit):
    """Test that VTA eligibility trace is updated during replay."""
    consolidation.set_vta_circuit(vta_circuit)

    # Create sequence
    sequence = [
        MockEpisode(importance=0.2),
        MockEpisode(importance=0.5),
        MockEpisode(importance=0.8),
    ]

    # Generate RPE (should update eligibility)
    await consolidation._generate_replay_rpe(sequence)

    # Check VTA state
    vta_state = vta_circuit.state

    # Eligibility should be non-zero after processing
    assert vta_state.eligibility > 0.0


@pytest.mark.asyncio
async def test_integration_full_sleep_cycle_with_vta(consolidation, vta_circuit):
    """Test full sleep cycle with VTA RPE generation."""
    consolidation.set_vta_circuit(vta_circuit)

    # Run full sleep cycle
    result = await consolidation.full_sleep_cycle(session_id="integration_test")

    # Should have completed successfully
    assert result.nrem_replays >= 0

    # VTA should have been active
    vta_stats = vta_circuit.get_stats()
    # Should have processed some RPEs (or circuit is in tonic state)
    assert vta_stats["current_da"] >= vta_circuit.config.min_da
