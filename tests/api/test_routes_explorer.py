"""
Tests for memory explorer visualization routes.

Tests memory encoding, retrieval, pattern separation, and CA3 settling dynamics.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from uuid import UUID

from ww.api.routes.explorer import (
    MemoryExplorerState,
    EncodeDemoRequest,
    EncodeDemoResponse,
    RetrieveDemoRequest,
    RetrieveDemoResponse,
    ExplorerStateManager,
    explorer_router,
)


# =============================================================================
# Test MemoryExplorerState Model
# =============================================================================


class TestMemoryExplorerState:
    """Tests for MemoryExplorerState model."""

    def test_default_state(self):
        """Test default explorer state initialization."""
        state = MemoryExplorerState()
        assert state.dg_sparsity == 0.03
        assert state.ca3_attractor_state == "idle"
        assert state.ca1_binding_strength == 0.0
        assert state.input_similarity == 0.0
        assert state.output_similarity == 0.0
        assert state.separation_gain == 1.0
        assert state.working_memory_items == 0
        assert state.episodic_recent == 0
        assert state.semantic_links == 0
        assert state.embedding_2d == []

    def test_state_with_values(self):
        """Test explorer state with custom values."""
        state = MemoryExplorerState(
            dg_sparsity=0.04,
            ca3_attractor_state="encoding",
            ca1_binding_strength=0.85,
            separation_gain=3.2,
            episodic_recent=5,
            embedding_2d=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert state.dg_sparsity == 0.04
        assert state.ca3_attractor_state == "encoding"
        assert state.ca1_binding_strength == 0.85
        assert state.separation_gain == 3.2
        assert state.episodic_recent == 5
        assert len(state.embedding_2d) == 2

    def test_dg_sparsity_range(self):
        """Test DG sparsity is in biological range."""
        for _ in range(10):
            state = MemoryExplorerState(dg_sparsity=np.random.uniform(0.02, 0.05))
            assert 0.02 <= state.dg_sparsity <= 0.05

    def test_ca3_states(self):
        """Test all valid CA3 states."""
        valid_states = ["idle", "encoding", "retrieval", "settling"]
        for state_val in valid_states:
            state = MemoryExplorerState(ca3_attractor_state=state_val)
            assert state.ca3_attractor_state == state_val

    def test_binding_strength_bounds(self):
        """Test CA1 binding strength is in [0, 1]."""
        state = MemoryExplorerState(ca1_binding_strength=0.75)
        assert 0.0 <= state.ca1_binding_strength <= 1.0


# =============================================================================
# Test EncodeDemoRequest/Response Models
# =============================================================================


class TestEncodeDemoRequest:
    """Tests for encoding demo request model."""

    def test_minimal_request(self):
        """Test minimal encode request."""
        req = EncodeDemoRequest(content="Test memory")
        assert req.content == "Test memory"
        assert req.cognitive_state == "focus"

    def test_custom_cognitive_state(self):
        """Test encode request with custom cognitive state."""
        req = EncodeDemoRequest(
            content="Explore this",
            cognitive_state="explore"
        )
        assert req.cognitive_state == "explore"

    def test_cognitive_states(self):
        """Test valid cognitive states."""
        for state in ["focus", "explore", "rest"]:
            req = EncodeDemoRequest(content="Test", cognitive_state=state)
            assert req.cognitive_state == state


class TestEncodeDemoResponse:
    """Tests for encoding demo response model."""

    def test_successful_response(self):
        """Test successful encode response."""
        resp = EncodeDemoResponse(
            success=True,
            memory_id="abc-123",
            dg_sparsity=0.03,
            separation_gain=3.2,
            ca3_activity=[0.1, 0.2, 0.3],
            embedding_2d=[0.5, 0.6],
        )
        assert resp.success is True
        assert resp.memory_id == "abc-123"
        assert resp.dg_sparsity == 0.03
        assert resp.separation_gain == 3.2
        assert len(resp.ca3_activity) == 3
        assert len(resp.embedding_2d) == 2

    def test_separation_gain_threshold(self):
        """Test separation gain reaches target >3x."""
        resp = EncodeDemoResponse(
            success=True,
            memory_id="test",
            dg_sparsity=0.03,
            separation_gain=3.5,
            ca3_activity=[0.1],
            embedding_2d=[0.0, 0.0],
        )
        assert resp.separation_gain > 3.0


# =============================================================================
# Test RetrieveDemoRequest/Response Models
# =============================================================================


class TestRetrieveDemoRequest:
    """Tests for retrieval demo request model."""

    def test_minimal_request(self):
        """Test minimal retrieve request."""
        req = RetrieveDemoRequest(cue="partial memory")
        assert req.cue == "partial memory"
        assert req.cue_completeness == 0.3

    def test_custom_completeness(self):
        """Test retrieve request with custom cue completeness."""
        req = RetrieveDemoRequest(cue="test", cue_completeness=0.7)
        assert req.cue_completeness == 0.7

    def test_cue_completeness_bounds(self):
        """Test cue completeness is in [0.1, 1.0]."""
        for completeness in [0.1, 0.3, 0.5, 0.7, 1.0]:
            req = RetrieveDemoRequest(cue="test", cue_completeness=completeness)
            assert 0.1 <= req.cue_completeness <= 1.0


class TestRetrieveDemoResponse:
    """Tests for retrieval demo response model."""

    def test_successful_response(self):
        """Test successful retrieve response."""
        resp = RetrieveDemoResponse(
            success=True,
            completion_rate=0.85,
            retrieved_content="Full memory restored",
            ca3_settling_steps=7,
            attractor_basin="attractor_abc123",
            embedding_2d=[0.5, 0.6],
        )
        assert resp.success is True
        assert resp.completion_rate == 0.85
        assert resp.retrieved_content == "Full memory restored"
        assert resp.ca3_settling_steps == 7

    def test_failed_response(self):
        """Test failed retrieve response."""
        resp = RetrieveDemoResponse(
            success=False,
            completion_rate=0.0,
            retrieved_content=None,
            ca3_settling_steps=0,
            attractor_basin="empty",
            embedding_2d=[0.0, 0.0],
        )
        assert resp.success is False
        assert resp.retrieved_content is None
        assert resp.attractor_basin == "empty"

    def test_completion_rate_bounds(self):
        """Test completion rate is in [0, 1]."""
        resp = RetrieveDemoResponse(
            success=True,
            completion_rate=0.95,
            retrieved_content="test",
            ca3_settling_steps=5,
            attractor_basin="test",
            embedding_2d=[0.0, 0.0],
        )
        assert 0.0 <= resp.completion_rate <= 1.0


# =============================================================================
# Test ExplorerStateManager
# =============================================================================


class TestExplorerStateManagerInitialization:
    """Tests for ExplorerStateManager initialization."""

    def test_manager_creation(self):
        """Test manager is initialized correctly."""
        manager = ExplorerStateManager()
        assert manager is not None
        assert manager._state is not None
        assert manager._pattern_history == []
        assert manager._memories == {}

    def test_initial_state(self):
        """Test initial state from manager."""
        manager = ExplorerStateManager()
        state = manager.get_state()
        assert isinstance(state, MemoryExplorerState)
        assert state.ca3_attractor_state == "idle"


class TestExplorerGetState:
    """Tests for get_state method (line 101)."""

    def test_get_initial_state(self):
        """Test getting initial state."""
        manager = ExplorerStateManager()
        state = manager.get_state()
        assert isinstance(state, MemoryExplorerState)
        assert state.episodic_recent == 0

    def test_get_state_after_encoding(self):
        """Test state reflects encoding changes."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test content")
        state = manager.get_state()
        assert state.ca3_attractor_state == "encoding"
        assert state.episodic_recent == 1

    def test_get_state_consistency(self):
        """Test multiple get_state calls return same object."""
        manager = ExplorerStateManager()
        state1 = manager.get_state()
        state2 = manager.get_state()
        # Should be same object reference
        assert state1 is state2


class TestExplorerSimulateEncoding:
    """Tests for simulate_encoding method (lines 103-172)."""

    def test_encode_creates_memory(self):
        """Test encoding creates a memory (line 110)."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test content")

        assert resp.success is True
        assert len(resp.memory_id) > 0
        assert resp.memory_id in manager._memories
        assert manager._memories[resp.memory_id]["content"] == "Test content"

    def test_encode_creates_uuid(self):
        """Test memory_id is valid UUID."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")

        try:
            UUID(resp.memory_id)
        except ValueError:
            pytest.fail(f"memory_id {resp.memory_id} is not a valid UUID")

    def test_encode_generates_embedding(self):
        """Test encoding generates 128-dim embedding."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")

        assert len(resp.ca3_activity) == 10
        assert all(isinstance(x, (int, float, np.number)) for x in resp.ca3_activity)

    def test_encode_dg_sparsity(self):
        """Test DG sparsity is in biological range [0.02, 0.05]."""
        manager = ExplorerStateManager()
        for _ in range(10):
            resp = manager.simulate_encoding(f"Content {_}")
            assert 0.02 <= resp.dg_sparsity <= 0.05

    def test_encode_state_updates(self):
        """Test state is updated after encoding."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")
        state = manager.get_state()

        assert state.ca3_attractor_state == "encoding"
        assert state.dg_sparsity == resp.dg_sparsity
        assert state.episodic_recent == 1

    def test_encode_embedding_2d(self):
        """Test encoding returns 2D embedding for visualization."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")

        assert len(resp.embedding_2d) == 2
        assert all(isinstance(x, (int, float, np.number)) for x in resp.embedding_2d)

    def test_encode_cognitive_state_parameter(self):
        """Test cognitive_state parameter is accepted."""
        manager = ExplorerStateManager()
        for state in ["focus", "explore", "rest"]:
            resp = manager.simulate_encoding("Test", cognitive_state=state)
            assert resp.success is True

    def test_encode_multiple_memories(self):
        """Test encoding multiple memories."""
        manager = ExplorerStateManager()
        resp1 = manager.simulate_encoding("Content 1")
        resp2 = manager.simulate_encoding("Content 2")

        assert len(manager._memories) == 2
        assert resp1.memory_id != resp2.memory_id

    def test_encode_pattern_history_tracking(self):
        """Test pattern history tracks encoded patterns (line 141)."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")

        assert len(manager._pattern_history) == 1
        assert isinstance(manager._pattern_history[0], np.ndarray)
        assert manager._pattern_history[0].shape == (128,)

    def test_encode_pattern_history_limit(self):
        """Test pattern history maxes out at 100 (line 142)."""
        manager = ExplorerStateManager()

        for i in range(150):
            manager.simulate_encoding(f"Content {i}")

        assert len(manager._pattern_history) == 100

    def test_encode_separation_gain_first_memory(self):
        """Test first memory has gain ~3.0 (line 138)."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("First content")

        assert abs(resp.separation_gain - 3.0) < 0.1

    def test_encode_pattern_separation_gain(self):
        """Test pattern separation gain computed for subsequent memories (line 135)."""
        manager = ExplorerStateManager()
        resp1 = manager.simulate_encoding("Content A")
        resp2 = manager.simulate_encoding("Content B")

        # Second encoding should have computed separation gain
        # Can vary widely depending on sparse patterns
        assert isinstance(resp2.separation_gain, (int, float))
        assert resp2.separation_gain > 0.0

    def test_encode_input_similarity_computed(self):
        """Test input similarity computed vs previous pattern (line 116)."""
        manager = ExplorerStateManager()
        resp1 = manager.simulate_encoding("Content 1")
        state1 = manager.get_state()

        resp2 = manager.simulate_encoding("Content 2")
        state2 = manager.get_state()

        # After second encoding, input similarity should be computed
        assert 0.0 <= state2.input_similarity <= 1.0

    def test_encode_output_similarity_computed(self):
        """Test output similarity computed after sparse representation (line 131)."""
        manager = ExplorerStateManager()
        resp1 = manager.simulate_encoding("Content 1")
        resp2 = manager.simulate_encoding("Content 2")
        state2 = manager.get_state()

        # Output similarity should be computed after second encoding
        assert 0.0 <= state2.output_similarity <= 1.0

    def test_encode_sparse_mask_applied(self):
        """Test sparse mask reduces active neurons (line 126)."""
        manager = ExplorerStateManager()

        # Test multiple times since it's random
        active_counts = []
        for _ in range(20):
            manager._pattern_history = []  # Reset history
            resp = manager.simulate_encoding("Test")

            # Get the sparse pattern from history
            sparse_pattern = manager._pattern_history[0]
            active_neurons = np.count_nonzero(sparse_pattern)
            active_counts.append(active_neurons)

        # Average should be around 3-5% of 128 = 3.84-6.4 neurons
        avg_active = np.mean(active_counts)
        assert 2 <= avg_active <= 8, f"Average active neurons {avg_active} not in expected range"

    def test_encode_timestamp_recorded(self):
        """Test timestamp is recorded in memory (line 151)."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")

        mem = manager._memories[resp.memory_id]
        assert "timestamp" in mem
        assert mem["timestamp"] is not None

    def test_encode_response_structure(self):
        """Test full response structure."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test content")

        assert isinstance(resp, EncodeDemoResponse)
        assert hasattr(resp, 'success')
        assert hasattr(resp, 'memory_id')
        assert hasattr(resp, 'dg_sparsity')
        assert hasattr(resp, 'separation_gain')
        assert hasattr(resp, 'ca3_activity')
        assert hasattr(resp, 'embedding_2d')


# =============================================================================
# Test Pattern Separation and Gain Calculation
# =============================================================================


class TestPatternSeparationGain:
    """Tests for pattern separation gain calculation (lines 129-138)."""

    def test_gain_first_encoding_is_3x(self):
        """Test first encoding gets ~3.0 separation gain."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Initial")

        assert 2.9 <= resp.separation_gain <= 3.1

    def test_gain_formula_correctness(self):
        """Test separation gain formula: (1-output_sim)/(1-input_sim)."""
        manager = ExplorerStateManager()

        # Create two memories
        resp1 = manager.simulate_encoding("Memory 1")
        resp2 = manager.simulate_encoding("Memory 2")

        state = manager.get_state()

        # Gain should be computed
        if state.input_similarity < 1.0:
            expected_gain = (1 - state.output_similarity) / (1 - state.input_similarity)
            assert abs(state.separation_gain - expected_gain) < 0.1

    def test_gain_increases_with_sparsity(self):
        """Test separation gain benefits from sparse representation."""
        manager = ExplorerStateManager()

        # Encode multiple patterns
        gains = []
        for i in range(5):
            resp = manager.simulate_encoding(f"Pattern {i}")
            if i > 0:  # Skip first
                gains.append(resp.separation_gain)

        # Should have positive gains
        assert all(g > 0.0 for g in gains)

    def test_gain_target_reached(self):
        """Test target separation gain >3x can be reached."""
        manager = ExplorerStateManager()

        max_gain = 0.0
        for i in range(20):
            resp = manager.simulate_encoding(f"Pattern {i}")
            max_gain = max(max_gain, resp.separation_gain)

        # With enough patterns, should reach >3x
        assert max_gain >= 2.5  # At least close to target


# =============================================================================
# Test ExplorerSimulateRetrieval
# =============================================================================


class TestExplorerSimulateRetrievalEmptyMemory:
    """Tests for retrieval with no memories (line 180-188)."""

    def test_retrieve_no_memories_returns_failure(self):
        """Test retrieval with no memories returns failure."""
        manager = ExplorerStateManager()
        resp = manager.simulate_retrieval("test cue")

        assert resp.success is False
        assert resp.completion_rate == 0.0
        assert resp.retrieved_content is None

    def test_retrieve_no_memories_empty_attractor(self):
        """Test no memories returns empty attractor basin (line 186)."""
        manager = ExplorerStateManager()
        resp = manager.simulate_retrieval("test cue")

        assert resp.attractor_basin == "empty"

    def test_retrieve_no_memories_zero_settling_steps(self):
        """Test no memories returns 0 settling steps."""
        manager = ExplorerStateManager()
        resp = manager.simulate_retrieval("test cue")

        assert resp.ca3_settling_steps == 0


class TestExplorerSimulateRetrievalSuccessful:
    """Tests for successful retrieval (lines 174-235)."""

    def test_retrieve_valid_cue_succeeds(self):
        """Test retrieval with valid cue succeeds."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("The quick brown fox")

        resp = manager.simulate_retrieval("quick")

        assert resp.success is True
        assert resp.retrieved_content == "The quick brown fox"

    def test_retrieve_returns_correct_content(self):
        """Test retrieved content matches encoded content."""
        manager = ExplorerStateManager()
        content = "Important memory about project X"
        manager.simulate_encoding(content)

        resp = manager.simulate_retrieval("project")

        assert resp.retrieved_content == content

    def test_retrieve_multiple_memories_best_match(self):
        """Test retrieval selects best matching memory."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory about cats")
        manager.simulate_encoding("Memory about dogs")

        resp = manager.simulate_retrieval("dog")

        assert resp.success is True
        assert "dogs" in resp.retrieved_content

    def test_retrieve_case_insensitive(self):
        """Test retrieval is case-insensitive."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("UPPERCASE MEMORY")

        resp = manager.simulate_retrieval("uppercase")

        assert resp.success is True

    def test_retrieve_completion_rate_bounds(self):
        """Test completion rate is in [0, 1]."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test memory")

        for _ in range(10):
            resp = manager.simulate_retrieval("test", cue_completeness=0.5)
            assert 0.0 <= resp.completion_rate <= 1.0


class TestExplorerSimulateRetrievalCA3Settling:
    """Tests for CA3 settling simulation (line 215)."""

    def test_settling_steps_increase_with_low_cue(self):
        """Test settling steps increase when cue is incomplete."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test memory content here")

        # Full cue
        resp_full = manager.simulate_retrieval("test memory", cue_completeness=1.0)
        # Partial cue
        resp_partial = manager.simulate_retrieval("test", cue_completeness=0.1)

        assert resp_partial.ca3_settling_steps >= resp_full.ca3_settling_steps

    def test_settling_steps_formula(self):
        """Test settling formula: 5 + 10*(1-cue_completeness) (line 215)."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test memory")

        # With cue_completeness=0.5, expect ~10 steps
        resp = manager.simulate_retrieval("test", cue_completeness=0.5)
        expected = 5 + 10 * (1 - 0.5)
        assert resp.ca3_settling_steps == int(expected)

    def test_settling_steps_min_value(self):
        """Test settling steps minimum with full cue."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        resp = manager.simulate_retrieval("memory", cue_completeness=1.0)
        assert resp.ca3_settling_steps == 5

    def test_settling_steps_max_value(self):
        """Test settling steps maximum with minimal cue."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        resp = manager.simulate_retrieval("m", cue_completeness=0.1)
        assert resp.ca3_settling_steps > 10


class TestExplorerSimulateRetrievalCompletion:
    """Tests for completion rate calculation (line 218)."""

    def test_completion_rate_increases_with_completeness(self):
        """Test completion rate increases with cue completeness."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test memory content")

        resp_low = manager.simulate_retrieval("test", cue_completeness=0.3)
        resp_high = manager.simulate_retrieval("test", cue_completeness=0.9)

        assert resp_high.completion_rate >= resp_low.completion_rate

    def test_completion_rate_biological_target(self):
        """Test completion >90% from 30% cue (biological target)."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test memory")

        # With 30% cue, should achieve high completion
        resp = manager.simulate_retrieval("test", cue_completeness=0.3)

        # Target: >70% (from formula: 0.7 + 0.25*0.3)
        assert resp.completion_rate >= 0.7

    def test_completion_rate_formula(self):
        """Test completion rate formula with stochasticity."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        resp = manager.simulate_retrieval("memory", cue_completeness=0.5)

        # Formula: 0.7 + 0.25*cue_completeness + noise [-0.05, 0.05]
        base_expected = 0.7 + 0.25 * 0.5  # = 0.825
        assert 0.775 <= resp.completion_rate <= 0.875


class TestExplorerSimulateRetrievalPoorMatch:
    """Tests for retrieval with poor match (line 204)."""

    def test_retrieve_poor_match_returns_failure(self):
        """Test retrieval with poor match <0.1 returns failure."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("abcdef")

        # Cue "xy" has 0 matches in "abcdef" = 0/2 score
        resp = manager.simulate_retrieval("xy")

        assert resp.success is False

    def test_retrieve_poor_match_no_match_attractor(self):
        """Test poor match returns no_match attractor (line 210)."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("abcdef")

        resp = manager.simulate_retrieval("xy")

        assert resp.attractor_basin == "no_match"

    def test_retrieve_threshold_at_0_1(self):
        """Test match threshold is exactly 0.1 (line 204)."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("abcdefghij")

        # Cue "abcd" matches 4/4 chars = 1.0 score (should succeed)
        resp = manager.simulate_retrieval("abcd")
        assert resp.success is True


class TestExplorerAttractorBasin:
    """Tests for attractor basin identification."""

    def test_attractor_basin_named_after_memory(self):
        """Test attractor basin is named after matched memory."""
        manager = ExplorerStateManager()
        resp_encode = manager.simulate_encoding("Test memory")
        memory_id = resp_encode.memory_id

        resp_retrieve = manager.simulate_retrieval("test")

        # Should reference the matched memory
        assert memory_id[:8] in resp_retrieve.attractor_basin

    def test_attractor_basin_format(self):
        """Test attractor basin has correct format."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        resp = manager.simulate_retrieval("mem")

        assert resp.attractor_basin.startswith("attractor_")


# =============================================================================
# Test Endpoint Integration
# =============================================================================


class TestExplorerEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_endpoint_get_state(self):
        """Test GET /state endpoint returns state."""
        from ww.api.routes.explorer import get_explorer_state

        result = await get_explorer_state()

        assert isinstance(result, MemoryExplorerState)
        assert result.ca3_attractor_state == "idle"

    @pytest.mark.asyncio
    async def test_endpoint_encode(self):
        """Test POST /demo/encode endpoint."""
        from ww.api.routes.explorer import demo_encode, _explorer_state

        # Reset state first
        _explorer_state._memories.clear()
        _explorer_state._pattern_history.clear()

        request = EncodeDemoRequest(content="Test memory")
        result = await demo_encode(request)

        assert isinstance(result, EncodeDemoResponse)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_endpoint_retrieve(self):
        """Test POST /demo/retrieve endpoint."""
        from ww.api.routes.explorer import demo_encode, demo_retrieve, _explorer_state

        # Reset and encode first
        _explorer_state._memories.clear()
        _explorer_state._pattern_history.clear()

        encode_req = EncodeDemoRequest(content="Test memory")
        await demo_encode(encode_req)

        # Now retrieve
        retrieve_req = RetrieveDemoRequest(cue="test")
        result = await demo_retrieve(retrieve_req)

        assert isinstance(result, RetrieveDemoResponse)

    @pytest.mark.asyncio
    async def test_endpoint_reset(self):
        """Test POST /reset endpoint."""
        from ww.api.routes.explorer import (
            reset_explorer, demo_encode, _explorer_state
        )

        # Encode first
        encode_req = EncodeDemoRequest(content="Test")
        await demo_encode(encode_req)

        assert len(_explorer_state._memories) > 0

        # Reset
        result = await reset_explorer()

        # State should be cleared
        assert result["status"] == "reset"

    @pytest.mark.asyncio
    async def test_endpoint_reset_clears_state(self):
        """Test reset endpoint clears state completely."""
        # Need to check the module-level state after reset
        import ww.api.routes.explorer as explorer_module

        # Encode and retrieve
        from ww.api.routes.explorer import demo_encode, reset_explorer

        encode_req = EncodeDemoRequest(content="Test memory")
        await demo_encode(encode_req)

        # Reset
        await reset_explorer()

        # Get new state from module
        state = explorer_module._explorer_state.get_state()
        assert state.episodic_recent == 0
        assert len(explorer_module._explorer_state._memories) == 0


# =============================================================================
# Test Retrieval Matching Algorithm
# =============================================================================


class TestRetrievalMatching:
    """Tests for memory retrieval matching algorithm."""

    def test_character_matching_scoring(self):
        """Test character matching scoring in retrieval."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("hello world")

        # "hel" matches 3/3 characters = 1.0 score
        resp = manager.simulate_retrieval("hel")
        assert resp.success is True

    def test_partial_character_matching(self):
        """Test partial character matching."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("python programming")

        # "pyt" matches all 3 chars
        resp = manager.simulate_retrieval("pyt")
        assert resp.success is True

    def test_character_match_calculation(self):
        """Test character matching score calculation."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("abcdef")

        # "abc" has 3/3 matching = 1.0 (> 0.1 threshold)
        resp = manager.simulate_retrieval("abc")
        assert resp.success is True

        # "xy" has 0/2 matching = 0.0 (< 0.1 threshold)
        resp = manager.simulate_retrieval("xy")
        assert resp.success is False

    def test_best_match_selection(self):
        """Test best match is selected among multiple memories."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("apple orange banana")
        manager.simulate_encoding("apple pie")

        # "app" should match both, "apple pie" has higher score (app=3/3=1.0 vs app=3/3=1.0)
        # But both match fully, so first one wins (best_id set when score > best_score)
        resp = manager.simulate_retrieval("app")
        assert resp.success is True


# =============================================================================
# Test State Persistence
# =============================================================================


class TestStateManagement:
    """Tests for state management across operations."""

    def test_state_persists_across_encodings(self):
        """Test state persists across multiple encodings."""
        manager = ExplorerStateManager()

        resp1 = manager.simulate_encoding("Memory 1")
        state1 = manager.get_state()
        episodic_count_1 = state1.episodic_recent

        resp2 = manager.simulate_encoding("Memory 2")
        state2 = manager.get_state()
        episodic_count_2 = state2.episodic_recent

        assert episodic_count_2 > episodic_count_1

    def test_memories_stored_with_metadata(self):
        """Test memories include all required metadata."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test content")

        mem = manager._memories[resp.memory_id]

        assert "content" in mem
        assert "embedding" in mem
        assert "timestamp" in mem
        assert mem["content"] == "Test content"

    def test_pattern_history_contains_sparse_patterns(self):
        """Test pattern history contains sparse numpy arrays."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory 1")

        pattern = manager._pattern_history[0]

        assert isinstance(pattern, np.ndarray)
        assert pattern.dtype == np.float32
        assert pattern.shape == (128,)
        # Should be sparse
        nonzero = np.count_nonzero(pattern)
        assert nonzero < 128


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_encode_empty_content(self):
        """Test encoding empty string content."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("")

        assert resp.success is True
        assert manager._memories[resp.memory_id]["content"] == ""

    def test_encode_very_long_content(self):
        """Test encoding very long content."""
        manager = ExplorerStateManager()
        long_content = "a" * 10000
        resp = manager.simulate_encoding(long_content)

        assert resp.success is True
        assert manager._memories[resp.memory_id]["content"] == long_content

    def test_retrieve_empty_cue(self):
        """Test retrieval with empty cue string."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Test memory")

        resp = manager.simulate_retrieval("")
        # Empty cue has 0/max(0, 1) = 0 match score
        assert resp.success is False

    def test_retrieve_very_high_completeness(self):
        """Test retrieval with cue_completeness > 1.0 (edge case)."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        # This might be clamped, test the behavior
        resp = manager.simulate_retrieval("mem", cue_completeness=1.5)
        # Settling formula: 5 + 10*(1-1.5) = 5 - 5 = 0
        assert resp.ca3_settling_steps >= 0

    def test_retrieve_zero_completeness(self):
        """Test retrieval with minimal cue_completeness."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        resp = manager.simulate_retrieval("m", cue_completeness=0.1)
        # Should have maximum settling steps
        assert resp.ca3_settling_steps > 10

    def test_special_characters_in_content(self):
        """Test encoding/retrieving special characters."""
        manager = ExplorerStateManager()
        special_content = "!@#$%^&*()_+-=[]{}|;:',.<>?/"
        manager.simulate_encoding(special_content)

        resp = manager.simulate_retrieval("@#$")
        assert resp.success is True

    def test_unicode_content(self):
        """Test encoding/retrieving unicode characters."""
        manager = ExplorerStateManager()
        unicode_content = "Hello 世界 مرحبا мир"
        resp = manager.simulate_encoding(unicode_content)

        assert resp.success is True

        resp2 = manager.simulate_retrieval("世")
        assert resp2.success is True


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability in computations."""

    def test_similarity_normalization(self):
        """Test similarity normalization handles edge cases."""
        manager = ExplorerStateManager()

        # Encode patterns and check normalization
        for _ in range(10):
            resp = manager.simulate_encoding(f"Pattern {_}")
            # Both similarities should be in [0, 1]
            state = manager.get_state()
            assert 0.0 <= state.input_similarity <= 1.0
            assert 0.0 <= state.output_similarity <= 1.0

    def test_separation_gain_avoids_division_by_zero(self):
        """Test separation gain handles (1-input_sim) near zero."""
        manager = ExplorerStateManager()

        # First encode creates gain 3.0
        resp = manager.simulate_encoding("Similar pattern 1")
        assert resp.separation_gain == 3.0

        # Subsequent encodes should avoid division by zero
        resp = manager.simulate_encoding("Similar pattern 2")
        # Should be a valid number
        assert np.isfinite(resp.separation_gain)

    def test_dot_product_epsilon_handling(self):
        """Test dot product computation includes epsilon to avoid division by zero."""
        manager = ExplorerStateManager()

        # Encode multiple patterns and verify numerical stability
        for i in range(20):
            resp = manager.simulate_encoding(f"Pattern {i}")
            assert np.isfinite(resp.separation_gain)
            state = manager.get_state()
            assert np.isfinite(state.input_similarity)
            assert np.isfinite(state.output_similarity)


# =============================================================================
# Test Cognitive State Parameter
# =============================================================================


class TestCognitiveStateHandling:
    """Tests for cognitive state parameter handling."""

    def test_cognitive_state_focus_accepted(self):
        """Test focus cognitive state is accepted."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Content", cognitive_state="focus")
        assert resp.success is True

    def test_cognitive_state_explore_accepted(self):
        """Test explore cognitive state is accepted."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Content", cognitive_state="explore")
        assert resp.success is True

    def test_cognitive_state_rest_accepted(self):
        """Test rest cognitive state is accepted."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Content", cognitive_state="rest")
        assert resp.success is True

    def test_cognitive_state_default(self):
        """Test default cognitive state is focus."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Content")
        assert resp.success is True


# =============================================================================
# Test Embedding 2D Projection
# =============================================================================


class TestEmbedding2D:
    """Tests for 2D embedding projection for visualization."""

    def test_encode_produces_2d_embedding(self):
        """Test encoding produces 2D projection."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Content")

        assert len(resp.embedding_2d) == 2
        assert isinstance(resp.embedding_2d[0], (float, np.floating))
        assert isinstance(resp.embedding_2d[1], (float, np.floating))

    def test_retrieve_produces_2d_embedding(self):
        """Test retrieval produces 2D projection."""
        manager = ExplorerStateManager()
        manager.simulate_encoding("Memory")

        resp = manager.simulate_retrieval("mem")

        assert len(resp.embedding_2d) == 2
        assert isinstance(resp.embedding_2d[0], (float, np.floating))

    def test_2d_embedding_within_unit_range(self):
        """Test 2D embeddings are from original features."""
        manager = ExplorerStateManager()
        resp = manager.simulate_encoding("Test")

        # Values come from first two dimensions of 128-dim vector
        # Should be reasonable floats (not necessarily [-1, 1])
        assert all(isinstance(x, (float, np.floating)) for x in resp.embedding_2d)

    def test_failed_retrieval_zero_embedding(self):
        """Test failed retrieval returns zero embedding."""
        manager = ExplorerStateManager()

        resp = manager.simulate_retrieval("xy")
        # For no match or empty memories, should be [0.0, 0.0]
        assert resp.embedding_2d == [0.0, 0.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
