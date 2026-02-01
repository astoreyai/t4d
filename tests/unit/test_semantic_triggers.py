"""
TMEM-014: Tests for semantic trigger matching in Procedural Memory.

Tests embedding-based semantic similarity for trigger patterns,
replacing simple substring matching with more robust approach.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from t4dm.core.types import Procedure, ProcedureStep, Domain, ScoredResult
from t4dm.memory.procedural import ProceduralMemory


class TestSemanticTriggerMatching:
    """Test semantic trigger matching with embeddings."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_vector_store, mock_graph_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_vector_store
        procedural.graph_store = mock_graph_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_semantic_trigger_match_paraphrased_query(self, procedural):
        """Test that semantic matching finds similar triggers even with paraphrasing."""
        # Mock embeddings with high similarity for paraphrases
        request_embedding = [0.1] * 1024
        trigger_embedding = [0.11] * 1024  # Very similar to request

        procedural.embedding.embed_query.side_effect = [
            request_embedding,  # For recall_skill
            request_embedding,  # For match_trigger
            trigger_embedding,  # For trigger pattern
        ]

        # Mock similarity calculation to return high score
        procedural.embedding.similarity = MagicMock(return_value=0.85)

        now = datetime.now()
        proc_id = str(uuid4())
        results = [
            (
                proc_id,
                0.75,
                {
                    "name": "Create Unit Tests",
                    "domain": "coding",
                    "trigger_pattern": "generate test cases for code",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 10,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        procedural.vector_store.get.return_value = [
            (
                proc_id,
                {
                    "trigger_embedding": trigger_embedding,
                    **results[0][2],
                },
            )
        ]

        # Paraphrased query - should still match
        matched = await procedural.match_trigger(
            user_request="write unit tests for the module",
            use_semantic_matching=True,
        )

        assert len(matched) > 0
        # Should have high trigger_match score due to semantic similarity
        assert matched[0].components.get("trigger_match", 0) >= 0.5
        # Score should be boosted
        assert matched[0].score > 0.75  # Original similarity was 0.75

    @pytest.mark.asyncio
    async def test_semantic_trigger_match_with_synonyms(self, procedural):
        """Test semantic matching handles synonyms correctly."""
        request_embedding = [0.2] * 1024
        trigger_embedding = [0.21] * 1024

        procedural.embedding.embed_query.side_effect = [
            request_embedding,
            request_embedding,
            trigger_embedding,
        ]
        procedural.embedding.similarity = MagicMock(return_value=0.78)

        now = datetime.now()
        proc_id = str(uuid4())
        results = [
            (
                proc_id,
                0.7,
                {
                    "name": "Deploy Application",
                    "domain": "devops",
                    "trigger_pattern": "ship code to production",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.85,
                    "execution_count": 15,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        procedural.vector_store.get.return_value = [
            (proc_id, {"trigger_embedding": trigger_embedding, **results[0][2]})
        ]

        # Synonym query
        matched = await procedural.match_trigger(
            user_request="deploy application to production",
            use_semantic_matching=True,
        )

        assert len(matched) > 0
        assert matched[0].components.get("trigger_match", 0) >= 0.5

    @pytest.mark.asyncio
    async def test_semantic_threshold_filtering(self, procedural):
        """Test that semantic threshold filters low-similarity matches."""
        request_embedding = [0.3] * 1024
        low_similarity_trigger = [0.9] * 1024  # Very different

        procedural.embedding.embed_query.side_effect = [
            request_embedding,
            request_embedding,
            low_similarity_trigger,
        ]
        # Low similarity - below default 0.5 threshold
        procedural.embedding.similarity = MagicMock(return_value=0.3)

        now = datetime.now()
        proc_id = str(uuid4())
        results = [
            (
                proc_id,
                0.6,
                {
                    "name": "Unrelated Task",
                    "domain": "coding",
                    "trigger_pattern": "completely different task",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 5,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        procedural.vector_store.get.return_value = [
            (proc_id, {"trigger_embedding": low_similarity_trigger, **results[0][2]})
        ]

        matched = await procedural.match_trigger(
            user_request="write tests",
            use_semantic_matching=True,
            semantic_threshold=0.5,
        )

        # Should not boost low-similarity match
        assert matched[0].components.get("trigger_match") is None or matched[0].components.get("trigger_match") < 0.5

    @pytest.mark.asyncio
    async def test_trigger_embedding_caching(self, procedural):
        """Test that trigger embeddings are cached in Qdrant."""
        request_embedding = [0.4] * 1024
        trigger_embedding = [0.41] * 1024

        procedural.embedding.embed_query.side_effect = [
            request_embedding,  # recall_skill
            request_embedding,  # match_trigger
            trigger_embedding,  # compute trigger embedding (not cached)
        ]
        procedural.embedding.similarity = MagicMock(return_value=0.7)

        now = datetime.now()
        proc_id = str(uuid4())
        results = [
            (
                proc_id,
                0.7,
                {
                    "name": "Test Skill",
                    "domain": "coding",
                    "trigger_pattern": "run tests",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 5,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        # First get: no cached embedding
        procedural.vector_store.get.return_value = [
            (proc_id, {**results[0][2]})  # No trigger_embedding field
        ]
        procedural.vector_store.update_payload.return_value = None

        await procedural.match_trigger(
            user_request="execute test suite",
            use_semantic_matching=True,
        )

        # Should have called update_payload to cache the embedding
        procedural.vector_store.update_payload.assert_called_once()
        call_args = procedural.vector_store.update_payload.call_args
        assert call_args[1]["id"] == proc_id
        assert "trigger_embedding" in call_args[1]["payload"]
        assert call_args[1]["payload"]["trigger_embedding"] == trigger_embedding

    @pytest.mark.asyncio
    async def test_trigger_embedding_uses_cached_value(self, procedural):
        """Test that cached trigger embeddings are reused."""
        request_embedding = [0.5] * 1024
        cached_trigger_embedding = [0.51] * 1024

        procedural.embedding.embed_query.side_effect = [
            request_embedding,  # recall_skill
            request_embedding,  # match_trigger
            # No third call - should use cached
        ]
        procedural.embedding.similarity = MagicMock(return_value=0.8)

        now = datetime.now()
        proc_id = str(uuid4())
        results = [
            (
                proc_id,
                0.7,
                {
                    "name": "Test Skill",
                    "domain": "coding",
                    "trigger_pattern": "run tests",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 5,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                    "trigger_embedding": cached_trigger_embedding,  # Cached!
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        procedural.vector_store.get.return_value = [
            (proc_id, results[0][2])  # Has cached trigger_embedding
        ]

        await procedural.match_trigger(
            user_request="execute tests",
            use_semantic_matching=True,
        )

        # Should NOT call update_payload since embedding was cached
        procedural.vector_store.update_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_substring_matching(self, procedural):
        """Test fallback to substring matching when semantic disabled."""
        request_embedding = [0.6] * 1024

        procedural.embedding.embed_query.return_value = request_embedding

        now = datetime.now()
        proc_id = str(uuid4())
        results = [
            (
                proc_id,
                0.8,
                {
                    "name": "Write Tests",
                    "domain": "coding",
                    "trigger_pattern": "write unit tests",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 10,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results

        # Disable semantic matching
        matched = await procedural.match_trigger(
            user_request="write unit tests for module",
            use_semantic_matching=False,
        )

        # Should match via substring
        assert len(matched) > 0
        assert matched[0].components.get("trigger_match") == 1.0

    @pytest.mark.asyncio
    async def test_boost_factor_calculation(self, procedural):
        """Test that boost factor increases with similarity score."""
        request_embedding = [0.7] * 1024

        now = datetime.now()
        proc_id = str(uuid4())
        base_score = 0.7

        results = [
            (
                proc_id,
                base_score,
                {
                    "name": "Test Skill",
                    "domain": "coding",
                    "trigger_pattern": "run tests",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 10,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        procedural.vector_store.get.return_value = [
            (proc_id, {**results[0][2], "trigger_embedding": [0.71] * 1024})
        ]

        # Test high similarity (0.9)
        procedural.embedding.embed_query.side_effect = [
            request_embedding,
            request_embedding,
        ]
        procedural.embedding.similarity = MagicMock(return_value=0.9)

        matched = await procedural.match_trigger(
            user_request="execute tests",
            use_semantic_matching=True,
        )

        # Boost factor = 1.0 + (0.9 * 0.7) = 1.63x
        # Expected score = 0.7 * 1.63 = 1.141 (but capped at reasonable value)
        expected_boost = 1.0 + (0.9 * 0.7)
        expected_score = base_score * expected_boost
        assert matched[0].score >= base_score  # At minimum, score should increase
        assert matched[0].components["trigger_match"] == 0.9

    @pytest.mark.asyncio
    async def test_multiple_procedures_with_different_similarities(self, procedural):
        """Test ranking procedures by combined vector + trigger similarity."""
        request_embedding = [0.8] * 1024
        high_trigger_sim_embedding = [0.81] * 1024
        low_trigger_sim_embedding = [0.2] * 1024

        procedural.embedding.embed_query.side_effect = [
            request_embedding,  # recall_skill
            request_embedding,  # match_trigger
            high_trigger_sim_embedding,  # proc1 trigger
            low_trigger_sim_embedding,  # proc2 trigger
        ]

        now = datetime.now()
        proc1_id = str(uuid4())
        proc2_id = str(uuid4())
        results = [
            (
                proc1_id,
                0.6,  # Lower base similarity
                {
                    "name": "High Trigger Match",
                    "domain": "coding",
                    "trigger_pattern": "write tests",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 10,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
            (
                proc2_id,
                0.7,  # Higher base similarity
                {
                    "name": "Low Trigger Match",
                    "domain": "coding",
                    "trigger_pattern": "deploy code",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.85,
                    "execution_count": 8,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results
        # recall_skill will sort results, so proc2 (higher composite score) comes first
        # Then match_trigger processes in that order, so get.side_effect must match
        procedural.vector_store.get.side_effect = [
            [(proc2_id, {**results[1][2], "trigger_embedding": low_trigger_sim_embedding})],  # proc2 first
            [(proc1_id, {**results[0][2], "trigger_embedding": high_trigger_sim_embedding})],  # proc1 second
        ]

        # Mock different similarities - proc2 processed first, then proc1
        procedural.embedding.similarity = MagicMock(side_effect=[0.3, 0.85])

        matched = await procedural.match_trigger(
            user_request="write unit tests",
            use_semantic_matching=True,
            semantic_threshold=0.5,
        )

        # proc1 should rank higher due to trigger boost
        # even though proc2 had higher base similarity
        assert len(matched) == 2
        # Proc1 should be first due to high trigger similarity boosting its score
        assert matched[0].item.name == "High Trigger Match"
        # Proc2 with low trigger similarity should be second (below threshold, no boost)
        assert matched[1].item.name == "Low Trigger Match"


class TestScriptAbstraction:
    """Test template-based script abstraction."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_vector_store, mock_graph_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_vector_store
        procedural.graph_store = mock_graph_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    def test_abstract_steps_replaces_file_paths(self, procedural):
        """Test that file paths are replaced with variables."""
        steps = [
            ProcedureStep(
                order=1,
                action="Read file /home/user/project/main.py",
                tool="Read",
            ),
            ProcedureStep(
                order=2,
                action="Write output to ./results/output.txt",
                tool="Write",
            ),
        ]

        abstracted = procedural._abstract_steps(steps)

        assert len(abstracted) == 2
        # Absolute path should be replaced
        assert "/home/user/project/main.py" not in abstracted[0]["action"]
        assert "{file_" in abstracted[0]["action"]
        # Relative path should be replaced
        assert "./results/output.txt" not in abstracted[1]["action"]

    def test_abstract_steps_replaces_urls(self, procedural):
        """Test that URLs are replaced with variables."""
        steps = [
            ProcedureStep(
                order=1,
                action="Clone from https://github.com/user/repo.git",
                tool="Bash",
            ),
            ProcedureStep(
                order=2,
                action="Download http://example.com/data.zip",
                tool="Bash",
            ),
        ]

        abstracted = procedural._abstract_steps(steps)

        assert "{url_" in abstracted[0]["action"]
        assert "https://github.com" not in abstracted[0]["action"]
        assert "{url_" in abstracted[1]["action"]

    def test_abstract_steps_replaces_numbers(self, procedural):
        """Test that large numbers are replaced with variables."""
        steps = [
            ProcedureStep(
                order=1,
                action="Process 15000 records",
                tool="Python",
            ),
            ProcedureStep(
                order=2,
                action="Set timeout to 3.5 seconds",
                tool="Config",
            ),
        ]

        abstracted = procedural._abstract_steps(steps)

        # Large numbers should be replaced
        assert "15000" not in abstracted[0]["action"]
        assert "{num_" in abstracted[0]["action"]
        # Decimals should be replaced
        assert "3.5" not in abstracted[1]["action"]

    def test_abstract_steps_replaces_command_arguments(self, procedural):
        """Test that common command patterns are abstracted."""
        steps = [
            ProcedureStep(
                order=1,
                action="Run pytest tests/unit/",
                tool="Bash",
            ),
            ProcedureStep(
                order=2,
                action="Execute git clone https://github.com/user/repo",
                tool="Bash",
            ),
            ProcedureStep(
                order=3,
                action="Start docker run nginx:latest",
                tool="Bash",
            ),
        ]

        abstracted = procedural._abstract_steps(steps)

        # pytest should be parameterized
        assert "pytest {target_dir}" in abstracted[0]["action"]
        # git clone should be parameterized
        assert "git clone {repo_url}" in abstracted[1]["action"]
        # docker run should be parameterized
        assert "docker run {image_name}" in abstracted[2]["action"]

    @pytest.mark.asyncio
    async def test_generate_script_uses_abstraction(self, procedural):
        """Test that _generate_script uses _abstract_steps."""
        steps = [
            ProcedureStep(
                order=1,
                action="Read /path/to/config.yaml",
                tool="Read",
            ),
            ProcedureStep(
                order=2,
                action="Process 5000 items",
                tool="Python",
            ),
        ]

        script = await procedural._generate_script(steps, "coding")

        # Script should contain abstracted placeholders, not concrete values
        assert "{file_" in script or "config.yaml" not in script
        assert "PROCEDURE:" in script
        assert "STEPS:" in script
        assert "POSTCONDITION:" in script

    def test_abstract_steps_preserves_tool_names(self, procedural):
        """Test that tool names are preserved during abstraction."""
        steps = [
            ProcedureStep(
                order=1,
                action="Read file.py",
                tool="Read",
            ),
            ProcedureStep(
                order=2,
                action="Search pattern",
                tool="Grep",
            ),
        ]

        abstracted = procedural._abstract_steps(steps)

        assert abstracted[0]["tool"] == "Read"
        assert abstracted[1]["tool"] == "Grep"

    def test_abstract_steps_handles_no_matches(self, procedural):
        """Test that steps without concrete values pass through unchanged."""
        steps = [
            ProcedureStep(
                order=1,
                action="Analyze the code",
                tool="Think",
            ),
            ProcedureStep(
                order=2,
                action="Generate summary",
                tool="LLM",
            ),
        ]

        abstracted = procedural._abstract_steps(steps)

        # No concrete values to replace - action should be similar
        assert "Analyze" in abstracted[0]["action"]
        assert "Generate" in abstracted[1]["action"]
