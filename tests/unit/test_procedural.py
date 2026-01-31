"""
TMEM-004: Comprehensive tests for Procedural Memory Service.

Tests skill creation, retrieval by relevance, deprecation,
skill versioning, and the MEMP algorithm.
"""

import pytest
from datetime import datetime, timedelta
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from t4dm.core.types import (
    Procedure, ProcedureStep, Domain, ScoredResult
)
from t4dm.memory.procedural import ProceduralMemory


class TestProceduralSkillCreation:
    """Test skill building from trajectories."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_qdrant_store
        procedural.graph_store = mock_neo4j_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_build_skill_from_successful_trajectory(self, procedural):
        """Test building skill from successful trajectory."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding
        procedural.vector_store.add.return_value = None
        procedural.graph_store.create_node.return_value = "test-node-id"

        trajectory = [
            {
                "action": "Read file",
                "tool": "Read",
                "parameters": {"file_path": "/path/to/file.py"},
                "result": "File content loaded",
            },
            {
                "action": "Analyze code",
                "tool": "Grep",
                "parameters": {"pattern": "def test_"},
                "result": "Found 5 test functions",
            },
            {
                "action": "Generate test",
                "tool": "Write",
                "parameters": {"file_path": "/path/to/test.py"},
                "result": "Test file created",
            },
        ]

        procedure = await procedural.create_skill(
            trajectory=trajectory,
            outcome_score=0.95,
            domain="coding",
            trigger_pattern="When: test file needed",
            name="Generate Unit Tests",
        )

        assert procedure is not None
        assert procedure.name == "Generate Unit Tests"
        assert procedure.domain == Domain.CODING
        assert procedure.trigger_pattern == "When: test file needed"
        assert len(procedure.steps) == 3
        assert procedure.success_rate == 1.0
        assert procedure.execution_count == 1

        procedural.vector_store.add.assert_called_once()
        procedural.graph_store.create_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_skill_insufficient_outcome_score(self, procedural):
        """Test that low outcome score skips skill building."""
        trajectory = [
            {
                "action": "Do something",
                "tool": "Tool",
                "parameters": {},
                "result": "Failed",
            }
        ]

        procedure = await procedural.create_skill(
            trajectory=trajectory,
            outcome_score=0.5,  # Below 0.7 threshold
            domain="testing",
        )

        assert procedure is None
        procedural.vector_store.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_skill_auto_generate_name(self, procedural):
        """Test automatic name generation when not provided."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding
        procedural.vector_store.add.return_value = None
        procedural.graph_store.create_node.return_value = "test-node-id"

        trajectory = [
            {
                "action": "Implement feature X",
                "tool": "Write",
                "parameters": {},
                "result": "Complete",
            }
        ]

        procedure = await procedural.create_skill(
            trajectory=trajectory,
            outcome_score=0.9,
            domain="coding",
        )

        # Should auto-generate name
        assert procedure.name is not None
        assert "coding" in procedure.name.lower() or "Implement" in procedure.name

    @pytest.mark.asyncio
    async def test_build_skill_auto_infer_trigger(self, procedural):
        """Test automatic trigger pattern inference."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding
        procedural.vector_store.add.return_value = None
        procedural.graph_store.create_node.return_value = "test-node-id"

        trajectory = [
            {
                "action": "Set up database",
                "tool": "Bash",
                "parameters": {},
                "result": "DB initialized",
            }
        ]

        procedure = await procedural.create_skill(
            trajectory=trajectory,
            outcome_score=0.9,
            domain="devops",
        )

        # Should infer trigger from first action
        assert procedure.trigger_pattern is not None
        assert "database" in procedure.trigger_pattern.lower() or "Set up" in procedure.trigger_pattern

    @pytest.mark.asyncio
    async def test_build_skill_all_domains(self, procedural):
        """Test building skills for all domains."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding
        procedural.vector_store.add.return_value = None
        procedural.graph_store.create_node.return_value = "test-node-id"

        trajectory = [{"action": "Do something", "tool": "Tool", "parameters": {}, "result": "Done"}]

        domains = ["coding", "research", "trading", "devops", "writing"]

        for domain in domains:
            procedure = await procedural.create_skill(
                trajectory=trajectory,
                outcome_score=0.9,
                domain=domain,
            )
            assert procedure.domain == Domain(domain)


class TestProceduralSkillRetrieval:
    """Test skill retrieval and ranking."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_qdrant_store
        procedural.graph_store = mock_neo4j_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_retrieve_skills_basic(self, procedural):
        """Test basic skill retrieval."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.95,
                {
                    "name": "Generate Tests",
                    "domain": "coding",
                    "trigger_pattern": "test",
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
                str(uuid4()),  # Must be valid UUID string
                0.72,
                {
                    "name": "Deploy Code",
                    "domain": "devops",
                    "trigger_pattern": "deploy",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.75,
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

        scored_results = await procedural.recall_skill(
            task="Write unit tests",
            limit=5,
        )

        assert isinstance(scored_results, list)
        assert len(scored_results) == 2
        assert all(isinstance(r, ScoredResult) for r in scored_results)

        # First result should have higher score
        assert scored_results[0].score >= scored_results[1].score

    @pytest.mark.asyncio
    async def test_retrieve_skills_scoring_formula(self, procedural):
        """Test skill retrieval scoring: 0.6*similarity + 0.3*success + 0.1*experience."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.9,  # similarity = 0.9
                {
                    "name": "Skill 1",
                    "domain": "coding",
                    "trigger_pattern": None,
                    "steps": [],
                    "script": None,
                    "success_rate": 1.0,  # success = 1.0
                    "execution_count": 5,  # experience = min(5/10, 1.0) = 0.5
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

        scored_results = await procedural.recall_skill(task="test", limit=5)

        assert len(scored_results) == 1
        result = scored_results[0]

        # Manual calculation: 0.6*0.9 + 0.3*1.0 + 0.1*0.5 = 0.54 + 0.3 + 0.05 = 0.89
        expected_score = 0.6 * 0.9 + 0.3 * 1.0 + 0.1 * 0.5
        assert abs(result.score - expected_score) < 0.01

    @pytest.mark.asyncio
    async def test_retrieve_skills_with_domain_filter(self, procedural):
        """Test skill retrieval with domain filtering."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding
        procedural.vector_store.search.return_value = []

        await procedural.recall_skill(
            task="test",
            domain="coding",
            limit=5,
        )

        # Verify domain filter was applied
        call_args = procedural.vector_store.search.call_args
        assert call_args[1]["filter"]["domain"] == "coding"

    @pytest.mark.asyncio
    async def test_retrieve_skills_excludes_deprecated(self, procedural):
        """Test that deprecated skills are excluded."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.9,
                {
                    "name": "Active Skill",
                    "domain": "coding",
                    "trigger_pattern": None,
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
                str(uuid4()),  # Must be valid UUID string
                0.88,
                {
                    "name": "Deprecated Skill",
                    "domain": "coding",
                    "trigger_pattern": None,
                    "steps": [],
                    "script": None,
                    "success_rate": 0.5,
                    "execution_count": 10,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": True,
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            ),
        ]

        procedural.vector_store.search.return_value = results

        scored_results = await procedural.recall_skill(task="test", limit=10)

        # Only active skill should be returned
        assert len(scored_results) == 1
        assert "Active" in scored_results[0].item.name

    @pytest.mark.asyncio
    async def test_retrieve_skills_respects_limit(self, procedural):
        """Test that retrieval respects the limit."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        # Create 10 results with valid UUID strings
        results = []
        for i in range(10):
            results.append((
                str(uuid4()),  # Must be valid UUID string
                0.9 - (i * 0.05),
                {
                    "name": f"Skill {i}",
                    "domain": "coding",
                    "trigger_pattern": None,
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
            ))

        procedural.vector_store.search.return_value = results

        scored_results = await procedural.recall_skill(task="test", limit=3)

        assert len(scored_results) == 3


class TestProceduralSkillUpdate:
    """Test skill execution tracking and updates."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_qdrant_store
        procedural.graph_store = mock_neo4j_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_update_success_rate_after_success(self, procedural):
        """Test success rate updates after successful execution."""
        proc_id = uuid4()
        now = datetime.now()

        payload = {
            "name": "Test Skill",
            "domain": "coding",
            "trigger_pattern": None,
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
        }

        procedural.vector_store.get.return_value = [(str(proc_id), payload)]
        procedural.vector_store.update_payload.return_value = None
        procedural.graph_store.update_node.return_value = None

        updated = await procedural.update(
            procedure_id=proc_id,
            success=True,
        )

        # Success rate should increase
        # (0.9 * 10 + 1) / (10 + 1) = 10.0 / 11 ≈ 0.909
        expected_rate = (0.9 * 10 + 1) / (10 + 1)
        assert abs(updated.success_rate - expected_rate) < 0.01
        assert updated.execution_count == 11

    @pytest.mark.asyncio
    async def test_update_success_rate_after_failure(self, procedural):
        """Test success rate updates after failed execution."""
        proc_id = uuid4()
        now = datetime.now()

        payload = {
            "name": "Test Skill",
            "domain": "coding",
            "trigger_pattern": None,
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
        }

        procedural.vector_store.get.return_value = [(str(proc_id), payload)]
        procedural.vector_store.update_payload.return_value = None
        procedural.graph_store.update_node.return_value = None

        updated = await procedural.update(
            procedure_id=proc_id,
            success=False,
        )

        # Success rate should decrease
        # (0.9 * 10 + 0) / (10 + 1) = 9.0 / 11 ≈ 0.818
        expected_rate = (0.9 * 10 + 0) / (10 + 1)
        assert abs(updated.success_rate - expected_rate) < 0.01
        assert updated.execution_count == 11

    @pytest.mark.asyncio
    async def test_procedure_deprecation_on_consistent_failures(self, procedural):
        """Test that procedure is deprecated after consistent failures."""
        proc_id = uuid4()
        now = datetime.now()

        # Create a procedure with many failures
        payload = {
            "name": "Bad Skill",
            "domain": "coding",
            "trigger_pattern": None,
            "steps": [],
            "script": None,
            "success_rate": 0.2,  # 20% success rate
            "execution_count": 15,  # More than 10 executions
            "last_executed": now.isoformat(),
            "version": 1,
            "deprecated": False,
            "consolidated_into": None,
            "created_at": now.isoformat(),
            "created_from": "trajectory",
        }

        procedural.vector_store.get.return_value = [(str(proc_id), payload)]
        procedural.vector_store.update_payload.return_value = None
        procedural.graph_store.update_node.return_value = None

        # One more failure
        updated = await procedural.update(
            procedure_id=proc_id,
            success=False,
        )

        # Should be deprecated (success_rate < 0.3 and execution_count > 10)
        assert updated.deprecated is True


class TestProceduralSkillDeprecation:
    """Test skill deprecation and consolidation."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_qdrant_store
        procedural.graph_store = mock_neo4j_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_deprecate_skill(self, procedural):
        """Test deprecating a skill."""
        proc_id = uuid4()
        now = datetime.now()

        # RACE-006 FIX: Mock get_procedure to return non-deprecated procedure
        procedural.vector_store.get.return_value = [
            (
                str(proc_id),
                {
                    "name": "Test Skill",
                    "domain": "coding",
                    "trigger_pattern": "test",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.8,
                    "execution_count": 5,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,  # Not yet deprecated
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            )
        ]
        procedural.vector_store.update_payload.return_value = None
        procedural.graph_store.update_node.return_value = None

        await procedural.deprecate(
            procedure_id=proc_id,
            reason="Replaced by better skill",
        )

        # Verify deprecation updates
        procedural.vector_store.update_payload.assert_called_once()
        call_args = procedural.vector_store.update_payload.call_args
        assert call_args[1]["payload"]["deprecated"] is True

    @pytest.mark.asyncio
    async def test_deprecate_with_consolidation(self, procedural):
        """Test deprecating skill and marking consolidation target."""
        proc_id = uuid4()
        consolidated_into = uuid4()
        now = datetime.now()

        # RACE-006 FIX: Mock get_procedure to return non-deprecated procedure
        procedural.vector_store.get.return_value = [
            (
                str(proc_id),
                {
                    "name": "Test Skill",
                    "domain": "coding",
                    "trigger_pattern": "test",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.8,
                    "execution_count": 5,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,  # Not yet deprecated
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            )
        ]
        procedural.vector_store.update_payload.return_value = None
        procedural.graph_store.update_node.return_value = None

        await procedural.deprecate(
            procedure_id=proc_id,
            reason="Consolidated into better skill",
            consolidated_into=consolidated_into,
        )

        # Verify consolidation reference
        call_args = procedural.vector_store.update_payload.call_args
        payload = call_args[1]["payload"]
        assert payload["deprecated"] is True
        assert payload["consolidated_into"] == str(consolidated_into)


class TestProceduralTriggerMatching:
    """Test skill trigger pattern matching."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_qdrant_store
        procedural.graph_store = mock_neo4j_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_match_trigger_pattern(self, procedural):
        """Test matching user request to trigger pattern."""
        test_embedding = [0.1] * 1024
        procedural.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.85,
                {
                    "name": "Write Tests",
                    "domain": "coding",
                    "trigger_pattern": "Write unit tests",
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
                str(uuid4()),  # Must be valid UUID string
                0.82,
                {
                    "name": "Deploy Service",
                    "domain": "devops",
                    "trigger_pattern": "Deploy to production",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.88,
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

        # Use substring matching (use_semantic_matching=False) to avoid
        # needing to mock embedding.similarity
        results = await procedural.match_trigger(
            user_request="Write unit tests for the new feature",
            use_semantic_matching=False,
        )

        # First result should be boosted due to substring trigger match
        assert results[0].item.trigger_pattern == "Write unit tests"
        assert results[0].components.get("trigger_match") == 1.0
        assert results[0].score > results[1].score


class TestProceduralStepExtraction:
    """Test step extraction and script generation."""

    @pytest_asyncio.fixture
    async def procedural(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create procedural memory instance."""
        procedural = ProceduralMemory(session_id=test_session_id)
        procedural.vector_store = mock_qdrant_store
        procedural.graph_store = mock_neo4j_store
        procedural.embedding = mock_embedding_provider
        procedural.vector_store.procedures_collection = "procedures"
        return procedural

    @pytest.mark.asyncio
    async def test_extract_steps_from_trajectory(self, procedural):
        """Test extracting structured steps from trajectory."""
        trajectory = [
            {
                "action": "Read configuration",
                "tool": "Read",
                "parameters": {"file_path": "config.yaml"},
                "result": "Config loaded",
            },
            {
                "action": "Parse settings",
                "tool": "Python",
                "parameters": {"code": "yaml.load(...)"},
                "result": "Settings dict",
            },
        ]

        steps = procedural._extract_steps(trajectory)

        assert len(steps) == 2
        assert steps[0].order == 1
        assert steps[0].action == "Read configuration"
        assert steps[0].tool == "Read"
        assert steps[1].order == 2

    def test_extract_steps_handles_missing_fields(self, procedural):
        """Test step extraction with missing optional fields."""
        trajectory = [
            {
                "description": "Do something",
                "parameters": {"key": "value"},
            },
            {
                "action": "Do next thing",
            },
        ]

        steps = procedural._extract_steps(trajectory)

        assert len(steps) == 2
        assert steps[0].action in ["Do something", ""]
        assert steps[1].action == "Do next thing"


