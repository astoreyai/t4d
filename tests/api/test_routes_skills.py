"""Tests for skills API routes."""

import pytest
from datetime import datetime
from uuid import uuid4

from t4dm.api.routes.skills import (
    StepCreate,
    SkillCreate,
    SkillResponse,
    SkillList,
    SkillRecallRequest,
    ExecutionRequest,
    HowToResponse,
)
from t4dm.core.types import Domain


class TestStepCreate:
    """Tests for StepCreate model."""

    def test_basic_step(self):
        """Create basic step."""
        step = StepCreate(order=1, action="Run tests")
        assert step.order == 1
        assert step.action == "Run tests"
        assert step.tool is None
        assert step.parameters == {}
        assert step.expected_outcome is None

    def test_step_with_all_fields(self):
        """Create step with all fields."""
        step = StepCreate(
            order=2,
            action="Compile code",
            tool="gcc",
            parameters={"flags": "-O2"},
            expected_outcome="Compiled binary",
        )
        assert step.tool == "gcc"
        assert step.parameters == {"flags": "-O2"}
        assert step.expected_outcome == "Compiled binary"

    def test_invalid_order_zero(self):
        """Order must be positive."""
        with pytest.raises(ValueError):
            StepCreate(order=0, action="Invalid")

    def test_invalid_order_negative(self):
        """Order must be positive."""
        with pytest.raises(ValueError):
            StepCreate(order=-1, action="Invalid")

    def test_empty_action(self):
        """Action cannot be empty."""
        with pytest.raises(ValueError):
            StepCreate(order=1, action="")


class TestSkillCreate:
    """Tests for SkillCreate model."""

    def test_basic_skill(self):
        """Create basic skill."""
        skill = SkillCreate(
            name="Test Skill",
            domain=Domain.CODING,
            task="Run unit tests",
        )
        assert skill.name == "Test Skill"
        assert skill.domain == Domain.CODING
        assert skill.task == "Run unit tests"
        assert skill.steps == []
        assert skill.script is None

    def test_skill_with_steps(self):
        """Create skill with steps."""
        skill = SkillCreate(
            name="Build Project",
            domain=Domain.CODING,
            task="Build and test the project",
            steps=[
                StepCreate(order=1, action="Run linter"),
                StepCreate(order=2, action="Run tests"),
                StepCreate(order=3, action="Build binary"),
            ],
        )
        assert len(skill.steps) == 3
        assert skill.steps[0].action == "Run linter"

    def test_skill_with_trigger_and_script(self):
        """Create skill with trigger pattern and script."""
        skill = SkillCreate(
            name="Deploy",
            domain=Domain.DEVOPS,
            task="Deploy application to production",
            trigger_pattern="deploy to production",
            script="git push origin main && kubectl apply -f deploy.yaml",
        )
        assert skill.trigger_pattern == "deploy to production"
        assert "kubectl" in skill.script

    def test_empty_name(self):
        """Name cannot be empty."""
        with pytest.raises(ValueError):
            SkillCreate(name="", domain=Domain.CODING, task="Test")

    def test_empty_task(self):
        """Task cannot be empty."""
        with pytest.raises(ValueError):
            SkillCreate(name="Test", domain=Domain.CODING, task="")


class TestSkillResponse:
    """Tests for SkillResponse model."""

    def test_basic_response(self):
        """Create basic skill response."""
        skill_id = uuid4()
        now = datetime.now()
        response = SkillResponse(
            id=skill_id,
            name="Test Skill",
            domain=Domain.CODING,
            trigger_pattern=None,
            steps=[],
            script=None,
            success_rate=0.0,
            execution_count=0,
            last_executed=None,
            version=1,
            deprecated=False,
            created_at=now,
        )
        assert response.id == skill_id
        assert response.success_rate == 0.0
        assert response.deprecated is False

    def test_response_with_execution_data(self):
        """Create response with execution data."""
        skill_id = uuid4()
        now = datetime.now()
        response = SkillResponse(
            id=skill_id,
            name="Build",
            domain=Domain.CODING,
            trigger_pattern="build project",
            steps=[
                StepCreate(order=1, action="Compile"),
                StepCreate(order=2, action="Test"),
            ],
            script="make all",
            success_rate=0.85,
            execution_count=100,
            last_executed=now,
            version=3,
            deprecated=False,
            created_at=now,
        )
        assert response.success_rate == 0.85
        assert response.execution_count == 100
        assert response.version == 3


class TestSkillList:
    """Tests for SkillList model."""

    def test_empty_list(self):
        """Create empty skill list."""
        skill_list = SkillList(skills=[], total=0)
        assert len(skill_list.skills) == 0
        assert skill_list.total == 0

    def test_list_with_skills(self):
        """Create list with skills."""
        now = datetime.now()
        skills = [
            SkillResponse(
                id=uuid4(),
                name=f"Skill {i}",
                domain=Domain.CODING,
                trigger_pattern=None,
                steps=[],
                script=None,
                success_rate=0.9,
                execution_count=10,
                last_executed=now,
                version=1,
                deprecated=False,
                created_at=now,
            )
            for i in range(5)
        ]
        skill_list = SkillList(skills=skills, total=5)
        assert len(skill_list.skills) == 5
        assert skill_list.total == 5


class TestSkillRecallRequest:
    """Tests for SkillRecallRequest model."""

    def test_basic_request(self):
        """Create basic recall request."""
        request = SkillRecallRequest(query="How to deploy?")
        assert request.query == "How to deploy?"
        assert request.domain is None
        assert request.limit == 5

    def test_request_with_domain(self):
        """Create request with domain filter."""
        request = SkillRecallRequest(
            query="Build project",
            domain=Domain.CODING,
            limit=10,
        )
        assert request.domain == Domain.CODING
        assert request.limit == 10

    def test_empty_query(self):
        """Query cannot be empty."""
        with pytest.raises(ValueError):
            SkillRecallRequest(query="")

    def test_limit_bounds(self):
        """Limit has bounds."""
        # Valid minimum
        request = SkillRecallRequest(query="test", limit=1)
        assert request.limit == 1

        # Valid maximum
        request = SkillRecallRequest(query="test", limit=50)
        assert request.limit == 50

        # Below minimum
        with pytest.raises(ValueError):
            SkillRecallRequest(query="test", limit=0)

        # Above maximum
        with pytest.raises(ValueError):
            SkillRecallRequest(query="test", limit=51)


class TestExecutionRequest:
    """Tests for ExecutionRequest model."""

    def test_success_execution(self):
        """Create successful execution request."""
        request = ExecutionRequest(success=True)
        assert request.success is True
        assert request.duration_ms is None
        assert request.notes is None

    def test_failed_execution(self):
        """Create failed execution request."""
        request = ExecutionRequest(
            success=False,
            duration_ms=5000,
            notes="Test failed due to network error",
        )
        assert request.success is False
        assert request.duration_ms == 5000
        assert "network error" in request.notes

    def test_negative_duration(self):
        """Duration cannot be negative."""
        with pytest.raises(ValueError):
            ExecutionRequest(success=True, duration_ms=-100)


class TestHowToResponse:
    """Tests for HowToResponse model."""

    def test_response_without_skill(self):
        """Create response when no skill found."""
        response = HowToResponse(
            query="How to do something unknown?",
            skill=None,
            steps=[],
            confidence=0.0,
        )
        assert response.skill is None
        assert response.confidence == 0.0

    def test_response_with_skill(self):
        """Create response with matching skill."""
        now = datetime.now()
        skill = SkillResponse(
            id=uuid4(),
            name="Build",
            domain=Domain.CODING,
            trigger_pattern=None,
            steps=[
                StepCreate(order=1, action="Run cmake"),
                StepCreate(order=2, action="Run make"),
            ],
            script=None,
            success_rate=0.95,
            execution_count=50,
            last_executed=now,
            version=1,
            deprecated=False,
            created_at=now,
        )
        response = HowToResponse(
            query="How to build the project?",
            skill=skill,
            steps=["Run cmake", "Run make"],
            confidence=0.92,
        )
        assert response.skill is not None
        assert response.skill.name == "Build"
        assert len(response.steps) == 2
        assert response.confidence == 0.92


class TestDomainEnum:
    """Tests for Domain enum values used in skills."""

    def test_all_domains(self):
        """Test all domain values."""
        domains = list(Domain)
        assert len(domains) == 5

        # All domains should exist
        assert Domain.CODING in domains
        assert Domain.RESEARCH in domains
        assert Domain.TRADING in domains
        assert Domain.DEVOPS in domains
        assert Domain.WRITING in domains

    def test_domain_values(self):
        """Test domain string values."""
        assert Domain.CODING.value == "coding"
        assert Domain.RESEARCH.value == "research"
        assert Domain.DEVOPS.value == "devops"
