"""
World Weaver REST API Skill Routes.

CRUD and search operations for procedural memory (skills/procedures).
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ww.api.deps import MemoryServices
from ww.api.errors import sanitize_error
from ww.core.types import Domain, Procedure, ProcedureStep

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class StepCreate(BaseModel):
    """Request model for procedure step."""

    order: int = Field(..., ge=1, description="Step order")
    action: str = Field(..., min_length=1, description="Action description")
    tool: str | None = Field(default=None, description="Tool to use")
    parameters: dict = Field(default_factory=dict, description="Step parameters")
    expected_outcome: str | None = Field(default=None, description="Expected result")


class SkillCreate(BaseModel):
    """Request model for creating a skill."""

    name: str = Field(..., min_length=1, description="Skill name")
    domain: Domain = Field(..., description="Skill domain")
    task: str = Field(..., min_length=1, description="Task description")
    steps: list[StepCreate] = Field(default_factory=list, description="Procedure steps")
    script: str | None = Field(default=None, description="High-level script")
    trigger_pattern: str | None = Field(default=None, description="When to invoke")


class SkillResponse(BaseModel):
    """Response model for skill data."""

    id: UUID
    name: str
    domain: Domain
    trigger_pattern: str | None
    steps: list[StepCreate]
    script: str | None
    success_rate: float
    execution_count: int
    last_executed: datetime | None
    version: int
    deprecated: bool
    created_at: datetime


class SkillList(BaseModel):
    """Response model for skill list."""

    skills: list[SkillResponse]
    total: int


class SkillUpdate(BaseModel):
    """Request model for updating a skill."""

    name: str | None = Field(default=None, max_length=500, description="Updated name")
    trigger_pattern: str | None = Field(default=None, max_length=1000, description="Updated trigger")
    script: str | None = Field(default=None, max_length=50000, description="Updated script")
    steps: list[StepCreate] | None = Field(default=None, description="Updated steps")


class SkillRecallRequest(BaseModel):
    """Request model for skill recall."""

    query: str = Field(..., min_length=1, description="Search query (task description)")
    domain: Domain | None = Field(default=None, description="Filter by domain")
    limit: int = Field(default=5, ge=1, le=50, description="Max results")


class ExecutionRequest(BaseModel):
    """Request model for recording skill execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    duration_ms: int | None = Field(default=None, ge=0, description="Execution duration")
    notes: str | None = Field(default=None, description="Execution notes")


class HowToResponse(BaseModel):
    """Response model for how-to query."""

    query: str
    skill: SkillResponse | None
    steps: list[str]
    confidence: float


# Endpoints

@router.post("", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
async def create_skill(
    request: SkillCreate,
    services: MemoryServices,
):
    """
    Create a new procedural skill.

    Stores a learned procedure with execution tracking.
    """
    procedural = services["procedural"]

    try:
        steps = [
            ProcedureStep(
                order=s.order,
                action=s.action,
                tool=s.tool,
                parameters=s.parameters,
                expected_outcome=s.expected_outcome,
            )
            for s in request.steps
        ]

        procedure = Procedure(
            name=request.name,
            domain=request.domain,
            trigger_pattern=request.trigger_pattern,
            steps=steps,
            script=request.script,
            created_from="api",
        )

        # Store pre-formed procedure directly
        stored = await procedural.store_skill_direct(
            name=procedure.name,
            domain=procedure.domain,
            task=request.task,
            steps=procedure.steps,
            trigger_pattern=procedure.trigger_pattern,
            script=procedure.script,
        )

        return SkillResponse(
            id=stored.id,
            name=stored.name,
            domain=stored.domain,
            trigger_pattern=stored.trigger_pattern,
            steps=[
                StepCreate(
                    order=s.order,
                    action=s.action,
                    tool=s.tool,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                )
                for s in stored.steps
            ],
            script=stored.script,
            success_rate=stored.success_rate,
            execution_count=stored.execution_count,
            last_executed=stored.last_executed,
            version=stored.version,
            deprecated=stored.deprecated,
            created_at=stored.created_at,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "create skill"),
        )


@router.get("/{skill_id}", response_model=SkillResponse)
async def get_skill(
    skill_id: UUID,
    services: MemoryServices,
):
    """
    Get a specific skill by ID.
    """
    procedural = services["procedural"]

    try:
        skill = await procedural.get_procedure(skill_id)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )

        return SkillResponse(
            id=skill.id,
            name=skill.name,
            domain=skill.domain,
            trigger_pattern=skill.trigger_pattern,
            steps=[
                StepCreate(
                    order=s.order,
                    action=s.action,
                    tool=s.tool,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                )
                for s in skill.steps
            ],
            script=skill.script,
            success_rate=skill.success_rate,
            execution_count=skill.execution_count,
            last_executed=skill.last_executed,
            version=skill.version,
            deprecated=skill.deprecated,
            created_at=skill.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "get skill"),
        )


@router.put("/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: UUID,
    request: SkillUpdate,
    services: MemoryServices,
):
    """
    Update a skill by ID.

    Only provided fields will be updated; others remain unchanged.
    """
    procedural = services["procedural"]

    try:
        skill = await procedural.get_procedure(skill_id)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )

        # Apply updates
        if request.name is not None:
            skill.name = request.name
        if request.trigger_pattern is not None:
            skill.trigger_pattern = request.trigger_pattern
        if request.script is not None:
            skill.script = request.script
        if request.steps is not None:
            skill.steps = [
                ProcedureStep(
                    order=s.order,
                    action=s.action,
                    tool=s.tool,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                )
                for s in request.steps
            ]

        # Re-store with updates
        updated = await procedural.update(skill)

        return SkillResponse(
            id=updated.id,
            name=updated.name,
            domain=updated.domain,
            trigger_pattern=updated.trigger_pattern,
            steps=[
                StepCreate(
                    order=s.order,
                    action=s.action,
                    tool=s.tool,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                )
                for s in updated.steps
            ],
            script=updated.script,
            success_rate=updated.success_rate,
            execution_count=updated.execution_count,
            last_executed=updated.last_executed,
            version=updated.version,
            deprecated=updated.deprecated,
            created_at=updated.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "update skill"),
        )


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_skill(
    skill_id: UUID,
    services: MemoryServices,
):
    """
    Delete a skill by ID.

    For soft-delete (preserving history), use /deprecate instead.
    """
    procedural = services["procedural"]

    try:
        deleted = await procedural.delete(skill_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "delete skill"),
        )


@router.get("", response_model=SkillList)
async def list_skills(
    services: MemoryServices,
    domain: Domain | None = Query(default=None, description="Filter by domain"),
    include_deprecated: bool = Query(default=False, description="Include deprecated skills"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
):
    """
    List skills with optional filtering.
    """
    procedural = services["procedural"]

    try:
        skills = await procedural.list_skills(
            domain=domain,
            include_deprecated=include_deprecated,
            limit=limit,
        )

        return SkillList(
            skills=[
                SkillResponse(
                    id=s.id,
                    name=s.name,
                    domain=s.domain,
                    trigger_pattern=s.trigger_pattern,
                    steps=[
                        StepCreate(
                            order=st.order,
                            action=st.action,
                            tool=st.tool,
                            parameters=st.parameters,
                            expected_outcome=st.expected_outcome,
                        )
                        for st in s.steps
                    ],
                    script=s.script,
                    success_rate=s.success_rate,
                    execution_count=s.execution_count,
                    last_executed=s.last_executed,
                    version=s.version,
                    deprecated=s.deprecated,
                    created_at=s.created_at,
                )
                for s in skills
            ],
            total=len(skills),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "list skills"),
        )


@router.post("/recall", response_model=SkillList)
async def recall_skills(
    request: SkillRecallRequest,
    services: MemoryServices,
):
    """
    Semantic search for skills by task description.

    Uses BGE-M3 embeddings to find matching procedures.
    """
    procedural = services["procedural"]

    try:
        results = await procedural.recall_skill(
            task=request.query,
            domain=request.domain,
            limit=request.limit,
        )
        # Extract procedures from ScoredResult objects
        skills = [r.item for r in results]

        return SkillList(
            skills=[
                SkillResponse(
                    id=s.id,
                    name=s.name,
                    domain=s.domain,
                    trigger_pattern=s.trigger_pattern,
                    steps=[
                        StepCreate(
                            order=st.order,
                            action=st.action,
                            tool=st.tool,
                            parameters=st.parameters,
                            expected_outcome=st.expected_outcome,
                        )
                        for st in s.steps
                    ],
                    script=s.script,
                    success_rate=s.success_rate,
                    execution_count=s.execution_count,
                    last_executed=s.last_executed,
                    version=s.version,
                    deprecated=s.deprecated,
                    created_at=s.created_at,
                )
                for s in skills
            ],
            total=len(skills),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "recall skills"),
        )


@router.post("/{skill_id}/execute", response_model=SkillResponse)
async def record_execution(
    skill_id: UUID,
    request: ExecutionRequest,
    services: MemoryServices,
):
    """
    Record a skill execution.

    Updates success rate and execution count.
    """
    procedural = services["procedural"]

    try:
        skill = await procedural.get_procedure(skill_id)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )

        # Update execution stats
        skill.update_success_rate(request.success)
        skill.last_executed = datetime.now()

        # Re-store with updates
        updated = await procedural.update(skill)

        return SkillResponse(
            id=updated.id,
            name=updated.name,
            domain=updated.domain,
            trigger_pattern=updated.trigger_pattern,
            steps=[
                StepCreate(
                    order=s.order,
                    action=s.action,
                    tool=s.tool,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                )
                for s in updated.steps
            ],
            script=updated.script,
            success_rate=updated.success_rate,
            execution_count=updated.execution_count,
            last_executed=updated.last_executed,
            version=updated.version,
            deprecated=updated.deprecated,
            created_at=updated.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "record execution"),
        )


@router.post("/{skill_id}/deprecate", response_model=SkillResponse)
async def deprecate_skill(
    skill_id: UUID,
    replacement_id: UUID | None = Query(default=None, description="Replacement skill ID"),
    services: MemoryServices = None,
):
    """
    Deprecate a skill, optionally pointing to replacement.
    """
    procedural = services["procedural"]

    try:
        skill = await procedural.get_procedure(skill_id)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )

        skill.deprecated = True
        if replacement_id:
            skill.consolidated_into = replacement_id

        updated = await procedural.update(skill)

        return SkillResponse(
            id=updated.id,
            name=updated.name,
            domain=updated.domain,
            trigger_pattern=updated.trigger_pattern,
            steps=[
                StepCreate(
                    order=s.order,
                    action=s.action,
                    tool=s.tool,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                )
                for s in updated.steps
            ],
            script=updated.script,
            success_rate=updated.success_rate,
            execution_count=updated.execution_count,
            last_executed=updated.last_executed,
            version=updated.version,
            deprecated=updated.deprecated,
            created_at=updated.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "deprecate skill"),
        )


@router.get("/how-to/{query:path}", response_model=HowToResponse)
async def how_to(
    query: str,
    domain: Domain | None = Query(default=None, description="Filter by domain"),
    services: MemoryServices = None,
):
    """
    Natural language query for procedural knowledge.

    Returns best matching skill with step-by-step instructions.
    """
    procedural = services["procedural"]

    try:
        results = await procedural.recall_skill(
            task=query,
            domain=domain,
            limit=1,
        )

        if not results:
            return HowToResponse(
                query=query,
                skill=None,
                steps=["No matching skill found. Try creating one or rephrasing your query."],
                confidence=0.0,
            )

        skill = results[0].item
        steps = [
            f"{s.order}. {s.action}" + (f" (using {s.tool})" if s.tool else "")
            for s in skill.steps
        ]

        return HowToResponse(
            query=query,
            skill=SkillResponse(
                id=skill.id,
                name=skill.name,
                domain=skill.domain,
                trigger_pattern=skill.trigger_pattern,
                steps=[
                    StepCreate(
                        order=s.order,
                        action=s.action,
                        tool=s.tool,
                        parameters=s.parameters,
                        expected_outcome=s.expected_outcome,
                    )
                    for s in skill.steps
                ],
                script=skill.script,
                success_rate=skill.success_rate,
                execution_count=skill.execution_count,
                last_executed=skill.last_executed,
                version=skill.version,
                deprecated=skill.deprecated,
                created_at=skill.created_at,
            ),
            steps=steps or [skill.script] if skill.script else ["No steps defined"],
            confidence=skill.success_rate,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "how-to query"),
        )
