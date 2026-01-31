"""
Procedural Memory Service for World Weaver.

Implements Memp-based skill storage with create-recall-update lifecycle.
"""

import logging
import warnings
from datetime import datetime
from functools import wraps
from typing import Any
from uuid import UUID

from opentelemetry.trace import SpanKind

from t4dm.core.config import get_settings
from t4dm.core.types import Domain, Procedure, ProcedureStep, ScoredResult
from t4dm.embedding.bge_m3 import get_embedding_provider
from t4dm.learning.dopamine import DopamineSystem
from t4dm.observability.tracing import traced
from t4dm.storage.neo4j_store import get_neo4j_store
from t4dm.storage.qdrant_store import get_qdrant_store
from t4dm.storage.saga import Saga, SagaState

logger = logging.getLogger(__name__)


def deprecated(replacement: str):
    """
    Decorator to mark a function as deprecated.

    Args:
        replacement: Name of the replacement function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated, use {replacement} instead",
                DeprecationWarning,
                stacklevel=2
            )
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class ProceduralMemory:
    """
    Procedural memory service.

    Stores learned skills with Memp build-retrieve-update lifecycle.
    Implements dual format storage (steps + script) and execution tracking.
    """

    def __init__(self, session_id: str | None = None):
        """
        Initialize procedural memory service.

        Args:
            session_id: Session identifier for instance isolation
        """
        settings = get_settings()
        self.session_id = session_id or settings.session_id

        self.embedding = get_embedding_provider()
        self.vector_store = get_qdrant_store(self.session_id)
        self.graph_store = get_neo4j_store(self.session_id)

        # Consolidation threshold
        self.skill_similarity = settings.consolidation_skill_similarity

        # Scoring weights
        self.similarity_weight = settings.procedural_weight_similarity
        self.success_weight = settings.procedural_weight_success
        self.experience_weight = settings.procedural_weight_experience

        # Optional dopamine system for RPE-modulated learning
        self.dopamine_system = None  # Lazy init
        self._dopamine_enabled = settings.procedural_dopamine_enabled

    async def initialize(self) -> None:
        """Initialize storage backends."""
        await self.vector_store.initialize()
        await self.graph_store.initialize()

    @traced("procedural.create_skill", kind=SpanKind.INTERNAL)
    async def create_skill(
        self,
        trajectory: list[dict[str, Any]],
        outcome_score: float,
        domain: str,
        trigger_pattern: str | None = None,
        name: str | None = None,
    ) -> Procedure | None:
        """
        CREATE: Create procedure from successful trajectory.

        Only learns from successful outcomes (score >= 0.7).

        Args:
            trajectory: List of action dicts with tool, parameters, result
            outcome_score: Success score [0, 1]
            domain: coding, research, trading, devops, writing
            trigger_pattern: When to invoke this procedure
            name: Optional procedure name (auto-generated if not provided)

        Returns:
            Created procedure, or None if outcome_score < 0.7
        """
        if outcome_score < 0.7:
            logger.debug(f"Skipping procedure build: outcome_score {outcome_score} < 0.7")
            return None

        # Extract steps from trajectory
        steps = self._extract_steps(trajectory)

        # Generate abstract script
        script = await self._generate_script(steps, domain)

        # Generate name if not provided
        if not name:
            name = await self._generate_name(steps, domain)

        # Infer trigger pattern if not provided
        if not trigger_pattern:
            trigger_pattern = await self._infer_trigger(trajectory, steps)

        # Generate embedding from script (more generalizable)
        embedding = await self.embedding.embed_query(script)

        # Create procedure
        procedure = Procedure(
            name=name,
            domain=Domain(domain),
            trigger_pattern=trigger_pattern,
            steps=steps,
            script=script,
            embedding=embedding,
            success_rate=1.0,  # First execution was successful
            execution_count=1,
            last_executed=datetime.now(),
            created_from="trajectory",
        )

        # Wrap in saga for atomicity
        saga = Saga(f"create_skill_{procedure.id}")

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.procedures_collection,
                ids=[str(procedure.id)],
                vectors=[embedding],
                payloads=[self._to_payload(procedure)],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.procedures_collection,
                ids=[str(procedure.id)],
            ),
        )

        # Step 2: Create graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Procedure",
                properties=self._to_graph_props(procedure),
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(procedure.id),
                label="Procedure",
            ),
        )

        # Execute saga
        result = await saga.execute()

        # Check saga result and raise on failure
        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Procedure creation failed: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.info(
            f"Created procedure '{name}' from trajectory ({len(steps)} steps) "
            f"(saga: {result.saga_id}, state: {result.state.value})"
        )
        return procedure

    @traced("procedural.store_skill_direct", kind=SpanKind.INTERNAL)
    async def store_skill_direct(
        self,
        name: str,
        domain: Domain | str,
        task: str,
        steps: list[ProcedureStep],
        trigger_pattern: str | None = None,
        script: str | None = None,
    ) -> Procedure:
        """
        Store a pre-formed procedure directly (no learning gate).

        Unlike create_skill(), this stores the procedure without requiring
        a trajectory or outcome score. Used for manual skill creation via API.

        Args:
            name: Procedure name
            domain: Skill domain
            task: Task description (used for embedding)
            steps: Pre-formed procedure steps
            trigger_pattern: When to invoke this procedure
            script: Optional high-level script

        Returns:
            Created procedure
        """
        # Normalize domain
        if isinstance(domain, str):
            domain = Domain(domain)

        # Generate script if not provided
        if not script:
            script = await self._generate_script(steps, domain.value)

        # Generate embedding from task description (better for recall matching)
        embedding = await self.embedding.embed_query(task)

        # Create procedure (execution_count=1 satisfies model constraint)
        procedure = Procedure(
            name=name,
            domain=domain,
            trigger_pattern=trigger_pattern,
            steps=steps,
            script=script,
            embedding=embedding,
            success_rate=1.0,
            execution_count=1,
            last_executed=None,
            created_from="api",
        )

        # Wrap in saga for atomicity
        saga = Saga(f"store_skill_{procedure.id}")

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.procedures_collection,
                ids=[str(procedure.id)],
                vectors=[embedding],
                payloads=[self._to_payload(procedure)],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.procedures_collection,
                ids=[str(procedure.id)],
            ),
        )

        # Step 2: Create graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Procedure",
                properties=self._to_graph_props(procedure),
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(procedure.id),
                label="Procedure",
            ),
        )

        # Execute saga
        result = await saga.execute()

        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Procedure storage failed: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.info(
            f"Stored procedure '{name}' directly ({len(steps)} steps) "
            f"(saga: {result.saga_id}, state: {result.state.value})"
        )
        return procedure

    @traced("procedural.recall_skill", kind=SpanKind.INTERNAL)
    async def recall_skill(
        self,
        task: str,
        domain: str | None = None,
        limit: int = 5,
        session_filter: str | None = None,
    ) -> list[ScoredResult]:
        """
        RECALL: Match task to stored procedures.

        Scoring: 0.6*similarity + 0.3*success_rate + 0.1*experience

        Args:
            task: Task description to match
            domain: Optional domain filter
            limit: Maximum results
            session_filter: Filter to specific session (defaults to current)

        Returns:
            Scored procedures with component breakdown
        """
        # Generate task embedding
        task_vec = await self.embedding.embed_query(task)

        # Build filter with session isolation
        filter_dict = {"deprecated": False}
        if session_filter:
            filter_dict["session_id"] = session_filter
        elif self.session_id != "default":
            filter_dict["session_id"] = self.session_id
        if domain:
            filter_dict["domain"] = domain

        # Vector search
        results = await self.vector_store.search(
            collection=self.vector_store.procedures_collection,
            vector=task_vec,
            limit=limit * 2,
            filter=filter_dict,
        )

        scored_results = []

        for id_str, similarity, payload in results:
            procedure = self._from_payload(id_str, payload)

            # Skip deprecated
            if procedure.deprecated:
                continue

            # Calculate score components
            success_score = procedure.success_rate
            experience_score = min(procedure.execution_count / 10, 1.0)

            # Combined score
            total_score = (
                self.similarity_weight * similarity +
                self.success_weight * success_score +
                self.experience_weight * experience_score
            )

            scored_results.append(ScoredResult(
                item=procedure,
                score=total_score,
                components={
                    "similarity": similarity,
                    "success_rate": success_score,
                    "experience": experience_score,
                },
            ))

        # Sort by score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:limit]

    async def update(
        self,
        procedure_id: UUID,
        success: bool,
        error: str | None = None,
        failed_step: int | None = None,
        context: str | None = None,
    ) -> Procedure:
        """
        UPDATE: Learn from execution outcomes.

        - Successful: Reinforce procedure
        - Failed: Reflect, potentially revise
        - Consistently failing: Deprecate

        Args:
            procedure_id: Procedure UUID
            success: Whether execution succeeded
            error: Error message if failed
            failed_step: Step number that failed
            context: Execution context

        Returns:
            Updated procedure
        """
        procedure = await self.get_procedure(procedure_id)
        if not procedure:
            raise ValueError(f"Procedure {procedure_id} not found")

        # Compute dopamine signal (RPE) for surprise-modulated learning
        rpe_signal = None
        if self._dopamine_enabled:
            try:
                # Lazy init
                if self.dopamine_system is None:
                    self.dopamine_system = DopamineSystem(
                        default_expected=0.5,
                        value_learning_rate=0.1,
                        surprise_threshold=0.05
                    )

                # Use procedure name as context hash
                actual_outcome = 1.0 if success else 0.0

                # Compute RPE: δ = actual - expected
                rpe = self.dopamine_system.compute_rpe(
                    memory_id=procedure_id,
                    actual_outcome=actual_outcome
                )
                rpe_signal = rpe.rpe

                # Update expectations for next time
                self.dopamine_system.update_expectations(
                    memory_id=procedure_id,
                    actual_outcome=actual_outcome
                )

                logger.debug(
                    f"Dopamine signal for '{procedure.name}': "
                    f"RPE={rpe_signal:.3f} (expected={rpe.expected:.2f}, actual={actual_outcome:.1f})"
                )

                # Use RPE to modulate consolidation
                # High |RPE| = surprising = more learning
                # Expected outcomes (RPE≈0) = less learning
                if abs(rpe_signal) > 0.1:
                    logger.info(
                        f"Surprising outcome for '{procedure.name}': RPE={rpe_signal:.3f} "
                        f"({'better' if rpe_signal > 0 else 'worse'} than expected)"
                    )

            except Exception as e:
                logger.warning(f"Dopamine RPE computation failed: {e}")


        # Store old values for compensation
        old_success_rate = procedure.success_rate
        old_execution_count = procedure.execution_count
        old_last_executed = procedure.last_executed

        # Update success rate
        procedure.update_success_rate(success)

        # Wrap in saga for atomicity
        saga = Saga(f"update_procedure_{procedure_id}")

        # Step 1: Update vector payload
        saga.add_step(
            name="update_vector_payload",
            action=lambda: self.vector_store.update_payload(
                collection=self.vector_store.procedures_collection,
                id=str(procedure_id),
                payload={
                    "success_rate": procedure.success_rate,
                    "execution_count": procedure.execution_count,
                    "last_executed": procedure.last_executed.isoformat(),
                },
            ),
            compensate=lambda: self.vector_store.update_payload(
                collection=self.vector_store.procedures_collection,
                id=str(procedure_id),
                payload={
                    "success_rate": old_success_rate,
                    "execution_count": old_execution_count,
                    "last_executed": old_last_executed.isoformat(),
                },
            ),
        )

        # Step 2: Update graph node
        saga.add_step(
            name="update_graph_node",
            action=lambda: self.graph_store.update_node(
                node_id=str(procedure_id),
                properties={
                    "successRate": procedure.success_rate,
                    "executionCount": procedure.execution_count,
                    "lastExecuted": procedure.last_executed.isoformat(),
                },
                label="Procedure",
            ),
            compensate=lambda: self.graph_store.update_node(
                node_id=str(procedure_id),
                properties={
                    "successRate": old_success_rate,
                    "executionCount": old_execution_count,
                    "lastExecuted": old_last_executed.isoformat(),
                },
                label="Procedure",
            ),
        )

        # Execute saga
        result = await saga.execute()

        # Check saga result and raise on failure
        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Failed to update procedure: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.debug(
            f"Updated procedure '{procedure.name}': "
            f"success_rate={procedure.success_rate:.2f}, "
            f"executions={procedure.execution_count} "
            f"(saga: {result.saga_id}, state: {result.state.value})"
        )

        # Check for deprecation - RACE-006 FIX: Check if already deprecated (idempotent)
        if not procedure.deprecated and procedure.should_deprecate():
            await self.deprecate(procedure_id, reason="Consistent failures")
            procedure.deprecated = True

        logger.info(
            f"Updated procedure '{procedure.name}': "
            f"success_rate={procedure.success_rate:.2f}, "
            f"executions={procedure.execution_count}"
        )

        return procedure

    async def deprecate(
        self,
        procedure_id: UUID,
        reason: str | None = None,
        consolidated_into: UUID | None = None,
    ) -> None:
        """
        Mark procedure as deprecated.

        RACE-006 FIX: This method is idempotent - calling it on an already
        deprecated procedure is a no-op.

        Args:
            procedure_id: Procedure UUID
            reason: Deprecation reason
            consolidated_into: If merged into another procedure
        """
        # RACE-006 FIX: Check if already deprecated (idempotent)
        procedure = await self.get_procedure(procedure_id)
        if procedure is None:
            logger.warning(f"Cannot deprecate {procedure_id}: not found")
            return
        if procedure.deprecated:
            logger.debug(f"Procedure {procedure_id} already deprecated, skipping")
            return

        # Prepare update payloads
        vector_updates = {
            "deprecated": True,
        }
        if consolidated_into:
            vector_updates["consolidated_into"] = str(consolidated_into)

        graph_updates = {
            "deprecated": True,
            "consolidatedInto": str(consolidated_into) if consolidated_into else "",
        }

        # Wrap in saga for atomicity
        saga = Saga(f"deprecate_procedure_{procedure_id}")

        # Step 1: Update vector payload
        saga.add_step(
            name="update_vector_payload",
            action=lambda: self.vector_store.update_payload(
                collection=self.vector_store.procedures_collection,
                id=str(procedure_id),
                payload=vector_updates,
            ),
            compensate=lambda: self.vector_store.update_payload(
                collection=self.vector_store.procedures_collection,
                id=str(procedure_id),
                payload={"deprecated": False, "consolidated_into": None},
            ),
        )

        # Step 2: Update graph node
        saga.add_step(
            name="update_graph_node",
            action=lambda: self.graph_store.update_node(
                node_id=str(procedure_id),
                properties=graph_updates,
                label="Procedure",
            ),
            compensate=lambda: self.graph_store.update_node(
                node_id=str(procedure_id),
                properties={"deprecated": False, "consolidatedInto": ""},
                label="Procedure",
            ),
        )

        # Execute saga
        result = await saga.execute()

        # Check saga result and raise on failure
        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Failed to deprecate procedure: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.info(
            f"Deprecated procedure {procedure_id}: {reason} "
            f"(saga: {result.saga_id}, state: {result.state.value})"
        )

    async def get_procedure(self, procedure_id: UUID) -> Procedure | None:
        """Get procedure by ID."""
        results = await self.vector_store.get(
            collection=self.vector_store.procedures_collection,
            ids=[str(procedure_id)],
        )

        if results:
            id_str, payload = results[0]
            return self._from_payload(id_str, payload)

        return None

    async def list_skills(
        self,
        domain: Domain | None = None,
        include_deprecated: bool = False,
        limit: int = 50,
        session_filter: str | None = None,
    ) -> list[Procedure]:
        """
        List skills with optional filtering.

        Args:
            domain: Filter by domain
            include_deprecated: Include deprecated skills
            limit: Maximum number of skills to return
            session_filter: Optional session ID filter

        Returns:
            List of procedures/skills
        """
        limit = min(limit, 500)

        # Build filter
        filter_conditions = {}
        if domain:
            filter_conditions["domain"] = domain.value
        if not include_deprecated:
            filter_conditions["deprecated"] = False
        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        try:
            results, _ = await self.vector_store.scroll(
                collection=self.vector_store.procedures_collection,
                scroll_filter=filter_conditions if filter_conditions else None,
                limit=limit,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )

            procedures = []
            for id_str, payload, _ in results:
                procedure = self._from_payload(id_str, payload)
                procedures.append(procedure)

            logger.debug(f"Listed {len(procedures)} skills")
            return procedures

        except Exception as e:
            logger.error(f"Error listing skills: {e}")
            raise

    async def match_trigger(
        self,
        user_request: str,
        use_semantic_matching: bool = True,
        semantic_threshold: float = 0.5,
    ) -> list[ScoredResult]:
        """
        Match user request against procedure trigger patterns.

        Uses embedding-based semantic similarity instead of substring matching.
        Much more robust to paraphrasing and synonyms.

        Args:
            user_request: User's request text
            use_semantic_matching: Use semantic similarity (default True)
            semantic_threshold: Minimum similarity score for semantic match (default 0.5)

        Returns:
            Procedures that match, boosted for semantic trigger match
        """
        # Get candidate procedures
        candidates = await self.recall_skill(user_request, limit=10)

        if not use_semantic_matching:
            # Fallback to substring matching
            boosted = []
            for result in candidates:
                proc = result.item
                if proc.trigger_pattern:
                    if proc.trigger_pattern.lower() in user_request.lower():
                        result.score *= 1.2  # Boost for trigger match
                        result.components["trigger_match"] = 1.0
                boosted.append(result)
            boosted.sort(key=lambda x: x.score, reverse=True)
            return boosted

        # Semantic trigger matching
        # Embed the user request once
        request_embedding = await self.embedding.embed_query(user_request)

        boosted = []
        for result in candidates:
            proc = result.item
            if proc.trigger_pattern:
                # Get or compute trigger embedding
                trigger_embedding = await self._get_trigger_embedding(proc)

                # Calculate semantic similarity
                similarity = self.embedding.similarity(request_embedding, trigger_embedding)

                if similarity >= semantic_threshold:
                    # Boost score proportional to similarity
                    # High similarity (0.8+) gives 1.5x boost
                    # Medium similarity (0.5-0.8) gives 1.2-1.5x boost
                    boost_factor = 1.0 + (similarity * 0.7)  # Max 1.7x boost
                    result.score *= boost_factor
                    result.components["trigger_match"] = similarity

            boosted.append(result)

        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    async def _get_trigger_embedding(self, procedure: Procedure) -> list[float]:
        """
        Get or compute trigger pattern embedding.

        Checks Qdrant payload for cached embedding, computes if missing.

        Args:
            procedure: Procedure with trigger_pattern

        Returns:
            Trigger embedding vector (1024-dim)
        """
        # Try to get cached embedding from Qdrant
        results = await self.vector_store.get(
            collection=self.vector_store.procedures_collection,
            ids=[str(procedure.id)],
        )

        if results:
            _, payload = results[0]
            cached = payload.get("trigger_embedding")
            if cached:
                return cached

        # No cached embedding - compute it
        trigger_text = procedure.trigger_pattern or procedure.name
        trigger_embedding = await self.embedding.embed_query(trigger_text)

        # Cache it in Qdrant payload
        await self.vector_store.update_payload(
            collection=self.vector_store.procedures_collection,
            id=str(procedure.id),
            payload={"trigger_embedding": trigger_embedding},
        )

        logger.debug(f"Cached trigger embedding for procedure '{procedure.name}'")
        return trigger_embedding

    def _extract_steps(self, trajectory: list[dict[str, Any]]) -> list[ProcedureStep]:
        """Convert action trajectory to structured steps."""
        steps = []
        for i, action in enumerate(trajectory):
            steps.append(ProcedureStep(
                order=i + 1,
                action=action.get("action", action.get("description", "")),
                tool=action.get("tool"),
                parameters=action.get("parameters", {}),
                expected_outcome=action.get("result", action.get("expected_outcome")),
            ))
        return steps

    async def _generate_script(self, steps: list[ProcedureStep], domain: str) -> str:
        """
        Generate high-level script from steps.

        Uses template-based abstraction to replace concrete values with
        variable placeholders, making procedures more reusable.
        """
        abstracted_steps = self._abstract_steps(steps)
        step_lines = [f"  {s['order']}. {s['action']}" for s in abstracted_steps]

        script = f"""PROCEDURE: {domain.title()} Workflow
TRIGGER: [Auto-detected pattern]
STEPS:
{chr(10).join(step_lines)}
POSTCONDITION: Task completed successfully"""

        return script

    def _abstract_steps(self, steps: list[ProcedureStep]) -> list[dict[str, Any]]:
        """
        Abstract concrete steps into reusable templates.

        Replaces concrete values with variable placeholders:
        - File paths -> {file_N}
        - URLs -> {url_N}
        - Numbers -> {num_N}
        - Commands with arguments -> parameterized form

        Args:
            steps: List of concrete procedure steps

        Returns:
            List of abstracted step dicts
        """
        import re

        abstracted = []
        variable_counter = {"file": 0, "url": 0, "num": 0, "var": 0}

        for step in steps:
            action = step.action

            # Replace URLs FIRST (before file paths, as URLs may contain /)
            url_pattern = r"https?://[^\s]+"
            matches = re.findall(url_pattern, action)
            for match in matches:
                variable_counter["url"] += 1
                action = action.replace(match, f"{{url_{variable_counter['url']}}}", 1)

            # Replace file paths (absolute and relative)
            file_pattern = r"(/[^\s]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_/-]+\.[a-zA-Z0-9]+)"
            matches = re.findall(file_pattern, action)
            for match in matches:
                if "." in match and ("/" in match or match.startswith(".")):
                    variable_counter["file"] += 1
                    action = action.replace(match, f"{{file_{variable_counter['file']}}}", 1)

            # Replace standalone numbers (but not in version strings or ports)
            num_pattern = r"\b(\d{4,}|\d+\.\d+)\b"
            matches = re.findall(num_pattern, action)
            for match in matches:
                variable_counter["num"] += 1
                action = action.replace(match, f"{{num_{variable_counter['num']}}}", 1)

            # Replace common argument patterns in commands
            # e.g., "pytest tests/" -> "pytest {target_dir}"
            if "pytest" in action.lower():
                action = re.sub(r"pytest\s+[\w/.-]+", "pytest {target_dir}", action)
            if "git clone" in action.lower():
                action = re.sub(r"git clone\s+\S+", "git clone {repo_url}", action)
            if "docker run" in action.lower():
                action = re.sub(r"docker run\s+[\w:.-]+", "docker run {image_name}", action)

            abstracted.append({
                "order": step.order,
                "action": action,
                "tool": step.tool,
            })

        return abstracted

    async def _generate_name(self, steps: list[ProcedureStep], domain: str) -> str:
        """Generate procedure name from steps."""
        if steps:
            # Use first action as basis
            first_action = steps[0].action[:50]
            return f"{domain.title()}: {first_action}"
        return f"{domain.title()} Procedure"

    async def _infer_trigger(
        self,
        trajectory: list[dict[str, Any]],
        steps: list[ProcedureStep],
    ) -> str:
        """Infer trigger pattern from trajectory."""
        if steps:
            return f"When: {steps[0].action}"
        return "General task"

    def _to_payload(self, procedure: Procedure) -> dict:
        """Convert procedure to Qdrant payload."""
        return {
            "session_id": self.session_id,
            "name": procedure.name,
            "domain": procedure.domain.value,
            "trigger_pattern": procedure.trigger_pattern,
            "steps": [s.model_dump() for s in procedure.steps],
            "script": procedure.script,
            "success_rate": procedure.success_rate,
            "execution_count": procedure.execution_count,
            "last_executed": procedure.last_executed.isoformat() if procedure.last_executed else None,
            "version": procedure.version,
            "deprecated": procedure.deprecated,
            "consolidated_into": str(procedure.consolidated_into) if procedure.consolidated_into else None,
            "created_at": procedure.created_at.isoformat(),
            "created_from": procedure.created_from,
        }

    def _from_payload(self, id_str: str, payload: dict) -> Procedure:
        """Reconstruct procedure from Qdrant payload."""
        return Procedure(
            id=UUID(id_str),
            name=payload["name"],
            domain=Domain(payload["domain"]),
            trigger_pattern=payload.get("trigger_pattern"),
            steps=[ProcedureStep(**s) for s in payload.get("steps", [])],
            script=payload.get("script"),
            embedding=None,
            success_rate=payload["success_rate"],
            execution_count=payload["execution_count"],
            last_executed=datetime.fromisoformat(payload["last_executed"]) if payload.get("last_executed") else None,
            version=payload.get("version", 1),
            deprecated=payload.get("deprecated", False),
            consolidated_into=UUID(payload["consolidated_into"]) if payload.get("consolidated_into") else None,
            created_at=datetime.fromisoformat(payload["created_at"]),
            created_from=payload.get("created_from", "manual"),
        )

    def _to_graph_props(self, procedure: Procedure) -> dict:
        """Convert procedure to Neo4j properties."""
        return {
            "id": str(procedure.id),
            "sessionId": self.session_id,
            "name": procedure.name,
            "domain": procedure.domain.value,
            "triggerPattern": procedure.trigger_pattern or "",
            "script": procedure.script or "",
            "successRate": procedure.success_rate,
            "executionCount": procedure.execution_count,
            "lastExecuted": procedure.last_executed.isoformat() if procedure.last_executed else "",
            "version": procedure.version,
            "deprecated": procedure.deprecated,
            "consolidatedInto": str(procedure.consolidated_into) if procedure.consolidated_into else "",
            "createdAt": procedure.created_at.isoformat(),
            "createdFrom": procedure.created_from,
        }

    # Deprecated backward-compatible methods
    @deprecated("create_skill")
    async def build(self, *args, **kwargs) -> Procedure | None:
        """Deprecated: Use create_skill instead."""
        return await self.create_skill(*args, **kwargs)

    @deprecated("recall_skill")
    async def retrieve(self, *args, **kwargs) -> list[ScoredResult]:
        """Deprecated: Use recall_skill instead."""
        return await self.recall_skill(*args, **kwargs)


# Factory function
def get_procedural_memory(session_id: str | None = None) -> ProceduralMemory:
    """
    Get procedural memory service instance.

    Args:
        session_id: Session identifier for instance isolation
    """
    return ProceduralMemory(session_id)
