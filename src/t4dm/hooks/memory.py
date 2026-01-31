"""
Memory module lifecycle hooks for World Weaver.

Provides hooks for:
- Memory creation (episodic, semantic, procedural)
- Memory recall and retrieval
- Memory updates and modifications
- Access tracking
- Memory decay
"""

import logging
from typing import Any

from t4dm.hooks.base import Hook, HookContext, HookPriority

logger = logging.getLogger(__name__)


class MemoryHook(Hook):
    """Base class for memory-related hooks."""

    def __init__(
        self,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
        memory_type: str | None = None,
    ):
        """
        Initialize memory hook.

        Args:
            name: Hook identifier
            priority: Execution priority
            enabled: Whether hook is active
            memory_type: Filter by memory type (episodic, semantic, procedural)
        """
        super().__init__(name, priority, enabled)
        self.memory_type = memory_type

    def should_execute(self, context: HookContext) -> bool:
        """Check if hook should execute based on memory type filter."""
        if not super().should_execute(context):
            return False

        if self.memory_type:
            mem_type = context.input_data.get("memory_type")
            return mem_type == self.memory_type

        return True


class CreateHook(MemoryHook):
    """
    Hook executed when creating new memories.

    Phases:
    - PRE: Before memory creation (validation, preprocessing)
    - POST: After memory creation (indexing, notifications)
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute memory creation hook.

        Context data (PRE):
        - input_data["memory_type"]: episodic/semantic/procedural
        - input_data["content"]: Memory content
        - input_data["metadata"]: Additional metadata
        - input_data["session_id"]: Session identifier

        Context data (POST):
        - output_data["memory_id"]: Created memory UUID
        - output_data["embedding"]: Generated embedding (if applicable)
        - output_data["success"]: Creation status

        Returns:
            Modified context
        """
        mem_type = context.input_data.get("memory_type", "unknown")
        phase = context.phase.value

        if phase == "pre":
            logger.debug(f"[{self.name}] Pre-create {mem_type} memory")
        else:
            memory_id = context.output_data.get("memory_id") if context.output_data else None
            logger.debug(f"[{self.name}] Post-create {mem_type}: {memory_id}")

        return context


class RecallHook(MemoryHook):
    """
    Hook executed when recalling/retrieving memories.

    Phases:
    - PRE: Before retrieval (query preprocessing, cache check)
    - POST: After retrieval (ranking adjustment, access tracking)
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute memory recall hook.

        Context data (PRE):
        - input_data["memory_type"]: episodic/semantic/procedural
        - input_data["query"]: Search query
        - input_data["filters"]: Query filters
        - input_data["limit"]: Result limit

        Context data (POST):
        - output_data["results"]: Retrieved memories
        - output_data["count"]: Number of results
        - output_data["scores"]: Relevance scores

        Returns:
            Modified context
        """
        mem_type = context.input_data.get("memory_type", "unknown")
        query = context.input_data.get("query", "")
        phase = context.phase.value

        if phase == "pre":
            logger.debug(f"[{self.name}] Pre-recall {mem_type}: {query[:50]}...")
        else:
            count = context.output_data.get("count", 0) if context.output_data else 0
            logger.debug(f"[{self.name}] Post-recall {mem_type}: {count} results")

        return context


class UpdateHook(MemoryHook):
    """
    Hook executed when updating existing memories.

    Phases:
    - PRE: Before update (validation, versioning)
    - POST: After update (reindexing, notifications)
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute memory update hook.

        Context data (PRE):
        - input_data["memory_id"]: Memory UUID
        - input_data["memory_type"]: episodic/semantic/procedural
        - input_data["updates"]: Fields to update
        - input_data["old_values"]: Previous values (optional)

        Context data (POST):
        - output_data["updated"]: Whether update succeeded
        - output_data["version"]: New version number

        Returns:
            Modified context
        """
        memory_id = context.input_data.get("memory_id")
        mem_type = context.input_data.get("memory_type", "unknown")
        phase = context.phase.value

        if phase == "pre":
            updates = context.input_data.get("updates", {})
            logger.debug(
                f"[{self.name}] Pre-update {mem_type} {memory_id}: "
                f"{len(updates)} fields"
            )
        else:
            updated = context.output_data.get("updated", False) if context.output_data else False
            logger.debug(f"[{self.name}] Post-update {mem_type} {memory_id}: {updated}")

        return context


class AccessHook(MemoryHook):
    """
    Hook executed when memory is accessed (read).

    Use for:
    - Access tracking
    - Usage statistics
    - Hebbian weight updates
    - Cache management
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute memory access hook.

        Context data:
        - input_data["memory_id"]: Memory UUID
        - input_data["memory_type"]: episodic/semantic/procedural
        - input_data["access_type"]: "read", "recall", "update"
        - input_data["context_ids"]: Other memories accessed together

        Returns:
            Modified context
        """
        memory_id = context.input_data.get("memory_id")
        access_type = context.input_data.get("access_type", "read")

        logger.debug(f"[{self.name}] Access: {memory_id} ({access_type})")
        return context


class DecayHook(MemoryHook):
    """
    Hook executed when memory decay is updated.

    Use for:
    - FSRS stability updates
    - Forgetting curve calculations
    - Retention tracking
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute memory decay hook.

        Context data:
        - input_data["memory_id"]: Memory UUID
        - input_data["old_stability"]: Previous stability value
        - input_data["new_stability"]: Updated stability value
        - input_data["retrievability"]: Current retrievability

        Returns:
            Modified context
        """
        memory_id = context.input_data.get("memory_id")
        old_stab = context.input_data.get("old_stability", 0)
        new_stab = context.input_data.get("new_stability", 0)

        logger.debug(
            f"[{self.name}] Decay update {memory_id}: "
            f"{old_stab:.2f} -> {new_stab:.2f}"
        )
        return context


# Example implementations

class CachingRecallHook(RecallHook):
    """Example: Cache recall results for frequent queries."""

    def __init__(self, cache_size: int = 1000):
        super().__init__(
            name="caching_recall",
            priority=HookPriority.HIGH,
        )
        self.cache: dict[str, Any] = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0

    async def execute(self, context: HookContext) -> HookContext:
        if context.phase.value == "pre":
            # Check cache
            query = context.input_data.get("query", "")
            cache_key = f"{context.input_data.get('memory_type')}:{query}"

            if cache_key in self.cache:
                self.hits += 1
                logger.debug(f"Cache HIT: {cache_key}")
                context.metadata["cache_hit"] = True
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {cache_key}")
                context.metadata["cache_hit"] = False

        elif context.phase.value == "post" and not context.metadata.get("cache_hit"):
            # Store in cache
            query = context.input_data.get("query", "")
            cache_key = f"{context.input_data.get('memory_type')}:{query}"

            if len(self.cache) >= self.cache_size:
                # Simple LRU: remove first item
                self.cache.pop(next(iter(self.cache)))

            self.cache[cache_key] = context.output_data

        return context

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = super().get_stats()
        total = self.hits + self.misses
        stats["cache"] = {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }
        return stats


class AuditTrailHook(MemoryHook):
    """Example: Create audit trail for all memory operations."""

    def __init__(self, audit_store: Any | None = None):
        super().__init__(
            name="audit_trail",
            priority=HookPriority.HIGH,
        )
        self.audit_store = audit_store or []

    async def execute(self, context: HookContext) -> HookContext:
        # Record all operations
        audit_entry = {
            "timestamp": context.start_time.isoformat(),
            "operation": context.operation,
            "phase": context.phase.value,
            "memory_type": context.input_data.get("memory_type"),
            "session_id": context.session_id,
            "user_id": context.user_id,
            "success": context.error is None,
        }

        if isinstance(self.audit_store, list):
            self.audit_store.append(audit_entry)
        else:
            # Could write to database, log file, etc.
            logger.info(f"Audit: {audit_entry}")

        return context


class HebbianUpdateHook(AccessHook):
    """Example: Update Hebbian weights on co-access."""

    def __init__(self, learning_rate: float = 0.1):
        super().__init__(
            name="hebbian_update",
            priority=HookPriority.NORMAL,
        )
        self.learning_rate = learning_rate

    async def execute(self, context: HookContext) -> HookContext:
        memory_id = context.input_data.get("memory_id")
        context_ids = context.input_data.get("context_ids", [])

        if context_ids:
            logger.debug(
                f"Hebbian update: {memory_id} co-accessed with {len(context_ids)} memories"
            )
            # Would update relationship weights in graph store
            context.metadata["hebbian_updates"] = len(context_ids)

        return context


class ValidationHook(CreateHook):
    """Example: Validate memory content before creation."""

    def __init__(
        self,
        max_content_length: int = 100000,
        required_fields: list[str] | None = None,
    ):
        super().__init__(
            name="validation",
            priority=HookPriority.CRITICAL,
        )
        self.max_content_length = max_content_length
        self.required_fields = required_fields or []

    async def execute(self, context: HookContext) -> HookContext:
        if context.phase.value != "pre":
            return context

        # Validate content length
        content = context.input_data.get("content", "")
        if len(content) > self.max_content_length:
            raise ValueError(
                f"Content too long: {len(content)} > {self.max_content_length}"
            )

        # Validate required fields
        for field in self.required_fields:
            if field not in context.input_data:
                raise ValueError(f"Required field missing: {field}")

        logger.debug("Memory validation passed")
        return context
