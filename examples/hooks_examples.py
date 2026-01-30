"""
World Weaver Hooks System - Usage Examples

Demonstrates common patterns for:
1. Observability (tracing, metrics, logging)
2. Caching (query cache, embedding cache)
3. Auditing (compliance, security)
4. Performance (timing, profiling)
5. Error handling (circuit breaker, retry)
"""

import asyncio
import logging
import time
from typing import Any, Optional

from ww.hooks import (
    get_global_registry,
    HookPriority,
    HookPhase,
    HookContext,
)
from ww.hooks.memory import CreateHook, RecallHook, AccessHook
from ww.hooks.storage import QueryHook, ErrorHook, RetryHook
from ww.hooks.mcp import ToolCallHook, RateLimitHook
from ww.hooks.consolidation import (
    PostConsolidateHook,
    EntityExtractedHook,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Observability Hooks
# ============================================================================

class OpenTelemetryTracingHook(CreateHook):
    """
    Integrate OpenTelemetry tracing with memory operations.

    Adds spans for memory creation with relevant attributes.
    """

    def __init__(self):
        super().__init__(
            name="otel_tracing",
            priority=HookPriority.HIGH,
        )

    async def execute(self, context: HookContext) -> HookContext:
        from ww.observability.tracing import (
            add_span_attribute,
            add_span_event,
        )

        if context.phase == HookPhase.PRE:
            # Add input attributes
            add_span_attribute(
                "memory.type",
                context.input_data.get("memory_type", "unknown"),
            )
            add_span_attribute(
                "content.length",
                len(context.input_data.get("content", "")),
            )
            add_span_event("memory_creation_started")

        elif context.phase == HookPhase.POST:
            # Add output attributes
            if context.output_data:
                add_span_attribute(
                    "memory.id",
                    str(context.output_data.get("memory_id")),
                )
                add_span_attribute(
                    "creation.duration_ms",
                    context.elapsed_ms(),
                )
            add_span_event("memory_creation_completed")

        return context


class PrometheusMetricsHook(RecallHook):
    """
    Export Prometheus metrics for recall operations.

    Tracks:
    - Recall latency histogram
    - Recall count by memory type
    - Cache hit rate
    """

    def __init__(self):
        super().__init__(
            name="prometheus_metrics",
            priority=HookPriority.HIGH,
        )
        # In production, use prometheus_client library
        self.recall_count = {}
        self.recall_latencies = []
        self.cache_hits = 0
        self.cache_misses = 0

    async def execute(self, context: HookContext) -> HookContext:
        mem_type = context.input_data.get("memory_type", "unknown")

        if context.phase == HookPhase.PRE:
            context.metadata["recall_start"] = time.time()

        elif context.phase == HookPhase.POST:
            # Record latency
            start = context.metadata.get("recall_start", time.time())
            latency_ms = (time.time() - start) * 1000
            self.recall_latencies.append(latency_ms)

            # Count recalls by type
            self.recall_count[mem_type] = self.recall_count.get(mem_type, 0) + 1

            # Track cache hits
            if context.metadata.get("cache_hit"):
                self.cache_hits += 1
            else:
                self.cache_misses += 1

        return context

    def get_metrics(self) -> dict:
        """Export metrics for Prometheus scraping."""
        total_recalls = sum(self.recall_count.values())
        total_cache = self.cache_hits + self.cache_misses

        return {
            "recall_total": total_recalls,
            "recall_by_type": self.recall_count,
            "recall_latency_avg_ms": (
                sum(self.recall_latencies) / len(self.recall_latencies)
                if self.recall_latencies else 0
            ),
            "cache_hit_rate": (
                self.cache_hits / total_cache
                if total_cache > 0 else 0
            ),
        }


class StructuredLoggingHook(ToolCallHook):
    """
    Structured logging for MCP tool calls.

    Logs JSON-formatted entries with:
    - Request ID
    - Tool name and arguments
    - Execution time
    - Success/failure
    """

    def __init__(self):
        super().__init__(
            name="structured_logging",
            priority=HookPriority.NORMAL,
        )

    async def execute(self, context: HookContext) -> HookContext:
        if context.phase == HookPhase.POST:
            log_entry = {
                "timestamp": context.start_time.isoformat(),
                "request_id": str(context.hook_id),
                "tool_name": context.input_data.get("tool_name"),
                "session_id": context.session_id,
                "duration_ms": context.elapsed_ms(),
                "success": context.output_data.get("success", False) if context.output_data else False,
                "error": str(context.error) if context.error else None,
            }

            logger.info("Tool call completed", extra=log_entry)

        return context


# ============================================================================
# Example 2: Caching Hooks
# ============================================================================

class EmbeddingCacheHook(CreateHook):
    """
    Cache embeddings to avoid redundant computations.

    Stores embeddings by content hash for reuse.
    """

    def __init__(self, max_size: int = 10000):
        super().__init__(
            name="embedding_cache",
            priority=HookPriority.HIGH,
        )
        self.cache: dict[str, list[float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_content(self, content: str) -> str:
        """Create stable hash of content."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()

    async def execute(self, context: HookContext) -> HookContext:
        content = context.input_data.get("content", "")
        cache_key = self._hash_content(content)

        if context.phase == HookPhase.PRE:
            # Check cache
            if cache_key in self.cache:
                self.hits += 1
                context.metadata["cached_embedding"] = self.cache[cache_key]
                context.metadata["embedding_cache_hit"] = True
                logger.debug(f"Embedding cache HIT: {cache_key[:16]}...")
            else:
                self.misses += 1
                context.metadata["embedding_cache_hit"] = False

        elif context.phase == HookPhase.POST:
            # Store embedding in cache
            if not context.metadata.get("embedding_cache_hit"):
                if context.output_data:
                    embedding = context.output_data.get("embedding")
                    if embedding:
                        # Evict if needed
                        if len(self.cache) >= self.max_size:
                            # Remove oldest (first) entry
                            self.cache.pop(next(iter(self.cache)))

                        self.cache[cache_key] = embedding
                        logger.debug(f"Embedding cache STORE: {cache_key[:16]}...")

        return context


class SemanticQueryCacheHook(RecallHook):
    """
    Cache semantic query results with TTL.

    Only caches read operations, with time-based expiration.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        super().__init__(
            name="query_cache",
            priority=HookPriority.HIGH,
        )
        self.cache: dict[str, tuple[float, Any]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def _make_cache_key(self, context: HookContext) -> str:
        """Generate cache key from query."""
        query = context.input_data.get("query", "")
        mem_type = context.input_data.get("memory_type", "")
        limit = context.input_data.get("limit", 10)
        return f"{mem_type}:{query}:{limit}"

    async def execute(self, context: HookContext) -> HookContext:
        cache_key = self._make_cache_key(context)
        now = time.time()

        if context.phase == HookPhase.PRE:
            # Check cache
            if cache_key in self.cache:
                timestamp, cached_result = self.cache[cache_key]

                # Check TTL
                if now - timestamp < self.ttl:
                    context.metadata["cache_hit"] = True
                    context.metadata["cached_result"] = cached_result
                    logger.debug(f"Query cache HIT: {cache_key[:50]}...")
                    return context
                else:
                    # Expired
                    del self.cache[cache_key]

            context.metadata["cache_hit"] = False

        elif context.phase == HookPhase.POST:
            # Store in cache
            if not context.metadata.get("cache_hit"):
                if context.output_data:
                    # Evict if needed
                    if len(self.cache) >= self.max_size:
                        # Remove oldest
                        oldest_key = min(
                            self.cache.keys(),
                            key=lambda k: self.cache[k][0],
                        )
                        del self.cache[oldest_key]

                    self.cache[cache_key] = (now, context.output_data)
                    logger.debug(f"Query cache STORE: {cache_key[:50]}...")

        return context


# ============================================================================
# Example 3: Audit and Compliance Hooks
# ============================================================================

class GDPRComplianceHook(AccessHook):
    """
    Track data access for GDPR compliance.

    Records:
    - Who accessed what data
    - When access occurred
    - Purpose of access
    """

    def __init__(self, audit_db: Optional[Any] = None):
        super().__init__(
            name="gdpr_compliance",
            priority=HookPriority.CRITICAL,
        )
        self.audit_db = audit_db or []

    async def execute(self, context: HookContext) -> HookContext:
        audit_record = {
            "timestamp": context.start_time.isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "memory_id": context.input_data.get("memory_id"),
            "memory_type": context.input_data.get("memory_type"),
            "access_type": context.input_data.get("access_type"),
            "purpose": context.metadata.get("access_purpose", "unspecified"),
        }

        # Store audit record
        if isinstance(self.audit_db, list):
            self.audit_db.append(audit_record)
        else:
            # Write to actual database
            await self._store_audit_record(audit_record)

        logger.info(f"GDPR audit: {audit_record}")
        return context

    async def _store_audit_record(self, record: dict) -> None:
        """Store audit record in compliance database."""
        # Implementation depends on audit storage backend
        pass


class SecurityAuditHook(ToolCallHook):
    """
    Security audit trail for tool calls.

    Tracks:
    - Authentication/authorization
    - Input validation failures
    - Rate limit violations
    - Suspicious patterns
    """

    def __init__(self):
        super().__init__(
            name="security_audit",
            priority=HookPriority.CRITICAL,
        )
        self.security_events = []

    async def execute(self, context: HookContext) -> HookContext:
        # Check for security-relevant events
        if context.phase == HookPhase.POST:
            security_event = None

            # Rate limit violation
            if context.metadata.get("rate_limited"):
                security_event = {
                    "type": "rate_limit_violation",
                    "severity": "warning",
                }

            # Validation error
            elif context.metadata.get("validation_error"):
                security_event = {
                    "type": "validation_error",
                    "severity": "warning",
                    "field": context.metadata.get("invalid_field"),
                }

            # Authentication failure
            elif context.error and "unauthorized" in str(context.error).lower():
                security_event = {
                    "type": "auth_failure",
                    "severity": "high",
                }

            if security_event:
                security_event.update({
                    "timestamp": context.start_time.isoformat(),
                    "session_id": context.session_id,
                    "tool_name": context.input_data.get("tool_name"),
                })
                self.security_events.append(security_event)
                logger.warning(f"Security event: {security_event}")

        return context


# ============================================================================
# Example 4: Performance Hooks
# ============================================================================

class QueryPerformanceHook(QueryHook):
    """
    Analyze query performance and identify optimization opportunities.

    Tracks:
    - Slow queries
    - Query patterns
    - Result set sizes
    """

    def __init__(self, slow_threshold_ms: float = 1000):
        super().__init__(
            name="query_performance",
            priority=HookPriority.NORMAL,
        )
        self.slow_threshold = slow_threshold_ms
        self.query_stats = []
        self.slow_queries = []

    async def execute(self, context: HookContext) -> HookContext:
        if context.phase == HookPhase.PRE:
            context.metadata["query_start"] = time.time()

        elif context.phase == HookPhase.POST:
            start = context.metadata.get("query_start", time.time())
            duration_ms = (time.time() - start) * 1000

            stats = {
                "duration_ms": duration_ms,
                "storage_type": context.input_data.get("storage_type"),
                "query_type": context.input_data.get("query_type"),
                "row_count": context.output_data.get("row_count", 0) if context.output_data else 0,
            }

            self.query_stats.append(stats)

            # Log slow queries
            if duration_ms > self.slow_threshold:
                query = context.input_data.get("query", "")
                slow_query = {
                    **stats,
                    "query": str(query)[:200],
                    "timestamp": context.start_time.isoformat(),
                }
                self.slow_queries.append(slow_query)
                logger.warning(
                    f"SLOW QUERY ({duration_ms:.2f}ms > {self.slow_threshold}ms): "
                    f"{slow_query['query']}"
                )

        return context

    def get_performance_report(self) -> dict:
        """Generate performance analysis report."""
        if not self.query_stats:
            return {}

        durations = [s["duration_ms"] for s in self.query_stats]
        sorted_durations = sorted(durations)

        return {
            "total_queries": len(self.query_stats),
            "slow_queries": len(self.slow_queries),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "p50_duration_ms": sorted_durations[len(sorted_durations) // 2],
            "p95_duration_ms": sorted_durations[int(len(sorted_durations) * 0.95)],
            "p99_duration_ms": sorted_durations[int(len(sorted_durations) * 0.99)],
        }


class HebbianWeightProfiler(AccessHook):
    """
    Profile Hebbian weight updates for optimization.

    Identifies:
    - Frequently co-accessed memories
    - Weight distribution patterns
    - Potential pruning candidates
    """

    def __init__(self):
        super().__init__(
            name="hebbian_profiler",
            priority=HookPriority.LOW,
        )
        self.co_access_patterns = {}
        self.access_frequencies = {}

    async def execute(self, context: HookContext) -> HookContext:
        memory_id = context.input_data.get("memory_id")
        context_ids = context.input_data.get("context_ids", [])

        # Track access frequency
        self.access_frequencies[memory_id] = (
            self.access_frequencies.get(memory_id, 0) + 1
        )

        # Track co-access patterns
        for ctx_id in context_ids:
            pair = tuple(sorted([str(memory_id), str(ctx_id)]))
            self.co_access_patterns[pair] = (
                self.co_access_patterns.get(pair, 0) + 1
            )

        return context

    def get_top_pairs(self, n: int = 10) -> list[tuple]:
        """Get top N co-accessed memory pairs."""
        return sorted(
            self.co_access_patterns.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]


# ============================================================================
# Example 5: Error Handling and Resilience
# ============================================================================

class AdaptiveCircuitBreakerHook(ErrorHook):
    """
    Adaptive circuit breaker with dynamic thresholds.

    Adjusts failure threshold based on:
    - Error rate trends
    - Recovery time
    - System load
    """

    def __init__(
        self,
        base_threshold: int = 5,
        reset_timeout: int = 60,
    ):
        super().__init__(
            name="adaptive_circuit_breaker",
            priority=HookPriority.CRITICAL,
        )
        self.base_threshold = base_threshold
        self.reset_timeout = reset_timeout
        self.failures = {}
        self.open_circuits = {}
        self.recovery_times = {}

    async def execute(self, context: HookContext) -> HookContext:
        storage_type = context.input_data.get("storage_type", "unknown")
        now = time.time()

        # Initialize tracking
        if storage_type not in self.failures:
            self.failures[storage_type] = []

        # Record failure
        self.failures[storage_type].append(now)

        # Clean old failures
        cutoff = now - self.reset_timeout
        self.failures[storage_type] = [
            t for t in self.failures[storage_type]
            if t > cutoff
        ]

        # Adaptive threshold based on recovery time
        threshold = self.base_threshold
        if storage_type in self.recovery_times:
            avg_recovery = sum(self.recovery_times[storage_type]) / len(
                self.recovery_times[storage_type]
            )
            # Increase threshold if recovery is fast
            if avg_recovery < 5:
                threshold = self.base_threshold * 2

        # Check threshold
        if len(self.failures[storage_type]) >= threshold:
            if storage_type not in self.open_circuits:
                self.open_circuits[storage_type] = now
                logger.error(
                    f"CIRCUIT BREAKER OPEN: {storage_type} "
                    f"({len(self.failures[storage_type])} failures, "
                    f"threshold: {threshold})"
                )
        else:
            # Close circuit and record recovery time
            if storage_type in self.open_circuits:
                open_time = self.open_circuits[storage_type]
                recovery_time = now - open_time

                if storage_type not in self.recovery_times:
                    self.recovery_times[storage_type] = []
                self.recovery_times[storage_type].append(recovery_time)

                del self.open_circuits[storage_type]
                logger.info(
                    f"CIRCUIT BREAKER CLOSED: {storage_type} "
                    f"(recovery time: {recovery_time:.2f}s)"
                )

        context.metadata["circuit_open"] = storage_type in self.open_circuits
        return context


class IntelligentRetryHook(RetryHook):
    """
    Intelligent retry with error classification.

    Decides retry strategy based on error type:
    - Transient errors: Retry with backoff
    - Permanent errors: Fail fast
    - Rate limit errors: Wait for reset
    """

    def __init__(self):
        super().__init__(
            name="intelligent_retry",
            priority=HookPriority.CRITICAL,
        )

    def _classify_error(self, error: Exception) -> str:
        """Classify error for retry decision."""
        error_str = str(error).lower()

        if any(x in error_str for x in ["timeout", "connection", "network"]):
            return "transient"
        elif "rate limit" in error_str:
            return "rate_limit"
        elif any(x in error_str for x in ["not found", "invalid", "forbidden"]):
            return "permanent"
        else:
            return "unknown"

    async def execute(self, context: HookContext) -> HookContext:
        error = context.input_data.get("error")
        attempt = context.input_data.get("attempt", 1)
        max_attempts = context.input_data.get("max_attempts", 3)

        # Classify error
        error_class = self._classify_error(error) if error else "unknown"

        # Decide retry strategy
        should_retry = False
        backoff_ms = 0

        if error_class == "transient" and attempt < max_attempts:
            should_retry = True
            # Exponential backoff
            backoff_ms = 100 * (2 ** (attempt - 1))
        elif error_class == "rate_limit":
            should_retry = True
            # Wait for rate limit reset
            backoff_ms = context.input_data.get("retry_after", 60) * 1000
        elif error_class == "permanent":
            should_retry = False
            logger.warning(f"Permanent error detected, not retrying: {error}")

        # Update context
        if context.input_data:
            context.input_data["should_retry"] = should_retry
            context.input_data["backoff_ms"] = backoff_ms

        if should_retry:
            logger.info(
                f"Retry decision: {error_class} error, "
                f"retry in {backoff_ms}ms (attempt {attempt}/{max_attempts})"
            )
            await asyncio.sleep(backoff_ms / 1000)

        return context


# ============================================================================
# Example 6: Consolidation Hooks
# ============================================================================

class ConsolidationDashboardHook(PostConsolidateHook):
    """
    Generate real-time consolidation dashboard metrics.

    Tracks and reports:
    - Processing rate
    - Memory efficiency gains
    - Quality metrics
    """

    def __init__(self):
        super().__init__(
            name="consolidation_dashboard",
            priority=HookPriority.NORMAL,
        )
        self.runs = []

    async def execute(self, context: HookContext) -> HookContext:
        if not context.output_data:
            return context

        metrics = {
            "timestamp": context.start_time.isoformat(),
            "type": context.input_data.get("consolidation_type"),
            "duration_s": context.elapsed_ms() / 1000,
            "episodes_processed": context.output_data.get("episodes_processed", 0),
            "duplicates_removed": context.output_data.get("duplicates_removed", 0),
            "entities_extracted": context.output_data.get("entities_extracted", 0),
            "clusters_formed": context.output_data.get("clusters_formed", 0),
        }

        # Calculate derived metrics
        if metrics["duration_s"] > 0:
            metrics["processing_rate"] = (
                metrics["episodes_processed"] / metrics["duration_s"]
            )

        if metrics["episodes_processed"] > 0:
            metrics["dedup_rate"] = (
                metrics["duplicates_removed"] / metrics["episodes_processed"]
            )

        self.runs.append(metrics)

        # Log dashboard update
        logger.info(
            f"Consolidation metrics: "
            f"{metrics['episodes_processed']} episodes in {metrics['duration_s']:.2f}s "
            f"({metrics.get('processing_rate', 0):.2f} eps/s), "
            f"{metrics['duplicates_removed']} duplicates removed, "
            f"{metrics['entities_extracted']} entities extracted"
        )

        return context


class EntityQualityHook(EntityExtractedHook):
    """
    Validate and score entity extraction quality.

    Checks:
    - Confidence thresholds
    - Entity name validity
    - Type consistency
    """

    def __init__(self, min_confidence: float = 0.7):
        super().__init__(
            name="entity_quality",
            priority=HookPriority.HIGH,
        )
        self.min_confidence = min_confidence
        self.quality_scores = []

    async def execute(self, context: HookContext) -> HookContext:
        entity_name = context.input_data.get("entity_name", "")
        entity_type = context.input_data.get("entity_type", "")
        confidence = context.input_data.get("confidence", 0.0)

        # Calculate quality score
        quality_score = 0.0

        # Confidence component (0-40 points)
        quality_score += min(confidence * 40, 40)

        # Name validity (0-30 points)
        if len(entity_name) >= 2 and entity_name[0].isupper():
            quality_score += 30

        # Type validity (0-30 points)
        valid_types = {"person", "organization", "location", "concept"}
        if entity_type in valid_types:
            quality_score += 30

        # Normalize to 0-1
        quality_score /= 100

        self.quality_scores.append(quality_score)

        context.metadata["quality_score"] = quality_score

        if quality_score < 0.7:
            logger.warning(
                f"Low quality entity: {entity_name} "
                f"(score: {quality_score:.2f})"
            )

        return context


# ============================================================================
# Setup Example
# ============================================================================

def setup_production_hooks():
    """
    Example: Set up comprehensive hooks for production deployment.

    Configures:
    - Observability (tracing, metrics, logging)
    - Performance (caching, profiling)
    - Security (audit, compliance)
    - Resilience (circuit breaker, retry)
    """
    from ww.hooks.registry import (
        get_global_registry,
        REGISTRY_EPISODIC,
        REGISTRY_SEMANTIC,
        REGISTRY_STORAGE_NEO4J,
        REGISTRY_MCP,
        REGISTRY_CONSOLIDATION,
    )

    # Episodic memory hooks
    episodic = get_global_registry(REGISTRY_EPISODIC)
    episodic.register(OpenTelemetryTracingHook(), HookPhase.PRE)
    episodic.register(OpenTelemetryTracingHook(), HookPhase.POST)
    episodic.register(EmbeddingCacheHook(), HookPhase.PRE)
    episodic.register(EmbeddingCacheHook(), HookPhase.POST)

    # Semantic memory hooks
    semantic = get_global_registry(REGISTRY_SEMANTIC)
    semantic.register(PrometheusMetricsHook(), HookPhase.PRE)
    semantic.register(PrometheusMetricsHook(), HookPhase.POST)
    semantic.register(SemanticQueryCacheHook(), HookPhase.PRE)
    semantic.register(SemanticQueryCacheHook(), HookPhase.POST)
    semantic.register(HebbianWeightProfiler(), HookPhase.ON)

    # Storage hooks
    storage = get_global_registry(REGISTRY_STORAGE_NEO4J)
    storage.register(QueryPerformanceHook(), HookPhase.PRE)
    storage.register(QueryPerformanceHook(), HookPhase.POST)
    storage.register(AdaptiveCircuitBreakerHook(), HookPhase.ERROR)
    storage.register(IntelligentRetryHook(), HookPhase.ON)

    # MCP hooks
    mcp = get_global_registry(REGISTRY_MCP)
    mcp.register(StructuredLoggingHook(), HookPhase.POST)
    mcp.register(SecurityAuditHook(), HookPhase.POST)
    mcp.register(GDPRComplianceHook(), HookPhase.ON)

    # Consolidation hooks
    consolidation = get_global_registry(REGISTRY_CONSOLIDATION)
    consolidation.register(ConsolidationDashboardHook(), HookPhase.POST)
    consolidation.register(EntityQualityHook(), HookPhase.ON)

    logger.info("Production hooks configured successfully")


if __name__ == "__main__":
    # Example: Set up and test hooks
    setup_production_hooks()

    # Test with example operation
    async def test_example():
        from ww.hooks.base import HookContext

        registry = get_global_registry("episodic")

        context = HookContext(
            operation="create",
            module="episodic",
            input_data={
                "memory_type": "episodic",
                "content": "This is a test memory",
            },
        )

        # Execute hooks
        context = await registry.execute_phase(HookPhase.PRE, context)
        print(f"PRE hooks executed: {context.metadata}")

    asyncio.run(test_example())
