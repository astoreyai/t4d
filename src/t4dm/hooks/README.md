# Hooks Module

**7 files | ~2,200 lines | Centrality: 0**

The hooks module provides a flexible event system for extending T4DM's memory lifecycle with custom processing, logging, caching, and validation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HOOK EXECUTION FLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PRE Phase          Operation          POST Phase         ERROR Phase  │
│   (Sequential)       (Execute)          (Parallel)         (If failed)  │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   ┌───────────┐ │
│  │ CRITICAL(0) │    │             │    │  LOW(1000)  │   │  ERROR    │ │
│  │ Validation  │───▶│   Memory    │───▶│  Caching    │──▶│  Handler  │ │
│  │ Auth check  │    │  Operation  │    │  Cleanup    │   │  Recovery │ │
│  ├─────────────┤    │             │    ├─────────────┤   └───────────┘ │
│  │  HIGH(100)  │    │             │    │ NORMAL(500) │                 │
│  │ Observability│    │             │    │ Logging     │                 │
│  ├─────────────┤    │             │    ├─────────────┤                 │
│  │ NORMAL(500) │    │             │    │  HIGH(100)  │                 │
│  │ Business    │    │             │    │ Metrics     │                 │
│  ├─────────────┤    │             │    ├─────────────┤                 │
│  │  LOW(1000)  │    │             │    │ CRITICAL(0) │                 │
│  │ Cache check │    │             │    │ Audit       │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `base.py` | ~530 | Core classes: Hook, HookRegistry, HookContext, HookPhase |
| `core.py` | ~265 | Lifecycle hooks: InitHook, ShutdownHook, HealthCheckHook |
| `memory.py` | ~385 | Memory hooks: CreateHook, RecallHook, UpdateHook, AccessHook |
| `storage.py` | ~450 | Storage hooks: QueryHook, ConnectionHook, ErrorHook, RetryHook |
| `consolidation.py` | ~425 | Consolidation hooks: DuplicateFoundHook, ClusterFormHook |
| `registry.py` | ~95 | Global registry management |
| `__init__.py` | ~85 | Public API exports |

## Hook Phases

| Phase | Execution | Order | Use Cases |
|-------|-----------|-------|-----------|
| **PRE** | Sequential | CRITICAL → LOW | Validation, auth, caching, preprocessing |
| **POST** | Parallel | LOW → CRITICAL | Logging, metrics, notifications, cleanup |
| **ON** | Parallel | Any | Event notifications, state changes |
| **ERROR** | Sequential | CRITICAL → LOW | Error handling, retry, recovery |

## Priority Levels

```python
from t4dm.hooks import HookPriority

CRITICAL = 0      # Security, validation, auth
HIGH = 100        # Observability, auditing
NORMAL = 500      # Business logic (default)
LOW = 1000        # Caching, cleanup
```

**PRE Phase**: Lower priority runs first (security before business logic)
**POST Phase**: Higher priority runs first (reversed order)

## Quick Start

### Creating a Hook

```python
from t4dm.hooks import Hook, HookContext, HookPhase, HookPriority

class LoggingHook(Hook):
    """Log all memory operations."""

    name = "logging_hook"
    phase = HookPhase.POST
    priority = HookPriority.HIGH

    async def execute(self, context: HookContext) -> None:
        print(f"Operation: {context.operation}")
        print(f"Result: {context.output_data}")

    def should_execute(self, context: HookContext) -> bool:
        # Optional: conditional execution
        return context.operation != "health_check"
```

### Registering Hooks

```python
from t4dm.hooks import HookRegistry

registry = HookRegistry(name="episodic_memory")

# Register hook
registry.register(LoggingHook())

# Execute phase
await registry.execute_phase(HookPhase.POST, context)
```

### Using the Decorator

```python
from t4dm.hooks import with_hooks

@with_hooks(registry_name="episodic")
async def store_episode(content: str, **kwargs) -> Episode:
    # PRE hooks run before
    episode = await _do_store(content)
    # POST hooks run after
    return episode
```

## Standard Registries

| Registry | Purpose | Location |
|----------|---------|----------|
| `REGISTRY_CORE` | System lifecycle | Init, shutdown, health |
| `REGISTRY_EPISODIC` | Episodic memory | Create, recall, update |
| `REGISTRY_SEMANTIC` | Semantic memory | Entity operations |
| `REGISTRY_PROCEDURAL` | Procedural memory | Skill operations |
| `REGISTRY_CONSOLIDATION` | Consolidation | Clustering, extraction |
| `REGISTRY_STORAGE_NEO4J` | Neo4j operations | Query, connection |
| `REGISTRY_STORAGE_QDRANT` | Qdrant operations | Vector operations |
| `REGISTRY_MCP` | MCP tool calls | Tool execution |

## Hook Types

### Core Hooks (core.py)

```python
from t4dm.hooks import InitHook, ShutdownHook, HealthCheckHook, ConfigChangeHook

# Initialization
class DatabaseInitHook(InitHook):
    async def execute(self, context: HookContext) -> None:
        # context.input_data: {module_name, config, session_id}
        await self.db.connect()

# Shutdown
class GracefulShutdownHook(ShutdownHook):
    async def execute(self, context: HookContext) -> None:
        # context.input_data: {module_name, reason, timeout}
        await self.flush_buffers()

# Health check
class DBHealthHook(HealthCheckHook):
    async def execute(self, context: HookContext) -> None:
        # context.input_data: {module_name, check_type}
        status = await self.db.ping()
        context.output_data = {"status": "healthy", "latency_ms": status.latency}
```

### Memory Hooks (memory.py)

```python
from t4dm.hooks import CreateHook, RecallHook, UpdateHook, AccessHook, DecayHook

# Create hook with type filtering
class ValidationHook(CreateHook):
    memory_type = "episodic"  # Only runs for episodic

    async def execute(self, context: HookContext) -> None:
        content = context.input_data["content"]
        if len(content) < 10:
            raise ValueError("Content too short")

# Recall hook for caching
class CachingRecallHook(RecallHook):
    def __init__(self, cache_size: int = 1000):
        self._cache = LRUCache(cache_size)

    async def execute(self, context: HookContext) -> None:
        query = context.input_data["query"]
        if query in self._cache:
            context.output_data = self._cache[query]
            context.metadata["cache_hit"] = True

# Access hook for Hebbian updates
class HebbianUpdateHook(AccessHook):
    async def execute(self, context: HookContext) -> None:
        # Strengthen co-accessed memories
        memory_ids = context.input_data["context_ids"]
        for id1, id2 in combinations(memory_ids, 2):
            await self.strengthen_link(id1, id2)
```

### Storage Hooks (storage.py)

```python
from t4dm.hooks import QueryHook, ConnectionHook, ErrorHook, RetryHook

# Query timing
class QueryTimingHook(QueryHook):
    storage_type = "neo4j"  # Filter by storage
    phase = HookPhase.POST

    async def execute(self, context: HookContext) -> None:
        duration = context.elapsed_ms()
        if duration > 100:
            logger.warning(f"Slow query: {duration}ms")

# Circuit breaker
class CircuitBreakerHook(ErrorHook):
    def __init__(self, threshold: int = 5, reset_after: int = 60):
        self._failures = {}
        self._threshold = threshold

    async def execute(self, context: HookContext) -> None:
        storage = context.input_data["storage_type"]
        self._failures[storage] = self._failures.get(storage, 0) + 1

        if self._failures[storage] >= self._threshold:
            context.metadata["circuit_open"] = True
```

### Consolidation Hooks (consolidation.py)

```python
from t4dm.hooks import (
    PreConsolidateHook, PostConsolidateHook,
    DuplicateFoundHook, ClusterFormHook, EntityExtractedHook
)

# Pre-consolidation
class ConsolidationMetricsHook(PreConsolidateHook):
    async def execute(self, context: HookContext) -> None:
        context.metadata["start_count"] = await self.count_episodes()

# Duplicate detection
class DuplicateMergeHook(DuplicateFoundHook):
    async def execute(self, context: HookContext) -> None:
        # context.input_data: {memory_id_1, memory_id_2, similarity}
        id1 = context.input_data["memory_id_1"]
        id2 = context.input_data["memory_id_2"]
        await self.merge_episodes(id1, id2)

# Cluster formation
class ClusterAnalysisHook(ClusterFormHook):
    async def execute(self, context: HookContext) -> None:
        # context.input_data: {cluster_id, memory_ids, coherence}
        if context.input_data["coherence"] > 0.8:
            await self.create_concept(context.input_data)
```

## HookContext

Complete execution context for hooks:

```python
@dataclass
class HookContext:
    # Identifiers
    hook_id: UUID
    session_id: str
    user_id: str | None

    # Operation info
    operation: str
    phase: HookPhase
    module: str

    # Data
    input_data: dict      # Operation inputs
    output_data: dict     # Operation results
    metadata: dict        # Inter-hook communication

    # Timing
    start_time: datetime
    end_time: datetime | None

    # Error tracking
    error: Exception | None
    error_context: dict

    # Methods
    def elapsed_ms(self) -> float
    def set_error(self, error: Exception, **context)
    def set_result(self, **data)
    def to_dict(self) -> dict
```

## Error Handling

```python
# Error isolation (default)
registry = HookRegistry(fail_fast=False)
# One hook failure doesn't stop others

# Fail fast mode
registry = HookRegistry(fail_fast=True)
# Stop on first error

# Custom error handler
class RecoveryHook(Hook):
    phase = HookPhase.ERROR

    async def execute(self, context: HookContext) -> None:
        if isinstance(context.error, ConnectionError):
            await self.reconnect()
            context.metadata["recovered"] = True

    async def on_error(self, context: HookContext) -> None:
        # Called if this hook itself fails
        logger.error(f"Recovery failed: {context.error}")
```

## Concurrency Control

```python
# Limit concurrent hook execution
registry = HookRegistry(max_concurrent=10)

# POST hooks run in parallel by default
# PRE hooks run sequentially (order matters)
# ERROR hooks run sequentially
```

## Common Patterns

### Caching Pattern

```python
class CacheHook(RecallHook):
    phase = HookPhase.PRE
    priority = HookPriority.LOW

    def __init__(self, ttl: int = 300):
        self._cache = TTLCache(ttl=ttl)

    async def execute(self, context: HookContext) -> None:
        key = hash(context.input_data["query"])
        if key in self._cache:
            context.output_data = self._cache[key]
            context.metadata["skip_operation"] = True
```

### Audit Trail Pattern

```python
class AuditHook(Hook):
    phase = HookPhase.POST
    priority = HookPriority.CRITICAL

    async def execute(self, context: HookContext) -> None:
        await self.log_audit({
            "timestamp": context.start_time,
            "operation": context.operation,
            "session_id": context.session_id,
            "success": context.error is None,
            "duration_ms": context.elapsed_ms()
        })
```

### Retry Pattern

```python
class ExponentialBackoffHook(RetryHook):
    def __init__(self, base_delay: float = 0.1, max_delay: float = 30.0):
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute(self, context: HookContext) -> None:
        attempt = context.input_data["attempt"]
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        context.metadata["backoff_ms"] = delay * 1000
        await asyncio.sleep(delay)
```

## Testing Hooks

```python
import pytest
from t4dm.hooks import HookRegistry, HookContext, HookPhase

@pytest.fixture
def registry():
    return HookRegistry(name="test")

@pytest.fixture
def context():
    return HookContext(
        operation="test_op",
        phase=HookPhase.PRE,
        module="test",
        input_data={"key": "value"}
    )

async def test_hook_execution(registry, context):
    hook = LoggingHook()
    registry.register(hook)

    await registry.execute_phase(HookPhase.POST, context)

    assert hook.get_stats()["executions"] == 1
```

## Installation

```bash
# Hooks included in core
pip install -e "."
```

## Public API

```python
# Base classes
Hook, HookRegistry, HookContext
HookPhase, HookPriority, HookError

# Core hooks
InitHook, ShutdownHook, HealthCheckHook, ConfigChangeHook

# Memory hooks
CreateHook, RecallHook, UpdateHook, AccessHook, DecayHook

# Storage hooks
QueryHook, ConnectionHook, ErrorHook, RetryHook

# Consolidation hooks
PreConsolidateHook, PostConsolidateHook
DuplicateFoundHook, ClusterFormHook, EntityExtractedHook

# Decorator
with_hooks

# Registry constants
REGISTRY_CORE, REGISTRY_EPISODIC, REGISTRY_SEMANTIC
REGISTRY_PROCEDURAL, REGISTRY_CONSOLIDATION
REGISTRY_STORAGE_NEO4J, REGISTRY_STORAGE_QDRANT, REGISTRY_MCP
```

## Design Patterns

| Pattern | Implementation |
|---------|---------------|
| Observer | Event-driven hook notifications |
| Strategy | Pluggable hook implementations |
| Decorator | `@with_hooks` function wrapper |
| Chain of Responsibility | Sequential PRE/ERROR execution |
| Circuit Breaker | ErrorHook with failure tracking |
