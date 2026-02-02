# Hook Development Guide

Extend T4DM's behavior with custom hooks.

## Overview

The hook system allows you to inject custom logic at various points in memory operations:

- **PRE**: Before operation (validation, preprocessing)
- **ON**: During operation (tracking, events)
- **POST**: After operation (logging, learning)
- **ERROR**: On failure (recovery, alerting)

## Creating a Hook

### Basic Hook

```python
from t4dm.hooks.base import Hook, HookContext, HookPhase, HookPriority

class MyValidationHook(Hook):
    """Validate memory content before storage."""

    name = "my_validation"
    priority = HookPriority.CRITICAL  # Run early

    async def execute(self, context: HookContext) -> HookContext:
        content = context.input_data.get("content", "")

        # Validate content
        if len(content) < 10:
            raise ValueError("Content too short")

        if any(word in content.lower() for word in ["secret", "password"]):
            raise ValueError("Content contains sensitive information")

        return context

    def should_execute(self, context: HookContext) -> bool:
        # Only run for create operations
        return context.operation == "create"
```

### Registering Hooks

```python
from t4dm.hooks import get_global_registry, HookPhase

# Get the episodic memory registry
registry = get_global_registry("episodic")

# Register for PRE phase
registry.register(MyValidationHook(), HookPhase.PRE)
```

## Hook Priorities

Hooks execute in priority order (lowest first):

| Priority | Value | Use Case |
|----------|-------|----------|
| `CRITICAL` | 0 | Security, validation |
| `HIGH` | 100 | Observability, auditing |
| `NORMAL` | 500 | Business logic |
| `LOW` | 1000 | Caching, cleanup |

## Built-in Hooks

### CachingRecallHook

Cache query results for faster repeated queries:

```python
from t4dm.hooks.memory import CachingRecallHook

hook = CachingRecallHook(
    cache_size=1000,
    ttl_seconds=300
)
registry.register(hook, HookPhase.PRE)
```

### AuditTrailHook

Log all memory operations:

```python
from t4dm.hooks.memory import AuditTrailHook

hook = AuditTrailHook(
    log_file="audit.log",
    include_content=False  # Privacy
)
registry.register(hook, HookPhase.POST)
```

### HebbianUpdateHook

Strengthen co-retrieved memories:

```python
from t4dm.hooks.memory import HebbianUpdateHook

hook = HebbianUpdateHook(
    learning_rate=0.01,
    decay_rate=0.001
)
registry.register(hook, HookPhase.POST)
```

### ValidationHook

Validate input data:

```python
from t4dm.hooks.memory import ValidationHook

hook = ValidationHook(
    max_content_length=50000,
    required_fields=["content"]
)
registry.register(hook, HookPhase.PRE)
```

## Advanced Patterns

### Conditional Execution

```python
class ConditionalHook(Hook):
    def should_execute(self, context: HookContext) -> bool:
        # Only run for specific sessions
        return context.session_id.startswith("production-")
```

### Async Operations

```python
class AsyncNotificationHook(Hook):
    async def execute(self, context: HookContext) -> HookContext:
        # Non-blocking notification
        asyncio.create_task(self._send_notification(context))
        return context

    async def _send_notification(self, context: HookContext):
        # Send to external service
        ...
```

### Error Handling

```python
class RecoveryHook(Hook):
    async def on_error(self, context: HookContext) -> None:
        error = context.error
        # Log error, alert, or attempt recovery
        logger.error(f"Operation failed: {error}")

        # Optionally re-raise or suppress
        if isinstance(error, RecoverableError):
            # Suppress and continue
            context.error = None
```

### Chaining Data

```python
class EnrichmentHook(Hook):
    """Add metadata to context for downstream hooks."""

    async def execute(self, context: HookContext) -> HookContext:
        # Add data for other hooks
        context.metadata["enriched_at"] = datetime.now()
        context.metadata["source"] = "api"
        return context
```

## Hook Registries

T4DM has multiple registries for different components:

| Registry | Purpose |
|----------|---------|
| `core` | System-wide hooks |
| `episodic` | Episodic memory operations |
| `semantic` | Semantic memory operations |
| `procedural` | Procedural memory operations |
| `consolidation` | Consolidation operations |
| `storage_neo4j` | Neo4j storage hooks |
| `storage_qdrant` | Qdrant storage hooks |

```python
from t4dm.hooks import get_global_registry

# Get specific registries
episodic_reg = get_global_registry("episodic")
semantic_reg = get_global_registry("semantic")
storage_reg = get_global_registry("storage_qdrant")
```

## Testing Hooks

```python
import pytest
from t4dm.hooks.base import HookContext, HookPhase

@pytest.mark.asyncio
async def test_validation_hook():
    hook = MyValidationHook()

    # Valid context
    context = HookContext(
        hook_id=uuid4(),
        session_id="test",
        operation="create",
        phase=HookPhase.PRE,
        module="episodic",
        input_data={"content": "Valid content here"}
    )

    result = await hook.execute(context)
    assert result.error is None

    # Invalid context
    context.input_data = {"content": "short"}
    with pytest.raises(ValueError):
        await hook.execute(context)
```

## Best Practices

1. **Keep hooks focused**: One responsibility per hook
2. **Use appropriate priority**: Critical operations first
3. **Handle errors gracefully**: Implement `on_error`
4. **Avoid blocking**: Use async for I/O operations
5. **Test thoroughly**: Unit test each hook
6. **Document behavior**: Clear docstrings and comments
