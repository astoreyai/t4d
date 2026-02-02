# T4DM Hooks - Quick Reference

## Installation

```python
from t4dm.hooks import get_global_registry, HookPriority, HookPhase
```

## Creating a Hook

```python
from t4dm.hooks.memory import CreateHook

class MyCustomHook(CreateHook):
    def __init__(self):
        super().__init__(
            name="my_custom_hook",
            priority=HookPriority.NORMAL,
            enabled=True,
        )

    async def execute(self, context: HookContext) -> HookContext:
        # Your logic here
        if context.phase == HookPhase.PRE:
            # Before operation
            pass
        elif context.phase == HookPhase.POST:
            # After operation
            pass
        return context
```

## Registering a Hook

```python
registry = get_global_registry("episodic")
registry.register(MyCustomHook(), HookPhase.PRE)
registry.register(MyCustomHook(), HookPhase.POST)
```

## Using Hook Decorator

```python
from t4dm.hooks.base import with_hooks

@with_hooks(registry, operation="create_episode", module="episodic")
async def create_episode(content: str, **kwargs) -> Episode:
    # Your function logic
    return episode
```

## Hook Priority Levels

```python
HookPriority.CRITICAL = 0      # Security, validation, auth
HookPriority.HIGH = 100        # Observability, auditing
HookPriority.NORMAL = 500      # Business logic
HookPriority.LOW = 1000        # Caching, cleanup
```

**Execution Order:**
- **PRE:** CRITICAL → HIGH → NORMAL → LOW
- **POST:** LOW → NORMAL → HIGH → CRITICAL

## Hook Phases

| Phase | When | Use For | Can Modify |
|-------|------|---------|------------|
| PRE | Before operation | Validation, auth, cache check | Input data |
| POST | After operation | Logging, metrics, cache update | Output data |
| ON | Event notification | Connections, state changes | Side effects |
| ERROR | Error occurred | Recovery, retry, alerts | Error handling |

## Context Data Access

### Input Data (PRE phase)

```python
async def execute(self, context: HookContext) -> HookContext:
    if context.phase == HookPhase.PRE:
        # Read input
        value = context.input_data.get("key")

        # Modify input (will affect operation)
        context.input_data["key"] = new_value
```

### Output Data (POST phase)

```python
async def execute(self, context: HookContext) -> HookContext:
    if context.phase == HookPhase.POST:
        # Read output
        result = context.output_data.get("result")

        # Add metadata
        context.output_data["_metadata"] = {...}
```

### Metadata (Any phase)

```python
# Store custom data
context.metadata["custom_key"] = value

# Read in later hooks
value = context.metadata.get("custom_key")
```

## Common Patterns

### 1. Caching

```python
class CacheHook(RecallHook):
    def __init__(self):
        super().__init__(name="cache", priority=HookPriority.HIGH)
        self.cache = {}

    async def execute(self, context):
        key = context.input_data.get("query")

        if context.phase == HookPhase.PRE:
            if key in self.cache:
                context.metadata["cached"] = self.cache[key]

        elif context.phase == HookPhase.POST:
            if not context.metadata.get("cached"):
                self.cache[key] = context.output_data

        return context
```

### 2. Timing

```python
class TimingHook(QueryHook):
    async def execute(self, context):
        import time

        if context.phase == HookPhase.PRE:
            context.metadata["start"] = time.time()

        elif context.phase == HookPhase.POST:
            duration = time.time() - context.metadata["start"]
            logger.info(f"Query took {duration:.3f}s")

        return context
```

### 3. Validation

```python
class ValidationHook(CreateHook):
    async def execute(self, context):
        if context.phase == HookPhase.PRE:
            content = context.input_data.get("content", "")
            if len(content) > 100000:
                raise ValueError("Content too long")
        return context
```

### 4. Audit Logging

```python
class AuditHook(MemoryHook):
    def __init__(self):
        super().__init__(name="audit", priority=HookPriority.HIGH)
        self.log = []

    async def execute(self, context):
        if context.phase == HookPhase.POST:
            self.log.append({
                "timestamp": context.start_time.isoformat(),
                "operation": context.operation,
                "session": context.session_id,
                "success": context.error is None,
            })
        return context
```

### 5. Error Handling

```python
class RetryHook(ErrorHook):
    async def execute(self, context):
        error_type = self._classify_error(context.error)

        if error_type == "transient":
            context.metadata["should_retry"] = True
            context.metadata["backoff_ms"] = 1000
        else:
            context.metadata["should_retry"] = False

        return context

    async def on_error(self, context):
        logger.error(f"Operation failed: {context.error}")
```

## Registry Operations

### Create Registry

```python
from t4dm.hooks import HookRegistry

registry = HookRegistry(
    name="my_registry",
    fail_fast=False,      # Continue on errors
    max_concurrent=10,    # Limit parallel hooks
)
```

### Execute Phase

```python
# Sequential (PRE hooks)
await registry.execute_phase(HookPhase.PRE, context)

# Parallel (POST hooks)
await registry.execute_phase(
    HookPhase.POST,
    context,
    parallel=True
)
```

### Get Statistics

```python
stats = registry.get_stats()
# {
#   "total_hooks": 10,
#   "hooks_by_phase": {...},
#   "hook_stats": [...]
# }
```

### Clear Hooks

```python
# Remove specific hook
registry.unregister("hook_name")

# Remove from specific phase
registry.unregister("hook_name", HookPhase.PRE)

# Clear all
registry.clear()
```

## Standard Registries

```python
from t4dm.hooks.registry import (
    get_global_registry,
    REGISTRY_EPISODIC,
    REGISTRY_SEMANTIC,
    REGISTRY_PROCEDURAL,
    REGISTRY_CONSOLIDATION,
    REGISTRY_STORAGE_NEO4J,
    REGISTRY_STORAGE_QDRANT,
    REGISTRY_MCP,
)

episodic = get_global_registry(REGISTRY_EPISODIC)
```

## Hook Base Classes

| Base Class | Module | Use For |
|------------|--------|---------|
| `Hook` | base | Generic hooks |
| `CoreHook` | core | Init, shutdown, health |
| `MemoryHook` | memory | Memory operations |
| `CreateHook` | memory | Memory creation |
| `RecallHook` | memory | Memory retrieval |
| `UpdateHook` | memory | Memory updates |
| `AccessHook` | memory | Access tracking |
| `DecayHook` | memory | FSRS decay |
| `StorageHook` | storage | Storage operations |
| `ConnectionHook` | storage | Connect/disconnect |
| `QueryHook` | storage | Query execution |
| `ErrorHook` | storage | Error handling |
| `RetryHook` | storage | Retry logic |
| `MCPHook` | mcp | MCP operations |
| `ToolCallHook` | mcp | Tool execution |
| `RateLimitHook` | mcp | Rate limiting |
| `ValidationErrorHook` | mcp | Validation errors |
| `ConsolidationHook` | consolidation | Consolidation |
| `ClusterFormHook` | consolidation | Clustering |
| `EntityExtractedHook` | consolidation | Entity extraction |

## Filtering

### By Memory Type

```python
class EpisodicOnlyHook(CreateHook):
    def __init__(self):
        super().__init__(
            name="episodic_only",
            memory_type="episodic"  # Filter
        )
```

### By Storage Type

```python
class Neo4jOnlyHook(QueryHook):
    def __init__(self):
        super().__init__(
            name="neo4j_only",
            storage_type="neo4j"  # Filter
        )
```

### By Tool Name

```python
class SpecificToolHook(ToolCallHook):
    def __init__(self):
        super().__init__(
            name="specific_tool",
            tool_filter="create_episode"  # Filter
        )
```

### Custom Filtering

```python
class CustomFilterHook(Hook):
    def should_execute(self, context):
        # Custom logic
        return context.session_id == "production"
```

## Error Handling

### Isolated Errors (Default)

```python
registry = HookRegistry(fail_fast=False)
# One hook failure doesn't stop others
```

### Fail Fast

```python
registry = HookRegistry(fail_fast=True)
# First hook error raises HookError
```

### Error Hooks

```python
class ErrorRecoveryHook(Hook):
    async def on_error(self, context):
        # Called when any hook fails
        logger.error(f"Hook error: {context.error}")
```

Register in ERROR phase:
```python
registry.register(ErrorRecoveryHook(), HookPhase.ERROR)
```

## Testing Hooks

```python
import pytest

@pytest.mark.asyncio
async def test_my_hook():
    hook = MyCustomHook()
    context = HookContext(
        operation="test",
        input_data={"key": "value"},
    )

    result = await hook.execute(context)

    assert result.metadata.get("processed") is True
```

## Best Practices

1. **Keep hooks lightweight** - Avoid expensive operations
2. **Use appropriate priority** - CRITICAL for security, LOW for cleanup
3. **Handle errors gracefully** - Don't crash on hook errors
4. **Document context data** - What inputs/outputs do you expect?
5. **Test in isolation** - Unit test each hook separately
6. **Monitor performance** - Use `get_stats()` to track execution
7. **Use filters** - Target hooks to specific operations
8. **Avoid dependencies** - Don't rely on other hooks' execution

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("t4dm.hooks").setLevel(logging.DEBUG)
```

### Inspect Context

```python
async def execute(self, context):
    print(f"Context: {context.to_dict()}")
    return context
```

### Check Hook Stats

```python
stats = hook.get_stats()
print(f"Executions: {stats['executions']}")
print(f"Errors: {stats['errors']}")
print(f"Error rate: {stats['error_rate']:.2%}")
```

## Performance Tips

1. **Use parallel POST hooks** - Independent operations run concurrently
2. **Cache expensive results** - Store in context.metadata or hook state
3. **Limit concurrency** - Set `max_concurrent` to prevent overload
4. **Profile slow hooks** - Monitor execution time
5. **Conditional execution** - Use `should_execute()` to skip unnecessary work

## Common Gotchas

❌ **Don't modify context in POST if parallel=True**
```python
# Multiple hooks modifying same data = race condition
```

✅ **Use metadata instead**
```python
context.metadata["hook_data"] = value
```

❌ **Don't rely on hook execution order in POST parallel**
```python
# Execution order is not guaranteed
```

✅ **Make POST hooks independent**
```python
# Each hook should work standalone
```

❌ **Don't raise exceptions for normal flow**
```python
raise ValueError("Invalid input")  # Stops all hooks
```

✅ **Use metadata for signaling**
```python
context.metadata["validation_failed"] = True
```

## File Locations

```
src/t4dm/hooks/
├── __init__.py              # Public API
├── base.py                  # Core classes
├── core.py                  # Core hooks
├── memory.py                # Memory hooks
├── storage.py               # Storage hooks
├── mcp.py                   # MCP hooks
├── consolidation.py         # Consolidation hooks
└── registry.py              # Registry management

docs/
├── hooks_design.md          # Full design doc
└── hooks_quick_reference.md # This file

examples/
└── hooks_examples.py        # Usage examples

tests/
└── test_hooks.py            # Test suite
```

## Resources

- **Design Document:** `/mnt/projects/t4d/t4dm/docs/hooks_design.md`
- **Examples:** `/mnt/projects/t4d/t4dm/examples/hooks_examples.py`
- **Tests:** `/mnt/projects/t4d/t4dm/tests/test_hooks.py`
- **Source Code:** `/mnt/projects/t4d/t4dm/src/t4dm/hooks/`
