# T4DM Hooks System Design

## Overview

The T4DM hooks system provides extensible lifecycle hooks for all modules with guaranteed execution order, error isolation, and comprehensive observability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Hook Registry                            │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │
│  │  PRE   │  │  POST  │  │   ON   │  │ ERROR  │            │
│  │ Hooks  │  │ Hooks  │  │ Hooks  │  │ Hooks  │            │
│  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘            │
│       │           │           │           │                 │
│       │ Priority  │ Priority  │ Priority  │ Priority        │
│       │  Order    │  Order    │  Order    │  Order          │
└───────┼───────────┼───────────┼───────────┼─────────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
   ┌────────────────────────────────────────────┐
   │          HookContext                       │
   │  • Operation metadata                      │
   │  • Input/output data                       │
   │  • Session/user context                    │
   │  • Timing information                      │
   │  • Error state                             │
   └────────────────────────────────────────────┘
```

## Hook Phases

### 1. PRE (Before Operation)

**Execution:** Sequential, priority-ordered (CRITICAL → HIGH → NORMAL → LOW)

**Use Cases:**
- Input validation
- Authentication/authorization
- Rate limiting
- Cache lookups
- Request preprocessing

**Can Modify:** Input data

### 2. POST (After Operation)

**Execution:** Parallel (default) or sequential

**Use Cases:**
- Response formatting
- Cache updates
- Metrics collection
- Audit logging
- Notifications

**Can Access:** Output data

### 3. ON (Event Notifications)

**Execution:** Parallel

**Use Cases:**
- Connection established
- Configuration changed
- Health check requested
- State transitions

**Can Trigger:** Side effects

### 4. ERROR (Error Handling)

**Execution:** Sequential

**Use Cases:**
- Error classification
- Retry logic
- Fallback mechanisms
- Alert generation
- Recovery procedures

**Can Access:** Error context

## Hook Priority Levels

```python
class HookPriority(int, Enum):
    CRITICAL = 0      # Security, validation, auth
    HIGH = 100        # Observability, auditing
    NORMAL = 500      # Business logic
    LOW = 1000        # Caching, cleanup
```

**Execution Order:**
- **PRE phase:** CRITICAL (0) → HIGH (100) → NORMAL (500) → LOW (1000)
- **POST phase:** LOW (1000) → NORMAL (500) → HIGH (100) → CRITICAL (0)

This ensures security checks run first and cleanup runs last.

## Module Hooks

### Core Module Hooks

#### `on_init` - Module Initialization
```python
class InitHook(CoreHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["module_name"]
        # - input_data["config"]
        # - input_data["session_id"]
```

**Examples:**
- Allocate resources
- Establish connections
- Warm caches
- Restore state

#### `on_shutdown` - Graceful Shutdown
```python
class ShutdownHook(CoreHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["module_name"]
        # - input_data["reason"]
        # - input_data["timeout"]
```

**Examples:**
- Close connections
- Flush buffers
- Persist state
- Release resources

#### `on_health_check` - Health Status
```python
class HealthCheckHook(CoreHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["module_name"]
        # - input_data["check_type"]  # liveness/readiness
        # Expected output:
        # - output_data["status"]  # healthy/degraded/unhealthy
        # - output_data["message"]
        # - output_data["metrics"]
```

**Examples:**
- Validate connections
- Check resource availability
- Report performance metrics

#### `on_config_change` - Configuration Reload
```python
class ConfigChangeHook(CoreHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["old_config"]
        # - input_data["new_config"]
        # - input_data["changed_keys"]
```

**Examples:**
- Reload settings
- Adjust resource allocations
- Invalidate caches

### Memory Module Hooks

#### `pre_create` / `post_create` - Memory Creation
```python
class CreateHook(MemoryHook):
    async def execute(self, context: HookContext) -> HookContext:
        # PRE context:
        # - input_data["memory_type"]  # episodic/semantic/procedural
        # - input_data["content"]
        # - input_data["metadata"]
        # POST context:
        # - output_data["memory_id"]
        # - output_data["embedding"]
```

**Examples:**
- **PRE:** Validate content, check duplicates
- **POST:** Index memory, send notifications

#### `pre_recall` / `post_recall` - Memory Retrieval
```python
class RecallHook(MemoryHook):
    async def execute(self, context: HookContext) -> HookContext:
        # PRE context:
        # - input_data["query"]
        # - input_data["filters"]
        # - input_data["limit"]
        # POST context:
        # - output_data["results"]
        # - output_data["count"]
        # - output_data["scores"]
```

**Examples:**
- **PRE:** Query preprocessing, cache check
- **POST:** Re-rank results, track access

#### `on_access` - Memory Access Tracking
```python
class AccessHook(MemoryHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["memory_id"]
        # - input_data["access_type"]  # read/recall/update
        # - input_data["context_ids"]  # Co-accessed memories
```

**Examples:**
- Update Hebbian weights
- Track usage statistics
- Update access timestamps

#### `on_decay` - Memory Decay Update
```python
class DecayHook(MemoryHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["memory_id"]
        # - input_data["old_stability"]
        # - input_data["new_stability"]
        # - input_data["retrievability"]
```

**Examples:**
- Log stability changes
- Trigger consolidation
- Archive old memories

### Storage Hooks

#### `on_connect` / `on_disconnect` - Connection Lifecycle
```python
class ConnectionHook(StorageHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["storage_type"]  # neo4j/qdrant
        # - input_data["connection_string"]
        # - input_data["event"]  # connect/disconnect
```

**Examples:**
- Monitor connection pool
- Log connection events
- Validate credentials

#### `pre_query` / `post_query` - Query Instrumentation
```python
class QueryHook(StorageHook):
    async def execute(self, context: HookContext) -> HookContext:
        # PRE context:
        # - input_data["query"]
        # - input_data["parameters"]
        # - input_data["query_type"]  # read/write/delete
        # POST context:
        # - output_data["result"]
        # - output_data["row_count"]
        # - output_data["duration_ms"]
```

**Examples:**
- **PRE:** Query optimization, cache lookup
- **POST:** Performance logging, cache update

#### `on_error` - Error Handling
```python
class ErrorHook(StorageHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["error_type"]  # connection/timeout/query
        # - input_data["error"]
        # - input_data["attempt"]
```

**Examples:**
- Classify errors
- Decide retry strategy
- Generate alerts

#### `on_retry` - Retry Attempts
```python
class RetryHook(StorageHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["attempt"]
        # - input_data["max_attempts"]
        # - input_data["backoff_ms"]
```

**Examples:**
- Exponential backoff
- Circuit breaker
- Retry logging

### MCP Hooks

#### `pre_tool_call` / `post_tool_call` - Tool Execution
```python
class ToolCallHook(MCPHook):
    async def execute(self, context: HookContext) -> HookContext:
        # PRE context:
        # - input_data["tool_name"]
        # - input_data["arguments"]
        # - input_data["session_id"]
        # POST context:
        # - output_data["result"]
        # - output_data["success"]
```

**Examples:**
- **PRE:** Authentication, input sanitization
- **POST:** Response formatting, audit logging

#### `on_rate_limit` - Rate Limit Triggered
```python
class RateLimitHook(MCPHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["session_id"]
        # - input_data["limit"]
        # - input_data["retry_after"]
```

**Examples:**
- Generate alerts
- Update metrics
- Log violations

#### `on_validation_error` - Validation Failed
```python
class ValidationErrorHook(MCPHook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["field"]
        # - input_data["value"]
        # - input_data["error_message"]
```

**Examples:**
- Security logging
- Error response formatting
- Input sanitization alerts

### Consolidation Hooks

#### `pre_consolidate` / `post_consolidate` - Consolidation Lifecycle
```python
class ConsolidationHook(Hook):
    async def execute(self, context: HookContext) -> HookContext:
        # PRE context:
        # - input_data["consolidation_type"]  # light/deep/skill/all
        # - input_data["session_filter"]
        # POST context:
        # - output_data["episodes_processed"]
        # - output_data["duplicates_removed"]
        # - output_data["entities_extracted"]
        # - output_data["clusters_formed"]
```

#### `on_duplicate_found` - Duplicate Detection
```python
class DuplicateFoundHook(Hook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["memory_id_1"]
        # - input_data["memory_id_2"]
        # - input_data["similarity"]
        # - input_data["merge_strategy"]
```

#### `on_cluster_formed` - Clustering Events
```python
class ClusterFormHook(Hook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["cluster_id"]
        # - input_data["memory_ids"]
        # - input_data["cluster_size"]
        # - input_data["coherence"]
```

#### `on_entity_extracted` - Entity Extraction
```python
class EntityExtractedHook(Hook):
    async def execute(self, context: HookContext) -> HookContext:
        # Context data:
        # - input_data["entity_id"]
        # - input_data["entity_type"]
        # - input_data["entity_name"]
        # - input_data["confidence"]
```

## Error Handling

### Error Isolation

Hooks are executed in isolation. A failure in one hook does not prevent other hooks from executing (unless `fail_fast=True`).

```python
registry = HookRegistry(
    name="episodic",
    fail_fast=False,  # Continue on hook errors
    max_concurrent=10,
)
```

### Error Recovery

Error hooks are executed when any hook fails:

```python
class RecoveryHook(Hook):
    async def on_error(self, context: HookContext) -> None:
        # Handle error from any hook
        logger.error(f"Hook error: {context.error}")
        # Trigger recovery logic
```

## Usage Examples

### Example 1: Observability Hooks

```python
from t4dm.hooks import get_global_registry, HookPriority
from t4dm.hooks.memory import CreateHook, RecallHook

# Get registry
registry = get_global_registry("episodic")

# Add tracing hook
class TracingCreateHook(CreateHook):
    def __init__(self):
        super().__init__(name="tracing", priority=HookPriority.HIGH)

    async def execute(self, context: HookContext) -> HookContext:
        from t4dm.observability.tracing import traced, add_span_attribute

        if context.phase == HookPhase.PRE:
            add_span_attribute("memory.type", context.input_data.get("memory_type"))
            add_span_attribute("content.length", len(context.input_data.get("content", "")))
        elif context.phase == HookPhase.POST:
            add_span_attribute("memory.id", str(context.output_data.get("memory_id")))

        return context

registry.register(TracingCreateHook(), HookPhase.PRE)
registry.register(TracingCreateHook(), HookPhase.POST)
```

### Example 2: Caching Hook

```python
from t4dm.hooks.memory import RecallHook

class CachingRecallHook(RecallHook):
    def __init__(self):
        super().__init__(name="cache", priority=HookPriority.HIGH)
        self.cache = {}

    async def execute(self, context: HookContext) -> HookContext:
        query = context.input_data.get("query")

        if context.phase == HookPhase.PRE:
            # Check cache
            if query in self.cache:
                context.metadata["cache_hit"] = True
                context.metadata["cached_result"] = self.cache[query]

        elif context.phase == HookPhase.POST:
            # Update cache
            if not context.metadata.get("cache_hit"):
                self.cache[query] = context.output_data.get("results")

        return context
```

### Example 3: Audit Trail Hook

```python
from t4dm.hooks.memory import MemoryHook

class AuditHook(MemoryHook):
    def __init__(self):
        super().__init__(name="audit", priority=HookPriority.HIGH)
        self.audit_log = []

    async def execute(self, context: HookContext) -> HookContext:
        if context.phase == HookPhase.POST:
            entry = {
                "timestamp": context.start_time.isoformat(),
                "operation": context.operation,
                "session_id": context.session_id,
                "success": context.error is None,
                "duration_ms": context.elapsed_ms(),
            }
            self.audit_log.append(entry)

        return context

# Register for all operations
registry.register(AuditHook(), HookPhase.POST)
```

### Example 4: Decorator Usage

```python
from t4dm.hooks.base import with_hooks

registry = get_global_registry("episodic")

@with_hooks(registry, operation="create_episode", module="episodic")
async def create(content: str, session_id: str = None, **kwargs) -> Episode:
    # Hooks will automatically:
    # 1. Execute PRE hooks (validation, caching, etc.)
    # 2. Run this function
    # 3. Execute POST hooks (logging, metrics, etc.)
    # 4. Execute ERROR hooks if exception occurs

    episode = Episode(content=content, session_id=session_id)
    # ... create logic ...
    return episode
```

## Performance Considerations

### Parallel Execution

POST and ON hooks can execute in parallel for better performance:

```python
# Execute POST hooks in parallel (default)
await registry.execute_phase(HookPhase.POST, context, parallel=True)
```

### Concurrency Control

Limit concurrent hook executions to prevent resource exhaustion:

```python
registry = HookRegistry(
    name="episodic",
    max_concurrent=10,  # Max 10 hooks executing simultaneously
)
```

### Hook Statistics

Monitor hook performance:

```python
stats = registry.get_stats()
# {
#     "registry": "episodic",
#     "total_hooks": 15,
#     "hooks_by_phase": {"pre": 5, "post": 7, "on": 2, "error": 1},
#     "hook_stats": [
#         {
#             "name": "tracing",
#             "priority": 100,
#             "executions": 1234,
#             "errors": 5,
#             "error_rate": 0.004,
#         },
#         ...
#     ]
# }
```

## Best Practices

### 1. Hook Priorities

- **CRITICAL (0):** Security, validation, authentication
- **HIGH (100):** Observability, auditing, rate limiting
- **NORMAL (500):** Business logic, custom processing
- **LOW (1000):** Caching, cleanup, non-critical tasks

### 2. Error Handling

- Hooks should handle their own errors gracefully
- Use `on_error()` for error recovery
- Don't rely on other hooks' execution
- Log errors comprehensively

### 3. Performance

- Keep hooks lightweight
- Use parallel execution for independent POST hooks
- Cache expensive operations
- Monitor hook execution time

### 4. Testing

- Test hooks in isolation
- Test hook execution order
- Test error scenarios
- Verify error isolation

### 5. Documentation

- Document hook purpose clearly
- Document expected context data
- Document side effects
- Provide usage examples

## Migration Path

For existing T4DM code:

1. **Identify lifecycle points:** Find initialization, shutdown, operation boundaries
2. **Create hook instances:** Implement relevant hook classes
3. **Register hooks:** Add to appropriate registries
4. **Add decorators:** Use `@with_hooks` for automatic execution
5. **Test thoroughly:** Verify hook execution and error handling

## File Locations

```
src/t4dm/hooks/
├── __init__.py              # Public API
├── base.py                  # Hook, HookRegistry, HookContext
├── core.py                  # Core lifecycle hooks
├── memory.py                # Memory module hooks
├── storage.py               # Storage module hooks
├── mcp.py                   # MCP module hooks
├── consolidation.py         # Consolidation hooks
└── registry.py              # Global registry management
```

## Future Enhancements

1. **Async generators:** Stream hook results
2. **Hook dependencies:** Declare execution dependencies
3. **Hook versioning:** Version compatibility checks
4. **Dynamic registration:** Register hooks at runtime
5. **Hook marketplace:** Share common hooks across projects
