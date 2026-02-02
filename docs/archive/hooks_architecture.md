# T4DM Hooks System - Architecture Diagrams

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      T4DM Application                            │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hook Registries                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  REGISTRY_CORE           │  Module initialization, shutdown, health     │
│  REGISTRY_EPISODIC       │  Episodic memory operations                  │
│  REGISTRY_SEMANTIC       │  Semantic memory operations                  │
│  REGISTRY_PROCEDURAL     │  Procedural memory operations                │
│  REGISTRY_CONSOLIDATION  │  Memory consolidation                        │
│  REGISTRY_STORAGE_NEO4J  │  Neo4j storage operations                    │
│  REGISTRY_STORAGE_QDRANT │  Qdrant storage operations                   │
│  REGISTRY_MCP            │  MCP tool calls                              │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ contains
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hook Collections                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  PRE Hooks (Sequential)                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Priority 0 (CRITICAL)  │ Security, Validation, Auth              │ │
│  │  Priority 100 (HIGH)    │ Observability, Rate Limiting            │ │
│  │  Priority 500 (NORMAL)  │ Business Logic, Preprocessing           │ │
│  │  Priority 1000 (LOW)    │ Caching, Optimization                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  POST Hooks (Parallel/Sequential)                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Priority 1000 (LOW)    │ Cleanup, Cache Updates                  │ │
│  │  Priority 500 (NORMAL)  │ Notifications, Side Effects             │ │
│  │  Priority 100 (HIGH)    │ Metrics, Audit Logging                  │ │
│  │  Priority 0 (CRITICAL)  │ Critical Operations, Validation         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ON Hooks (Event-driven)                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Connection events, State changes, Health checks                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ERROR Hooks (Exception handling)                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Error classification, Recovery, Retry, Alerting                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hook Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Operation Initiated                                 │
│                    (e.g., create_episode)                                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Create Context │
                    │  • operation   │
                    │  • module      │
                    │  • input_data  │
                    │  • session_id  │
                    └────────┬───────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRE Phase (Sequential)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Hook 1 (CRITICAL)                                                │   │
│  │  - Validate session_id                                           │   │
│  │  - Check authentication                                          │   │
│  │  ✓ Can modify input_data                                         │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                            │                                              │
│  ┌─────────────────────────▼────────────────────────────────────────┐   │
│  │ Hook 2 (HIGH)                                                     │   │
│  │  - Check rate limits                                             │   │
│  │  - Start tracing span                                            │   │
│  │  ✓ Can modify input_data                                         │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                            │                                              │
│  ┌─────────────────────────▼────────────────────────────────────────┐   │
│  │ Hook 3 (NORMAL)                                                   │   │
│  │  - Preprocess content                                            │   │
│  │  - Extract metadata                                              │   │
│  │  ✓ Can modify input_data                                         │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                            │                                              │
│  ┌─────────────────────────▼────────────────────────────────────────┐   │
│  │ Hook 4 (LOW)                                                      │   │
│  │  - Check embedding cache                                         │   │
│  │  - Check query cache                                             │   │
│  │  ✓ Store cache hit in metadata                                   │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                            │                                              │
└────────────────────────────┼─────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Execute Core Operation                              │
│                   (e.g., store episode in DB)                            │
│                                                                           │
│  • Use potentially modified input_data from PRE hooks                    │
│  • Generate result                                                       │
│  • Set output_data in context                                            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        POST Phase (Parallel)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ Hook A (LOW)   │  │ Hook B (NORMAL)│  │ Hook C (HIGH)  │            │
│  │ - Cache result │  │ - Send notif.  │  │ - Log metrics  │            │
│  │ - Cleanup temp │  │ - Update graph │  │ - Update trace │            │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘            │
│           │                   │                   │                     │
│           └───────────────────┴───────────────────┘                     │
│                               │                                          │
│                    (All execute in parallel)                             │
│                               │                                          │
│  ┌─────────────────────────────▼────────────────────────────────────┐   │
│  │ Hook D (CRITICAL)                                                 │   │
│  │  - Validate result integrity                                     │   │
│  │  - Final security checks                                         │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Return Result  │
                    └────────────────┘

                    (If error occurs)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ERROR Phase (Sequential)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Hook 1                                                           │   │
│  │  - Classify error (transient/permanent)                         │   │
│  │  - Decide retry strategy                                        │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                            │                                              │
│  ┌─────────────────────────▼────────────────────────────────────────┐   │
│  │ Hook 2                                                            │   │
│  │  - Circuit breaker check                                         │   │
│  │  - Update failure metrics                                        │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                            │                                              │
│  ┌─────────────────────────▼────────────────────────────────────────┐   │
│  │ Hook 3                                                            │   │
│  │  - Generate alerts                                               │   │
│  │  - Log to audit trail                                            │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hook Context Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HookContext Object                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Creation (at operation start):                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  hook_id: UUID                  (unique request ID)         │    │
│  │  session_id: str                (session identifier)        │    │
│  │  operation: str                 (e.g., "create_episode")    │    │
│  │  module: str                    (e.g., "episodic")          │    │
│  │  phase: HookPhase               (PRE/POST/ON/ERROR)         │    │
│  │  start_time: datetime           (operation start)           │    │
│  │  input_data: dict               (operation inputs)          │    │
│  │  metadata: dict                 (inter-hook communication)  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  During PRE phase:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Hooks can READ and MODIFY input_data                       │    │
│  │  Hooks can WRITE to metadata                                │    │
│  │  Hooks can SET validation errors                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  During operation:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Operation uses (possibly modified) input_data               │    │
│  │  Operation produces result                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  After operation (before POST):                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  output_data: dict              (operation result)           │    │
│  │  end_time: datetime             (operation completion)       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  During POST phase:                                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Hooks can READ input_data                                   │    │
│  │  Hooks can READ and MODIFY output_data                       │    │
│  │  Hooks can READ/WRITE metadata                               │    │
│  │  Hooks can ADD metrics, logs, traces                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  On error:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  error: Exception               (caught exception)           │    │
│  │  error_context: dict            (error metadata)             │    │
│  │  end_time: datetime             (error occurrence time)      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  Final state:                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  elapsed_ms(): float            (total execution time)       │    │
│  │  to_dict(): dict                (serialized for logging)     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Module-Specific Hook Hierarchies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Hook Class Hierarchy                          │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────┐
                              │   Hook   │
                              │  (base)  │
                              └────┬─────┘
                                   │
                 ┌─────────────────┼──────────────────┬──────────────┐
                 │                 │                  │              │
            ┌────▼────┐       ┌────▼────┐       ┌────▼────┐    ┌────▼────┐
            │ CoreHook│       │ Memory  │       │ Storage │    │  MCP    │
            │         │       │  Hook   │       │  Hook   │    │  Hook   │
            └────┬────┘       └────┬────┘       └────┬────┘    └────┬────┘
                 │                 │                  │              │
        ┌────────┼──────┐          │         ┌────────┼──────┐       │
        │        │      │          │         │        │      │       │
   ┌────▼──┐ ┌──▼───┐ │     ┌────▼────┐ ┌──▼───┐ ┌──▼───┐ │  ┌────▼────┐
   │ Init  │ │Shut  │ │     │ Create  │ │Query │ │Error │ │  │ToolCall │
   │ Hook  │ │down  │ │     │  Hook   │ │ Hook │ │ Hook │ │  │  Hook   │
   └───────┘ └──────┘ │     └─────────┘ └──────┘ └──────┘ │  └─────────┘
                       │                                    │
                  ┌────▼────┐                          ┌────▼────┐
                  │ Health  │                          │ Connect │
                  │  Check  │                          │  Hook   │
                  └─────────┘                          └─────────┘

          More specialized hooks:

          CreateHook → RecallHook → UpdateHook → AccessHook → DecayHook
             │              │           │            │            │
             └──────────────┴───────────┴────────────┴────────────┘
                              (Memory operations)

          QueryHook → ConnectionHook → RetryHook → ErrorHook
             │              │              │           │
             └──────────────┴──────────────┴───────────┘
                        (Storage operations)
```

## Data Flow Through Hooks

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Memory Creation with Hooks                          │
└──────────────────────────────────────────────────────────────────────┘

Input:
  content: "User performed action X"
  metadata: {project: "ww", file: "test.py"}

         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRE Hook 1: ValidationHook (CRITICAL)                           │
│  • Checks content length < 100000                               │
│  • Validates required fields exist                              │
│  • input_data unchanged                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRE Hook 2: EmbeddingCacheHook (HIGH)                           │
│  • Hashes content → "a1b2c3..."                                 │
│  • Checks cache: MISS                                           │
│  • metadata["cache_hit"] = False                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRE Hook 3: TracingHook (HIGH)                                  │
│  • Adds span attributes                                         │
│  • metadata["span_id"] = "xyz123"                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────┐
         │   Core Operation          │
         │  • Generate embedding     │
         │  • Create Episode object  │
         │  • Store in Neo4j/Qdrant  │
         │  • Return episode         │
         └──────────┬────────────────┘
                    │
                    ▼
         output_data:
           memory_id: UUID("...")
           embedding: [0.1, 0.2, ...]
           success: true

                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ POST Hook 1: EmbeddingCacheHook (HIGH) [parallel]              │
│  • Stores embedding in cache                                    │
│  • cache["a1b2c3..."] = embedding                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ POST Hook 2: MetricsHook (HIGH) [parallel]                     │
│  • Increments creation counter                                  │
│  • Records latency histogram                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ POST Hook 3: AuditHook (HIGH) [parallel]                       │
│  • Logs to audit trail                                          │
│  • Records: who, what, when                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ POST Hook 4: TracingHook (NORMAL) [parallel]                   │
│  • Adds result to span                                          │
│  • Completes trace                                              │
└─────────────────────────────────────────────────────────────────┘

                    │
                    ▼
         Return Episode to caller
```

## Error Handling Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Error in Hook Execution                           │
└──────────────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │ Hook throws │
                    │  exception  │
                    └──────┬──────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Registry catches│
                  │    exception    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │  fail_fast=True?   │
                  └──┬──────────────┬──┘
                     │              │
                 Yes │              │ No
                     │              │
                     ▼              ▼
          ┌──────────────┐   ┌───────────────────┐
          │ Raise Hook   │   │ Log error         │
          │    Error     │   │ Continue execution│
          │ Stop all     │   │ Track in context  │
          │   hooks      │   └────────┬──────────┘
          └──────────────┘            │
                                      ▼
                           ┌──────────────────────┐
                           │ After all hooks done │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ context.error set?   │
                           └──┬──────────────┬────┘
                              │              │
                          Yes │              │ No
                              │              │
                              ▼              ▼
                   ┌────────────────┐   ┌─────────┐
                   │ Execute ERROR  │   │ Success │
                   │  phase hooks   │   └─────────┘
                   └────────┬───────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ ERROR Hook 1         │
                 │  - Classify error    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ ERROR Hook 2         │
                 │  - Circuit breaker   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ ERROR Hook 3         │
                 │  - Generate alerts   │
                 └────────────────────────┘
```

## Concurrency Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Hook Concurrency and Parallelism                    │
└──────────────────────────────────────────────────────────────────────┘

PRE Phase (Sequential - Order Matters):

  Time →
  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  │ Hook 1 │→ │ Hook 2 │→ │ Hook 3 │→ │ Hook 4 │
  └────────┘  └────────┘  └────────┘  └────────┘
   CRITICAL      HIGH       NORMAL       LOW

  Total time = T1 + T2 + T3 + T4


POST Phase (Parallel - Independent Operations):

  Time →
  ┌────────┐
  │ Hook A │  (LOW)
  └────────┘
  ┌────────┐
  │ Hook B │  (NORMAL)
  └────────┘
  ┌────────┐
  │ Hook C │  (HIGH)
  └────────┘
  ┌────────┐
  │ Hook D │  (HIGH)
  └────────┘

  All execute concurrently
  Total time ≈ max(Ta, Tb, Tc, Td)


Concurrency Control (Semaphore):

  ┌──────────────────────────────────────────┐
  │         Semaphore (max_concurrent=10)    │
  ├──────────────────────────────────────────┤
  │  [•] [•] [•] [•] [•] [•] [•] [•] [•] [•] │  10 slots
  └──────────────────────────────────────────┘
       │   │   │   │   │   │   │   │   │   │
       ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
      H1  H2  H3  H4  H5  H6  H7  H8  H9  H10

  Hooks 11+ wait for slot to become available
```

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  T4DM System Integration                      │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          MCP Gateway                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  @with_hooks(mcp_registry, "tool_call")                       │  │
│  │  async def handle_tool_call(tool_name, args):                 │  │
│  │    # PRE: auth, rate limit, validation                        │  │
│  │    # POST: audit, metrics, logging                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Memory Services                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Episodic Memory                                              │  │
│  │  @with_hooks(episodic_registry, "create")                     │  │
│  │  @with_hooks(episodic_registry, "recall")                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Semantic Memory                                              │  │
│  │  @with_hooks(semantic_registry, "create")                     │  │
│  │  @with_hooks(semantic_registry, "search")                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Storage Layer                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Neo4j Store                                                  │  │
│  │  @with_hooks(neo4j_registry, "query")                         │  │
│  │  @with_hooks(neo4j_registry, "connect")                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Qdrant Store                                                 │  │
│  │  @with_hooks(qdrant_registry, "query")                        │  │
│  │  @with_hooks(qdrant_registry, "connect")                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

Cross-cutting concerns handled by hooks:
  • OpenTelemetry tracing
  • Prometheus metrics
  • Structured logging
  • Caching (embeddings, queries)
  • Audit trails (GDPR compliance)
  • Error handling (circuit breaker, retry)
  • Performance profiling
```

This architecture provides:
- **Separation of concerns:** Core logic independent of cross-cutting concerns
- **Extensibility:** New hooks added without modifying core code
- **Observability:** Comprehensive tracing, metrics, and logging
- **Reliability:** Error isolation and recovery mechanisms
- **Performance:** Parallel execution and caching strategies
