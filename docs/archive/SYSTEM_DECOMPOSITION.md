# T4DM (ww) System Decomposition

**Generated**: 2025-12-06
**Version**: 0.1.0
**Test Status**: 2011 passed, 19 skipped, 7 xfailed

---

## 1. MEMORY SYSTEMS (`t4dm/memory/`)
The core memory architecture implementing biologically-inspired memory types.

### 1.1 Episodic Memory
**File**: `episodic.py` (2,561 LOC - largest file)
- **Classes**:
  - `EpisodicMemory` - Primary episodic memory implementation
  - `LearnedFusionWeights` - Adaptive score fusion weights
  - `LearnedReranker` - Neural reranking for retrieval

### 1.2 Semantic Memory
**File**: `semantic.py` (906 LOC)
- **Classes**:
  - `SemanticMemory` - Entity and relationship storage (knowledge graph)

### 1.3 Procedural Memory
**File**: `procedural.py`
- **Classes**:
  - `ProceduralMemory` - Skills/procedures storage

### 1.4 Working Memory
**File**: `working_memory.py`
- **Classes**:
  - `WorkingMemory[T]` - Capacity-limited active memory buffer
  - `WorkingMemoryItem[T]` - Individual item wrapper
  - `AttentionalBlink` - Attention gating mechanism
  - `EvictionEvent` - Eviction tracking
  - `ItemState` (Enum) - Item lifecycle states

### 1.5 Buffer Management
**File**: `buffer_manager.py`
- **Classes**:
  - `BufferManager` - Manages memory promotion/demotion
  - `BufferedItem` - Item with evidence scoring
  - `PromotionDecision` - Promotion decision container
  - `PromotionAction` (Enum) - STORE, DEFER, DISCARD

### 1.6 Unified Memory Service
**File**: `unified.py`
- **Classes**:
  - `UnifiedMemoryService` - Cross-memory search facade

### 1.7 HSA-Inspired Indexing (Phases 1-3)
**Files**: `cluster_index.py`, `learned_sparse_index.py`, `feature_aligner.py`, `pattern_separation.py`
- **Classes**:
  - `ClusterIndex` - CA3-like hierarchical clustering (Phase 1)
  - `ClusterMeta` - Cluster metadata
  - `LearnedSparseIndex` - Adaptive sparse addressing (Phase 2)
  - `SparseAddressingResult` - Sparse query result
  - `PendingUpdate` - Deferred update tracking
  - `FeatureAligner` - Joint optimization (Phase 3)
  - `AlignmentResult` - Alignment output
  - `JointLossWeights` - Loss weight configuration
  - `DentateGyrus` - Pattern separation (biological)
  - `PatternCompletion` - Pattern completion
  - `SeparationResult` - Separation output

---

## 2. LEARNING SYSTEMS (`t4dm/learning/`)
Biologically-grounded learning and neuromodulation.

### 2.1 Neuromodulator Systems
**Files**: `neuromodulators.py`, `dopamine.py`, `norepinephrine.py`, `serotonin.py`, `acetylcholine.py`
- **Classes**:
  - `NeuromodulatorOrchestra` - Coordinates all neuromodulators
  - `NeuromodulatorState` - Combined state snapshot
  - `DopamineSystem` - Reward prediction error
  - `RewardPredictionError` - RPE calculation
  - `NorepinephrineSystem` - Arousal/attention modulation
  - `ArousalState` - Arousal level tracking
  - `SerotoninSystem` - Temporal discount / mood
  - `EligibilityTrace` - Eligibility trace mechanism
  - `TemporalContext` - Temporal context tracking
  - `AcetylcholineSystem` - Attention/mode switching
  - `AcetylcholineState` - ACh state snapshot
  - `CognitiveMode` (Enum) - ENCODING, RETRIEVAL, BALANCED

### 2.2 Neuro-Symbolic Integration
**File**: `neuro_symbolic.py` (1,064 LOC)
- **Classes**:
  - `NeuroSymbolicMemory` - Hybrid neural-symbolic memory
  - `NeuroSymbolicReasoner` - Symbolic reasoning engine
  - `LearnedFusion` (nn.Module) - Neural fusion network
  - `Triple` - Subject-predicate-object triple
  - `TripleSet` - Collection of triples
  - `TripleExtractor` (ABC) - Base extractor
  - `SimilarityExtractor` - Similarity-based extraction
  - `CoRetrievalExtractor` - Co-retrieval pattern extraction
  - `CausalExtractor` - Causal relationship extraction
  - `PredicateType` (Enum) - SIMILAR_TO, CO_RETRIEVED, CAUSED_BY, etc.

### 2.3 Retrieval Scoring
**File**: `scorer.py`
- **Classes**:
  - `LearnedRetrievalScorer` (nn.Module) - Neural ranking model
  - `ScorerTrainer` - Online training logic
  - `TrainerConfig` - Training hyperparameters
  - `PrioritizedReplayBuffer` - Prioritized experience replay
  - `ReplayItem` - Replay buffer item
  - `ListMLELoss` (nn.Module) - Listwise ranking loss

### 2.4 Event Collection
**File**: `collector.py` (994 LOC)
- **Classes**:
  - `EventCollector` - Collects learning signals
  - `EventStore` - Persistent event storage
  - `CollectorConfig` - Collector configuration

### 2.5 Learning Events
**File**: `events.py`
- **Classes**:
  - `Experience` - Learning experience record
  - `RetrievalEvent` - Retrieval event
  - `OutcomeEvent` - Outcome feedback event
  - `RepresentationFormat` (ABC) - Serialization format base
  - `FullJSON`, `ToonJSON`, `NeuroSymbolicTriples` - Format implementations
  - `MemoryType`, `OutcomeType`, `FeedbackSignal` (Enums)

### 2.6 Synaptic Plasticity
**File**: `plasticity.py`
- **Classes**:
  - `PlasticityManager` - Coordinates plasticity mechanisms
  - `MetaplasticityController` - Metaplasticity control
  - `HomeostaticScaler` - Homeostatic scaling
  - `LTDEngine` - Long-term depression
  - `SynapticTagger` - Synaptic tagging
  - `SynapticTag` - Individual tag
  - `SynapseState` - Synapse state
  - `PlasticityEvent` - Plasticity change event
  - `PlasticityType` (Enum) - LTP, LTD, HOMEOSTATIC

### 2.7 Memory Reconsolidation
**File**: `reconsolidation.py`
- **Classes**:
  - `ReconsolidationEngine` - Memory update on reactivation
  - `ReconsolidationUpdate` - Update record

### 2.8 Inhibition Networks
**File**: `inhibition.py`
- **Classes**:
  - `InhibitoryNetwork` - Lateral inhibition
  - `SparseRetrieval` - Sparse activation
  - `InhibitionResult` - Inhibition output

### 2.9 Cold Start
**File**: `cold_start.py`
- **Classes**:
  - `ColdStartManager` - Bootstrap new users
  - `PopulationPrior` - Population-level priors
  - `ContextLoader` - Context loading
  - `ContextSignals` - Context signal container

### 2.10 State Persistence
**File**: `persistence.py`
- **Classes**:
  - `StatePersister` - Save/load learned state
  - `LearnedGateState`, `ScorerState`, `NeuromodulatorState` - State containers

---

## 3. CORE INFRASTRUCTURE (`t4dm/core/`)
Foundation classes, types, and configuration.

### 3.1 Memory Gating
**Files**: `memory_gate.py`, `learned_gate.py` (789 LOC)
- **Classes**:
  - `MemoryGate` - Rule-based storage decision
  - `LearnedMemoryGate` - Bayesian adaptive gating
  - `GateContext` - Context for gating decisions
  - `GateResult` - Gating decision result
  - `GateDecision` - Learned gate output
  - `StorageDecision` (Enum) - STORE, SKIP, DEFER
  - `TemporalBatcher` - Batch temporal events

### 3.2 Type Definitions
**File**: `types.py`
- **Classes**:
  - `Episode`, `Entity`, `Procedure`, `Relationship` - Core data types
  - `EpisodeContext`, `EpisodeQuery` - Query types
  - `EntityQuery`, `ProcedureQuery` - Query types
  - `ProcedureStep` - Procedure step
  - `ScoredResult[T]` - Generic scored result
  - `ConsolidationEvent` - Consolidation record
  - Enums: `Domain`, `EntityType`, `RelationType`, `Outcome`, `ConsolidationType`

### 3.3 Protocols (Interfaces)
**File**: `protocols.py`
- **Classes** (Protocol):
  - `VectorStore` - Vector storage interface
  - `GraphStore` - Graph storage interface
  - `EpisodicStore`, `SemanticStore`, `ProceduralStore` - Memory interfaces
  - `EmbeddingProvider` - Embedding interface

### 3.4 Configuration
**File**: `config.py` (793 LOC)
- **Classes**:
  - `Settings` (BaseSettings) - Pydantic settings with environment loading

### 3.5 Actions System
**File**: `actions.py` (1,045 LOC)
- **Classes**:
  - `ActionRegistry` - Register actions
  - `ActionExecutor` - Execute actions
  - `ActionDefinition` - Action definition
  - `ActionRequest`, `ActionResult` - Request/response
  - Enums: `ActionCategory`, `ActionStatus`, `PermissionLevel`, `RiskLevel`

### 3.6 Privacy & Security
**File**: `privacy_filter.py`
- **Classes**:
  - `PrivacyFilter` - Content filtering
  - `ContentClassifier` - Classify sensitive content
  - `PrivacyRule` - Filtering rule
  - `RedactionResult` - Redaction output
  - `SensitivityLevel` (Enum)

### 3.7 Personal Entities
**File**: `personal_entities.py`
- **Classes**:
  - `PersonalContext` - User context manager
  - `Contact`, `ContactInfo` - Contact data
  - `Email`, `EmailAddress`, `EmailAttachment` - Email data
  - `CalendarEvent`, `EventAttendee`, `EventReminder` - Calendar data
  - `Task` - Task data
  - `Location` - Location data
  - Enums: `ContactType`, `RelationshipType`, `EmailCategory`, `EmailImportance`, `EventType`, `EventStatus`, `TaskPriority`, `TaskStatus`

### 3.8 Serialization
**File**: `serialization.py`
- **Classes**:
  - `Serializer[T]` (ABC) - Base serializer
  - `EpisodeSerializer`, `EntitySerializer`, `ProcedureSerializer` - Type serializers
  - `DateTimeSerializer` - DateTime handling

### 3.9 Dependency Container
**File**: `container.py`
- **Classes**:
  - `Container` - Dependency injection container

---

## 4. STORAGE BACKENDS (`t4dm/storage/`)
Persistent storage implementations.

### 4.1 Vector Store (Qdrant)
**File**: `t4dx_vector_adapter.py` (1,058 LOC)
- **Classes**:
  - `T4DXVectorAdapter` - Qdrant vector database
  - `DatabaseTimeoutError` - Timeout exception

### 4.2 Graph Store (Neo4j)
**File**: `t4dx_graph_adapter.py` (1,011 LOC)
- **Classes**:
  - `T4DXGraphAdapter` - Neo4j graph database
  - `DatabaseConnectionError`, `DatabaseTimeoutError` - Exceptions

### 4.3 Saga Pattern (Transactions)
**File**: `saga.py`
- **Classes**:
  - `Saga` - Distributed transaction coordinator
  - `MemorySaga` - Memory-specific saga
  - `SagaStep` - Individual saga step
  - `SagaResult` - Saga outcome
  - `SagaState` (Enum) - PENDING, COMMITTED, COMPENSATED
  - `CompensationError` - Compensation failure

---

## 5. CONSOLIDATION (`t4dm/consolidation/`)
Sleep-inspired memory consolidation.

### 5.1 Sleep Consolidation
**File**: `sleep.py` (797 LOC)
- **Classes**:
  - `SleepConsolidation` - Sleep cycle simulation
  - `SleepCycleResult` - Cycle outcome
  - `ReplayEvent` - Memory replay
  - `AbstractionEvent` - Pattern abstraction
  - `SleepPhase` (Enum) - NREM, REM, WAKE
  - Protocol stubs: `EpisodicMemory`, `SemanticMemory`, `GraphStore`

### 5.2 Consolidation Service
**File**: `service.py` (1,118 LOC)
- **Classes**:
  - `ConsolidationService` - Orchestrates consolidation

---

## 6. EMBEDDING (`t4dm/embedding/`)
Neural embedding generation.

### 6.1 BGE-M3 Embeddings
**File**: `bge_m3.py`
- **Classes**:
  - `BGEM3Embedding` - BGE-M3 embedding provider
  - `TTLCache` - Time-to-live cache

---

## 7. MCP SERVER (`t4dm/mcp/`)
Model Context Protocol server for Claude integration.

### 7.1 Gateway
**File**: `gateway.py`
- **Classes**:
  - `RateLimiter` - API rate limiting

### 7.2 Tools
**Files**: `tools/episodic.py`, `tools/semantic.py`, `tools/procedural.py`, `tools/system.py`
- MCP tool implementations for each memory type

### 7.3 Validation
**File**: `validation.py`
- **Classes**:
  - `ValidationError`, `SessionValidationError` - Validation exceptions

### 7.4 Types
**File**: `types.py`
- **Classes** (TypedDict):
  - `EpisodeData`, `EntityData`, `ProcedureData` - Data containers
  - `EpisodeRecallResponse`, `EntityRecallResponse`, etc. - Response types
  - `BatchCreateResponse`, `BatchRecallResponse` - Batch operations
  - `MemoryStatsResponse`, `ConsolidationResult` - System responses

### 7.5 Errors
**File**: `errors.py`
- **Classes**:
  - `ErrorCode` (Enum) - Standardized error codes

### 7.6 Compatibility
**File**: `compat.py`
- **Classes**:
  - `StubMCPApp` - Fallback when MCP unavailable

---

## 8. API SERVER (`t4dm/api/`)
REST API for web/client access.

### 8.1 Route Handlers
**Files**: `routes/episodes.py`, `routes/entities.py`, `routes/skills.py`, `routes/system.py`, `routes/visualization.py` (1,918 LOC)

**Episode Routes** (`episodes.py`):
- `EpisodeCreate`, `EpisodeResponse`, `EpisodeList` - CRUD models
- `RecallRequest`, `RecallResponse` - Recall models

**Entity Routes** (`entities.py`):
- `EntityCreate`, `EntityResponse`, `EntityList` - CRUD models
- `RelationCreate`, `RelationResponse` - Relationship models
- `SemanticRecallRequest`, `SpreadActivationRequest` - Query models

**Skill Routes** (`skills.py`):
- `SkillCreate`, `SkillResponse`, `SkillList` - CRUD models
- `StepCreate`, `SkillRecallRequest` - Detail models
- `ExecutionRequest`, `HowToResponse` - Execution models

**System Routes** (`system.py`):
- `HealthResponse`, `StatsResponse` - Health models
- `ConsolidationRequest`, `ConsolidationResponse` - Consolidation models

**Visualization Routes** (`visualization.py`):
- Graph visualization models: `GraphResponse`, `MemoryNodeResponse`, `MemoryEdgeResponse`
- Timeline models: `TimelineResponse`, `TimelineEvent`
- Embedding visualization: `EmbeddingsResponse`, `EmbeddingPoint`, `Position3D`
- Activity models: `ActivityResponse`, `ActivityMetrics`
- Export models: `ExportRequest`, `ExportResponse`
- Biological mechanism visualizations: `BiologicalMechanismsResponse`
  - `DopamineRPEMetrics`, `LearnedGateMetrics`, `LearnedFusionMetrics`
  - `PatternSeparationMetrics`, `PatternCompletionMetrics`
  - `ReconsolidationMetrics`, `WorkingMemoryState`
  - `SleepConsolidationState`, `HebbianWeight`, `CreditPath`
  - Neuromodulator states: `NorepinephrineState`, `SerotoninState`, `AcetylcholineState`, `InhibitionState`
- Enums: `MemoryType`, `EdgeType`, `SleepPhaseViz`

### 8.2 Server & Dependencies
**Files**: `server.py`, `deps.py`
- FastAPI application setup and dependency injection

---

## 9. HOOKS SYSTEM (`t4dm/hooks/`)
Event-driven extension points.

### 9.1 Base Framework
**File**: `base.py`
- **Classes**:
  - `Hook` (ABC) - Base hook class
  - `HookRegistry` - Register/dispatch hooks
  - `HookContext` - Execution context
  - `HookError` - Hook exception
  - `HookPhase`, `HookPriority` (Enums)

### 9.2 Memory Hooks
**File**: `memory.py`
- **Classes**:
  - `MemoryHook` - Base memory hook
  - `CreateHook`, `UpdateHook`, `RecallHook`, `AccessHook`, `DecayHook` - Lifecycle hooks
  - `ValidationHook`, `CachingRecallHook`, `HebbianUpdateHook`, `AuditTrailHook` - Specialized hooks

### 9.3 Storage Hooks
**File**: `storage.py`
- **Classes**:
  - `StorageHook` - Base storage hook
  - `ConnectionHook`, `QueryHook`, `RetryHook`, `ErrorHook` - Operation hooks
  - `QueryTimingHook`, `QueryCacheHook` - Performance hooks
  - `ExponentialBackoffRetryHook`, `CircuitBreakerHook` - Resilience hooks
  - `ConnectionPoolMonitorHook` - Monitoring hook

### 9.4 Consolidation Hooks
**File**: `consolidation.py`
- **Classes**:
  - `ConsolidationHook` - Base consolidation hook
  - `PreConsolidateHook`, `PostConsolidateHook` - Lifecycle hooks
  - `ClusterFormHook`, `DuplicateFoundHook`, `EntityExtractedHook` - Event hooks
  - Implementations: `ConsolidationProgressHook`, `ConsolidationMetricsHook`, `ClusterAnalysisHook`, `DuplicateMergeHook`, `EntityValidationHook`

### 9.5 MCP Hooks
**File**: `mcp.py`
- **Classes**:
  - `MCPHook` - Base MCP hook
  - `ToolCallHook`, `RateLimitHook`, `ValidationErrorHook` - MCP hooks
  - Implementations: `AuthenticationHook`, `InputSanitizationHook`, `ToolCallAuditHook`, `ToolCallTimingHook`, `ResponseFormatterHook`, `RateLimitAlertHook`

### 9.6 Core Hooks
**File**: `core.py`
- **Classes**:
  - `CoreHook` - Base core hook
  - `InitHook`, `ShutdownHook`, `ConfigChangeHook`, `HealthCheckHook` - Lifecycle hooks
  - Implementations: `LoggingInitHook`, `GracefulShutdownHook`, `ConfigValidationHook`, `HealthMetricsHook`

---

## 10. INTEGRATIONS

### 10.1 Claude Code Integration (`t4dm/integration/`)
**Files**: `ccapi_memory.py`, `ccapi_observer.py`, `ccapi_routes.py`
- **Classes**:
  - `WWMemory` - Memory interface for Claude Code
  - `Message` - Message container
  - `WWObserver` - Event observation
  - `Event`, `EventType`, `Span` - Event tracking
  - Route models: `ContextRequest`, `ContextResponse`, `MemorySearchRequest`, etc.

### 10.2 Kymera Voice Assistant (`t4dm/integrations/kymera/`)
**Files**: Multiple (855 LOC in `advanced_features.py`)
- **Classes**:
  - `VoiceMemoryBridge` - Bridge voice to memory
  - `VoiceContext`, `MemoryContext` - Context containers
  - `VoiceIntentParser`, `ParsedIntent`, `TimeParser` - Intent parsing
  - `VoiceActionRouter`, `VoiceActionExecutor` - Action handling
  - `VoiceActionResult`, `VoiceResponse`, `VoiceExecutionConfig` - Results
  - `ContextInjector`, `ConversationContextManager`, `InjectionConfig` - Context injection
  - `JarvisMemoryHook`, `EnhancedContext`, `JarvisIntegration` - Jarvis integration
  - `ConversationCapture`, `ConversationSummarizer`, `MemoryConsolidator` - Continuity
  - `Conversation`, `ConversationTurn`, `ProactiveContext` - Conversation tracking
  - `PersonalDataManager`, `PersonalDataConfig`, `ContactCache` - Personal data
  - `MultiModalContext`, `ContextItem`, `ContentType` - Multi-modal
  - `NotificationManager`, `Notification`, `NotificationPriority`, `NotificationType` - Notifications
  - `PreferenceLearner`, `UserPreference` - Preference learning
  - `VoiceTriggerManager`, `VoiceTrigger` - Voice triggers

### 10.3 Google Workspace (`t4dm/integrations/`)
**File**: `google_workspace.py`
- **Classes**:
  - `GoogleWorkspaceSync` - Sync with Google services
  - `PersonalDataStore` - Store personal data

---

## 11. EXTRACTION (`t4dm/extraction/`)
Entity and information extraction.

**File**: `entity_extractor.py`
- **Classes**:
  - `EntityExtractor` (Protocol) - Extractor interface
  - `RegexEntityExtractor` - Pattern-based extraction
  - `LLMEntityExtractor` - LLM-based extraction
  - `CompositeEntityExtractor` - Combined extractors
  - `ExtractedEntity` - Extracted entity container

---

## 12. OBSERVABILITY (`t4dm/observability/`)
Monitoring, logging, and tracing.

### 12.1 Health Checking
**File**: `health.py`
- **Classes**:
  - `HealthChecker` - System health monitoring
  - `ComponentHealth` - Component status
  - `SystemHealth` - Overall system status
  - `HealthStatus` (Enum) - HEALTHY, DEGRADED, UNHEALTHY

### 12.2 Metrics
**File**: `metrics.py`
- **Classes**:
  - `MetricsCollector` - Collect metrics
  - `OperationMetrics` - Operation stats
  - `Timer`, `AsyncTimer` - Timing utilities

### 12.3 Logging
**File**: `logging.py`
- **Classes**:
  - `OperationLogger` - Structured logging
  - `LogContext` - Log context
  - `StructuredFormatter` - JSON formatting
  - `ContextAdapter` - Logger adapter

### 12.4 Tracing
**File**: `tracing.py`
- OpenTelemetry integration with `@traced` decorator

---

## 13. SDK (`t4dm/sdk/`)
Client libraries for external consumers.

**Files**: `client.py`, `models.py`
- **Classes**:
  - `T4DMClient` - Sync client
  - `AsyncT4DMClient` - Async client
  - Exceptions: `T4DMError`, `ConnectionError`, `NotFoundError`, `RateLimitError`
  - Models: `Episode`, `Entity`, `Skill`, `Step`, `Relationship`
  - `EpisodeContext`, `RecallResult`, `ActivationResult`
  - `MemoryStats`, `HealthStatus`

---

## Summary Statistics

| Area | Files | Classes | LOC (approx) |
|------|-------|---------|--------------|
| Memory Systems | 8 | 28 | ~5,500 |
| Learning Systems | 12 | 58 | ~4,800 |
| Core Infrastructure | 9 | 52 | ~5,200 |
| Storage | 3 | 10 | ~2,600 |
| Consolidation | 2 | 9 | ~1,900 |
| Embedding | 1 | 2 | ~500 |
| MCP Server | 8 | 35 | ~1,800 |
| API Server | 6 | 65 | ~3,500 |
| Hooks | 6 | 42 | ~1,500 |
| Integrations | 11 | 48 | ~3,500 |
| Extraction | 1 | 5 | ~500 |
| Observability | 4 | 12 | ~500 |
| SDK | 2 | 14 | ~600 |
| **TOTAL** | **73** | **~380** | **~32,000** |

---

## Dependency Graph (High-Level)

```
                    ┌─────────────┐
                    │   SDK       │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   API Server  │  │  MCP Server   │  │ Integrations  │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Memory Systems    │
                │  (Episodic/Semantic │
                │   /Procedural/WM)   │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Learning    │  │ Consolidation │  │    Hooks      │
│   Systems     │  │               │  │               │
└───────┬───────┘  └───────┬───────┘  └───────────────┘
        │                  │
        └──────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Core Infrastructure│
                │  (Types/Protocols/  │
                │   Config/Gate)      │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    Storage    │  │   Embedding   │  │ Observability │
│(Qdrant/Neo4j) │  │   (BGE-M3)    │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## Key Architectural Patterns

1. **Protocol-based Interfaces**: Core uses Python Protocols for dependency inversion
2. **Biologically-Inspired**: Learning systems model neuromodulators (DA, NE, 5-HT, ACh)
3. **HSA Architecture**: Hippocampal-Striatal-ACC inspired memory indexing
4. **Event-Driven Hooks**: Extensible hook system for cross-cutting concerns
5. **Saga Pattern**: Distributed transactions across vector/graph stores
6. **Cold Start Handling**: Population priors bootstrap new users
7. **Online Learning**: Bayesian updates for memory gating decisions
