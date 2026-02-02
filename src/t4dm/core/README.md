# Core Module

**13 files | ~4,500 lines | Centrality: 16 (Hub)**

The core module is the central hub of T4DM, providing types, configuration, validation, protocols, and services that all other modules depend upon.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CORE HUB                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │     Types      │  │  Configuration │  │       Validation           │ │
│  │ Episode/Entity │  │  80+ params    │  │  XSS, injection, sanitize  │ │
│  │ Procedure/Rel  │  │  YAML + env    │  │  Session ID, UUID, enum    │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │   Protocols    │  │    Services    │  │    Memory Gate             │ │
│  │ EmbeddingProv  │  │  get_services  │  │  Heuristic + Learned       │ │
│  │ VectorStore    │  │  RateLimiter   │  │  Thompson sampling         │ │
│  │ GraphStore     │  │  cleanup       │  │  Three-factor learning     │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ Serialization  │  │ Privacy Filter │  │    Action Framework        │ │
│  │ Episode/Entity │  │ PII redaction  │  │  50+ actions, permissions  │ │
│  │ Procedure      │  │ Voice commands │  │  Risk levels, confirm      │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `types.py` | ~350 | Pydantic models: Episode, Entity, Procedure, Relationship |
| `config.py` | ~1100 | Settings with 80+ parameters, YAML + env support |
| `validation.py` | ~550 | Input validation, XSS prevention, sanitization |
| `protocols.py` | ~200 | Provider-agnostic interfaces (EmbeddingProvider, VectorStore, GraphStore) |
| `services.py` | ~180 | Thread-safe service lifecycle, RateLimiter |
| `container.py` | ~210 | Dependency injection container |
| `serialization.py` | ~300 | Domain ↔ storage format converters |
| `privacy_filter.py` | ~340 | PII redaction, credential detection |
| `memory_gate.py` | ~360 | Heuristic storage decisions |
| `learned_gate.py` | ~920 | Online Bayesian learning for storage |
| `actions.py` | ~950 | Permission-based action framework |
| `personal_entities.py` | ~460 | Google Workspace integration models |
| `__init__.py` | ~30 | Public API exports |

## Core Types

### Episode (Episodic Memory)

```python
@dataclass
class Episode:
    id: UUID
    session_id: str
    content: str
    embedding: list[float]  # BGE-M3 1024-dim
    timestamp: datetime      # Event time (T_ref)
    ingested_at: datetime    # System time (T_sys)
    context: EpisodeContext  # Project, file, tool
    outcome: Outcome         # SUCCESS, FAILURE, PARTIAL, NEUTRAL
    emotional_valence: float # [0,1] importance

    # FSRS tracking
    access_count: int
    last_accessed: datetime
    stability: float  # Days

    # Temporal links (P5.2)
    previous_episode_id: UUID | None
    next_episode_id: UUID | None

    def retrievability(self, current_time=None) -> float:
        """R(t,S) = (1 + 0.9*t/S)^-0.5"""
```

### Entity (Semantic Memory)

```python
@dataclass
class Entity:
    id: UUID
    name: str
    entity_type: EntityType  # CONCEPT, PERSON, PROJECT, TOOL, TECHNIQUE, FACT
    summary: str
    details: str | None
    embedding: list[float]
    source: str | None

    # Bi-temporal versioning
    valid_from: datetime
    valid_to: datetime | None

    def is_valid(self, at_time=None) -> bool
```

### Procedure (Procedural Memory)

```python
@dataclass
class Procedure:
    id: UUID
    name: str
    domain: Domain  # CODING, RESEARCH, TRADING, DEVOPS, WRITING
    trigger_pattern: str | None
    steps: list[ProcedureStep]
    script: str | None

    success_rate: float
    execution_count: int
    version: int
    deprecated: bool

    def update_success_rate(self, success: bool) -> float
    def should_deprecate(self, min_executions=10, min_success=0.3) -> bool
```

### Relationship (Hebbian-Weighted)

```python
@dataclass
class Relationship:
    source_id: UUID
    target_id: UUID
    relation_type: RelationType  # USES, PRODUCES, REQUIRES, CAUSES, PART_OF
    weight: float  # [0,1] Hebbian strength
    co_access_count: int

    def strengthen(self, learning_rate=0.1) -> float:
        """w' = w + lr*(1-w)"""
```

## Configuration

### Settings Class

80+ configurable parameters with strong validation:

```python
from t4dm.core import get_settings

settings = get_settings()  # Cached singleton

# Storage
settings.neo4j_uri           # bolt://localhost:7687
settings.qdrant_url          # http://localhost:6333

# Embedding
settings.embedding_model     # BAAI/bge-m3
settings.embedding_dimension # 1024
settings.embedding_device    # cuda:0

# FSRS (memory decay)
settings.fsrs_decay_factor   # 0.9
settings.fsrs_retention_target  # 0.9

# ACT-R (spreading activation)
settings.actr_spreading_strength  # 1.6
settings.actr_decay               # 0.5

# Hebbian learning
settings.hebbian_learning_rate  # 0.1
settings.hebbian_decay_rate     # 0.01

# Retrieval weights (sum to 1.0)
settings.episodic_weight_semantic    # 0.4
settings.episodic_weight_recency     # 0.25
settings.episodic_weight_outcome     # 0.2
settings.episodic_weight_importance  # 0.15
```

### Configuration Load Order

1. **Environment Variables** (`T4DM_*` prefix)
2. **YAML Config File** (searched in order):
   - `T4DM_CONFIG_FILE` env var
   - `./t4dm.yaml`
   - `~/.t4dm/config.yaml`
   - `/etc/t4dm/config.yaml`
3. **Default Values**

## Validation

Security-focused input validation:

```python
from t4dm.core import (
    validate_session_id,
    sanitize_string,
    validate_uuid,
    ValidationError
)

# Session ID validation (injection prevention)
session = validate_session_id(user_input)  # Raises ValidationError

# String sanitization (XSS, null bytes)
clean = sanitize_string(user_content)

# UUID validation
uuid = validate_uuid(id_string)

# Numeric validation
value = validate_positive_int(num, "count", max_val=1000)
score = validate_range(value, 0.0, 1.0, "score")
```

**Security Features**:
- Session ID: Alphanumeric + `-_` only, 1-128 chars
- Path traversal: `..`, `/`, `\` rejected
- Null bytes: Explicitly checked
- XSS patterns: 12 regex patterns blocked
- Control characters: Removed (except newline/tab)

## Protocols

Provider-agnostic interfaces:

```python
from t4dm.core import EmbeddingProvider, VectorStore, GraphStore

@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int
    async def embed(self, texts: list[str]) -> list[list[float]]
    async def embed_query(self, query: str) -> list[float]

@runtime_checkable
class VectorStore(Protocol):
    async def add(self, collection, ids, vectors, payloads)
    async def search(self, collection, vector, limit, filter=None)
    async def delete(self, collection, ids)

@runtime_checkable
class GraphStore(Protocol):
    async def create_node(self, label, properties) -> str
    async def create_relationship(self, source_id, target_id, rel_type, properties)
    async def query(self, cypher, parameters=None) -> list[dict]
```

## Services

Thread-safe service lifecycle:

```python
from t4dm.core import get_services, cleanup_services, RateLimiter

# Get memory services (lazy initialization)
episodic, semantic, procedural = await get_services(session_id)

# Cleanup on shutdown
await cleanup_services(session_id)

# Rate limiting
limiter = RateLimiter(requests_per_minute=60)
if limiter.allow(session_id):
    # Process request
    pass
```

## Memory Gate

### Heuristic Gate

```python
from t4dm.core import MemoryGate, GateContext

gate = MemoryGate(store_threshold=0.4, buffer_threshold=0.2)

context = GateContext(
    session_id="session-1",
    project="t4dm",
    is_voice=False
)

result = gate.evaluate("Deployed to production", context)
if result.decision == StorageDecision.STORE:
    # Store with suggested_importance
    pass
```

**Scoring factors**: Novelty, outcome signals, entity density, action significance

### Learned Gate (Online Bayesian)

```python
from t4dm.core import LearnedMemoryGate

gate = LearnedMemoryGate(
    store_threshold=0.6,
    cold_start_threshold=100
)

# Predict with Thompson sampling
decision = gate.predict(
    content_embedding=embedding,
    context=context,
    neuromod_state=neuromod,
    explore=True
)

# Update from feedback
gate.update(memory_id, utility=0.8)

# Get stats
stats = gate.get_stats()  # n_obs, ECE, accuracy
```

**Features**:
- 247-dimensional feature vector
- Thompson sampling for exploration
- Cold start blending with heuristics
- Three-factor learning integration

## Action Framework

50+ built-in actions with permission model:

```python
from t4dm.core import ActionRegistry, ActionExecutor, ActionRequest

registry = ActionRegistry()
executor = ActionExecutor(registry)

request = ActionRequest(
    action_name="email.send",
    category=ActionCategory.EMAIL,
    parameters={"to": ["user@example.com"], "subject": "Hello"}
)

result = await executor.execute(request)
if result.requires_confirmation:
    # Ask user, then confirm
    executor.confirm(result.action_id)
```

**Permission Levels**: ALLOWED, LOGGED, CONFIRM, VERIFY, BLOCKED
**Risk Levels**: NONE, LOW, MEDIUM, HIGH, CRITICAL

## Privacy Filter

PII redaction and credential detection:

```python
from t4dm.core import PrivacyFilter, ContentClassifier

filter = PrivacyFilter()
redacted = filter.redact("My SSN is 123-45-6789")
# "My SSN is [REDACTED_SSN]"

classifier = ContentClassifier()
level = classifier.classify(content)
# SensitivityLevel.CONFIDENTIAL
```

**Patterns**: SSN, credit cards, phone numbers, API keys, AWS keys, tokens
**Voice commands**: "off the record", "forget that", "don't store this"

## Public API

```python
# Types
Episode, Entity, Procedure, Relationship
Outcome, EntityType, RelationType, Domain
EpisodeContext, ProcedureStep

# Configuration
Settings, get_settings, reset_settings

# Validation
validate_session_id, sanitize_string, validate_uuid
validate_positive_int, validate_range, ValidationError

# Protocols
EmbeddingProvider, VectorStore, GraphStore
EpisodicStore, SemanticStore, ProceduralStore

# Services
get_services, cleanup_services, RateLimiter

# Gates
MemoryGate, LearnedMemoryGate, GateContext, GateResult

# Actions
ActionRegistry, ActionExecutor, ActionRequest, ActionResult
```

## Dependency Graph

Core is the hub with highest centrality (16):

```
Core ← Storage (Neo4j, Qdrant)
Core ← Memory (Episodic, Semantic, Procedural)
Core ← Consolidation
Core ← Learning
Core ← API, CLI
Core ← Embedding
```

**No circular dependencies**: Clean architecture

## Testing

```bash
# Run core tests
pytest tests/core/ -v

# With coverage
pytest tests/core/ --cov=t4dm.core --cov-report=term-missing
```

## Security Checklist

- [ ] Neo4j password: 12+ chars, 3 of 4 char classes
- [ ] API key required in production
- [ ] CORS origins: HTTPS in production (no wildcard)
- [ ] File permissions: 0o600 on config files
- [ ] No credentials in logs
