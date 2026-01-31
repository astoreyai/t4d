# Core Module
**Path**: `/mnt/projects/t4d/t4dm/src/ww/core/`

## What
Central hub of the system providing domain types (Episode, Entity, Procedure, Relationship), configuration (80+ parameters), input validation, provider-agnostic protocols, service lifecycle, memory gating, action framework, privacy filtering, and production infrastructure (circuit breakers, feature flags, secrets management).

## How
- **Types** (Pydantic): Episode with FSRS decay (`R(t,S) = (1 + 0.9*t/S)^-0.5`), Entity with bi-temporal versioning, Procedure with success tracking, Relationship with Hebbian strengthening (`w' = w + lr*(1-w)`)
- **Config**: Settings class with 80+ params, loaded from env vars (`WW_*`) -> YAML -> defaults
- **Protocols**: `EmbeddingProvider`, `VectorStore`, `GraphStore` -- runtime-checkable abstract interfaces
- **Services**: Thread-safe lazy initialization via `get_services(session_id)`, RateLimiter
- **Validation**: XSS pattern blocking (12 regexes), path traversal prevention, null byte checks, session ID sanitization
- **Memory Gate**: Heuristic gate (novelty/outcome/entity density scoring) + learned gate (247-dim features, Thompson sampling, three-factor learning)
- **Emergency** (Phase 9): CircuitBreaker, EmergencyManager with PanicLevel
- **Feature Flags** (Phase 9): Runtime toggles with percentage rollout
- **Privacy Filter**: PII redaction (SSN, credit cards, API keys), voice command detection ("off the record")

## Why
All other modules depend on core for types, config, and protocols. Clean architecture with no circular dependencies. Centrality: 16 (highest in the codebase).

## Key Files
| File | Purpose |
|------|---------|
| `types.py` | Episode, Entity, Procedure, Relationship (~350 lines) |
| `config.py` | Settings with 80+ params, YAML + env support (~1,100 lines) |
| `validation.py` | Input sanitization, XSS prevention (~550 lines) |
| `protocols.py` | EmbeddingProvider, VectorStore, GraphStore interfaces (~200 lines) |
| `services.py` | Thread-safe service lifecycle, RateLimiter (~180 lines) |
| `memory_gate.py` | Heuristic storage decisions (~360 lines) |
| `learned_gate.py` | Online Bayesian learned gate (~920 lines) |
| `actions.py` | 50+ actions with permission/risk framework (~950 lines) |
| `privacy_filter.py` | PII redaction, credential detection (~340 lines) |
| `emergency.py` | CircuitBreaker, EmergencyManager |
| `feature_flags.py` | Runtime feature toggles |
| `secrets.py` | SecretsManager with multiple backends |

## Data Flow
```
All modules -> core.types (Episode, Entity, Procedure)
All modules -> core.config (get_settings())
API/CLI -> core.services (get_services()) -> memory stores
API/CLI -> core.validation (sanitize inputs)
Memory ops -> core.memory_gate (store/buffer/discard decision)
```

## Integration Points
- **Every module** depends on core (hub architecture)
- **storage**: Implements VectorStore, GraphStore protocols
- **embedding**: Implements EmbeddingProvider protocol
- **api/cli**: Uses services, config, validation
