# World Weaver System Diagrams

**Generated**: 2025-12-06
**Purpose**: Visual reference for system architecture and data flows

---

## 1. REST API Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REST API LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         SECURITY MIDDLEWARE                             │ │
│  │                                                                         │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │ │
│  │  │ CORS Validator│  │ Error         │  │ Rate Limiter  │              │ │
│  │  │ - No wildcards│  │ Sanitizer     │  │ (100 req/min) │              │ │
│  │  │   in prod     │  │ - Mask paths  │  │               │              │ │
│  │  │ - Restricted  │  │ - Mask URIs   │  │               │              │ │
│  │  │   headers     │  │ - Mask creds  │  │               │              │ │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘              │ │
│  │          │                  │                  │                       │ │
│  │          └──────────────────┼──────────────────┘                       │ │
│  │                             │                                          │ │
│  └─────────────────────────────┼──────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         SESSION VALIDATION                              │ │
│  │                                                                         │ │
│  │  X-Session-ID Header → validate_session_id() → MemoryServices          │ │
│  │                                                                         │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                │                                             │
│         ┌──────────────────────┼──────────────────────┐                     │
│         ▼                      ▼                      ▼                     │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐               │
│  │  /episodes  │       │  /entities  │       │   /skills   │               │
│  ├─────────────┤       ├─────────────┤       ├─────────────┤               │
│  │ POST /      │       │ POST /      │       │ POST /      │               │
│  │ GET /{id}   │       │ GET /{id}   │       │ GET /{id}   │               │
│  │ GET /       │       │ GET /       │       │ GET /       │               │
│  │ POST /recall│       │ POST /recall│       │ POST /recall│               │
│  │ DELETE /{id}│       │ POST /relat.│       │ POST /{}/exe│               │
│  │ POST /mark  │       │ POST /spread│       │ POST /deprec│               │
│  │             │       │ POST /supers│       │ GET /how-to │               │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘               │
│         │                     │                     │                       │
│         └─────────────────────┼─────────────────────┘                       │
│                               │                                             │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        MEMORY SERVICES                                  │ │
│  │                                                                         │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │ │
│  │  │  Episodic     │  │   Semantic    │  │  Procedural   │              │ │
│  │  │   Memory      │  │    Memory     │  │    Memory     │              │ │
│  │  │               │  │               │  │               │              │ │
│  │  │ .create()     │  │ .create_ent() │  │ .create_skill │              │ │
│  │  │ .recall()     │  │ .recall()     │  │ .recall_skill │              │ │
│  │  │ .get()        │  │ .get_entity() │  │ .get_proced() │              │ │
│  │  │ .recent()     │  │ .list_ent()   │  │ .list_skills  │              │ │
│  │  │ .delete()     │  │ .supersede()  │  │ .update()     │              │ │
│  │  │ .store()      │  │ .spread_act() │  │               │              │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘              │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Learning Systems Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LEARNING SYSTEMS ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    NEUROMODULATOR ORCHESTRA                             │ │
│  │                                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │  DOPAMINE   │  │NOREPINEPHRI.│  │  SEROTONIN  │  │ACETYLCHOLINE│  │ │
│  │  │    (DA)     │  │    (NE)     │  │   (5-HT)    │  │   (ACh)     │  │ │
│  │  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤  │ │
│  │  │ Reward      │  │ Arousal/    │  │ Temporal    │  │ Encoding/   │  │ │
│  │  │ prediction  │  │ attention   │  │ discounting │  │ retrieval   │  │ │
│  │  │ error       │  │ modulation  │  │             │  │ mode        │  │ │
│  │  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤  │ │
│  │  │ RPE = R -   │  │ gain =      │  │ discount =  │  │ mode = enc  │  │ │
│  │  │   V(s)      │  │ 1 + NE*m    │  │ e^(-5HT*t)  │  │ if ACh>0.5  │  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │ │
│  │         │                │                │                │         │ │
│  │         └────────────────┴────────────────┴────────────────┘         │ │
│  │                                  │                                    │ │
│  │                                  ▼                                    │ │
│  │                    ┌─────────────────────────┐                       │ │
│  │                    │  COMBINED MODULATION    │                       │ │
│  │                    │  M = DA * NE * 5HT * ACh│                       │ │
│  │                    └─────────────────────────┘                       │ │
│  │                                                                       │ │
│  └───────────────────────────────────┬───────────────────────────────────┘ │
│                                      │                                      │
│         ┌────────────────────────────┼────────────────────────────────┐    │
│         ▼                            ▼                                ▼    │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐  │
│  │ HEBBIAN LEARNING│       │ FSRS DECAY      │       │ ACT-R ACTIVATION│  │
│  ├─────────────────┤       ├─────────────────┤       ├─────────────────┤  │
│  │                 │       │                 │       │                 │  │
│  │ Synaptic weight │       │ Forgetting      │       │ Memory          │  │
│  │ strengthening   │       │ curve           │       │ activation      │  │
│  │                 │       │                 │       │                 │  │
│  │ Δw = η * pre *  │       │ R(t) = (1 +     │       │ A = B + Σ W*S  │  │
│  │      post * DA  │       │  t/9S)^(-0.5)   │       │                 │  │
│  │                 │       │                 │       │ B = base level  │  │
│  │ η = learning    │       │ S = stability   │       │ W = weight      │  │
│  │     rate        │       │ R = retriev.    │       │ S = spreading   │  │
│  │                 │       │                 │       │                 │  │
│  │ co-access       │       │ review →        │       │ fan effect:     │  │
│  │ strengthens     │       │ stability++     │       │ more links =    │  │
│  │ connections     │       │                 │       │ lower per-link  │  │
│  │                 │       │ decay →         │       │ activation      │  │
│  │                 │       │ retriev. ↓      │       │                 │  │
│  └────────┬────────┘       └────────┬────────┘       └────────┬────────┘  │
│           │                         │                         │            │
│           └─────────────────────────┼─────────────────────────┘            │
│                                     │                                      │
│                                     ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      PLASTICITY MANAGER                                 │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    SYNAPTIC TAGGING                              │  │ │
│  │  │                                                                  │  │ │
│  │  │  Early-phase LTP:  Short-lived, protein-independent             │  │ │
│  │  │  Late-phase LTP:   Long-lasting, requires protein synthesis     │  │ │
│  │  │                                                                  │  │ │
│  │  │  DA signal → tags early LTP → promotes → late LTP conversion    │  │ │
│  │  │                                                                  │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    MEMORY GATE ADAPTATION                        │  │ │
│  │  │                                                                  │  │ │
│  │  │  Bayesian update: P(store|outcome) = P(outcome|store) * P(store)│  │ │
│  │  │                   ──────────────────────────────────────────────│  │ │
│  │  │                              P(outcome)                          │  │ │
│  │  │                                                                  │  │ │
│  │  │  Success → increase store probability for similar inputs        │  │ │
│  │  │  Failure → decrease store probability                           │  │ │
│  │  │                                                                  │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Memory Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY FLOW DIAGRAM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     ENCODE PATH                  RETRIEVE PATH              UPDATE PATH     │
│                                                                              │
│    ┌─────────┐                 ┌─────────┐                ┌─────────┐       │
│    │ Content │                 │  Query  │                │Feedback │       │
│    └────┬────┘                 └────┬────┘                └────┬────┘       │
│         │                           │                          │            │
│         ▼                           ▼                          ▼            │
│    ┌─────────┐                 ┌─────────┐                ┌─────────┐       │
│    │ BGE-M3  │                 │ BGE-M3  │                │  Parse  │       │
│    │ Embed   │                 │ Embed   │                │ Outcome │       │
│    └────┬────┘                 └────┬────┘                └────┬────┘       │
│         │                           │                          │            │
│         ▼                           ▼                          ▼            │
│    ┌─────────┐                 ┌─────────┐                ┌─────────┐       │
│    │ Pattern │                 │  Index  │                │   DA    │       │
│    │  Sep.   │                 │ Lookup  │                │   RPE   │       │
│    │  (DG)   │                 │ (HSA)   │                │ Compute │       │
│    └────┬────┘                 └────┬────┘                └────┬────┘       │
│         │                           │                          │            │
│         ▼                           ▼                          ▼            │
│    ┌─────────┐                 ┌─────────┐                ┌─────────┐       │
│    │ Memory  │                 │  Multi  │                │ Weight  │       │
│    │  Gate   │──── decides ───▶│ Signal  │                │ Update  │       │
│    │Decision │   to store?     │ Fusion  │                │ Hebbian │       │
│    └────┬────┘                 └────┬────┘                └────┬────┘       │
│         │                           │                          │            │
│    ┌────┴────┐                      ▼                          ▼            │
│    │         │                 ┌─────────┐                ┌─────────┐       │
│    ▼         ▼                 │Inhibit. │                │ Stabil. │       │
│ ┌──────┐ ┌──────┐              │ Network │                │ Update  │       │
│ │STORE │ │ GATE │              │ (WTA)   │                │  FSRS   │       │
│ │      │ │ OUT  │              └────┬────┘                └────┬────┘       │
│ └──┬───┘ └──────┘                   │                          │            │
│    │                                ▼                          ▼            │
│    │                           ┌─────────┐                ┌─────────┐       │
│    │                           │ Rerank  │                │ Reconst │       │
│    │                           │(Learned)│                │ -olidat │       │
│    │                           └────┬────┘                └─────────┘       │
│    │                                │                                       │
│    └────────────────────────────────┼────────────────────────────────┐      │
│                                     │                                │      │
│                                     ▼                                ▼      │
│                            ┌────────────────────────────────────────────┐   │
│                            │              PERSISTENCE                   │   │
│                            │                                            │   │
│                            │    Qdrant (vectors)    Neo4j (graph)      │   │
│                            │         ▲                  ▲              │   │
│                            │         │                  │              │   │
│                            │         └──────────────────┘              │   │
│                            │              │                             │   │
│                            │              ▼                             │   │
│                            │        Saga (ACID)                         │   │
│                            └────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. ScoredResult Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ScoredResult DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Memory Service                    API Route                  Response       │
│                                                                              │
│  ┌───────────────┐           ┌───────────────┐           ┌───────────────┐ │
│  │               │           │               │           │               │ │
│  │  recall()     │──────────▶│   results =   │──────────▶│   JSON        │ │
│  │  recall_skill │           │  [ScoredRes.] │           │  Response     │ │
│  │               │           │               │           │               │ │
│  └───────────────┘           └───────────────┘           └───────────────┘ │
│         │                           │                           │          │
│         │                           │                           │          │
│         ▼                           ▼                           ▼          │
│  ┌───────────────┐           ┌───────────────┐           ┌───────────────┐ │
│  │ Returns:      │           │ Access via:   │           │ Return:       │ │
│  │               │           │               │           │               │ │
│  │ list[Scored   │           │ r.item →      │           │ {"entities":  │ │
│  │   Result[T]]  │           │   Episode/    │           │   [...],      │ │
│  │               │           │   Entity/     │           │  "total": N}  │ │
│  │ where:        │           │   Procedure   │           │               │ │
│  │  .item = T    │           │               │           │ or            │ │
│  │  .score =float│           │ r.score →     │           │               │ │
│  │               │           │   similarity  │           │ {"episodes":  │ │
│  └───────────────┘           └───────────────┘           │   [...],      │ │
│                                                          │  "scores":    │ │
│                                                          │   [...]}      │ │
│                                                          └───────────────┘ │
│                                                                              │
│  CORRECT PATTERN:                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  results = await memory.recall(query)                                  │ │
│  │  items = [r.item for r in results]      # Extract items               │ │
│  │  scores = [r.score for r in results]    # Extract scores              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  INCORRECT PATTERN (will fail):                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  results = await memory.recall(query)                                  │ │
│  │  items = results  # WRONG - results are ScoredResult, not raw items   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INCOMING REQUEST                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         CORS MIDDLEWARE                                  ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  DEVELOPMENT:                                                    │   ││
│  │  │  - Allow localhost origins                                       │   ││
│  │  │  - Wildcards permitted                                           │   ││
│  │  │                                                                  │   ││
│  │  │  PRODUCTION (WW_ENVIRONMENT=production):                         │   ││
│  │  │  - REJECT wildcards (*)                                          │   ││
│  │  │  - Only explicit origins allowed                                 │   ││
│  │  │  - Restricted headers: Authorization, Content-Type,             │   ││
│  │  │    X-Session-ID, X-Request-ID                                   │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         RATE LIMITING                                    ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  - 100 requests / 60 seconds / session                          │   ││
│  │  │  - Sliding window algorithm                                      │   ││
│  │  │  - Returns 429 + Retry-After header when exceeded               │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         SESSION VALIDATION                               ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  - UUID format check                                             │   ││
│  │  │  - Alphanumeric character validation                             │   ││
│  │  │  - Reserved ID blocking (optional)                               │   ││
│  │  │  - Null byte detection                                           │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         INPUT VALIDATION                                 ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  - XSS pattern removal                                           │   ││
│  │  │  - Enum value validation (EntityType, Domain, Outcome)          │   ││
│  │  │  - Range validation (0-1 for floats, positive ints)             │   ││
│  │  │  - Pydantic model validation                                     │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         ERROR SANITIZATION                               ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  PATTERNS MASKED:                                                │   ││
│  │  │  - bolt://user:pass@host  →  [database connection]              │   ││
│  │  │  - /home/aaron/path       →  [path]                              │   ││
│  │  │  - api_key=xxxxx          →  [credential]                        │   ││
│  │  │  - stack traces           →  logged only (not returned)         │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│     MEMORY SERVICES (protected)                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Consolidation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONSOLIDATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LIGHT CONSOLIDATION (dedupe)          DEEP CONSOLIDATION (abstraction)     │
│                                                                              │
│  ┌───────────────────────┐             ┌───────────────────────┐            │
│  │ 1. Recent episodes    │             │ 1. Recall 500 episodes│            │
│  │    (last N hours)     │             │                       │            │
│  └──────────┬────────────┘             └──────────┬────────────┘            │
│             │                                     │                          │
│             ▼                                     ▼                          │
│  ┌───────────────────────┐             ┌───────────────────────┐            │
│  │ 2. Pairwise similarity│             │ 2. HDBSCAN clustering │            │
│  │    comparison         │             │    (min_cluster=3)    │            │
│  └──────────┬────────────┘             └──────────┬────────────┘            │
│             │                                     │                          │
│             ▼                                     ▼                          │
│  ┌───────────────────────┐             ┌───────────────────────┐            │
│  │ 3. Mark duplicates    │             │ 3. Extract entities   │            │
│  │    (sim > 0.95)       │             │    from clusters      │            │
│  └──────────┬────────────┘             └──────────┬────────────┘            │
│             │                                     │                          │
│             ▼                                     ▼                          │
│  ┌───────────────────────┐             ┌───────────────────────┐            │
│  │ 4. Keep best (highest │             │ 4. Create/supersede   │            │
│  │    valence/stability) │             │    entities           │            │
│  └───────────────────────┘             └──────────┬────────────┘            │
│                                                   │                          │
│                                                   ▼                          │
│                                        ┌───────────────────────┐            │
│                                        │ 5. Link episodes to   │            │
│                                        │    entities (SOURCE_OF)            │
│                                        └───────────────────────┘            │
│                                                                              │
│  SKILL CONSOLIDATION                   SLEEP CONSOLIDATION                  │
│                                                                              │
│  ┌───────────────────────┐             ┌───────────────────────┐            │
│  │ 1. Retrieve procedures│             │ 1. WAKE → NREM        │            │
│  └──────────┬────────────┘             │    (encoding active)  │            │
│             │                          └──────────┬────────────┘            │
│             ▼                                     │                          │
│  ┌───────────────────────┐                        ▼                          │
│  │ 2. HDBSCAN on scripts │             ┌───────────────────────┐            │
│  │    (min_cluster=2)    │             │ 2. NREM replay        │            │
│  └──────────┬────────────┘             │    (hippocampal)      │            │
│             │                          └──────────┬────────────┘            │
│             ▼                                     │                          │
│  ┌───────────────────────┐                        ▼                          │
│  │ 3. Sort by success_rate            ┌───────────────────────┐            │
│  │    descending         │             │ 3. REM integration    │            │
│  └──────────┬────────────┘             │    (cortical)         │            │
│             │                          └──────────┬────────────┘            │
│             ▼                                     │                          │
│  ┌───────────────────────┐                        ▼                          │
│  │ 4. Merge steps into   │             ┌───────────────────────┐            │
│  │    best procedure     │             │ 4. WAKE retrieval     │            │
│  └──────────┬────────────┘             │    (consolidated mem) │            │
│             │                          └───────────────────────┘            │
│             ▼                                                                │
│  ┌───────────────────────┐                                                   │
│  │ 5. Deprecate others   │                                                   │
│  │    (CONSOLIDATED_INTO)│                                                   │
│  └───────────────────────┘                                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Diagram | Purpose | Key Components |
|---------|---------|----------------|
| REST API | API structure and security | Routes, middleware, services |
| Learning Systems | Neuromodulation and adaptation | DA, NE, 5-HT, ACh, Hebbian, FSRS |
| Memory Flow | Data path through system | Encode, retrieve, update paths |
| ScoredResult | API result handling | Wrapper type for scored items |
| Security | Protection layers | CORS, rate limit, validation, sanitization |
| Consolidation | Memory stabilization | Light, deep, skill, sleep cycles |

---

**See Also**:
- `FUNCTIONAL_ARCHITECTURE.md` - Detailed functional decomposition
- `architecture.md` - System architecture overview
- `API.md` - REST API documentation
