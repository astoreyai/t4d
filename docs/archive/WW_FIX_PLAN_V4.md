# World Weaver Fix Plan v4.0

**Created**: 2025-11-27 | **Based On**: Gap Analysis Post-Phase 9 | **Priority**: Production Hardening

> Phases 10-12 address CRITICAL, HIGH, and MEDIUM issues identified in gap analysis after Phases 1-9 completion.

---

## Status Dashboard

| Phase | Status | Priority | Tasks | Description |
|-------|--------|----------|-------|-------------|
| Phase 1-9 | COMPLETE | - | 58 | Initial implementation through feature gaps |
| **Phase 10: Critical Fixes** | PENDING | P0 | 5 | Performance O(n^2), security passwords |
| **Phase 11: High Priority** | PENDING | P1 | 7 | Testing gaps, N+1 queries, gateway security |
| **Phase 12: Medium Priority** | PENDING | P2 | 8 | Architecture, testing, performance polish |

### Total New Tasks: 20
### Estimated Duration: 4-5 weeks

---

## Gap Analysis Summary

### CRITICAL (Phase 10) - 5 Issues
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| C1 | O(n^2) duplicate detection | `consolidation/service.py:397` | Performance degradation at scale |
| C2 | Unbounded query "*" | `consolidation/service.py:148` | Memory exhaustion |
| C3 | Sequential Hebbian decay (N queries) | `memory/semantic.py:686` | Slow consolidation |
| C4 | Hardcoded password in docker-compose | `docker-compose.yml:14` | Security vulnerability |
| C5 | Default password "password" | `core/config.py:135` | Security vulnerability |

### HIGH (Phase 11) - 7 Issues
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| H1 | Incomplete LLM entity extraction | `extraction/entity_extractor.py:252` | Feature incomplete |
| H2 | Missing saga compensation tests | `storage/saga.py` | Data consistency risk |
| H3 | No tests for MCP modules | `mcp/errors.py, server.py, resources.py` | Regression risk |
| H4 | N+1 access updates on recall | `memory/episodic.py:249` | Performance issue |
| H5 | Spreading activation graph explosion | `memory/semantic.py:597` | Memory exhaustion |
| H6 | Session ID validation not at gateway | `mcp/gateway.py:301` | Security bypass risk |
| H7 | Qdrant/Neo4j exposed without auth | `docker-compose.yml` | Security vulnerability |

### MEDIUM (Phase 12) - 8 Issues
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| M1 | Serialization duplication (150 lines) | `memory/*.py` | Maintainability |
| M2 | Singleton pattern limits testability | Multiple files | Testing difficulty |
| M3 | No embedding provider tests | `embedding/bge_m3.py` | Regression risk |
| M4 | No chaos/fault injection tests | Tests missing | Resilience unknown |
| M5 | HDBSCAN memory at scale | `consolidation/service.py` | Scalability limit |
| M6 | Embedding cache no TTL | `embedding/bge_m3.py` | Memory leak potential |
| M7 | Sequential batch updates | `storage/qdrant_store.py` | Performance issue |
| M8 | OTLP exporter insecure | `observability/tracing.py` | Security risk |

---

## Phase 10: Critical Fixes (P0)

**Goal**: Eliminate performance blockers and security vulnerabilities that block production deployment.

**Estimated Duration**: 1.5 weeks

---

### TASK-P10-001: Replace O(n^2) Duplicate Detection with LSH

**Files**:
- `src/ww/consolidation/service.py` (modify)
- `src/ww/storage/qdrant_store.py` (add method)

**Description**: The `_find_duplicates()` method at line 397 uses O(n^2) pairwise comparison for finding near-duplicate episodes. This becomes prohibitively slow with >1000 episodes.

**Current Code** (lines 397-433):
```python
async def _find_duplicates(
    self,
    episodes: list[Episode],
    threshold: float = 0.95,
) -> list[tuple[str, str]]:
    # Compare all pairs (O(n^2) - for production, use approximate NN)
    ep_list = list(embeddings.keys())
    for i, id1 in enumerate(ep_list):
        for id2 in ep_list[i + 1:]:
            # ...pairwise comparison
```

**Solution**: Use Locality-Sensitive Hashing (LSH) via Qdrant's native search with high threshold.

**Implementation**:
```python
async def _find_duplicates(
    self,
    episodes: list[Episode],
    threshold: float = 0.95,
) -> list[tuple[str, str]]:
    """Find near-duplicate episode pairs using approximate NN."""
    if len(episodes) < 2:
        return []

    duplicates = []
    seen_pairs = set()

    # Use Qdrant's native search for approximate NN
    for ep in episodes:
        # Skip if no embedding available
        if ep.embedding is None:
            continue

        # Search for similar episodes
        results = await self.vector_store.search(
            collection=self.vector_store.episodes_collection,
            vector=ep.embedding,
            limit=5,  # Check top 5 candidates
            score_threshold=threshold,
        )

        for id_str, score, payload in results:
            if id_str != str(ep.id):
                pair = tuple(sorted([str(ep.id), id_str]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    # Determine which to keep (older)
                    ep_time = ep.timestamp
                    other_time = datetime.fromisoformat(payload.get("timestamp", ep_time.isoformat()))
                    if ep_time < other_time:
                        duplicates.append((str(ep.id), id_str))
                    else:
                        duplicates.append((id_str, str(ep.id)))

    return duplicates
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_duplicate_detection_performance` | Benchmark with 1000+ episodes | < 5s for 1000 episodes |
| `test_duplicate_detection_accuracy` | Verify duplicates found correctly | 95%+ precision |
| `test_duplicate_detection_empty` | Empty input handling | Returns [] |
| `test_duplicate_detection_single` | Single episode | Returns [] |

**Hooks**: Pre-consolidation hook (validate data)

**Validation**:
- [ ] Performance benchmark: 1000 episodes in < 5 seconds
- [ ] Unit tests pass
- [ ] No regression in duplicate detection accuracy

---

### TASK-P10-002: Add Pagination to Unbounded Queries

**Files**:
- `src/ww/consolidation/service.py` (modify)
- `src/ww/memory/episodic.py` (add paginated recall)

**Description**: The `_consolidate_light()` method uses `query="*"` with limit=1000, which can exhaust memory with large datasets.

**Current Code** (lines 148-152):
```python
results = await episodic.recall(
    query="*",
    limit=1000,
    session_filter=session_filter,
)
```

**Solution**: Implement cursor-based pagination for consolidation queries.

**Implementation**:
```python
# In episodic.py - add paginated recall
async def recall_paginated(
    self,
    query: str,
    page_size: int = 100,
    cursor: Optional[str] = None,
    session_filter: Optional[str] = None,
) -> tuple[list[ScoredResult], Optional[str]]:
    """
    Paginated recall for large result sets.

    Args:
        query: Search query
        page_size: Results per page (max 100)
        cursor: Continuation token from previous page
        session_filter: Session filter

    Returns:
        Tuple of (results, next_cursor)
    """
    # Decode cursor to get offset
    offset = 0
    if cursor:
        try:
            offset = int(cursor)
        except ValueError:
            offset = 0

    # Fetch page + 1 to check for more
    results = await self.recall(
        query=query,
        limit=page_size + 1,
        session_filter=session_filter,
    )

    # Check if there are more results
    has_more = len(results) > page_size
    if has_more:
        results = results[:page_size]
        next_cursor = str(offset + page_size)
    else:
        next_cursor = None

    return results, next_cursor

# In consolidation/service.py - use pagination
async def _consolidate_light_paginated(
    self,
    session_filter: Optional[str] = None,
    page_size: int = 100,
) -> dict:
    """Light consolidation with pagination."""
    episodic, _, _ = await self._get_services()

    total_scanned = 0
    duplicates_found = 0
    cleaned = 0
    cursor = None
    all_episodes = []

    # Paginate through all episodes
    while True:
        results, next_cursor = await episodic.recall_paginated(
            query="*",
            page_size=page_size,
            cursor=cursor,
            session_filter=session_filter,
        )

        episodes = [r.item for r in results]
        all_episodes.extend(episodes)
        total_scanned += len(episodes)

        if next_cursor is None:
            break
        cursor = next_cursor

        # Memory safety: process in batches of 1000
        if len(all_episodes) >= 1000:
            batch_dups = await self._find_duplicates(all_episodes, threshold=0.95)
            duplicates_found += len(batch_dups)
            # Process duplicates...
            all_episodes = []

    # Process final batch
    if all_episodes:
        batch_dups = await self._find_duplicates(all_episodes, threshold=0.95)
        duplicates_found += len(batch_dups)

    return {
        "episodes_scanned": total_scanned,
        "duplicates_found": duplicates_found,
        "cleaned": cleaned,
    }
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_paginated_recall_basic` | Basic pagination works | Cursor increments correctly |
| `test_paginated_recall_memory` | Memory usage stays bounded | < 500MB for 10k episodes |
| `test_paginated_consolidation` | Full consolidation with pagination | Completes without OOM |
| `test_pagination_edge_cases` | Empty, single, exact page size | All return correctly |

**Hooks**: None

**Validation**:
- [ ] Memory profiling shows bounded usage
- [ ] All pagination tests pass
- [ ] Consolidation works with 10k+ episodes

---

### TASK-P10-003: Batch Hebbian Decay Updates

**Files**:
- `src/ww/memory/semantic.py` (modify)
- `src/ww/storage/neo4j_store.py` (add batch method)

**Description**: The `apply_hebbian_decay()` method at line 686 iterates through stale relationships with N individual update queries. This should use batch operations.

**Current Code** (lines 686-706):
```python
for rel in stale_rels:
    current_weight = rel.get("weight", 1.0)
    new_weight = current_weight * (1 - decay_rate)

    if new_weight < min_weight:
        await self.graph_store.delete_relationship(rel["id"])
        pruned += 1
    else:
        await self.graph_store.update_relationship_by_id(
            rel["id"],
            properties={...}
        )
        decayed += 1
```

**Solution**: Add batch update/delete methods and use them for Hebbian decay.

**Implementation**:
```python
# In neo4j_store.py - add batch methods
async def batch_update_relationships(
    self,
    updates: list[tuple[str, dict]],
) -> int:
    """
    Batch update multiple relationships.

    Args:
        updates: List of (relationship_id, properties) tuples

    Returns:
        Count of updated relationships
    """
    if not updates:
        return 0

    async with self._get_session() as session:
        # Build batch UNWIND query
        query = """
        UNWIND $updates AS update
        MATCH ()-[r]->()
        WHERE elementId(r) = update.id
        SET r += update.props
        RETURN count(r) as updated
        """

        update_params = [
            {"id": rel_id, "props": props}
            for rel_id, props in updates
        ]

        result = await session.run(query, updates=update_params)
        record = await result.single()
        return record["updated"] if record else 0

async def batch_delete_relationships(
    self,
    relationship_ids: list[str],
) -> int:
    """
    Batch delete multiple relationships.

    Args:
        relationship_ids: List of relationship IDs to delete

    Returns:
        Count of deleted relationships
    """
    if not relationship_ids:
        return 0

    async with self._get_session() as session:
        query = """
        UNWIND $ids AS relId
        MATCH ()-[r]->()
        WHERE elementId(r) = relId
        DELETE r
        RETURN count(*) as deleted
        """

        result = await session.run(query, ids=relationship_ids)
        record = await result.single()
        return record["deleted"] if record else 0

# In semantic.py - use batch operations
async def apply_hebbian_decay(self, ...) -> dict:
    # ...existing code to find stale_rels...

    # Separate into update and delete lists
    to_update = []
    to_delete = []

    for rel in stale_rels:
        current_weight = rel.get("weight", 1.0)
        new_weight = current_weight * (1 - decay_rate)

        if new_weight < min_weight:
            to_delete.append(rel["id"])
        else:
            to_update.append((
                rel["id"],
                {
                    "weight": new_weight,
                    "lastDecay": datetime.now().isoformat(),
                }
            ))

    # Batch operations
    pruned = await self.graph_store.batch_delete_relationships(to_delete)
    decayed = await self.graph_store.batch_update_relationships(to_update)

    return {
        "decayed_count": decayed,
        "pruned_count": pruned,
        "total_processed": len(stale_rels),
    }
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_batch_update_relationships` | Batch updates work | All relationships updated |
| `test_batch_delete_relationships` | Batch deletes work | All relationships deleted |
| `test_hebbian_decay_batch` | Decay uses batching | Single query per operation type |
| `test_batch_empty` | Empty input handling | Returns 0, no errors |

**Hooks**: None

**Validation**:
- [ ] Query count reduced from N to 2 (one update, one delete)
- [ ] Performance: 1000 relationships in < 1 second
- [ ] All tests pass

---

### TASK-P10-004: Remove Hardcoded Passwords from Docker Compose

**Files**:
- `docker-compose.yml` (modify)
- `docker-compose.example.yml` (create)
- `.env.example` (create/update)

**Description**: The docker-compose.yml contains hardcoded password at line 14 (`NEO4J_AUTH=neo4j/wwpassword`). This is a security vulnerability.

**Current Code** (line 14):
```yaml
environment:
  - NEO4J_AUTH=neo4j/wwpassword
```

**Solution**: Use environment variables with secure defaults and documentation.

**Implementation**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5-community
    container_name: ww-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:?NEO4J_PASSWORD required}
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=1G
    volumes:
      - ww_neo4j_data:/data
      - ww_neo4j_logs:/logs
      - ww_neo4j_plugins:/plugins
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: ww-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY:-}
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    volumes:
      - ww_qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  ww_neo4j_data:
    driver: local
  ww_neo4j_logs:
    driver: local
  ww_neo4j_plugins:
    driver: local
  ww_qdrant_storage:
    driver: local

networks:
  default:
    name: ww-network
```

```bash
# .env.example
# World Weaver Environment Variables
# Copy to .env and update with secure values

# Neo4j Configuration (REQUIRED)
NEO4J_USER=neo4j
NEO4J_PASSWORD=  # REQUIRED: Set a strong password (min 8 chars)

# Qdrant Configuration (Optional)
QDRANT_API_KEY=  # Optional: Set for production

# World Weaver Configuration
WW_SESSION_ID=default
WW_NEO4J_URI=bolt://localhost:7687
WW_NEO4J_USER=neo4j
WW_NEO4J_PASSWORD=  # Must match NEO4J_PASSWORD above
WW_QDRANT_URL=http://localhost:6333
WW_QDRANT_API_KEY=  # Must match QDRANT_API_KEY above
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| Manual | Docker compose fails without password | Error message clear |
| Manual | Docker compose works with .env | Services start correctly |
| CI/CD | No hardcoded secrets in yaml | grep finds no passwords |

**Hooks**: Configuration validation hook (startup check)

**Validation**:
- [ ] `grep -r "password" docker-compose.yml` returns empty
- [ ] `grep -r "wwpassword" .` returns empty
- [ ] Documentation updated with setup instructions

---

### TASK-P10-005: Remove Default Password from Config

**Files**:
- `src/ww/core/config.py` (modify)

**Description**: The config.py has `default="password"` at line 135 for `neo4j_password`. This should require explicit configuration.

**Current Code** (lines 135-138):
```python
neo4j_password: str = Field(
    default="password",
    description="Neo4j password",
)
```

**Solution**: Make password required or validate against weak defaults.

**Implementation**:
```python
# In config.py

# Weak passwords to reject
WEAK_PASSWORDS = frozenset([
    "password", "Password", "PASSWORD",
    "neo4j", "admin", "root", "test",
    "123456", "12345678", "qwerty",
    "password123", "admin123",
])


class Settings(BaseSettings):
    # ... existing fields ...

    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j bolt connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: str = Field(
        default="",  # Empty default requires explicit config
        description="Neo4j password (required)",
    )

    @field_validator("neo4j_password")
    @classmethod
    def validate_neo4j_password(cls, v: str) -> str:
        """Validate Neo4j password is set and not weak."""
        if not v:
            raise ValueError(
                "WW_NEO4J_PASSWORD environment variable is required. "
                "Set a strong password (min 8 characters)."
            )
        if len(v) < 8:
            raise ValueError(
                "Neo4j password must be at least 8 characters."
            )
        if v.lower() in WEAK_PASSWORDS:
            raise ValueError(
                f"Password '{v}' is too weak. Choose a stronger password."
            )
        return v

    @field_validator("qdrant_api_key")
    @classmethod
    def validate_qdrant_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Warn if Qdrant API key not set in production-like environment."""
        if not v:
            import os
            env = os.getenv("WW_ENVIRONMENT", "development")
            if env in ("production", "staging"):
                logger.warning(
                    "QDRANT_API_KEY not set in production environment. "
                    "Consider enabling authentication."
                )
        return v
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_config_requires_password` | Empty password raises error | ValidationError raised |
| `test_config_rejects_weak_passwords` | "password" rejected | ValidationError raised |
| `test_config_accepts_strong_password` | Valid password works | Settings created |
| `test_config_password_min_length` | < 8 chars rejected | ValidationError raised |

**Hooks**: Configuration validation hook (startup check)

**Validation**:
- [ ] Application fails to start without WW_NEO4J_PASSWORD
- [ ] Weak passwords are rejected with clear error message
- [ ] All configuration tests pass

---

## Phase 11: High Priority Fixes (P1)

**Goal**: Complete feature implementations, add missing tests, and harden security.

**Estimated Duration**: 2 weeks

---

### TASK-P11-001: Complete LLM Entity Extraction

**Files**:
- `src/ww/extraction/entity_extractor.py` (modify)
- `tests/extraction/test_entity_extractor.py` (create)

**Description**: The `LLMEntityExtractor._call_llm()` method at line 252 is a placeholder returning empty. Implement actual LLM integration.

**Current Code** (lines 252-268):
```python
async def _call_llm(self, text: str) -> str:
    # This is a placeholder - implement based on actual LLM client
    logger.debug("LLM call would be made here")
    return "[]"
```

**Solution**: Implement with OpenAI-compatible async client.

**Implementation**:
```python
# In entity_extractor.py

class LLMEntityExtractor:
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        max_text_length: int = 2000,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_text_length = max_text_length

        # Initialize client
        if llm_client:
            self.llm_client = llm_client
        elif api_key:
            try:
                from openai import AsyncOpenAI
                self.llm_client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed")
                self.llm_client = None
        else:
            # Try environment variable
            import os
            key = os.getenv("OPENAI_API_KEY")
            if key:
                try:
                    from openai import AsyncOpenAI
                    self.llm_client = AsyncOpenAI(api_key=key)
                except ImportError:
                    self.llm_client = None
            else:
                self.llm_client = None

    async def _call_llm(self, text: str) -> str:
        """
        Call LLM API for entity extraction.

        Args:
            text: Input text to extract from

        Returns:
            JSON string of extracted entities
        """
        if not self.llm_client:
            logger.warning("LLM client not configured, returning empty")
            return "[]"

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(text=text)}
                ],
                temperature=0.0,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            logger.debug(f"LLM extraction response: {content[:100]}...")
            return content or "[]"

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return "[]"
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_llm_extractor_no_client` | No client returns empty | Returns [] |
| `test_llm_extractor_mock_client` | Mock client works | Entities extracted |
| `test_llm_extractor_parse_response` | JSON parsing | Valid entities returned |
| `test_llm_extractor_error_handling` | API errors handled | Graceful fallback |

**Hooks**: Post-create hook (trigger extraction after episode)

**Validation**:
- [ ] LLM extraction works with OpenAI API
- [ ] Fallback to regex when LLM unavailable
- [ ] All tests pass

---

### TASK-P11-002: Add Saga Compensation Tests

**Files**:
- `src/ww/storage/saga.py` (reference)
- `tests/storage/test_saga.py` (create)

**Description**: The saga pattern in `storage/saga.py` lacks comprehensive compensation tests for failure scenarios.

**Implementation**:
```python
# tests/storage/test_saga.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ww.storage.saga import Saga, SagaState, SagaResult


class TestSagaCompensation:
    """Test saga compensation (rollback) behavior."""

    @pytest.mark.asyncio
    async def test_compensation_on_step_failure(self):
        """When a step fails, all previous steps are compensated."""
        compensated = []

        saga = Saga("test_saga")

        # Step 1: succeeds
        saga.add_step(
            name="step1",
            action=AsyncMock(return_value="result1"),
            compensate=AsyncMock(side_effect=lambda: compensated.append("step1")),
        )

        # Step 2: succeeds
        saga.add_step(
            name="step2",
            action=AsyncMock(return_value="result2"),
            compensate=AsyncMock(side_effect=lambda: compensated.append("step2")),
        )

        # Step 3: fails
        saga.add_step(
            name="step3",
            action=AsyncMock(side_effect=RuntimeError("Step 3 failed")),
            compensate=AsyncMock(),  # Never called (step didn't execute)
        )

        result = await saga.execute()

        assert result.state == SagaState.COMPENSATED
        assert "step1" in compensated
        assert "step2" in compensated
        assert len(compensated) == 2

    @pytest.mark.asyncio
    async def test_compensation_order_is_reverse(self):
        """Compensation happens in reverse order (LIFO)."""
        compensation_order = []

        saga = Saga("test_saga")

        for i in range(5):
            step_name = f"step{i}"
            saga.add_step(
                name=step_name,
                action=AsyncMock(return_value=f"result{i}"),
                compensate=AsyncMock(side_effect=lambda n=step_name: compensation_order.append(n)),
            )

        # Add failing step
        saga.add_step(
            name="failing_step",
            action=AsyncMock(side_effect=RuntimeError("Failure")),
            compensate=AsyncMock(),
        )

        await saga.execute()

        # Should be reversed: step4, step3, step2, step1, step0
        assert compensation_order == ["step4", "step3", "step2", "step1", "step0"]

    @pytest.mark.asyncio
    async def test_compensation_failure_logged(self):
        """When compensation fails, it's logged but other compensations continue."""
        compensated = []

        saga = Saga("test_saga")

        saga.add_step(
            name="step1",
            action=AsyncMock(return_value="result1"),
            compensate=AsyncMock(side_effect=lambda: compensated.append("step1")),
        )

        saga.add_step(
            name="step2",
            action=AsyncMock(return_value="result2"),
            compensate=AsyncMock(side_effect=RuntimeError("Compensation failed")),
        )

        saga.add_step(
            name="step3",
            action=AsyncMock(side_effect=RuntimeError("Step failed")),
            compensate=AsyncMock(),
        )

        result = await saga.execute()

        # Step 1 should still be compensated even if step 2 compensation failed
        assert "step1" in compensated
        assert result.state == SagaState.COMPENSATED

    @pytest.mark.asyncio
    async def test_saga_timeout(self):
        """Saga times out if step takes too long."""
        saga = Saga("test_saga", timeout=0.1)

        async def slow_action():
            await asyncio.sleep(1.0)
            return "result"

        saga.add_step(
            name="slow_step",
            action=slow_action,
            compensate=AsyncMock(),
        )

        result = await saga.execute()

        assert result.state in (SagaState.FAILED, SagaState.COMPENSATED)

    @pytest.mark.asyncio
    async def test_concurrent_saga_isolation(self):
        """Multiple sagas run concurrently without interference."""
        results = []

        async def run_saga(name: str, delay: float):
            saga = Saga(name)
            saga.add_step(
                name="step1",
                action=lambda: asyncio.sleep(delay),
                compensate=AsyncMock(),
            )
            result = await saga.execute()
            results.append((name, result.state))

        await asyncio.gather(
            run_saga("saga1", 0.1),
            run_saga("saga2", 0.05),
            run_saga("saga3", 0.15),
        )

        assert len(results) == 3
        assert all(state == SagaState.COMMITTED for _, state in results)
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_compensation_on_step_failure` | Compensation triggered | All completed steps compensated |
| `test_compensation_order_is_reverse` | LIFO order | Last step compensated first |
| `test_compensation_failure_logged` | Graceful degradation | Other compensations continue |
| `test_saga_timeout` | Timeout handling | Saga fails/compensates on timeout |
| `test_concurrent_saga_isolation` | Concurrent execution | No interference |

**Hooks**: None

**Validation**:
- [ ] All compensation tests pass
- [ ] Coverage > 90% for saga.py
- [ ] Integration tests verify database consistency

---

### TASK-P11-003: Add MCP Module Tests

**Files**:
- `tests/mcp/test_errors.py` (create)
- `tests/mcp/test_server.py` (create)
- `tests/mcp/test_resources.py` (create)
- `tests/mcp/test_gateway.py` (expand)

**Description**: MCP modules lack test coverage: `errors.py`, `server.py`, `resources.py`.

**Implementation**:
```python
# tests/mcp/test_errors.py
import pytest
from ww.mcp.errors import make_error, ErrorCode, rate_limited, validation_error


class TestMCPErrors:
    def test_make_error_basic(self):
        error = make_error(ErrorCode.VALIDATION, "Invalid input")
        assert error["error"]["code"] == "validation_error"
        assert error["error"]["message"] == "Invalid input"

    def test_make_error_with_field(self):
        error = make_error(ErrorCode.VALIDATION, "Required", field="name")
        assert error["error"]["field"] == "name"

    def test_rate_limited_response(self):
        error = rate_limited(30.0)
        assert error["error"]["code"] == "rate_limited"
        assert error["error"]["retry_after"] == 30.0

    def test_validation_error_helper(self):
        error = validation_error("name", "too long")
        assert error["error"]["field"] == "name"
        assert "too long" in error["error"]["message"]


# tests/mcp/test_gateway.py
import pytest
from unittest.mock import AsyncMock, patch
from ww.mcp.gateway import (
    RateLimiter,
    rate_limited,
    with_request_id,
    get_services,
    cleanup_services,
)


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        for _ in range(10):
            assert limiter.allow("session1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.allow("session1")
        assert limiter.allow("session1") is False

    def test_sessions_isolated(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.allow("session1")
        limiter.allow("session1")
        assert limiter.allow("session1") is False
        assert limiter.allow("session2") is True  # Different session

    def test_time_until_allowed(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.allow("session1")
        retry_after = limiter.time_until_allowed("session1")
        assert 0 < retry_after <= 60

    def test_reset_session(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.allow("session1")
        limiter.reset("session1")
        assert limiter.allow("session1") is True


class TestRateLimitedDecorator:
    @pytest.mark.asyncio
    async def test_rate_limited_allows_request(self):
        @rate_limited
        async def my_func(session_id="test"):
            return "success"

        result = await my_func(session_id="new_session")
        assert result == "success"


class TestServiceManagement:
    @pytest.mark.asyncio
    async def test_get_services_initializes(self):
        with patch("ww.mcp.gateway.get_episodic_memory") as mock_ep, \
             patch("ww.mcp.gateway.get_semantic_memory") as mock_sem, \
             patch("ww.mcp.gateway.get_procedural_memory") as mock_proc:

            mock_ep.return_value.initialize = AsyncMock()
            mock_sem.return_value.initialize = AsyncMock()
            mock_proc.return_value.initialize = AsyncMock()

            ep, sem, proc = await get_services("test_session")

            mock_ep.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_services(self):
        with patch("ww.mcp.gateway.close_qdrant_store") as mock_qdrant, \
             patch("ww.mcp.gateway.close_neo4j_store") as mock_neo4j:

            mock_qdrant.return_value = AsyncMock()
            mock_neo4j.return_value = AsyncMock()

            await cleanup_services("test_session")

            mock_qdrant.assert_called_once()
            mock_neo4j.assert_called_once()
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_make_error_*` | Error response format | Correct structure |
| `test_rate_limiter_*` | Rate limiting logic | Blocks/allows correctly |
| `test_service_management` | Service lifecycle | Init/cleanup works |
| `test_gateway_security` | Security decorators | Auth/role checks work |

**Hooks**: None

**Validation**:
- [ ] MCP module coverage > 80%
- [ ] All tests pass
- [ ] Error responses follow standard format

---

### TASK-P11-004: Batch Access Updates on Recall

**Files**:
- `src/ww/memory/episodic.py` (modify)

**Description**: The `recall()` method at line 249 updates access counts in a loop (N+1 pattern).

**Current Code** (lines 249-251):
```python
for result in results:
    await self._update_access(result.item.id)
```

**Solution**: Batch the access updates.

**Implementation**:
```python
# In episodic.py

async def _batch_update_access(
    self,
    episode_ids: list[UUID],
    success: bool = True,
) -> None:
    """
    Batch update access statistics for multiple episodes.

    Args:
        episode_ids: List of episode UUIDs
        success: Whether recall was successful
    """
    if not episode_ids:
        return

    current_time = datetime.now()

    # Batch fetch current episode data
    results = await self.vector_store.get(
        collection=self.vector_store.episodes_collection,
        ids=[str(eid) for eid in episode_ids],
    )

    # Calculate updates
    updates = []
    for id_str, payload in results:
        episode = self._from_payload(id_str, payload)
        elapsed_days = (current_time - episode.last_accessed).total_seconds() / 86400
        R = episode.retrievability(current_time)

        if success:
            new_stability = episode.stability * (1 + 0.1 * (1 - R))
        else:
            new_stability = episode.stability * 0.8

        updates.append((
            id_str,
            {
                "stability": new_stability,
                "last_accessed": current_time.isoformat(),
                "access_count": episode.access_count + 1,
            }
        ))

    # Batch update
    if updates:
        await self.vector_store.batch_update_payloads(
            collection=self.vector_store.episodes_collection,
            updates=updates,
        )
        logger.debug(f"Batch updated access for {len(updates)} episodes")

# Update recall method
async def recall(self, ...) -> list[ScoredResult]:
    # ... existing code ...

    # Batch update access counts (replaces loop)
    episode_ids = [result.item.id for result in results]
    await self._batch_update_access(episode_ids)

    return results
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_batch_access_update` | Batch updates work | All episodes updated |
| `test_recall_updates_access` | Recall triggers batch | Single batch call |
| `test_empty_batch` | Empty input handled | No errors |

**Hooks**: None

**Validation**:
- [ ] Query count reduced from N to 2 (fetch + update)
- [ ] All tests pass
- [ ] No regression in recall behavior

---

### TASK-P11-005: Add Spreading Activation Limits

**Files**:
- `src/ww/memory/semantic.py` (modify)

**Description**: The `spread_activation()` method at line 597 can explode on highly connected graphs without limits.

**Current Code** (lines 597-648):
```python
async def spread_activation(
    self,
    seed_entities: list[str],
    steps: int = 3,
    # No limit on activation size
```

**Solution**: Add visited set and max activation limits.

**Implementation**:
```python
async def spread_activation(
    self,
    seed_entities: list[str],
    steps: int = 3,
    retention: float = 0.5,
    decay: float = 0.1,
    threshold: float = 0.01,
    max_nodes: int = 1000,
    max_neighbors_per_node: int = 50,
) -> dict[str, float]:
    """
    Spread activation through knowledge graph with explosion prevention.

    Args:
        seed_entities: Entity IDs to start from
        steps: Number of propagation steps
        retention: Activation retained at each node
        decay: Activation decay per step
        threshold: Minimum activation to continue
        max_nodes: Maximum nodes in activation map (prevents explosion)
        max_neighbors_per_node: Maximum neighbors to consider per node

    Returns:
        Dict of entity_id -> activation level
    """
    activation = {eid: 1.0 for eid in seed_entities}
    visited = set(seed_entities)  # Track visited to prevent cycles

    for step in range(steps):
        new_activation = {}
        nodes_to_process = sorted(
            activation.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_nodes]  # Only process highest activation nodes

        for entity_id, act in nodes_to_process:
            if act < threshold:
                continue

            # Retain some activation
            retained = act * retention
            new_activation[entity_id] = new_activation.get(entity_id, 0) + retained

            # Get weighted neighbors (limited)
            neighbors = await self.graph_store.get_relationships(
                node_id=entity_id,
                direction="both",
                limit=max_neighbors_per_node,  # Limit neighbors
            )

            if neighbors:
                spread_amount = act * (1 - retention)
                # Filter to unvisited or high-activation neighbors
                valid_neighbors = [
                    n for n in neighbors
                    if n["other_id"] not in visited or act > 0.5
                ][:max_neighbors_per_node]

                for neighbor in valid_neighbors:
                    weight = neighbor["properties"].get("weight", 0.1)
                    spread = spread_amount * weight / len(valid_neighbors)
                    neighbor_id = neighbor["other_id"]
                    new_activation[neighbor_id] = new_activation.get(neighbor_id, 0) + spread
                    visited.add(neighbor_id)

        # Apply decay and enforce max_nodes limit
        activation = dict(sorted(
            ((k, v * (1 - decay)) for k, v in new_activation.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:max_nodes])

        logger.debug(f"Spreading step {step + 1}: {len(activation)} nodes")

    logger.info(f"Spreading activation complete: {len(activation)} nodes from {len(seed_entities)} seeds")
    return activation
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_spread_activation_limits` | Max nodes enforced | <= max_nodes returned |
| `test_spread_activation_cycle` | Cycles handled | No infinite loop |
| `test_spread_activation_performance` | Large graph performance | < 5s for 10k nodes |
| `test_spread_activation_basic` | Basic spreading works | Activation decreases with distance |

**Hooks**: None

**Validation**:
- [ ] Memory usage bounded
- [ ] Performance acceptable for large graphs
- [ ] All tests pass

---

### TASK-P11-006: Validate Session ID at Gateway Level

**Files**:
- `src/ww/mcp/gateway.py` (modify)
- `src/ww/mcp/validation.py` (add session validation)

**Description**: Session ID validation happens deep in service calls, not at the gateway. This allows potential bypass.

**Solution**: Add session validation decorator at gateway entry points.

**Implementation**:
```python
# In validation.py - add session validation

import re
from typing import Optional

# Valid session ID pattern: alphanumeric, dash, underscore, 1-64 chars
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def validate_session_id(session_id: Optional[str]) -> str:
    """
    Validate and normalize session ID.

    Args:
        session_id: Session ID to validate

    Returns:
        Validated session ID

    Raises:
        ValueError: If session ID is invalid
    """
    if session_id is None:
        return "default"

    session_id = str(session_id).strip()

    if not session_id:
        return "default"

    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(
            f"Invalid session_id '{session_id}'. "
            "Must be 1-64 alphanumeric characters, dashes, or underscores."
        )

    return session_id


# In gateway.py - add validation decorator

from ww.mcp.validation import validate_session_id
from ww.mcp.errors import make_error, ErrorCode


def validated_session(func):
    """
    Decorator to validate session_id at gateway entry point.

    Validates session_id parameter before passing to function.
    Returns error response for invalid sessions.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        session_id = kwargs.get('session_id')

        try:
            validated = validate_session_id(session_id)
            kwargs['session_id'] = validated
        except ValueError as e:
            return make_error(
                ErrorCode.VALIDATION,
                str(e),
                field="session_id",
            )

        return await func(*args, **kwargs)

    return wrapper


# Apply to all MCP tools
@mcp_app.tool()
@rate_limited
@with_request_id
@validated_session  # New: validate session at gateway
async def create_episode(
    content: str,
    session_id: str = None,
    ...
) -> dict:
    ...
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_session_validation_valid` | Valid IDs pass | No error |
| `test_session_validation_invalid` | Invalid IDs rejected | Error response |
| `test_session_validation_special_chars` | Special chars rejected | Error response |
| `test_session_validation_length` | > 64 chars rejected | Error response |
| `test_gateway_validates_session` | Gateway uses validator | Validation at entry |

**Hooks**: None

**Validation**:
- [ ] All MCP tools validate session at entry
- [ ] Invalid sessions return proper error response
- [ ] All tests pass

---

### TASK-P11-007: Add Database Authentication

**Files**:
- `docker-compose.yml` (modify)
- `src/ww/storage/qdrant_store.py` (modify)
- `scripts/setup_auth.sh` (create)

**Description**: Qdrant and Neo4j are exposed without authentication in docker-compose.

**Solution**: Enable authentication with secure configuration.

**Implementation**:
```yaml
# docker-compose.yml - add Qdrant authentication
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: ww-qdrant
    ports:
      - "127.0.0.1:6333:6333"  # Bind to localhost only
      - "127.0.0.1:6334:6334"
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY:-}
      - QDRANT__SERVICE__ENABLE_TLS=${QDRANT_ENABLE_TLS:-false}
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    volumes:
      - ww_qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "-H", "api-key: ${QDRANT_API_KEY:-}", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  neo4j:
    image: neo4j:5-community
    container_name: ww-neo4j
    ports:
      - "127.0.0.1:7474:7474"  # Bind to localhost only
      - "127.0.0.1:7687:7687"
    environment:
      - NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:?NEO4J_PASSWORD required}
      # ... rest of config
```

```bash
#!/bin/bash
# scripts/setup_auth.sh

set -e

echo "World Weaver Authentication Setup"
echo "=================================="

# Generate secure API key for Qdrant
generate_api_key() {
    openssl rand -base64 32 | tr -d '/+=' | head -c 32
}

# Check for existing .env
if [ -f .env ]; then
    echo "Warning: .env already exists. Backup will be created."
    cp .env .env.backup.$(date +%Y%m%d%H%M%S)
fi

# Create .env from example
if [ -f .env.example ]; then
    cp .env.example .env
else
    touch .env
fi

# Prompt for Neo4j password
echo ""
read -s -p "Enter Neo4j password (min 8 chars): " neo4j_pass
echo ""
if [ ${#neo4j_pass} -lt 8 ]; then
    echo "Error: Password must be at least 8 characters"
    exit 1
fi

# Generate Qdrant API key
qdrant_key=$(generate_api_key)

# Update .env
cat >> .env << EOF

# Generated by setup_auth.sh on $(date)
NEO4J_PASSWORD=$neo4j_pass
WW_NEO4J_PASSWORD=$neo4j_pass
QDRANT_API_KEY=$qdrant_key
WW_QDRANT_API_KEY=$qdrant_key
EOF

echo ""
echo "Authentication configured successfully!"
echo "- Neo4j password set"
echo "- Qdrant API key generated: ${qdrant_key:0:8}..."
echo ""
echo "To start services: docker-compose up -d"
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| Manual | Services reject unauthenticated requests | 401/403 response |
| Manual | Services accept authenticated requests | 200 response |
| CI/CD | setup_auth.sh works | Creates valid .env |

**Hooks**: Configuration validation hook (verify auth on startup)

**Validation**:
- [ ] Services bound to localhost only
- [ ] Authentication required for access
- [ ] Setup script works correctly

---

## Phase 12: Medium Priority Fixes (P2)

**Goal**: Improve maintainability, testability, and polish performance.

**Estimated Duration**: 1.5 weeks

---

### TASK-P12-001: Extract Common Serialization

**Files**:
- `src/ww/core/serialization.py` (create)
- `src/ww/memory/episodic.py` (modify)
- `src/ww/memory/semantic.py` (modify)
- `src/ww/memory/procedural.py` (modify)

**Description**: `_to_payload()` and `_from_payload()` methods are duplicated across memory modules (~150 lines).

**Solution**: Extract to shared serialization module with type-specific handlers.

**Implementation**:
```python
# src/ww/core/serialization.py
"""
Shared serialization utilities for World Weaver.

Provides consistent conversion between domain objects and storage formats.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TypeVar, Generic, Any, Optional
from uuid import UUID

T = TypeVar('T')


class Serializer(ABC, Generic[T]):
    """Abstract base for type-specific serializers."""

    @abstractmethod
    def to_payload(self, obj: T, session_id: str) -> dict[str, Any]:
        """Convert domain object to storage payload."""
        ...

    @abstractmethod
    def from_payload(self, id_str: str, payload: dict[str, Any]) -> T:
        """Reconstruct domain object from storage payload."""
        ...

    @abstractmethod
    def to_graph_props(self, obj: T, session_id: str) -> dict[str, Any]:
        """Convert domain object to graph node properties."""
        ...


class DateTimeSerializer:
    """Shared datetime serialization utilities."""

    @staticmethod
    def to_iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    @staticmethod
    def from_iso(iso_str: Optional[str]) -> Optional[datetime]:
        return datetime.fromisoformat(iso_str) if iso_str else None


class EpisodeSerializer(Serializer):
    """Serializer for Episode objects."""

    def to_payload(self, episode: 'Episode', session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "content": episode.content,
            "timestamp": DateTimeSerializer.to_iso(episode.timestamp),
            "ingested_at": DateTimeSerializer.to_iso(episode.ingested_at),
            "context": episode.context.model_dump(),
            "outcome": episode.outcome.value,
            "emotional_valence": episode.emotional_valence,
            "access_count": episode.access_count,
            "last_accessed": DateTimeSerializer.to_iso(episode.last_accessed),
            "stability": episode.stability,
        }

    def from_payload(self, id_str: str, payload: dict[str, Any]) -> 'Episode':
        from ww.core.types import Episode, EpisodeContext, Outcome
        return Episode(
            id=UUID(id_str),
            session_id=payload["session_id"],
            content=payload["content"],
            embedding=None,
            timestamp=DateTimeSerializer.from_iso(payload["timestamp"]),
            ingested_at=DateTimeSerializer.from_iso(payload["ingested_at"]),
            context=EpisodeContext(**payload["context"]),
            outcome=Outcome(payload["outcome"]),
            emotional_valence=payload["emotional_valence"],
            access_count=payload["access_count"],
            last_accessed=DateTimeSerializer.from_iso(payload["last_accessed"]),
            stability=payload["stability"],
        )

    def to_graph_props(self, episode: 'Episode', session_id: str) -> dict[str, Any]:
        return {
            "id": str(episode.id),
            "sessionId": session_id,
            "content": episode.content[:500],
            "timestamp": DateTimeSerializer.to_iso(episode.timestamp),
            "ingestedAt": DateTimeSerializer.to_iso(episode.ingested_at),
            "outcome": episode.outcome.value,
            "emotionalValence": episode.emotional_valence,
            "accessCount": episode.access_count,
            "lastAccessed": DateTimeSerializer.to_iso(episode.last_accessed),
            "stability": episode.stability,
        }


# Factory function
def get_serializer(type_name: str) -> Serializer:
    """Get serializer for given type."""
    serializers = {
        "episode": EpisodeSerializer(),
        "entity": EntitySerializer(),
        "procedure": ProcedureSerializer(),
    }
    return serializers[type_name]
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_episode_roundtrip` | Serialize/deserialize | Object unchanged |
| `test_entity_roundtrip` | Serialize/deserialize | Object unchanged |
| `test_procedure_roundtrip` | Serialize/deserialize | Object unchanged |
| `test_datetime_edge_cases` | None, UTC, timezone | All handled |

**Hooks**: None

**Validation**:
- [ ] All memory modules use shared serialization
- [ ] No duplication in memory/*.py
- [ ] All tests pass

---

### TASK-P12-002: Add Dependency Injection Container

**Files**:
- `src/ww/core/container.py` (create)
- `src/ww/memory/*.py` (modify)
- `src/ww/storage/*.py` (modify)

**Description**: Singleton patterns limit testability. Add DI container for better testing.

**Implementation**:
```python
# src/ww/core/container.py
"""
Dependency Injection Container for World Weaver.

Provides centralized service registration and resolution,
enabling better testability and configuration.
"""

from typing import TypeVar, Type, Optional, Callable, Any
import threading

T = TypeVar('T')


class Container:
    """
    Simple dependency injection container.

    Supports:
    - Singleton registration
    - Factory registration
    - Instance registration
    - Scoped services (per session)
    """

    _instance: Optional['Container'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Callable] = {}
        self._instances: dict[str, Any] = {}
        self._scoped: dict[str, dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> 'Container':
        """Get singleton container instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Container()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset container (for testing)."""
        with cls._lock:
            cls._instance = None

    def register_singleton(self, key: str, factory: Callable[[], T]) -> None:
        """Register a singleton factory."""
        self._factories[key] = factory

    def register_instance(self, key: str, instance: T) -> None:
        """Register a pre-created instance."""
        self._instances[key] = instance

    def register_scoped(self, key: str, factory: Callable[[str], T]) -> None:
        """Register a scoped factory (creates per session)."""
        self._factories[f"scoped:{key}"] = factory

    def resolve(self, key: str, scope: Optional[str] = None) -> T:
        """
        Resolve a dependency.

        Args:
            key: Service key
            scope: Optional scope (e.g., session_id)

        Returns:
            Resolved service instance
        """
        # Check instances first
        if key in self._instances:
            return self._instances[key]

        # Check singletons
        if key in self._singletons:
            return self._singletons[key]

        # Check scoped
        if scope and f"scoped:{key}" in self._factories:
            if scope not in self._scoped:
                self._scoped[scope] = {}
            if key not in self._scoped[scope]:
                factory = self._factories[f"scoped:{key}"]
                self._scoped[scope][key] = factory(scope)
            return self._scoped[scope][key]

        # Create singleton
        if key in self._factories:
            self._singletons[key] = self._factories[key]()
            return self._singletons[key]

        raise KeyError(f"No service registered for key: {key}")

    def clear_scope(self, scope: str) -> None:
        """Clear all services for a scope."""
        if scope in self._scoped:
            del self._scoped[scope]


# Convenience functions
def get_container() -> Container:
    """Get global container instance."""
    return Container.get_instance()


def configure_production() -> None:
    """Configure container for production."""
    c = get_container()

    from ww.storage.qdrant_store import QdrantStore
    from ww.storage.neo4j_store import Neo4jStore
    from ww.embedding.bge_m3 import BGE_M3Provider

    c.register_singleton("embedding", BGE_M3Provider)
    c.register_scoped("qdrant", lambda session: QdrantStore())
    c.register_scoped("neo4j", lambda session: Neo4jStore())


def configure_testing() -> None:
    """Configure container for testing with mocks."""
    c = get_container()
    Container.reset()
    # Tests register their own mocks
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_container_singleton` | Singleton returns same instance | Same object |
| `test_container_scoped` | Scoped returns per-scope instance | Different per scope |
| `test_container_reset` | Reset clears all | Container empty |
| `test_container_testing` | Testing mode works | Mocks injected |

**Hooks**: None

**Validation**:
- [ ] Services use DI container
- [ ] Tests can inject mocks easily
- [ ] All tests pass

---

### TASK-P12-003: Add Embedding Provider Tests

**Files**:
- `tests/embedding/test_bge_m3.py` (create)

**Description**: Embedding provider lacks test coverage.

**Implementation**:
```python
# tests/embedding/test_bge_m3.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ww.embedding.bge_m3 import BGE_M3Provider, get_embedding_provider


class TestBGE_M3Provider:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.encode.return_value = np.random.rand(1, 1024).astype(np.float32)
        return model

    def test_embed_query_returns_list(self, mock_model):
        with patch("ww.embedding.bge_m3.SentenceTransformer", return_value=mock_model):
            provider = BGE_M3Provider()
            result = provider.embed_query("test query")

            assert isinstance(result, list)
            assert len(result) == 1024

    def test_embed_batch_returns_lists(self, mock_model):
        mock_model.encode.return_value = np.random.rand(3, 1024).astype(np.float32)

        with patch("ww.embedding.bge_m3.SentenceTransformer", return_value=mock_model):
            provider = BGE_M3Provider()
            result = provider.embed(["text1", "text2", "text3"])

            assert len(result) == 3
            assert all(len(v) == 1024 for v in result)

    def test_embed_empty_input(self, mock_model):
        mock_model.encode.return_value = np.array([]).reshape(0, 1024)

        with patch("ww.embedding.bge_m3.SentenceTransformer", return_value=mock_model):
            provider = BGE_M3Provider()
            result = provider.embed([])

            assert result == []

    def test_embedding_is_normalized(self, mock_model):
        # Return non-normalized embedding
        mock_model.encode.return_value = np.array([[2.0] * 1024], dtype=np.float32)

        with patch("ww.embedding.bge_m3.SentenceTransformer", return_value=mock_model):
            provider = BGE_M3Provider(normalize=True)
            result = provider.embed_query("test")

            # Check normalized (L2 norm ~= 1)
            norm = np.linalg.norm(result)
            assert 0.99 < norm < 1.01

    def test_caching_same_text(self, mock_model):
        with patch("ww.embedding.bge_m3.SentenceTransformer", return_value=mock_model):
            provider = BGE_M3Provider()

            result1 = provider.embed_query("same text")
            result2 = provider.embed_query("same text")

            # Should hit cache - encode called once
            assert mock_model.encode.call_count == 1

    def test_singleton_instance(self):
        with patch("ww.embedding.bge_m3.SentenceTransformer"):
            provider1 = get_embedding_provider()
            provider2 = get_embedding_provider()

            assert provider1 is provider2
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_embed_query_returns_list` | Single embed | Correct dimension |
| `test_embed_batch` | Batch embed | All correct dimension |
| `test_embedding_normalized` | Normalization | L2 norm ~= 1 |
| `test_caching` | Cache works | Single model call |

**Hooks**: None

**Validation**:
- [ ] Embedding provider coverage > 80%
- [ ] All tests pass
- [ ] Integration tests with real model (optional)

---

### TASK-P12-004: Add Chaos/Fault Injection Tests

**Files**:
- `tests/chaos/test_database_failures.py` (create)
- `tests/chaos/test_network_failures.py` (create)
- `tests/chaos/conftest.py` (create)

**Description**: No tests verify system behavior under failure conditions.

**Implementation**:
```python
# tests/chaos/conftest.py
import pytest
import asyncio
from unittest.mock import AsyncMock


class ChaosMonkey:
    """Utility for injecting failures in tests."""

    def __init__(self):
        self.failure_rate = 0.0
        self.failure_type = Exception
        self.delay_ms = 0

    def set_failure_rate(self, rate: float, exception: type = Exception):
        """Set random failure rate (0.0 to 1.0)."""
        self.failure_rate = rate
        self.failure_type = exception

    def set_delay(self, ms: int):
        """Set artificial delay in milliseconds."""
        self.delay_ms = ms

    async def maybe_fail(self):
        """Maybe raise an exception based on failure rate."""
        import random
        if random.random() < self.failure_rate:
            raise self.failure_type("Chaos monkey strikes!")
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)


@pytest.fixture
def chaos_monkey():
    return ChaosMonkey()


# tests/chaos/test_database_failures.py
import pytest
from unittest.mock import AsyncMock, patch

from ww.memory.episodic import EpisodicMemory
from ww.storage.saga import Saga, SagaState


class TestDatabaseFailures:
    @pytest.mark.asyncio
    async def test_vector_store_failure_triggers_compensation(self):
        """Vector store failure should trigger saga compensation."""
        episodic = EpisodicMemory("test")

        with patch.object(episodic.vector_store, "add") as mock_add:
            mock_add.side_effect = ConnectionError("Database unavailable")

            with pytest.raises(RuntimeError, match="Episode creation failed"):
                await episodic.create(content="test content")

    @pytest.mark.asyncio
    async def test_graph_store_failure_rolls_back_vector(self):
        """Graph failure should delete vector entry."""
        episodic = EpisodicMemory("test")
        deleted_ids = []

        with patch.object(episodic.vector_store, "add", AsyncMock()), \
             patch.object(episodic.vector_store, "delete") as mock_delete, \
             patch.object(episodic.graph_store, "create_node") as mock_create:

            mock_delete.side_effect = lambda **kw: deleted_ids.extend(kw.get("ids", []))
            mock_create.side_effect = ConnectionError("Graph unavailable")

            with pytest.raises(RuntimeError):
                await episodic.create(content="test")

            # Vector should have been cleaned up
            assert len(deleted_ids) == 1

    @pytest.mark.asyncio
    async def test_partial_batch_failure_recovery(self, chaos_monkey):
        """Batch operations should handle partial failures."""
        from ww.storage.qdrant_store import QdrantStore

        store = QdrantStore()
        chaos_monkey.set_failure_rate(0.3)

        # Mock with chaos
        original_add = store._add_batch
        async def chaotic_add(*args, **kwargs):
            await chaos_monkey.maybe_fail()
            return await original_add(*args, **kwargs)

        with patch.object(store, "_add_batch", chaotic_add):
            # Should either succeed or fail cleanly (no partial state)
            try:
                await store.add(
                    collection="test",
                    ids=["1", "2", "3"],
                    vectors=[[0.1] * 1024] * 3,
                    payloads=[{}] * 3,
                )
            except Exception:
                # Verify rollback attempted
                pass


# tests/chaos/test_network_failures.py
class TestNetworkFailures:
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Operations should timeout and not hang."""
        from ww.storage.qdrant_store import QdrantStore, DatabaseTimeoutError

        store = QdrantStore(timeout=0.1)

        async def slow_operation():
            await asyncio.sleep(10)

        with patch.object(store, "_get_client") as mock_client:
            mock_client.return_value.query_points = slow_operation

            with pytest.raises(DatabaseTimeoutError):
                await store.search(
                    collection="test",
                    vector=[0.1] * 1024,
                )

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Transient failures should be retried."""
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        # Test with retry wrapper
        from ww.core.retry import with_retry

        result = await with_retry(flaky_operation, max_attempts=5)
        assert result == "success"
        assert call_count == 3
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_vector_store_failure` | Vector failure compensation | Saga compensates |
| `test_graph_failure_rollback` | Graph failure rolls back | Vector cleaned |
| `test_partial_batch` | Partial failure recovery | Clean state |
| `test_timeout` | Timeout handling | Exception raised |

**Hooks**: None

**Validation**:
- [ ] System recovers from failures
- [ ] No data corruption on partial failure
- [ ] All chaos tests pass

---

### TASK-P12-005: Add HDBSCAN Memory Limit

**Files**:
- `src/ww/consolidation/service.py` (modify)

**Description**: HDBSCAN can exhaust memory with large datasets.

**Solution**: Add sampling for large datasets.

**Implementation**:
```python
async def _cluster_episodes(
    self,
    episodes: list[Episode],
    threshold: float = 0.75,
    min_cluster_size: int = 3,
    max_samples: int = 5000,  # New: memory limit
) -> list[list[Episode]]:
    """
    Cluster episodes with memory-safe sampling.

    For datasets > max_samples, uses stratified sampling to maintain
    cluster representativeness while bounding memory usage.
    """
    if not episodes:
        return []

    if len(episodes) < min_cluster_size:
        return []

    # Sample if too large
    sampled = False
    original_episodes = episodes
    if len(episodes) > max_samples:
        logger.info(
            f"Sampling {max_samples} from {len(episodes)} episodes for clustering"
        )
        # Stratified sample by time (preserve temporal distribution)
        episodes = self._stratified_sample(episodes, max_samples)
        sampled = True

    try:
        # ... existing HDBSCAN clustering code ...

        # If sampled, assign non-sampled episodes to nearest cluster
        if sampled:
            clusters = await self._assign_to_clusters(
                original_episodes,
                clusters,
                episodes,  # sampled episodes
            )

        return clusters
    except MemoryError:
        logger.error(f"HDBSCAN memory error with {len(episodes)} episodes")
        # Fallback: return episodes as single cluster
        return [episodes] if len(episodes) >= min_cluster_size else []

def _stratified_sample(
    self,
    episodes: list[Episode],
    n_samples: int,
) -> list[Episode]:
    """Stratified sampling preserving temporal distribution."""
    import random

    # Sort by timestamp
    sorted_eps = sorted(episodes, key=lambda e: e.timestamp)

    # Sample evenly across time
    step = len(sorted_eps) / n_samples
    indices = [int(i * step) for i in range(n_samples)]

    return [sorted_eps[i] for i in indices]
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_clustering_large_dataset` | > 5000 episodes | Completes without OOM |
| `test_stratified_sampling` | Sample distribution | Even temporal spread |
| `test_assign_to_clusters` | Non-sampled assignment | All episodes clustered |

**Hooks**: None

**Validation**:
- [ ] Memory usage bounded
- [ ] Clustering quality maintained with sampling
- [ ] All tests pass

---

### TASK-P12-006: Add Embedding Cache TTL

**Files**:
- `src/ww/embedding/bge_m3.py` (modify)

**Description**: Embedding cache has no TTL, leading to potential memory leaks.

**Solution**: Add TTL-based cache eviction.

**Implementation**:
```python
from functools import lru_cache
from datetime import datetime, timedelta
import threading


class TTLCache:
    """Thread-safe cache with TTL eviction."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now() - timestamp < self.ttl:
                    return value
                else:
                    del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value with current timestamp."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1],
                )
                del self._cache[oldest_key]

            self._cache[key] = (value, datetime.now())

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

    def evict_expired(self) -> int:
        """Evict all expired entries. Returns count evicted."""
        with self._lock:
            now = datetime.now()
            expired = [
                k for k, (_, ts) in self._cache.items()
                if now - ts >= self.ttl
            ]
            for k in expired:
                del self._cache[k]
            return len(expired)


class BGE_M3Provider:
    def __init__(self, ...):
        # ... existing init ...
        self._cache = TTLCache(
            max_size=get_settings().embedding_cache_size,
            ttl_seconds=get_settings().embedding_cache_ttl,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed with caching."""
        cache_key = hash(text)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = await self._embed_impl(text)
        self._cache.set(cache_key, embedding)
        return embedding
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_cache_ttl_expiry` | Entries expire | Old entries removed |
| `test_cache_max_size` | Size limit | Evicts when full |
| `test_cache_thread_safety` | Concurrent access | No race conditions |

**Hooks**: Health check hook (report cache stats)

**Validation**:
- [ ] Memory usage stable over time
- [ ] Cache hit rate logged
- [ ] All tests pass

---

### TASK-P12-007: Parallelize Batch Updates

**Files**:
- `src/ww/storage/qdrant_store.py` (modify)

**Description**: `batch_update_payloads()` at line 507 uses sequential loop.

**Current Code** (lines 503-518):
```python
async def _batch_update():
    count = 0
    for id_str, payload in updates:
        await client.set_payload(...)
        count += 1
    return count
```

**Solution**: Parallelize with controlled concurrency.

**Implementation**:
```python
async def batch_update_payloads(
    self,
    collection: str,
    updates: list[tuple[str, dict[str, Any]]],
    max_concurrency: int = 10,
) -> int:
    """
    Update multiple payloads with parallel execution.

    Args:
        collection: Collection name
        updates: List of (id, payload) tuples
        max_concurrency: Maximum parallel updates

    Returns:
        Count of updates applied
    """
    if not updates:
        return 0

    async def _batch_update():
        client = await self._get_client()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def update_one(id_str: str, payload: dict) -> bool:
            async with semaphore:
                try:
                    await client.set_payload(
                        collection_name=collection,
                        payload=payload,
                        points=[id_str],
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Failed to update {id_str}: {e}")
                    return False

        # Execute all updates in parallel
        results = await asyncio.gather(*[
            update_one(id_str, payload)
            for id_str, payload in updates
        ])

        success_count = sum(1 for r in results if r)
        logger.debug(
            f"Batch updated {success_count}/{len(updates)} payloads in '{collection}'"
        )
        return success_count

    return await self._with_timeout(_batch_update(), f"batch_update_payloads({collection})")
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_parallel_update` | Parallel execution | Faster than sequential |
| `test_concurrency_limit` | Semaphore works | Max concurrent respected |
| `test_partial_failure` | Some updates fail | Others succeed |

**Hooks**: None

**Validation**:
- [ ] Performance improvement measurable
- [ ] Concurrency limit prevents overload
- [ ] All tests pass

---

### TASK-P12-008: Secure OTLP Exporter

**Files**:
- `src/ww/observability/tracing.py` (modify)
- `src/ww/core/config.py` (add settings)

**Description**: OTLP exporter uses insecure channel by default.

**Solution**: Add TLS configuration option.

**Implementation**:
```python
# In config.py - add settings
otel_insecure: bool = Field(
    default=True,
    description="Use insecure gRPC for OTLP (set False for production)",
)
otel_cert_file: Optional[str] = Field(
    default=None,
    description="Path to TLS certificate file for OTLP",
)

# In tracing.py - use secure channel
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing() -> None:
    """Configure OpenTelemetry tracing."""
    settings = get_settings()

    if not settings.otel_enabled:
        return

    # Configure exporter with security
    exporter_kwargs = {
        "endpoint": settings.otel_endpoint,
    }

    if settings.otel_insecure:
        exporter_kwargs["insecure"] = True
        if settings.otel_environment == "production":
            logger.warning(
                "OTLP exporter using insecure channel in production. "
                "Set WW_OTEL_INSECURE=false and provide WW_OTEL_CERT_FILE."
            )
    else:
        if settings.otel_cert_file:
            with open(settings.otel_cert_file, "rb") as f:
                exporter_kwargs["credentials"] = grpc.ssl_channel_credentials(f.read())
        else:
            exporter_kwargs["credentials"] = grpc.ssl_channel_credentials()

    exporter = OTLPSpanExporter(**exporter_kwargs)

    # ... rest of tracing configuration ...
```

**Testing Requirements**:
| Test | Description | Validation |
|------|-------------|------------|
| `test_insecure_warning` | Warning in production | Log warning emitted |
| `test_secure_config` | TLS configuration | Credentials used |
| `test_cert_file` | Custom cert | File loaded |

**Hooks**: Configuration validation hook (warn on insecure in prod)

**Validation**:
- [ ] TLS works with secure endpoint
- [ ] Warning emitted for insecure in production
- [ ] All tests pass

---

## Module Lifecycle Hooks

### Hook Specification

```python
# src/ww/core/hooks.py
"""
Lifecycle hooks for World Weaver modules.

Provides extensible hook system for:
- Pre/post operation hooks
- Validation hooks
- Health check hooks
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Any
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class HookPhase(Enum):
    """Hook execution phases."""
    PRE_CONSOLIDATION = "pre_consolidation"
    POST_CREATE = "post_create"
    PRE_DELETE = "pre_delete"
    HEALTH_CHECK = "health_check"
    CONFIG_VALIDATION = "config_validation"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class Hook(ABC):
    """Abstract base for hooks."""

    @property
    @abstractmethod
    def phase(self) -> HookPhase:
        """Hook execution phase."""
        ...

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute hook.

        Args:
            context: Hook context data

        Returns:
            Modified context or results
        """
        ...


class HookRegistry:
    """Registry for lifecycle hooks."""

    def __init__(self):
        self._hooks: dict[HookPhase, list[Hook]] = {
            phase: [] for phase in HookPhase
        }

    def register(self, hook: Hook) -> None:
        """Register a hook."""
        self._hooks[hook.phase].append(hook)
        logger.debug(f"Registered hook: {hook.__class__.__name__} for {hook.phase}")

    async def execute(
        self,
        phase: HookPhase,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute all hooks for a phase.

        Args:
            phase: Hook phase
            context: Initial context

        Returns:
            List of hook results
        """
        results = []
        for hook in self._hooks[phase]:
            try:
                result = await hook.execute(context)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__} failed: {e}")
                results.append({"error": str(e)})
        return results


# Singleton registry
_hook_registry = HookRegistry()


def get_hook_registry() -> HookRegistry:
    return _hook_registry
```

### Hook Implementations

```python
# src/ww/hooks/consolidation.py

class PreConsolidationHook(Hook):
    """Validate data before consolidation."""

    @property
    def phase(self) -> HookPhase:
        return HookPhase.PRE_CONSOLIDATION

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        episodes = context.get("episodes", [])

        # Validate episode count
        if len(episodes) > 10000:
            logger.warning(f"Large consolidation: {len(episodes)} episodes")

        # Check memory availability
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            logger.warning(f"High memory usage: {mem.percent}%")

        return {
            "validated": True,
            "episode_count": len(episodes),
            "memory_percent": mem.percent,
        }


class PostCreateHook(Hook):
    """Trigger entity extraction after episode creation."""

    @property
    def phase(self) -> HookPhase:
        return HookPhase.POST_CREATE

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        episode = context.get("episode")
        if not episode:
            return {"extracted": False}

        settings = get_settings()
        if not settings.auto_extraction_enabled:
            return {"extracted": False, "reason": "disabled"}

        # Trigger async extraction
        from ww.extraction.entity_extractor import create_default_extractor
        extractor = create_default_extractor(use_llm=settings.extraction_use_llm)

        entities = await extractor.extract(episode.content)

        return {
            "extracted": True,
            "entity_count": len(entities),
        }


class HealthCheckHook(Hook):
    """Extensible health checking."""

    @property
    def phase(self) -> HookPhase:
        return HookPhase.HEALTH_CHECK

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        checks = {}

        # Database connectivity
        try:
            from ww.storage.qdrant_store import get_qdrant_store
            store = get_qdrant_store()
            await asyncio.wait_for(store.count("ww_episodes"), timeout=5.0)
            checks["qdrant"] = "healthy"
        except Exception as e:
            checks["qdrant"] = f"unhealthy: {e}"

        # Memory usage
        import psutil
        mem = psutil.virtual_memory()
        checks["memory"] = {
            "used_percent": mem.percent,
            "available_gb": mem.available / (1024**3),
        }

        return checks


class ConfigValidationHook(Hook):
    """Validate configuration on startup."""

    @property
    def phase(self) -> HookPhase:
        return HookPhase.CONFIG_VALIDATION

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        settings = context.get("settings", get_settings())
        issues = []

        # Check password strength
        if len(settings.neo4j_password) < 8:
            issues.append("Neo4j password too short")

        # Check production settings
        env = os.getenv("WW_ENVIRONMENT", "development")
        if env == "production":
            if settings.otel_insecure:
                issues.append("OTLP insecure in production")
            if not settings.qdrant_api_key:
                issues.append("Qdrant API key not set in production")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }
```

---

## Testing Matrix

| Phase | Task | Unit Tests | Integration Tests | Performance Tests |
|-------|------|------------|-------------------|-------------------|
| P10 | P10-001 LSH Duplicates | `test_duplicate_*` | Consolidation flow | Benchmark 1000 eps |
| P10 | P10-002 Pagination | `test_paginated_*` | Full consolidation | Memory profiling |
| P10 | P10-003 Batch Hebbian | `test_batch_*` | Decay flow | Query count |
| P10 | P10-004 Docker Secrets | Manual | Docker compose up | - |
| P10 | P10-005 Config Password | `test_config_*` | Startup validation | - |
| P11 | P11-001 LLM Extraction | `test_llm_*` | Extraction flow | - |
| P11 | P11-002 Saga Tests | `test_saga_*` | Multi-store tx | - |
| P11 | P11-003 MCP Tests | `test_mcp_*` | Tool execution | - |
| P11 | P11-004 Batch Access | `test_batch_access_*` | Recall flow | Query count |
| P11 | P11-005 Spread Limits | `test_spread_*` | Graph traversal | Memory/time |
| P11 | P11-006 Session Valid | `test_session_*` | Gateway flow | - |
| P11 | P11-007 DB Auth | Manual | Docker auth | - |
| P12 | P12-001 Serialization | `test_*_roundtrip` | - | - |
| P12 | P12-002 DI Container | `test_container_*` | Service injection | - |
| P12 | P12-003 Embedding Tests | `test_embed_*` | - | - |
| P12 | P12-004 Chaos Tests | `test_chaos_*` | Failure injection | - |
| P12 | P12-005 HDBSCAN Limit | `test_clustering_*` | Large dataset | Memory |
| P12 | P12-006 Cache TTL | `test_cache_*` | - | Memory |
| P12 | P12-007 Parallel Update | `test_parallel_*` | - | Throughput |
| P12 | P12-008 Secure OTLP | `test_otel_*` | Trace export | - |

---

## Dependency Graph

```
Phase 10 (Critical) ─────────────────────────────────────────────────
    │
    ├── P10-001 LSH Duplicates ◄── P10-002 Pagination (uses search)
    │
    ├── P10-003 Batch Hebbian (independent)
    │
    ├── P10-004 Docker Secrets ◄── P10-005 Config Password (both security)
    │
    └── P10-005 Config Password ◄── P11-007 DB Auth (config used)
                                          │
Phase 11 (High) ─────────────────────────────────────────────────────
    │
    ├── P11-001 LLM Extraction (independent)
    │
    ├── P11-002 Saga Tests ◄── P11-004 Batch Access (saga pattern)
    │
    ├── P11-003 MCP Tests ◄── P11-006 Session Valid (gateway tests)
    │
    ├── P11-004 Batch Access (independent)
    │
    ├── P11-005 Spread Limits (independent)
    │
    ├── P11-006 Session Valid (independent)
    │
    └── P11-007 DB Auth ◄── P10-004, P10-005
                   │
Phase 12 (Medium) ───────────────────────────────────────────────────
    │
    ├── P12-001 Serialization (independent)
    │
    ├── P12-002 DI Container ◄── P12-003 Embedding Tests (uses DI)
    │                        ◄── P12-004 Chaos Tests (uses DI)
    │
    ├── P12-003 Embedding Tests (depends on P12-002)
    │
    ├── P12-004 Chaos Tests ◄── P11-002 Saga Tests (failure testing)
    │
    ├── P12-005 HDBSCAN Limit ◄── P10-002 Pagination (large data)
    │
    ├── P12-006 Cache TTL (independent)
    │
    ├── P12-007 Parallel Update ◄── P11-004 Batch Access (pattern)
    │
    └── P12-008 Secure OTLP (independent)
```

---

## Quick Reference

| Task | File(s) | Severity | Est. |
|------|---------|----------|------|
| P10-001 | consolidation/service.py | CRITICAL | 4h |
| P10-002 | consolidation/service.py, episodic.py | CRITICAL | 4h |
| P10-003 | semantic.py, neo4j_store.py | CRITICAL | 3h |
| P10-004 | docker-compose.yml, .env.example | CRITICAL | 2h |
| P10-005 | core/config.py | CRITICAL | 2h |
| P11-001 | extraction/entity_extractor.py | HIGH | 4h |
| P11-002 | tests/storage/test_saga.py | HIGH | 4h |
| P11-003 | tests/mcp/*.py | HIGH | 6h |
| P11-004 | memory/episodic.py | HIGH | 3h |
| P11-005 | memory/semantic.py | HIGH | 3h |
| P11-006 | mcp/gateway.py, validation.py | HIGH | 3h |
| P11-007 | docker-compose.yml, scripts/ | HIGH | 3h |
| P12-001 | core/serialization.py, memory/*.py | MEDIUM | 4h |
| P12-002 | core/container.py | MEDIUM | 4h |
| P12-003 | tests/embedding/*.py | MEDIUM | 3h |
| P12-004 | tests/chaos/*.py | MEDIUM | 4h |
| P12-005 | consolidation/service.py | MEDIUM | 2h |
| P12-006 | embedding/bge_m3.py | MEDIUM | 2h |
| P12-007 | storage/qdrant_store.py | MEDIUM | 2h |
| P12-008 | observability/tracing.py | MEDIUM | 2h |

---

## Estimated Effort

| Phase | Tasks | Days | Cumulative |
|-------|-------|------|------------|
| Phase 10 | 5 | 5 | 5 days |
| Phase 11 | 7 | 8 | 13 days |
| Phase 12 | 8 | 7 | 20 days |

**Total to completion**: 4 weeks (conservative)

---

## Success Criteria

### Phase 10 Complete When:
- [ ] Duplicate detection < 5s for 1000 episodes
- [ ] Pagination prevents memory exhaustion
- [ ] Hebbian decay uses batch operations (2 queries max)
- [ ] No hardcoded passwords in codebase
- [ ] Config rejects weak passwords

### Phase 11 Complete When:
- [ ] LLM entity extraction functional
- [ ] Saga compensation coverage > 90%
- [ ] MCP module coverage > 80%
- [ ] Recall uses batch access updates
- [ ] Spreading activation memory bounded
- [ ] Session validation at gateway
- [ ] Database authentication enabled

### Phase 12 Complete When:
- [ ] Serialization deduplicated
- [ ] DI container functional
- [ ] Embedding provider tested
- [ ] Chaos tests passing
- [ ] HDBSCAN memory bounded
- [ ] Cache TTL working
- [ ] Parallel updates faster
- [ ] OTLP security configurable

---

## Appendix: File Index

| File | Tasks |
|------|-------|
| `src/ww/consolidation/service.py` | P10-001, P10-002, P12-005 |
| `src/ww/memory/episodic.py` | P10-002, P11-004 |
| `src/ww/memory/semantic.py` | P10-003, P11-005 |
| `src/ww/storage/neo4j_store.py` | P10-003 |
| `src/ww/storage/qdrant_store.py` | P12-007 |
| `src/ww/core/config.py` | P10-005, P12-008 |
| `docker-compose.yml` | P10-004, P11-007 |
| `src/ww/extraction/entity_extractor.py` | P11-001 |
| `src/ww/storage/saga.py` | P11-002 |
| `src/ww/mcp/gateway.py` | P11-003, P11-006 |
| `src/ww/mcp/validation.py` | P11-006 |
| `src/ww/core/serialization.py` | P12-001 (new) |
| `src/ww/core/container.py` | P12-002 (new) |
| `src/ww/embedding/bge_m3.py` | P12-003, P12-006 |
| `src/ww/observability/tracing.py` | P12-008 |
| `src/ww/core/hooks.py` | All (new) |
| `tests/storage/test_saga.py` | P11-002 (new) |
| `tests/mcp/*.py` | P11-003 (new/expand) |
| `tests/embedding/*.py` | P12-003 (new) |
| `tests/chaos/*.py` | P12-004 (new) |

---

**Status**: PLAN READY - Awaiting Implementation
