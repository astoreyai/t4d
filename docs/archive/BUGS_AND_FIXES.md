# World Weaver - Critical Bugs and Fixes

**Report Date**: 2025-12-07  
**Status**: 4 Critical Bugs Identified

---

## Bug #1: Missing EpisodicMemory.recent() Method

**Severity**: CRITICAL  
**File**: `/mnt/projects/ww/src/ww/memory/episodic.py`  
**Error Message**: `'EpisodicMemory' object has no attribute 'recent'`

### Where It Breaks
- API Route: `GET /api/v1/episodes` → 500 Internal Server Error
- Test: Integration test for listing episodes

### Root Cause
The API routes file (`src/ww/api/routes/episodes.py`) calls `episodic.recent()` but the method doesn't exist on the EpisodicMemory class.

### Fix Required
Add the `recent()` method to EpisodicMemory class that returns recently accessed episodes, probably with some limit.

Expected signature:
```python
def recent(self, limit: int = 20) -> List[Episode]:
    """Return recently accessed episodes for current session."""
    # Implementation needed
```

---

## Bug #2: Missing SemanticMemory Methods

**Severity**: CRITICAL  
**File**: `/mnt/projects/ww/src/ww/memory/semantic.py`  
**Error Messages**:
- `'SemanticMemory' object has no attribute 'store_entity'`
- `'SemanticMemory' object has no attribute 'list_entities'`

### Where It Breaks
- API Route: `POST /api/v1/entities` → 422 Unprocessable Entity
- API Route: `GET /api/v1/entities` → 500 Internal Server Error
- Tests: Entity creation and listing tests

### Root Cause
The API routes file (`src/ww/api/routes/entities.py`) calls:
- `semantic.store_entity()` - doesn't exist
- `semantic.list_entities()` - doesn't exist

### Fix Required
Add two methods to SemanticMemory class:

```python
def store_entity(self, entity: Entity, session_id: Optional[str] = None) -> Entity:
    """Create or update an entity in semantic memory."""
    # Implementation needed

def list_entities(self, session_id: Optional[str] = None) -> List[Entity]:
    """List all entities in semantic memory for current/specified session."""
    # Implementation needed
```

---

## Bug #3: Pattern Separation State Not Being Tracked

**Severity**: CRITICAL  
**File**: `/mnt/projects/ww/src/ww/memory/pattern_separation.py`  
**Class**: `DentateGyrus`

### Failing Tests (7 failures)
```
test_encode_no_similar
  - Expected: 1 vector search call recorded
  - Got: 0 (empty search_calls list)

test_encode_with_similar_applies_separation
  - Expected: 1 separation applied
  - Got: 0

test_encode_separation_produces_different_embedding
  - Expected: similarity difference > 0.01
  - Got: 0.0 (vectors identical)

test_get_separation_history
  - Expected: 3 history entries
  - Got: 0 (empty list)

test_get_separation_history_only_separated
  - Expected: 1 separated pattern
  - Got: 0

test_get_stats
  - Expected: stats with counts > 0
  - Got: empty dict or 0 values

test_clear_history
  - Expected: history cleared after method call
  - Got: still empty (was never populated)
```

### Root Cause
The DentateGyrus class has methods that should:
1. Record vector store search calls (`search_calls` list)
2. Maintain separation history (`separation_history`)
3. Calculate and return statistics (`stats`)

But these are either:
- Not being called from `encode()` method
- Not being updated properly
- Not being returned correctly

### Fix Required
Review the DentateGyrus implementation and ensure:
1. `encode()` method populates `self.search_calls` when it queries the vector store
2. `encode()` method populates `self.separation_history` when it applies separation
3. `get_separation_history()` properly returns the maintained history
4. `get_stats()` calculates real statistics from collected data
5. `clear_history()` actually clears the internal state

---

## Bug #4: Joint Optimization Loss Diverging Instead of Converging

**Severity**: CRITICAL  
**File**: `/mnt/projects/ww/src/ww/learning/` (location TBD - likely in optimization/learning code)  
**Test**: `tests/unit/test_joint_optimization.py::TestConsistencyLoss::test_consistency_loss_convergence`

### What's Happening
```
Early loss:  0.4055
Late loss:   0.5313
Direction:   DIVERGING (increasing)
Expected:    Loss should decrease to <= 1.2
Actual:      Loss = 1.3101704323934527
```

### Root Cause
The consistency loss between memory systems should decrease over time as they converge. Instead it's increasing, suggesting:
1. The loss calculation might be inverted
2. The optimization direction might be wrong
3. There might be a sign error in gradients
4. The learning rate might be too high/unstable

### Fix Required
1. Review loss calculation in consistency loss function
2. Verify optimization direction (minimization vs maximization)
3. Check learning rate scaling
4. Verify gradient flow and signs
5. Consider adding loss smoothing or annealing

---

## Bug #5: Missing Search Endpoints

**Severity**: MEDIUM  
**File**: `/mnt/projects/ww/src/ww/api/routes/entities.py` and `skills.py`  
**Error**: 405 Method Not Allowed

### Missing Routes
- `GET /api/v1/entities/search` - Search entities by name/type
- `GET /api/v1/skills/search` - Search skills by name/domain

### Fix Required
Add route handlers:
```python
# In entities.py
@router.get("/search")
async def search_entities(q: str, entity_type: Optional[str] = None) -> SearchResponse:
    """Search entities by query and optional type filter."""
    # Implementation needed

# In skills.py
@router.get("/search")
async def search_skills(q: str, domain: Optional[str] = None) -> SearchResponse:
    """Search skills by query and optional domain filter."""
    # Implementation needed
```

---

## Bug #6: Invalid Database Enum Values

**Severity**: MEDIUM  
**Issue**: Database constraint violation
**Error**: `domain='testing' not in ['coding', 'research', 'trading', 'devops', 'writing']`

### Root Cause
Database contains at least one skill with `domain='testing'` which is not in the allowed enum values.

### Fix Required
1. Query database for skills with invalid domain values:
   ```cypher
   MATCH (s:Skill) WHERE s.domain = 'testing' RETURN s
   ```

2. Either:
   - Update the skill's domain to valid value
   - Or update the enum to include 'testing' as valid value
   - Or delete the invalid skill record

---

## Bug #7: AsyncMock Serialization in Integration Tests

**Severity**: MEDIUM  
**File**: `/mnt/projects/ww/tests/integration/conftest.py`  
**Affected Tests**: 8 API integration tests

### Error Pattern
```
500 Internal Server Error
6 validation errors for EpisodeResponse
  - id: UUID input should be a string, bytes or UUID object [type=uuid_type, input_value=<AsyncMock ...>]
  - session_id: Input should be a valid string [type=string_type, input_value=<AsyncMock ...>]
  - content: Input should be a valid string [type=string_type, input_value=<AsyncMock ...>]
  - outcome: Input should be 'success', 'failure', 'partial' or 'neutral' [type=enum, input_value=<AsyncMock ...>]
  - context: Input should be a valid dictionary [type=model_type, input_value=<AsyncMock ...>]
  - retrievability: Input should be a valid number [type=float_type, input_value=<coroutine ...>]
```

### Root Cause
The test fixtures are creating AsyncMock objects but not properly:
1. Awaiting them before returning
2. Configuring return_value with concrete values
3. The route is receiving Mock objects instead of actual data

### Fix Required
In conftest.py, update mock fixtures to use `AsyncMock` with `return_value` set to actual model instances:

```python
# BEFORE (broken):
mock_episodic.create = AsyncMock()  # Returns AsyncMock itself

# AFTER (fixed):
mock_episodic.create = AsyncMock(return_value=Episode(...))  # Returns concrete instance
```

Or configure mock return values properly:
```python
mock_episodic.create.return_value = Episode(
    id=UUID("..."),
    session_id="default",
    content="test",
    outcome="success",
    # ... other fields
)
```

---

## Summary Table

| Bug | File | Type | Impact | Fix Time |
|-----|------|------|--------|----------|
| 1 | episodic.py | Missing method | Cannot list episodes | 30 min |
| 2 | semantic.py | Missing methods | Cannot manage entities | 30 min |
| 3 | pattern_separation.py | State tracking | Pattern separation broken | 1-2 hrs |
| 4 | learning/* | Algorithm | Loss diverging | 2-3 hrs |
| 5 | routes/entities.py, skills.py | Missing routes | No search functionality | 30 min |
| 6 | Neo4j database | Data integrity | Enum constraint violated | 15 min |
| 7 | conftest.py | Test fixture | 8 tests failing | 30 min |

**Total Estimated Fix Time**: 5-7 hours

---

## Verification Steps After Fixes

For each bug, verify with:

```bash
# Unit tests
pytest tests/unit/ -v --tb=short

# Integration tests
pytest tests/integration/test_api_flows.py -v --tb=short

# API validation
curl http://localhost:8765/api/v1/health
curl http://localhost:8765/api/v1/episodes
curl http://localhost:8765/api/v1/entities
curl http://localhost:8765/api/v1/skills

# Coverage
pytest tests/ --cov=src/ww --cov-report=term-missing
```

