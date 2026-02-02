# World Weaver Test Implementation Checklist

**Status**: Ready to implement | **Est. Time**: 60 hours | **Priority**: P1

---

## Phase 0: Fix Async Issues (1-2 hours) - CRITICAL

### Tasks
- [ ] Create `/mnt/projects/t4d/t4dm/tests/conftest.py`
  - [ ] Add session-scoped event loop fixture
  - [ ] Add cleanup_memory_services fixture
  - [ ] Add mock fixtures (embedding, qdrant, neo4j)
  - [ ] Configure pytest markers
- [ ] Update `/mnt/projects/t4d/t4dm/pyproject.toml`
  - [ ] Set `asyncio_mode = "auto"`
  - [ ] Add pytest markers
- [ ] Run tests: `pytest tests/ -v`
  - [ ] Expected: 237 passed (5 previously failing)
  - [ ] Check: No `RuntimeError: Task got Future attached to a different loop`

### Verification
```bash
pytest tests/test_memory.py::test_episodic_memory_recall -v
# Should PASS (was failing before)
```

---

## Phase 1: Consolidation Tests (8-10 hours) - CRITICAL

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_consolidation.py`

#### Consolidation Service Tests
- [ ] `TestLightConsolidation` class
  - [ ] `test_consolidate_light_returns_status()`
  - [ ] `test_consolidate_light_deduplicates_episodes()`
  - [ ] `test_consolidate_light_cleanup_old_episodes()`

- [ ] `TestDeepConsolidation` class
  - [ ] `test_consolidate_deep_extracts_entities()`
  - [ ] `test_consolidate_deep_respects_session_filter()`
  - [ ] `test_consolidate_deep_returns_metrics()`

- [ ] `TestSkillConsolidation` class
  - [ ] `test_consolidate_skills_merges_similar_procedures()`
  - [ ] `test_consolidate_skills_updates_success_rates()`

- [ ] `TestConsolidationErrorHandling` class
  - [ ] `test_consolidation_handles_embedding_error()`
  - [ ] `test_consolidation_handles_database_error()`
  - [ ] `test_consolidation_handles_storage_error()`

- [ ] `TestConsolidationEdgeCases` class
  - [ ] `test_consolidation_with_empty_episodes()`
  - [ ] `test_consolidation_with_very_large_episode()`
  - [ ] `test_consolidation_timeout_recovery()`

- [ ] `TestConsolidationIntegration` class
  - [ ] `test_full_consolidation_pipeline()`
  - [ ] `test_consolidation_preserves_session_isolation()`

### Verification
```bash
pytest tests/unit/test_consolidation.py -v
# Should have 15+ tests, coverage: 18% → 60%+
```

---

## Phase 2: MCP Gateway Tests (10-12 hours) - CRITICAL

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_mcp_gateway.py`

#### Episodic Memory Tools
- [ ] `TestEpisodicMemoryTools` class
  - [ ] `test_episodic_create_with_all_parameters()`
  - [ ] `test_episodic_create_validates_valence()`
  - [ ] `test_episodic_create_validates_outcome()`
  - [ ] `test_episodic_create_returns_episode_id()`
  - [ ] `test_episodic_recall_with_query()`
  - [ ] `test_episodic_recall_default_limit()`
  - [ ] `test_episodic_cleanup_removes_old_episodes()`

#### Semantic Memory Tools
- [ ] `TestSemanticMemoryTools` class
  - [ ] `test_semantic_create_entity_validates_type()`
  - [ ] `test_semantic_create_entity_invalid_type()`
  - [ ] `test_semantic_create_relationship()`
  - [ ] `test_semantic_recall_with_activation()`
  - [ ] `test_semantic_get_entity()`

#### Procedural Memory Tools
- [ ] `TestProceduralMemoryTools` class
  - [ ] `test_procedural_build_from_trajectory()`
  - [ ] `test_procedural_build_validates_trajectory()`
  - [ ] `test_procedural_build_validates_outcome_score()`
  - [ ] `test_procedural_retrieve_by_task()`
  - [ ] `test_procedural_retrieve_without_domain()`

#### Error Handling
- [ ] `TestErrorHandling` class
  - [ ] `test_invalid_uuid_returns_error()`
  - [ ] `test_missing_required_parameter_returns_error()`
  - [ ] `test_validation_error_format()`
  - [ ] `test_service_unavailable_error()`

#### Session Management
- [ ] `TestSessionManagement` class
  - [ ] `test_get_services_initializes_once()`
  - [ ] `test_multiple_sessions_isolated()`

#### Tool Documentation
- [ ] `TestToolDocumentation` class
  - [ ] `test_mcp_app_defined()`
  - [ ] `test_mcp_app_has_instructions()`

### Verification
```bash
pytest tests/unit/test_mcp_gateway.py -v
# Should have 25+ tests, coverage: 18% → 60%+
```

---

## Phase 3: Observability Tests (8-10 hours) - HIGH PRIORITY

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_observability.py`

#### Logging Tests
- [ ] `TestLogging` class
  - [ ] `test_configure_logging()`
  - [ ] `test_get_logger()`
  - [ ] `test_set_context()`
  - [ ] `test_clear_context()`
  - [ ] `test_log_operation()`

- [ ] `TestOperationLogger` class
  - [ ] `test_operation_logger_success()`
  - [ ] `test_operation_logger_timing()`

#### Metrics Tests
- [ ] `TestMetrics` class
  - [ ] `test_get_metrics()`
  - [ ] `test_metrics_collector_initialization()`
  - [ ] `test_count_operation()`
  - [ ] `test_timed_operation()`
  - [ ] `test_timer_context_manager()`
  - [ ] `test_async_timer_context_manager()`

#### Health Check Tests
- [ ] `TestHealthChecker` class
  - [ ] `test_health_status_enum()`
  - [ ] `test_component_health()`
  - [ ] `test_system_health()`
  - [ ] `test_health_checker()`
  - [ ] `test_health_check_all_components()`
  - [ ] `test_health_check_individual_component()`

### Verification
```bash
pytest tests/unit/test_observability.py -v
# Should have 15+ tests, coverage: 0% → 35%+
```

---

## Phase 4: Storage & Edge Cases (12-15 hours) - MEDIUM PRIORITY

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_storage.py`

#### Neo4j Storage Tests
- [ ] `TestNeo4jStore` class
  - [ ] `test_initialize_creates_connection()`
  - [ ] `test_create_node_with_properties()`
  - [ ] `test_update_node_properties()`
  - [ ] `test_delete_node()`
  - [ ] `test_create_relationship()`
  - [ ] `test_get_relationships()`
  - [ ] `test_strengthen_relationship_updates_weight()`
  - [ ] `test_connection_timeout_error()`
  - [ ] `test_query_error_recovery()`
  - [ ] `test_batch_operations()`

#### Qdrant Storage Tests
- [ ] `TestQdrantStore` class
  - [ ] `test_initialize_creates_collections()`
  - [ ] `test_add_vector_with_payload()`
  - [ ] `test_search_returns_scored_results()`
  - [ ] `test_update_payload()`
  - [ ] `test_delete_vector()`
  - [ ] `test_batch_operations()`
  - [ ] `test_collection_cleanup()`

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_edge_cases.py`

#### Memory Edge Cases
- [ ] `TestMemoryEdgeCases` class
  - [ ] `test_recall_empty_results()`
  - [ ] `test_create_very_large_episode()`
  - [ ] `test_create_very_large_entity()`
  - [ ] `test_create_deeply_nested_context()`
  - [ ] `test_recall_with_zero_similarity()`
  - [ ] `test_recall_with_identical_similarity()`

#### Concurrency Edge Cases
- [ ] `TestConcurrentOperations` class
  - [ ] `test_concurrent_creates_same_session()`
  - [ ] `test_concurrent_recalls()`
  - [ ] `test_concurrent_entity_creation()`
  - [ ] `test_race_condition_relationship_update()`

#### Resource Limit Edge Cases
- [ ] `TestResourceLimits` class
  - [ ] `test_many_relationships_entity()`
  - [ ] `test_deep_entity_hierarchy()`
  - [ ] `test_very_long_context_path()`
  - [ ] `test_unicode_content_handling()`

#### Timeout Edge Cases
- [ ] `TestTimeoutHandling` class
  - [ ] `test_embedding_service_timeout()`
  - [ ] `test_database_query_timeout()`
  - [ ] `test_network_timeout_recovery()`
  - [ ] `test_operation_cancellation()`

### Verification
```bash
pytest tests/unit/test_storage.py tests/unit/test_edge_cases.py -v
# Should have 30+ tests, coverage: 41%/56% → 65%/70%+
```

---

## Phase 5: Integration & Documentation (6-8 hours) - MEDIUM PRIORITY

### Documentation
- [ ] Update `/mnt/projects/t4d/t4dm/README.md` - Add test section
- [ ] Create test execution guide
- [ ] Document test markers and categories
- [ ] Add coverage report instructions

### CI/CD Setup
- [ ] Add pytest coverage checks to CI
- [ ] Set coverage thresholds (75%+)
- [ ] Add test reporting
- [ ] Setup coverage dashboard

### Cleanup
- [ ] Remove duplicate tests from `tests/test_memory.py`
- [ ] Consolidate into `tests/integration/test_*.py`
- [ ] Review test organization
- [ ] Ensure no test interdependencies

### Final Verification
- [ ] All 237+ tests passing
- [ ] Coverage >= 75%
- [ ] No warnings or deprecations
- [ ] Documentation complete

---

## Coverage Targets Checklist

### Current State
- [ ] Total Coverage: 47% ✓
- [ ] Tests Passing: 232/237 (5 async failures)

### Target State
- [ ] Total Coverage: 75%+
  - [ ] consolidation/service.py: 18% → 75%+
  - [ ] mcp/memory_gateway.py: 18% → 75%+
  - [ ] observability/*: 0% → 50%+
  - [ ] storage/neo4j_store.py: 41% → 70%+
  - [ ] storage/qdrant_store.py: 56% → 70%+
  - [ ] memory/semantic.py: 53% → 75%+
  - [ ] memory/procedural.py: 64% → 75%+
  - [ ] memory/episodic.py: 87% → 90%

- [ ] Tests Passing: 237/237 (100%)
- [ ] All error paths tested
- [ ] All edge cases covered
- [ ] Async operations validated

---

## Testing Best Practices

### During Implementation
- [ ] Use provided test templates
- [ ] Follow existing test patterns
- [ ] Run tests frequently: `pytest -x` (stop on first failure)
- [ ] Check coverage: `pytest --cov=src/ww --cov-report=term-missing`
- [ ] Use fixtures to avoid duplication
- [ ] Mock external services appropriately

### Code Review
- [ ] Tests are clear and well-documented
- [ ] One assertion per test (or logically grouped)
- [ ] Test names describe what's being tested
- [ ] Use meaningful variable names
- [ ] Follow naming convention: `test_<function>_<scenario>`

### Quality Standards
- [ ] Coverage > 80% for critical modules
- [ ] All async tests marked with `@pytest.mark.asyncio`
- [ ] Error tests verify message content
- [ ] Edge case tests use boundary values
- [ ] Performance tests marked with `@pytest.mark.slow`

---

## Quick Commands

```bash
# Run specific phase
pytest tests/unit/test_consolidation.py -v
pytest tests/unit/test_mcp_gateway.py -v
pytest tests/unit/test_observability.py -v
pytest tests/unit/test_storage.py -v
pytest tests/unit/test_edge_cases.py -v

# Run with coverage
pytest --cov=src/ww --cov-report=term-missing

# Run and show slowest tests
pytest --durations=10

# Run and stop on first failure
pytest -x

# Run specific test
pytest tests/unit/test_consolidation.py::TestLightConsolidation::test_consolidate_light_returns_status -v

# Generate HTML coverage report
pytest --cov=src/ww --cov-report=html
# Open htmlcov/index.html
```

---

## Timeline

```
Week 1:
  Day 1: Phase 0 + Phase 1 (conftest + consolidation tests)
  Day 2: Phase 1 (consolidation tests) + Phase 2 start
  Day 3: Phase 2 (MCP gateway tests)
  Day 4: Phase 2 (MCP gateway tests) + Phase 3 start
  Day 5: Phase 3 (observability tests)

Week 2:
  Day 1: Phase 4 (storage & edge cases)
  Day 2: Phase 4 (storage & edge cases)
  Day 3: Phase 4 (storage & edge cases) + Phase 5
  Day 4: Phase 5 (documentation & CI/CD)
  Day 5: Final verification & polish
```

---

## Success Criteria

- [ ] 237 tests passing (100%)
- [ ] Overall coverage: 75%+
- [ ] Consolidation: 75%+
- [ ] MCP Gateway: 75%+
- [ ] Observability: 50%+
- [ ] All error paths tested
- [ ] All edge cases covered
- [ ] Documentation complete
- [ ] CI/CD coverage checks enabled

---

## Resources

**Analysis Documents**:
- `/mnt/projects/t4d/t4dm/TEST_COVERAGE_ANALYSIS.md` - Detailed analysis
- `/mnt/projects/t4d/t4dm/TEST_IMPLEMENTATION_ROADMAP.md` - Full roadmap with code
- `/mnt/projects/t4d/t4dm/TEST_SUMMARY.md` - Executive summary

**Test Templates**:
- `/mnt/projects/t4d/t4dm/tests/unit/test_validation.py` - 100% coverage example
- `/mnt/projects/t4d/t4dm/tests/unit/test_saga.py` - 96% coverage example
- `/mnt/projects/t4d/t4dm/tests/integration/test_session_isolation.py` - Integration test example

**Configuration**:
- `/mnt/projects/t4d/t4dm/pyproject.toml` - Pytest settings

---

## Notes

- All test code is provided in the roadmap document
- Copy-paste test implementations, don't rewrite
- Use fixtures to reduce duplication
- Follow naming conventions from existing tests
- Run tests frequently to catch issues early
- Update this checklist as you progress

