# World Weaver P2/P3 Issues Summary

**Generated**: 2025-12-09
**Full Details**: See `/mnt/projects/t4d/t4dm/P2_P3_ISSUES_INVENTORY.md`

---

## Overview

After comprehensive P0/P1 bug remediation (37 issues fixed), 52 medium and low priority issues remain:

- **P2 Medium**: 32 issues
- **P3 Low**: 20 issues
- **Status**: All critical functionality working (4,162 tests passing, 76% coverage)

---

## P2 Issues Breakdown (32 total)

### Already Fixed in Recent Sessions (17 issues)
All memory leak and logic error issues from BUG_REMEDIATION_PLAN.md have been resolved:

**Memory Leaks (9)**: MEM-001 through MEM-009 - ALL FIXED
- Metrics histograms, plasticity history, dopamine RPE history, eligibility traces, visualization figures, three_factor signals, reconsolidation cooldowns, working memory eviction history

**Logic Errors (8)**: LOGIC-004 through LOGIC-011 - ALL FIXED/VERIFIED
- FSRS implementation, serotonin double counting, collector ordering, eligibility double decay, credit flow sign preservation, expected value calculation
- Hebbian and BCM verified as correct implementations

### Remaining P2 Issues (15 issues)

#### Security (6 issues) - 13 hours
1. **M1**: Weak password validation (8 char min → 12 char, pattern detection) - 2h
2. **M2**: Session ID allows reserved IDs ("admin", "system") - 1h
3. **M3**: Rate limiter not distributed (ineffective with multiple workers) - 4h or 1h (doc)
4. **M4**: No API key authentication (session ID only) - 3h
5. **M5**: Docker ports audit (may be exposed in full deployment) - 1h
6. **M6**: No CSP headers (/docs, /redoc vulnerable to XSS) - 2h

#### Performance (9 issues) - 24 hours
1. **B1.1**: Cache eviction O(n) scan → O(log n) heap - 3h
2. **B2.1**: Session ID filtering without index (O(n) → O(log n)) - 2h HIGH IMPACT
3. **B2.2**: Hybrid search over-fetching (4x candidates) - 4h
4. **B2.3**: Batch parallelism not controlled (semaphore needed) - 2h
5. **B3.2**: Cypher query composition (defense-in-depth review) - 2h
6. **B3.3**: Connection pool utilization unknown (add metrics) - 3h
7. **B4.2.1**: Context cache misses (add LRU fallback) - 3h
8. **B4.3.1**: Sorted node processing (optimization opportunity) - 2h
9. **B5.2**: Worker configuration not documented - 2h

---

## P3 Issues Breakdown (20 total)

#### Security (8 issues) - 10.5 hours
1. **L1**: .env permissions not enforced (warning only) - 1h
2. **L2**: Cypher queries use f-strings (defense-in-depth hardening) - 1h
3. **L3**: XSS sanitization uses pattern removal (document HTML escaping requirement) - 0.5h
4. **L4**: Embedding cache limits verification needed - 1h
5. **L5**: No audit logging (session ops, bulk deletes, auth failures) - 3h
6. **L6**: Docker container runs as root - 1h
7. **L7**: No request size limits (DoS risk) - 2h
8. **L8**: LLM timeout verification needed - 1h

#### Code Quality (12 issues) - 33 hours
1. **Q001**: dynamics.py TODO (get original embedding) - 2h
2. **Q002**: eligibility_security.py TODOs (3 validation items) - 2h
3. **Q003**: API input validation audit (UUIDs, ranges, lengths) - 4h
4. **Q004**: Replace broad exception handlers - 4h
5. **Q005**: Add network call timeouts - 3h
6. **Q006**: Remove dead code paths - 2h
7. **Q007**: Standardize naming conventions (lr_modulation vs effective_lr) - 3h
8. **Q008**: Add type hints throughout - 4h
9. **Q009**: Extract magic numbers to constants - 2h
10. **Q010**: Add missing docstrings - 6h
11. **Q011**: Update outdated comments - 2h
12. **Q012**: Performance profiling (hot paths) - 6h

---

## Recommended Fix Order

### PHASE 1: Quick Security Wins (6 hours)
Priority: Address highest-risk security issues with minimal effort
- P2-SEC-M1: Password validation (2h)
- P2-SEC-M2: Session ID validation (1h)
- P2-SEC-M6: CSP headers (2h)
- P3-SEC-L1: .env permissions (1h)

**Impact**: Production deployment security significantly improved

---

### PHASE 2: Quick Performance Wins (9 hours)
Priority: High-impact optimizations
- **P2-OPT-B2.1**: Session ID index (2h) ⭐ HIGH IMPACT
- P2-OPT-B2.3: Batch semaphore (2h)
- P2-OPT-B1.1: Cache eviction heap (3h)
- P2-OPT-B5.2: Worker docs (2h)

**Impact**: 2-5x performance improvement for multi-session deployments

---

### PHASE 3: Code Quality Quick Fixes (10 hours)
Priority: Close TODO gaps and verification items
- P3-QUALITY-001: dynamics.py TODO (2h)
- P3-QUALITY-002: Security TODOs (2h)
- P3-SEC-L3-L8: Various verifications and docs (5.5h)
- P3-QUALITY-006: Dead code audit (1.5h)

**Impact**: Code completeness and reliability improved

---

### PHASE 4: Medium Security Fixes (8 hours)
Priority: Production hardening
- P2-SEC-M4: API key auth (3h) - critical if public deployment
- P2-SEC-M3: Distributed rate limiter (4h or 1h doc)
- P2-SEC-M5: Docker ports audit (1h)

**Impact**: Production-grade access control

---

### PHASE 5: Performance Optimization (14 hours)
Priority: Advanced optimizations
- P2-OPT-B2.2: Adaptive hybrid search (4h)
- P2-OPT-B3.3: Pool metrics (3h)
- P2-OPT-B4.2.1: Context cache (3h)
- P2-OPT-B4.3.1: Sorted processing (2h)
- P2-OPT-B3.2: Cypher review (2h)

**Impact**: 10-40% latency reduction in search operations

---

### PHASE 6: Code Quality Improvements (23 hours)
Priority: Long-term maintainability
- P3-QUALITY-003: API validation (4h)
- P3-QUALITY-004: Exception handlers (4h)
- P3-QUALITY-005: Network timeouts (3h)
- P3-QUALITY-007: Naming (3h)
- P3-QUALITY-008: Type hints (4h)
- P3-QUALITY-009: Magic numbers (2h)
- P3-QUALITY-011: Comments (2h)
- P3-SEC-L5: Audit logging (3h)

**Impact**: Codebase maintainability and type safety

---

### PHASE 7: Documentation & Long-term (14 hours)
Priority: Developer experience
- P3-QUALITY-010: Docstrings (6h)
- P3-QUALITY-012: Performance profiling (6h+)

**Impact**: Developer onboarding and optimization insights

---

## Critical Path for Production

If preparing for production deployment, focus on these HIGH PRIORITY items (19 hours):

1. **Security Must-Haves** (6 hours):
   - P2-SEC-M1: Password validation (2h)
   - P2-SEC-M4: API key auth (3h)
   - P2-SEC-M6: CSP headers (2h)
   - P3-SEC-L1: .env permissions (1h)

2. **Performance Must-Haves** (4 hours):
   - P2-OPT-B2.1: Session ID index (2h) ⭐ CRITICAL
   - P2-OPT-B2.3: Batch semaphore (2h)

3. **Rate Limiting Fix** (1-4 hours):
   - P2-SEC-M3: Document single-worker limitation (1h) OR implement Redis (4h)

4. **Security Hardening** (5 hours):
   - P2-SEC-M2: Session validation (1h)
   - P3-SEC-L6: Docker non-root (1h)
   - P3-SEC-L7: Request size limits (2h)
   - P3-SEC-L5: Audit logging (3h) - optional but recommended

**Total Critical Path**: 16-19 hours (depending on rate limiter approach)

---

## Quick Reference: File Locations

### Security Issues
- `/mnt/projects/t4d/t4dm/src/t4dm/core/config.py` - Password validation, .env permissions
- `/mnt/projects/t4d/t4dm/src/t4dm/api/deps.py` - Session validation, API key auth
- `/mnt/projects/t4d/t4dm/src/t4dm/mcp/gateway.py` - Rate limiter
- `/mnt/projects/t4d/t4dm/src/t4dm/api/server.py` - CSP headers, request size limits
- `/mnt/projects/t4d/t4dm/Dockerfile` - Docker non-root user

### Performance Issues
- `/mnt/projects/t4d/t4dm/src/t4dm/embedding/bge_m3.py` - Cache eviction
- `/mnt/projects/t4d/t4dm/src/t4dm/storage/qdrant_store.py` - Session index, hybrid search, batch ops
- `/mnt/projects/t4d/t4dm/src/t4dm/storage/neo4j_store.py` - Connection pool metrics
- `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` - Context cache, sorted processing

### Code Quality Issues
- `/mnt/projects/t4d/t4dm/src/t4dm/temporal/dynamics.py:350` - TODO
- `/mnt/projects/t4d/t4dm/tests/security/test_eligibility_security.py` - Security TODOs

---

## Status Tracking

| Category | Total | Fixed | Remaining | Est. Hours |
|----------|-------|-------|-----------|------------|
| **P2 Medium** | 32 | 17 | 15 | 37h |
| **P3 Low** | 20 | 0 | 20 | 43.5h |
| **TOTAL** | 52 | 17 | 35 | 80.5h |

**Critical Path Only**: 16-19 hours for production readiness

---

## Next Steps

1. **Review this summary** with team/stakeholders
2. **Prioritize phases** based on deployment timeline
3. **Start with Phase 1** (Quick Security Wins) - 6 hours
4. **Run full test suite** after each phase: `pytest tests/ --cov=src/ww`
5. **Update progress** in `/mnt/projects/t4d/t4dm/P2_P3_ISSUES_INVENTORY.md`

---

## References

- Full Inventory: `/mnt/projects/t4d/t4dm/P2_P3_ISSUES_INVENTORY.md`
- Bug Remediation Report: `/mnt/projects/t4d/t4dm/BUG_REMEDIATION_REPORT.md`
- Security Audit: `/mnt/projects/t4d/t4dm/SECURITY_AUDIT_REPORT.md`
- Optimization Analysis: `/mnt/projects/t4d/t4dm/OPTIMIZATION_ANALYSIS_REPORT.md`
- Bug Remediation Plan: `/mnt/projects/t4d/t4dm/BUG_REMEDIATION_PLAN.md`

---

**Last Updated**: 2025-12-09
