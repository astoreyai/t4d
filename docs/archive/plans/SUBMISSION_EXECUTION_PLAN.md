# World Weaver Publication Execution Plan
**Created**: 2024-12-05 | **Target**: PhD-Level Journal Submission

## Executive Summary
- **Main Paper**: IEEE Transactions on Artificial Intelligence (10 pages, 59 citations)
- **Companion Paper**: Philosophy of AI/Cognitive Science venue (10 pages, 25 citations)
- **Source Code**: 29,020 lines Python, 79% test coverage
- **Target AI Detection**: <10%

---

## Phase 1: Codebase Cleanup [DISCOVERY COMPLETE]

### Files to REMOVE (~12GB):
| Category | Path | Size | Action |
|----------|------|------|--------|
| Virtual envs | `.venv/`, `venv/` | 7.9GB | DELETE |
| Models | `models/` | 4.3GB | DELETE (downloadable) |
| Cache | `.hypothesis/`, `.pytest_cache/`, `__pycache__/`, `.ruff_cache/` | 2.5MB | DELETE |
| LaTeX aux | `*.aux`, `*.log`, `*.out` | 100KB | DELETE |
| Review artifacts | `docs/papers/reviews/`, `docs/papers/qa_reports/` | 500KB | ARCHIVE |
| Test artifacts | `*.csv`, `*SUMMARY.txt`, `VALIDATION_*.json` | 50KB | DELETE |
| Temp files | `=2.6`, `texput.log`, `test_pagination.py` | 10KB | DELETE |

### Files to KEEP:
| Category | Path | Notes |
|----------|------|-------|
| Source | `src/t4dm/` | 29,020 lines core implementation |
| Tests | `tests/` | 1,259 tests |
| Papers | `docs/world_weaver_final.tex`, `docs/papers/philosophy_of_ai_memory.tex` | PRIMARY |
| Config | `pyproject.toml`, `requirements.txt`, `.env.example` | Build files |
| Docker | `Dockerfile`, `docker-compose.yml` | Deployment |
| Docs | `README.md`, `ARCHITECTURE.md` | Essential only |

---

## Phase 2: Citation Verification

### Main Paper (59 citations claimed, 31 \cite{} calls):
**CRITICAL**: Extract all bibitem entries and verify:
1. Author names (exact spelling, initials)
2. Publication year
3. Journal/conference name
4. Volume, issue, page numbers
5. DOI where available

### Philosophy Paper (25 citations, 21 \citep{} calls):
Same verification protocol.

**Tools**: CrossRef API, Google Scholar, Semantic Scholar

---

## Phase 3: Statistical Validation

### Hardware Specification CORRECTION:
- **Current (WRONG)**: Intel i9-12900K, 64GB RAM, NVIDIA RTX 4090
- **Actual (CORRECT)**: Intel i9, 128GB RAM, NVIDIA RTX 3090, 4TB HD

### Performance Claims to Verify:
| Claim | Value | Source | Status |
|-------|-------|--------|--------|
| Hybrid Recall@10 | 84% ± 0.02 | Experiments | VERIFY |
| Dense-only Recall@10 | 72% | Experiments | VERIFY |
| p-value | < 0.001 | Statistical test | VERIFY |
| Embedding latency | ~1.2s/episode | Benchmark | VERIFY |
| Retrieval latency (10K) | 52ms | Benchmark | VERIFY |
| Retrieval latency (50K) | 180ms | Benchmark | VERIFY |
| Survey papers | 52 | Literature review | VERIFY |
| Test coverage | 79% | pytest-cov | VERIFY |

---

## Phase 4: AI Detection Remediation

### Target Patterns to Eliminate:
1. **Filler phrases**: "It is worth noting", "In essence", "Notably"
2. **Hedging**: "may potentially", "could possibly"
3. **AI superlatives**: "remarkable", "significant", "crucial"
4. **Passive constructions**: Convert to active voice
5. **Contractions**: Expand where academic (don't → do not)
6. **Perfect grammar**: Introduce acceptable variations
7. **Sentence uniformity**: Vary length and structure

### Verification Method:
- GPTZero analysis
- Originality.ai check
- Manual linguistic review
- Target: <10% AI probability

---

## Phase 5: Final Polish

### IEEE Formatting:
- [ ] Author affiliations complete
- [ ] ORCID verified
- [ ] Keywords optimized
- [ ] Abstract word count (150-250)
- [ ] References IEEE style
- [ ] Figure captions complete
- [ ] Table formatting consistent

### Philosophy Paper Formatting:
- [ ] APA/Chicago style citations
- [ ] Proper subsection hierarchy
- [ ] Abstract structured
- [ ] Keywords for philosophy venues

---

## Checkpoint Verification

### After Each Phase:
```bash
# Compile papers
cd /mnt/projects/t4d/t4dm/docs && pdflatex -interaction=nonstopmode world_weaver_final.tex
cd /mnt/projects/t4d/t4dm/docs/papers && pdflatex philosophy_of_ai_memory.tex

# Run tests
cd /mnt/projects/ww && pytest tests/ -v --tb=short

# Check AI detection (manual)
# Upload to GPTZero/Originality.ai
```

---

## Execution Timeline

| Phase | Task | Checkpoint |
|-------|------|------------|
| 1 | Codebase cleanup | Files reduced to essential |
| 2 | Citation verification | All 84 citations validated |
| 3 | Statistical validation | All numbers confirmed |
| 4 | AI remediation | <10% detection |
| 5 | Final polish | Submission-ready |
| 6 | Archive | Package for repository |

---

## Notes
- Hardware: archimedes (i9, 128GB RAM, 4TB, RTX 3090)
- All statistics must trace to reproducible experiments
- Zero tolerance for fabricated or unverified claims
