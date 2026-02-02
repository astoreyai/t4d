# T4DM Paper - Revision Checklist

**Target Journal**: IEEE Transactions on Artificial Intelligence
**Current Status**: MINOR REVISION REQUIRED
**Overall Score**: 7.8/10
**Estimated Revision Time**: 1-2 weeks for critical items

---

## CRITICAL (Must Complete Before Submission)

### 1. Create Actual Figures
- [ ] Figure 1: System architecture diagram showing all three memory stores
  - Show episodic, semantic, procedural memory
  - Illustrate consolidation pathways
  - Label BGE-M3, GLiNER, HDBSCAN components
- [ ] Figure 2: Hybrid retrieval pipeline
  - Show parallel dense/sparse search
  - Illustrate RRF fusion process
  - Include timing/latency annotations

**Estimated Time**: 4-6 hours using draw.io or similar
**Priority**: HIGHEST - reviewers will reject if figures are placeholders

---

### 2. Resolve Expert Review Claims
**Current Problem**: Lines 63 and 630 mention "six specialized expert reviews" but don't include them

**Options** (choose one):
- [ ] **Option A**: Remove all mentions of expert reviews (FASTEST - 5 minutes)
- [ ] **Option B**: Include reviews as supplementary appendix with attribution
- [ ] **Option C**: Convert to acknowledgments ("we thank X reviewers for feedback")

**Recommendation**: Option A - the paper stands on its own merits without these claims
**Priority**: HIGH - could be seen as unsubstantiated claims

---

### 3. Document Experimental Methodology
**Current Problem**: Results are unverifiable without understanding how ground truth was created

Add new subsection under Section 5 (Evaluation):

#### 5.1.1 Evaluation Methodology

- [ ] **Relevance Judgments for Recall@10**
  ```
  Write 2-3 paragraphs describing:
  - How "relevant" memories were identified for each query
  - Who performed judgments (you, independent annotators, both?)
  - Criteria used (exact match, semantic similarity, threshold?)
  - Example: "A memory was judged relevant if it mentioned the same
    function name, file path, or error code as the query"
  ```

- [ ] **Entity Extraction Ground Truth**
  ```
  Describe:
  - How you created gold-standard entity annotations
  - What entities were annotated (functions, classes, files, variables?)
  - Inter-annotator reliability if multiple annotators
  - Sample size (N episodes annotated)
  ```

- [ ] **Human Evaluation Protocol**
  ```
  Clarify:
  - Are "10 software developers" (line 360) same as "10 annotators" (Table 4)?
  - How were participants recruited?
  - What instructions were they given?
  - Example task with screenshot
  ```

- [ ] **Add Example Annotations**
  ```
  Include 2-3 concrete examples:
  - Example query + relevant/non-relevant memories with rationale
  - Example episode with entity annotations
  - Example task evaluation with success criteria
  ```

**Estimated Time**: 3-4 hours to document existing process
**Priority**: CRITICAL - without this, results cannot be verified

---

### 4. Release Code and Data
**Current Problem**: Line 648 promises release "upon acceptance" - outdated practice

**Required Artifacts**:

- [ ] Create GitHub repository (can be anonymous during review)
  - Core memory system code (MCP server)
  - BGE-M3 embedding integration
  - GLiNER entity extraction
  - Consolidation pipeline
  - Database schema SQL

- [ ] Add evaluation scripts
  - Retrieval evaluation (Recall@10 calculation)
  - Behavioral task evaluation
  - Ablation study scripts
  - Statistical testing code

- [ ] Provide sample dataset
  - [ ] **Option A**: 1000 real episodes (anonymized - remove company/personal info)
  - [ ] **Option B**: 1000 synthetic episodes following same distribution
  - Include sample queries with relevance judgments

- [ ] Complete README with:
  - Installation instructions (Docker compose preferred)
  - How to reproduce Table 2 results
  - How to reproduce Table 3 results
  - How to reproduce Table 4 results
  - Expected output format

- [ ] Update paper line 648:
  ```latex
  Code and data are available at: \url{https://github.com/astoreyai/t4dm}
  ```

**Estimated Time**: 1-2 days (code exists, needs cleanup + documentation)
**Priority**: CRITICAL - reviewers may require this per IEEE T-AI policy

---

### 5. Address Single-User Limitation
**Current Problem**: All experiments by one developer - generalizability unclear

**Quick Fix** (choose based on time available):

- [ ] **Option A**: Recruit 3-5 additional users (2-4 weeks)
  - Run same evaluation protocol
  - Report aggregate + per-user results
  - Strengthens paper significantly

- [ ] **Option B**: Reframe as case study (1 hour)
  - Change title: "...Agentic AI: A Case Study"
  - Abstract: "We present a case study of..."
  - Throughout: "In this deployment..." instead of claims about general performance

- [ ] **Option C**: Add strong disclaimers (30 minutes)
  - Abstract: Add sentence about single-user evaluation
  - Introduction: Explicit scope limitation
  - Limitations section: Emphasize generalization unknown
  - Conclusion: Hedge claims appropriately

**Recommendation**: Option C now, Option A for camera-ready if accepted
**Priority**: CRITICAL - major validity threat

**Specific Text Changes**:
```latex
% Abstract - add after line 38:
Evaluation was conducted as a single-user case study; generalization
to other users and domains requires future validation.

% Limitations section (line 494) - strengthen:
\textbf{Single-User Evaluation}: All experiments were conducted by one
developer across personal projects. While results demonstrate feasibility,
we cannot claim these findings generalize to other users, coding styles,
programming languages, or application domains without further validation.
Different users may exhibit different memory patterns, query behaviors,
and task success rates.
```

---

### 6. Clarify Hyperparameter Selection
**Current Problem**: Risk of test-set contamination if hyperparameters were tuned on evaluation queries

Add one clear statement to Section 5.1 or Implementation:

- [ ] Add paragraph:
  ```latex
  \subsection{Hyperparameter Selection}

  All hyperparameters (Table X) were selected [CHOOSE ONE]:
  - based on prior work and theoretical considerations
  - using a separate validation set of 100 queries not included in evaluation
  - through grid search on a held-out validation set

  No hyperparameters were tuned on the evaluation query sets reported
  in Tables 2-4 to avoid test set contamination.
  ```

**Estimated Time**: 15 minutes (if you can honestly make this claim)
**Priority**: HIGH - if you DID tune on test set, this is a major problem

---

## STRONGLY RECOMMENDED (Should Complete)

### 7. Add Statistical Rigor

- [ ] Report effect sizes
  ```latex
  % For each significant result in Tables 2-3, add Cohen's d:
  Hybrid retrieval improved recall for mixed queries
  (0.72 → 0.84, p<0.001, d=3.2, large effect)
  ```

- [ ] Multiple comparison correction
  ```latex
  % Add footnote to Table 2:
  With Bonferroni correction for 4 comparisons (α=0.0125),
  all results except conceptual queries (p=0.042) remain significant.
  ```

- [ ] Acknowledge power analysis limitation
  ```latex
  % In limitations:
  No a priori power analysis was conducted; future work should
  pre-register sample sizes based on minimum detectable effect sizes.
  ```

**Estimated Time**: 2-3 hours for calculations + text updates
**Priority**: HIGH - strengthens statistical claims

---

### 8. MemGPT Comparison or Claim Revision

**Current Problem**: MemGPT is most comparable system but no direct comparison

**Options**:

- [ ] **Option A**: Implement MemGPT baseline (1-2 weeks)
  - Install MemGPT
  - Run same evaluation protocol
  - Add results to tables
  - Add to Table 1 comparison

- [ ] **Option B**: Revise claims to focus on absolute performance (1 hour)
  - Remove comparative language ("better than", "outperforms")
  - Focus on "demonstrates feasibility" rather than "superior to"
  - Acknowledge MemGPT comparison as future work

**Recommendation**: Option B for initial submission, Option A if reviewers request it
**Priority**: MEDIUM-HIGH

---

### 9. Expand Reproducibility Documentation

- [ ] Add appendix with complete database schema
  ```sql
  -- Full PostgreSQL schema
  CREATE TABLE episodes (...);
  CREATE TABLE semantic_entities (...);
  CREATE INDEX ... USING hnsw (...);
  ```

- [ ] Document GLiNER entity types
  ```
  We extract the following entity types:
  - FUNCTION: Python/JavaScript function names
  - CLASS: Class definitions
  - FILE: File paths and names
  - VARIABLE: Variable names in scope
  - ERROR: Error messages and codes
  - CONCEPT: Domain concepts (e.g., "authentication", "database")
  ```

- [ ] Add complete dependency list with versions
  ```
  BGE-M3: BAAI/bge-m3 (version 2024.01)
  GLiNER: urchade/gliner-base (version 0.1.5)
  PostgreSQL: 15.3
  pgvector: 0.5.1
  HNSW: ef_construction=200, M=16
  ```

**Estimated Time**: 3-4 hours
**Priority**: MEDIUM - improves reproducibility

---

## RECOMMENDED FOR STRENGTHENING

### 10. Add Failure Case Examples

Currently failures are described abstractly (lines 437-445). Add concrete examples:

- [ ] False memory retrieval example
  ```
  \textbf{Example}: When debugging JWT validation in Project B,
  the system retrieved an episode about RSA key handling from Project A.
  While both involved authentication, Project B used HMAC signing.
  The retrieved memory's suggestion to "check RSA key permissions"
  was misleading. Root cause: high semantic similarity (0.87)
  between "JWT validation" contexts despite different cryptographic approaches.
  ```

- [ ] Skill overfitting example
- [ ] Temporal confusion example

**Estimated Time**: 2 hours
**Priority**: LOW-MEDIUM - strengthens paper but not critical

---

### 11. Timing Breakdown

Add table breaking down where time is spent:

- [ ] Create performance analysis table
  ```
  Component                  | Latency (10K episodes) | Latency (50K episodes)
  ---------------------------|------------------------|------------------------
  Query embedding (BGE-M3)   | 12ms                   | 12ms
  Dense search (pgvector)    | 28ms                   | 95ms
  Sparse search (inverted)   | 8ms                    | 35ms
  RRF fusion                 | 4ms                    | 12ms
  Context filtering          | 3ms                    | 8ms
  Total                      | 52ms                   | 180ms
  ```

**Estimated Time**: 1 hour if you have profiling data
**Priority**: LOW - nice to have

---

### 12. Sample Episodes in Appendix

Add appendix with real examples:

- [ ] Appendix A: Sample Episodes
  - 3-5 representative episodes with full metadata
  - Show dense embedding (first 10 dimensions)
  - Show sparse embedding (top 20 terms)
  - Show FSRS stability evolution

- [ ] Appendix B: Sample Semantic Entities
  - Example entity graph for one concept
  - Show relationships and activation values

- [ ] Appendix C: Sample Skills
  - 3-5 skills with usefulness trajectories
  - Show how usefulness evolved over time

**Estimated Time**: 2-3 hours
**Priority**: LOW - improves understanding but not critical

---

## Completion Timeline

### Week 1 (Critical Items Only)
- **Day 1-2**: Create figures, document methodology, clarify hyperparameters
- **Day 3-4**: Prepare code repository, anonymize sample data
- **Day 5**: Address single-user limitation (Option C - disclaimers)
- **Day 6**: Remove expert review claims, add statistical notes
- **Day 7**: Final review and submission preparation

### Week 2 (If Adding Recommended Items)
- **Day 8-9**: Add failure examples, timing breakdown
- **Day 10-11**: Expand reproducibility docs, create appendices
- **Day 12-14**: MemGPT comparison study (if choosing this option)

---

## Quality Checklist Before Submission

- [ ] All figures are present and professional quality
- [ ] No placeholder text remains (search for "[FIGURE")
- [ ] Code repository is public and includes README
- [ ] Sample data is available (at least 1000 episodes)
- [ ] Experimental methodology is documented
- [ ] Single-user limitation is acknowledged prominently
- [ ] Expert review claims removed or substantiated
- [ ] Hyperparameter selection process stated
- [ ] Statistical claims are accurate (effect sizes, corrections)
- [ ] All references are complete and formatted correctly
- [ ] Supplementary materials are prepared
- [ ] Paper compiles without errors
- [ ] Abstract accurately reflects limitations
- [ ] Title is accurate (add "A Case Study" if appropriate)

---

## Expected Review Outcome

**With all CRITICAL items completed**:
- **Most Likely**: MINOR REVISION (2-3 month review cycle)
- **Best Case**: ACCEPT with minor changes
- **Worst Case**: MAJOR REVISION if reviewers want multi-user validation

**Without CRITICAL items**:
- **Most Likely**: REJECT or MAJOR REVISION
- Reviewers will likely request exactly these items

---

## Contact for Questions

Issues? Uncertainties? Consider:
1. Check IEEE T-AI author guidelines: https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9078688
2. Review similar papers in IEEE T-AI for formatting standards
3. Consult with advisor/colleagues on methodology documentation

---

**Document Version**: 1.0
**Created**: 2025-12-04
**Based on**: Comprehensive quality assessment of t4dm_final.tex
