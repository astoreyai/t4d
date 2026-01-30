# World Weaver Paper Generation Workflow

**Document**: Comprehensive workflow for generating publication-ready academic papers
**Source**: world_weaver_final.tex generation process
**Date**: 2024-12-04

---

## Overview

This workflow transforms draft papers into publication-ready manuscripts through systematic review, revision, and quality assurance. It was developed during the creation of `world_weaver_final.tex` and is designed to be reproducible using research-assistant agents.

---

## Phase 1: Source Material Preparation

### Inputs
- Draft paper(s) in LaTeX format
- Any supplementary documentation
- Target venue requirements (page limit, format, scope)

### Actions
1. Identify all source documents
2. Extract key content from each source
3. Determine target format and constraints
4. Create consolidated working document

### Outputs
- Primary working document
- Content inventory
- Target specifications

---

## Phase 2: Literature Review

### Agent: `research-assistant:literature-reviewer`

### Actions
1. **Systematic Search**: Query databases (OpenAlex, arXiv, PubMed) for related work
2. **Screening**: Apply inclusion/exclusion criteria
3. **Data Extraction**: Extract key findings, methods, claims from each paper
4. **Gap Analysis**: Identify what existing work doesn't address
5. **Citation Verification**: Verify all citations are accurate and current

### Outputs
- `literature_review_results.md` - Comprehensive review findings
- `research_gaps_analysis.md` - Identified gaps and opportunities
- `citation_verification_report.md` - Citation accuracy report
- Updated bibliography with 50+ verified references

### Quality Criteria
- Minimum 50 papers reviewed
- Coverage of last 5 years (2020-2024)
- All major related work cited
- No retracted or superseded citations

---

## Phase 3: Expert Reviews (6 Specialized Perspectives)

### Agent: `research-assistant:quality-assurance` (orchestrating multiple perspectives)

### Review 1: Domain Theory Review
**Focus**: Theoretical foundations and claims accuracy
- Are theoretical claims accurate?
- Are foundational citations correct?
- Is the theoretical framing appropriate?
- Score: X/10 with specific issues

### Review 2: Technical Accuracy Review
**Focus**: Methods, algorithms, and technical claims
- Are algorithms correctly described?
- Are complexity claims accurate?
- Are implementation details sufficient?
- Score: X/10 with specific issues

### Review 3: Empirical Rigor Review
**Focus**: Experimental design and statistical validity
- Are experiments well-designed?
- Are statistics appropriate?
- Are results properly interpreted?
- Score: X/10 with specific issues

### Review 4: Critical Perspective Review
**Focus**: Philosophical tensions and limitations
- Are limitations honestly addressed?
- Are alternative viewpoints acknowledged?
- Is the work appropriately positioned?
- Score: X/10 with specific issues

### Review 5: AI Detection Review
**Skill**: `research-assistant:ai-check`
**Focus**: AI-generated text patterns
- List parallelism issues
- AI-typical vocabulary
- Sentence structure uniformity
- Risk score: X/10 with specific fixes

### Review 6: Journal Editor Review
**Focus**: Publication readiness
- Format compliance
- Completeness (figures, tables, references)
- Clarity and organization
- Decision: Accept/Minor/Major/Reject with required changes

### Outputs
- `review_[perspective].md` for each review
- `revision_synthesis.md` - Consolidated prioritized action items

---

## Phase 4: Revision Synthesis

### Actions
1. Collect all review findings
2. Categorize issues by priority:
   - **Priority 1 (Critical)**: Must fix for acceptance
   - **Priority 2 (Important)**: Should fix for quality
   - **Priority 3 (Recommended)**: Nice to have
3. Identify cross-cutting themes
4. Create actionable revision checklist

### Outputs
- `revision_synthesis_final.md` with:
  - Executive summary
  - Scores from all reviewers
  - Priority 1 issues with specific fixes
  - Priority 2 issues with specific fixes
  - Missing citations to add
  - Expected outcomes after revision

---

## Phase 5: Paper Revision

### Agent: `research-assistant:manuscript-writer`

### Priority 1 Fixes (Critical)
1. **Add Figures**: Create publication-quality diagrams
   - Architecture overview
   - Process flow diagrams
   - Results visualizations

2. **Statistical Rigor**: Add to all tables
   - Error bars (95% CI)
   - Sample sizes (n=X)
   - Significance tests (p-values)
   - Effect sizes where appropriate

3. **Implementation Details**: Add reproducibility section
   - Model specifications
   - Hyperparameters table
   - Computational requirements
   - Code availability statement

4. **Fix AI-Typical Patterns**: Per ai-check findings
   - Break list parallelism
   - Replace AI-typical vocabulary
   - Vary sentence structure

### Priority 2 Fixes (Important)
5. **Theoretical Additions**: Address reviewer concerns
   - Add missing theoretical concepts
   - Clarify terminology
   - Strengthen citations

6. **Expand Weak Sections**: Based on editor feedback
   - Add case studies
   - Expand methodology
   - Deepen discussion

### Priority 3 Fixes (Recommended)
7. **Polish**: Final improvements
   - Tighten prose
   - Improve transitions
   - Ensure consistency

### Outputs
- Revised paper meeting all Priority 1 requirements
- Most Priority 2 requirements addressed
- Clean LaTeX with no compilation errors

---

## Phase 6: Quality Assurance

### Agent: `research-assistant:quality-assurance`

### Checks
1. **Compilation**: Paper compiles without errors
2. **Page Count**: Within target range
3. **Reference Count**: Meets minimum (50+)
4. **Figure/Table Count**: Appropriate for venue
5. **AI Detection Re-check**: Risk reduced to acceptable level
6. **Citation Format**: Correct for venue (IEEE, APA, etc.)
7. **Reproducibility**: Implementation details sufficient

### Skill: `research-assistant:citation-format`
- Verify all citations properly formatted
- Check for missing required fields
- Ensure consistency

### Outputs
- `quality_assurance_report.md`
- Final compiled PDF
- Verification checklist

---

## Phase 7: Folder Organization

### Actions
1. Archive source materials
2. Archive review documents
3. Keep only final deliverables at top level
4. Clean up auxiliary files

### Final Structure
```
docs/
├── [paper_name].tex          # Final source
├── [paper_name].pdf          # Final PDF
├── papers/                   # Additional papers
└── archive/
    ├── source_papers/        # Original versions
    ├── reviews/              # Expert reviews
    └── research/             # Literature review docs
```

---

## Workflow Execution Summary

| Phase | Agent/Skill | Key Output |
|-------|-------------|------------|
| 1. Preparation | Manual | Content inventory |
| 2. Literature | literature-reviewer | 50+ citations |
| 3. Reviews | quality-assurance | 6 review reports |
| 4. Synthesis | Manual | Prioritized fixes |
| 5. Revision | manuscript-writer | Revised paper |
| 6. QA | quality-assurance, ai-check | Final verification |
| 7. Organization | Manual | Clean folder |

---

## Metrics for Success

- [ ] All Priority 1 issues resolved
- [ ] 80%+ Priority 2 issues resolved
- [ ] Page count within target range
- [ ] 50+ properly formatted references
- [ ] AI detection risk < 5/10
- [ ] All figures/tables present
- [ ] Compiles without errors
- [ ] Reproducibility details complete

---

## Application to New Papers

To apply this workflow to a new paper:

```bash
# 1. Place draft in docs/papers/
# 2. Run literature review agent
# 3. Run quality-assurance with 6 perspectives
# 4. Create revision synthesis
# 5. Apply revisions via manuscript-writer
# 6. Run final QA checks
# 7. Organize outputs
```

---

**End of Workflow Document**
