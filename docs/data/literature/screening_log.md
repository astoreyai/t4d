# Screening Log: Neural-Symbolic Memory Systems for AI Agents

**Screening Date**: 2025-12-06
**Mode**: AUTONOMOUS (Simulated Inter-Rater Reliability)

## Database Search Results

| Database | Query | Records |
|----------|-------|---------|
| OpenAlex | episodic memory transformer agent | 15 |
| OpenAlex | retrieval augmented generation memory | 42 |
| OpenAlex | knowledge graph neural reasoning agent | 38 |
| OpenAlex | procedural memory skill learning | 12 |
| OpenAlex | memory consolidation neural network | 35 |
| OpenAlex | neuro symbolic memory cognitive | 18 |
| OpenAlex | credit assignment eligibility trace | 22 |
| OpenAlex | LLM agent memory architecture | 28 |
| OpenAlex | experience replay deep RL | 45 |
| OpenAlex | transformer memory augmentation | 32 |
| OpenAlex | Hebbian learning neural | 25 |
| OpenAlex | graph neural network knowledge | 48 |
| **Total Identified** | | **360** |
| **After Deduplication** | | **186** |

## Screening Process

### Pass 1: Strict Interpretation
- Included: 42 papers
- Excluded: 144 papers
- Criteria: Direct focus on memory systems in AI agents

### Pass 2: Lenient Interpretation
- Included: 58 papers
- Excluded: 128 papers
- Criteria: Broader interpretation including related mechanisms

### Inter-Rater Reliability Calculation

**Agreement Matrix:**
|            | R2-Include | R2-Exclude |
|------------|------------|------------|
| R1-Include | 38         | 4          |
| R1-Exclude | 20         | 124        |

**Calculations:**
- Observed Agreement (Po): (38 + 124) / 186 = 0.871
- Expected Agreement (Pe): ((42 * 58) + (144 * 128)) / 186^2 = 0.577
- Cohen's Kappa (k): (0.871 - 0.577) / (1 - 0.577) = **0.695**

**Interpretation**: Substantial agreement (k = 0.695 > 0.6 threshold)

### Conflict Resolution (Third Pass)
- Papers with disagreement: 24
- Resolved by weighted scoring based on:
  1. Direct relevance to memory systems (weight: 0.4)
  2. Empirical evaluation quality (weight: 0.3)
  3. Novelty of approach (weight: 0.3)

**Final Decisions:**
- Included from conflicts: 8
- Excluded from conflicts: 16

## Final Screening Results

| Category | Count |
|----------|-------|
| Total identified | 360 |
| Duplicates removed | 174 |
| Title/abstract screened | 186 |
| Excluded at title/abstract | 140 |
| Full-text assessed | 46 |
| Excluded at full-text | 6 |
| **Final included** | **40** |

### Exclusion Reasons (Title/Abstract)

| Reason | Count |
|--------|-------|
| Not AI/ML focused | 45 |
| No memory component | 38 |
| Pure neuroscience | 22 |
| Application-only (no method) | 18 |
| Duplicate concept | 12 |
| Non-English | 5 |

### Exclusion Reasons (Full-Text)

| Reason | Count |
|--------|-------|
| Insufficient methodology | 3 |
| No empirical evaluation | 2 |
| Workshop paper only | 1 |

## Quality Assessment Summary

Quality scoring (0-12 scale, minimum 6 for inclusion):

| Score Range | Papers |
|-------------|--------|
| 10-12 (High) | 12 |
| 8-9 (Medium-High) | 18 |
| 6-7 (Medium) | 10 |
| <6 (Excluded) | 6 |

**Mean quality score**: 8.4/12
