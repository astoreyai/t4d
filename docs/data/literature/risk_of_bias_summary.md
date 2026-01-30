# Risk of Bias Assessment Summary

## Assessment Methodology

Using adapted criteria for computational research:
- **Methodology Clarity** (0-3): Is the method clearly described and reproducible?
- **Experimental Rigor** (0-3): Are experiments comprehensive with appropriate baselines?
- **Reproducibility** (0-3): Is code/data available? Can results be replicated?
- **Novelty/Contribution** (0-3): Does the work advance the field?

**Total Score**: 0-12 (Minimum 6 for inclusion)

---

## Assessment Results by Focus Area

### Episodic Memory Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| MuRAG (Chen 2022) | 3 | 3 | 2 | 3 | 11 | Low |
| SERAC (Mitchell 2022) | 3 | 2 | 3 | 3 | 11 | Low |
| RECITE (Sun 2022) | 2 | 3 | 2 | 3 | 10 | Low |
| RA-CM3 (Yasunaga 2022) | 3 | 2 | 2 | 3 | 10 | Low |
| TrMRL (Melo 2022) | 2 | 2 | 2 | 2 | 8 | Medium |
| Memory Gym (Pleines 2023) | 3 | 3 | 3 | 2 | 11 | Low |
| Interactive AI RAG (Zhang 2024) | 2 | 2 | 1 | 2 | 7 | Medium |

### Semantic Memory Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| Multi-Modal GNNs (Holzinger 2021) | 3 | 3 | 2 | 3 | 11 | Low |
| KG Alignment (Sun 2020) | 3 | 3 | 3 | 3 | 12 | Low |
| Heterogeneous KG (Li 2021) | 3 | 3 | 2 | 3 | 11 | Low |
| KGNN (Lin 2020) | 3 | 3 | 3 | 3 | 12 | Low |
| Radiology KG (Zhang 2020) | 3 | 2 | 2 | 3 | 10 | Low |
| Vector DB Survey (Pan 2024) | 3 | 2 | N/A | 2 | 7* | Medium |

*Survey papers assessed differently (reproducibility N/A)

### Procedural Memory Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| Integrated HRL (Eppe 2022) | 3 | 3 | 2 | 3 | 11 | Low |
| DRL IRS (Yang 2020) | 3 | 3 | 2 | 2 | 10 | Low |
| IoT Edge DRL (Xiong 2020) | 2 | 3 | 2 | 2 | 9 | Low |
| UAV-MEC DRL (Wang 2021) | 3 | 3 | 2 | 2 | 10 | Low |
| Production RL (Brasoveanu 2021) | 2 | 2 | 2 | 2 | 8 | Medium |

### Consolidation Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| CL Survey (Delange 2021) | 3 | 3 | N/A | 3 | 9* | Low |
| Brain-Inspired Replay (van de Ven 2020) | 3 | 3 | 3 | 3 | 12 | Low |
| Triple-Memory (Wang 2021) | 3 | 3 | 2 | 3 | 11 | Low |
| Hindsight Anchor (Chaudhry 2021) | 3 | 3 | 3 | 3 | 12 | Low |
| Sleep Consolidation (Singh 2022) | 3 | 2 | 2 | 3 | 10 | Low |
| Sleep Catastrophic (Gonzalez 2020) | 3 | 3 | 2 | 3 | 11 | Low |
| Memory Recall (Zhang 2021) | 2 | 3 | 2 | 2 | 9 | Low |
| Offline Replay Review (Roscow 2021) | 3 | 2 | N/A | 2 | 7* | Medium |

### Credit Assignment Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| One-Shot RL (Gallistel 2024) | 3 | 3 | 2 | 3 | 11 | Low |
| Expected Traces (van Hasselt 2021) | 3 | 3 | 2 | 3 | 11 | Low |
| Predecessor Features (Bailey 2022) | 2 | 2 | 2 | 3 | 9 | Low |
| Info-Theoretic CA (Arumugam 2021) | 3 | 2 | 2 | 3 | 10 | Low |

### Neural-Symbolic Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| CoALA (Sumers 2023) | 3 | 1 | 1 | 3 | 8 | Medium |
| MemGPT (Packer 2023) | 3 | 3 | 3 | 3 | 12 | Low |
| Human-like Memory (Hou 2024) | 2 | 2 | 2 | 3 | 9 | Low |
| FinMem (Yu 2024) | 2 | 2 | 2 | 2 | 8 | Medium |
| Cognitive LLMs (Wu 2024) | 2 | 2 | 2 | 3 | 9 | Low |
| RAISE (Liu 2024) | 2 | 2 | 1 | 2 | 7 | Medium |

### Hebbian Learning Papers

| Paper | Methodology | Rigor | Reproducibility | Novelty | Total | RoB |
|-------|-------------|-------|-----------------|---------|-------|-----|
| STDP Robot (Lobov 2020) | 3 | 3 | 2 | 3 | 11 | Low |
| Ferroelectric Synaptic (Yan 2021) | 3 | 3 | 1 | 3 | 10 | Low |
| Active Inference (Isomura 2022) | 3 | 2 | 2 | 3 | 10 | Low |

---

## Summary Statistics

### Overall Risk of Bias Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| Low | 30 | 75% |
| Medium | 10 | 25% |
| High | 0 | 0% |

### Mean Scores by Domain

| Domain | Mean Score | SD |
|--------|------------|-----|
| Episodic Memory | 9.7 | 1.5 |
| Semantic Memory | 10.5 | 1.7 |
| Procedural Memory | 9.6 | 1.1 |
| Consolidation | 10.1 | 1.5 |
| Credit Assignment | 10.3 | 0.8 |
| Neural-Symbolic | 8.8 | 1.7 |
| Hebbian Learning | 10.3 | 0.6 |

### Common Weaknesses

1. **Reproducibility**: Many papers lack code release (especially 2020-2022)
2. **Baselines**: Some domain-specific papers compare only to field-specific methods
3. **Framework papers**: CoALA-style papers lack empirical validation

### Strength of Evidence by Finding

| Finding | Supporting Papers | Evidence Quality |
|---------|-------------------|------------------|
| RAG is dominant paradigm | 7 | Strong (multiple low-RoB) |
| KGNNs effective for semantic | 5 | Strong (all low-RoB, high citations) |
| Replay prevents forgetting | 6 | Strong (Nature Comms, PNAS) |
| Tripartite architectures needed | 3 | Moderate (conceptual papers) |
| Hebbian learning viable | 3 | Moderate (hardware-focused) |
| Credit assignment unsolved | 4 | Strong (consistent findings) |

---

## GRADE Evidence Assessment

### Overall Certainty of Evidence

| Outcome | Certainty | Rationale |
|---------|-----------|-----------|
| RAG effectiveness | High | Multiple RCT-equivalent studies |
| KG neural reasoning | High | Extensive benchmarks |
| Consolidation mechanisms | Moderate | Biological grounding, limited AI implementation |
| Tripartite architecture | Low | Mostly conceptual, limited implementations |
| Credit assignment | Moderate | Theoretical advances, practical gaps |

### Downgrade Factors Applied

- Risk of bias: No downgrades (overall low)
- Inconsistency: No downgrades (findings consistent)
- Indirectness: -1 for tripartite (conceptual only)
- Imprecision: -1 for Hebbian (small sample of hardware papers)
- Publication bias: Cannot assess

---

*Assessment Date: 2025-12-06*
*Reviewer: Literature Review Agent (AUTONOMOUS)*
