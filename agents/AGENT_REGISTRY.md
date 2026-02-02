# WW Specialized Bug Hunting Agent Registry

Unified registry for bio-inspired memory system debugging agents, grounded in Hinton's learning theory and neuroscience principles.

## Agent Inventory

| Agent | File | Purpose | Primary Bug Types |
|-------|------|---------|-------------------|
| Bio-Memory Auditor | `bio-memory-auditor.md` | Validate biological plausibility | CLS violations, plasticity bugs, neuromodulator errors |
| Race Condition Hunter | `race-condition-hunter.md` | Detect concurrency bugs | TOCTOU, deadlocks, async state corruption |
| Memory Leak Hunter | `memory-leak-hunter.md` | Find resource exhaustion | Unbounded growth, circular refs, cache leaks |
| Hinton Learning Validator | `hinton-learning-validator.md` | Validate learning algorithms | FF violations, backprop misuse, locality bugs |
| Cache Coherence Analyzer | `cache-coherence-analyzer.md` | Analyze cache correctness | Staleness, stampede, poisoning, invalidation |
| Eligibility Trace Debugger | `eligibility-trace-debugger.md` | Debug temporal credit | Decay bugs, accumulation errors, trace misuse |

## Quick Reference: When to Use Each Agent

```
User reports: "Learning isn't improving"
  → Run: Bio-Memory Auditor, Hinton Learning Validator, Eligibility Trace Debugger

User reports: "Memory usage keeps growing"
  → Run: Memory Leak Hunter, Cache Coherence Analyzer

User reports: "Intermittent failures/corruption"
  → Run: Race Condition Hunter, Cache Coherence Analyzer

User reports: "Consolidation not working"
  → Run: Bio-Memory Auditor, Eligibility Trace Debugger

User reports: "Neuromodulators not affecting learning"
  → Run: Bio-Memory Auditor, Hinton Learning Validator
```

## Unified Audit Protocol

### Phase 1: Structural Audit
```bash
# Run all agents in parallel on core learning code
paths=(
  "src/t4dm/learning/"
  "src/t4dm/memory/"
  "src/t4dm/consolidation/"
)
```

**Agents**: Bio-Memory Auditor, Hinton Learning Validator

### Phase 2: Concurrency Audit
```bash
# Focus on async and shared state
paths=(
  "src/t4dm/mcp/"
  "src/t4dm/core/"
  "src/t4dm/storage/"
)
```

**Agents**: Race Condition Hunter, Memory Leak Hunter

### Phase 3: Caching Audit
```bash
# Focus on caching layers
paths=(
  "src/t4dm/indexes/"
  "src/t4dm/storage/"
  "src/t4dm/core/"
)
```

**Agents**: Cache Coherence Analyzer, Memory Leak Hunter

### Phase 4: Learning Dynamics Audit
```bash
# Focus on temporal learning
paths=(
  "src/t4dm/learning/eligibility_traces.py"
  "src/t4dm/learning/neuromodulators.py"
  "src/t4dm/learning/hebbian.py"
  "src/t4dm/core/learned_gate.py"
)
```

**Agents**: Eligibility Trace Debugger, Hinton Learning Validator, Bio-Memory Auditor

## Common Bug Signatures

### Critical (P0) - System Broken

| Signature | Agent | Bug Pattern |
|-----------|-------|-------------|
| `method() is not defined` | Bio-Memory | Missing strengthen_relationship |
| `return 0.0` in neuromodulator | Bio-Memory | Hardcoded neuromodulator output |
| `+= 1` without decay | Trace Debugger | Trace never decays |
| `self.state =` in async | Race Hunter | Shared state mutation |
| `_cache = {}` without maxsize | Leak Hunter | Unbounded cache |

### High (P1) - Silent Data Corruption

| Signature | Agent | Bug Pattern |
|-----------|-------|-------------|
| `* decay` after `+= 1` | Trace Debugger | Wrong decay order |
| `if key in dict:` then `dict[key]` | Race Hunter | TOCTOU race |
| `cache[key] = None` in except | Cache Analyzer | Cache poisoning |
| `.backward()` in local learning | Hinton Validator | Backprop in FF context |
| Same LR for episodic/semantic | Bio-Memory | CLS violation |

### Medium (P2) - Performance Issues

| Signature | Agent | Bug Pattern |
|-----------|-------|-------------|
| `.append()` without trim | Leak Hunter | Unbounded list growth |
| No lock on lazy init | Race Hunter | Double initialization |
| TTL without jitter | Cache Analyzer | Cache stampede |
| Dense activations (ReLU) | Hinton Validator | Missing sparsity |

## Integration with Task Tool

### Claude Code Task Agent Types

Add these to your agent definitions:

```yaml
ww-bio-auditor:
  description: "Audit biological memory implementations against neuroscience principles"
  tools: [Read, Grep, Glob, Write]
  prompt_file: "/mnt/projects/t4d/t4dm/agents/bio-memory-auditor.md"

ww-race-hunter:
  description: "Hunt race conditions and concurrency bugs in async code"
  tools: [Read, Grep, Glob, Write, Bash]
  prompt_file: "/mnt/projects/t4d/t4dm/agents/race-condition-hunter.md"

ww-leak-hunter:
  description: "Find memory leaks and unbounded resource growth"
  tools: [Read, Grep, Glob, Write, Bash]
  prompt_file: "/mnt/projects/t4d/t4dm/agents/memory-leak-hunter.md"

ww-hinton-validator:
  description: "Validate learning against Hinton's principles"
  tools: [Read, Grep, Glob, Write]
  prompt_file: "/mnt/projects/t4d/t4dm/agents/hinton-learning-validator.md"

ww-cache-analyzer:
  description: "Analyze cache coherence and invalidation"
  tools: [Read, Grep, Glob, Write]
  prompt_file: "/mnt/projects/t4d/t4dm/agents/cache-coherence-analyzer.md"

ww-trace-debugger:
  description: "Debug eligibility trace implementations"
  tools: [Read, Grep, Glob, Write]
  prompt_file: "/mnt/projects/t4d/t4dm/agents/eligibility-trace-debugger.md"
```

## Full System Audit Command

```
Conduct comprehensive WW bug audit using all specialized agents:

1. Bio-Memory Auditor on src/t4dm/learning/, src/t4dm/memory/, src/t4dm/consolidation/
2. Race Condition Hunter on src/t4dm/mcp/, src/t4dm/core/
3. Memory Leak Hunter on all Python files
4. Hinton Learning Validator on src/t4dm/learning/
5. Cache Coherence Analyzer on src/t4dm/indexes/, src/t4dm/storage/
6. Eligibility Trace Debugger on src/t4dm/learning/eligibility_traces.py

Generate reports to /home/aaron/mem/T4DM_AUDIT_{agent}_{timestamp}.md
Consolidate findings to /home/aaron/mem/T4DM_MASTER_AUDIT.md
```

## Report Output Locations

All agents write reports to `/home/aaron/mem/`:

```
/home/aaron/mem/
├── BIO_AUDIT_{filename}.md
├── RACE_AUDIT_{filename}.md
├── LEAK_AUDIT_{filename}.md
├── HINTON_VALIDATION_{filename}.md
├── CACHE_AUDIT_{filename}.md
├── TRACE_DEBUG_{filename}.md
└── T4DM_MASTER_AUDIT.md (consolidated)
```

## Theoretical Foundations

### Hinton's Core Principles (All Agents Reference)

1. **Forward-Forward Algorithm**: Layer-local learning via goodness G(h) = Σh² - θ
2. **GLOM**: Part-whole hierarchies through islands of agreement
3. **Modern Hopfield**: Exponential capacity associative memory
4. **Biological Plausibility**: Local learning rules, no weight transport

### Neuroscience Foundations

1. **Complementary Learning Systems**: Fast episodic + slow semantic stores
2. **Three-Factor Learning**: ΔW = η × activity × eligibility × neuromodulator
3. **Eligibility Traces**: TD(λ) temporal credit assignment
4. **Neuromodulation**: DA (reward), NE (novelty), ACh (encoding), 5-HT (patience)

### Key Equations

```
Eligibility Trace:
e(t) = γλ × e(t-1) + ∇θ log π(a|s)

Three-Factor Update:
ΔW = η × pre × post × neuromod × eligibility

Hopfield Energy:
E = -Σᵢ softmax(βXᵀξ)ᵢ xᵢ

CLS Learning Rate Ratio:
episodic_lr / semantic_lr ≥ 10 (typically 100x)
```

## Maintenance

### Adding New Agents

1. Create `/mnt/projects/t4d/t4dm/agents/{agent-name}.md`
2. Follow template: Identity → Mission → Bug Patterns → Checklist → Audit Commands → Report Format
3. Add to this registry
4. Add Task agent type definition

### Updating Bug Patterns

When new bugs discovered:
1. Add pattern to relevant agent's Bug Patterns section
2. Update Detection Checklist
3. Add to Common Bug Signatures table above
4. Update Audit Commands if needed

---

*Registry Version: 1.0.0*
*Last Updated: 2025-12-08*
*Agents: 6 | Bug Patterns: 60+ | Checklists: 30+*
