# Bioinspired Tools Design

## Overview

This document identifies tool gaps in World Weaver for bioinspired neural memory functionality and proposes new tools to fill those gaps across MCP operations, development, debugging, and analysis.

---

## 1. Gap Analysis

### 1.1 Current Tool Inventory

| Category | Existing Tools | Coverage |
|----------|----------------|----------|
| Memory Operations | `remember`, `recall`, `forget`, `consolidate` | 70% |
| Graph Operations | `entity_create`, `relation_add`, `query_graph` | 80% |
| Configuration | `config_get`, `config_set` | 60% |
| Monitoring | `status`, `metrics_get` | 40% |
| Learning | `reward`, `feedback` | 50% |
| Debugging | None | 0% |

### 1.2 Identified Gaps

```
┌────────────────────────────────────────────────────────────────┐
│                     TOOL GAP ANALYSIS                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ❌ No sparse encoding inspection tools                        │
│  ❌ No attractor network debugging                             │
│  ❌ No eligibility trace visualization                         │
│  ❌ No neuromodulator manual override                          │
│  ❌ No pattern capacity analysis                               │
│  ❌ No biological validation tooling                           │
│  ❌ No fast episodic store management                          │
│  ❌ No consolidation pipeline control                          │
│  ❌ No learning rate experimentation                           │
│  ❌ No dendritic compartment inspection                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. New MCP Tools

### 2.1 Sparse Encoding Tools

#### `bio_encode` - Encode content with sparse encoder

```python
@mcp.tool()
async def bio_encode(
    content: str,
    encoding_type: Literal["default", "hippocampal", "cortical"] = "default",
    return_visualization: bool = False
) -> Dict[str, Any]:
    """
    Encode content using bioinspired sparse encoder.

    Args:
        content: Text content to encode
        encoding_type: Encoder variant (different sparsity targets)
        return_visualization: Include ASCII visualization of pattern

    Returns:
        {
            "encoding": List[float],  # Sparse vector
            "sparsity": float,        # Actual sparsity achieved
            "active_dims": List[int], # Non-zero dimension indices
            "biological_valid": bool, # Within target range
            "visualization": str      # ASCII pattern (if requested)
        }
    """
    encoder = get_encoder(encoding_type)
    embedding = await embed_text(content)
    sparse = encoder.encode(embedding)

    result = {
        "encoding": sparse.tolist(),
        "sparsity": (sparse != 0).float().mean().item(),
        "active_dims": torch.nonzero(sparse).flatten().tolist(),
        "biological_valid": 0.01 < result["sparsity"] < 0.05
    }

    if return_visualization:
        result["visualization"] = visualize_sparse(sparse)

    return result
```

#### `bio_decode` - Decode sparse pattern back to content

```python
@mcp.tool()
async def bio_decode(
    encoding: List[float],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Attempt to decode sparse encoding back to semantic content.

    Returns nearest neighbors in embedding space.
    """
    sparse = torch.tensor(encoding)
    reconstructed = decoder.decode(sparse)

    neighbors = await find_nearest_embeddings(reconstructed, k=top_k)

    return {
        "reconstruction_quality": compute_quality(sparse, reconstructed),
        "nearest_content": [n.content for n in neighbors],
        "confidence_scores": [n.score for n in neighbors]
    }
```

### 2.2 Attractor Network Tools

#### `bio_attractor_store` - Store pattern in attractor network

```python
@mcp.tool()
async def bio_attractor_store(
    pattern_id: str,
    encoding: Optional[List[float]] = None,
    content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store pattern as attractor in Hopfield-style network.

    Args:
        pattern_id: Unique identifier for pattern
        encoding: Pre-computed sparse encoding (optional)
        content: Raw content to encode (if no encoding provided)

    Returns:
        {
            "stored": bool,
            "capacity_usage": float,
            "pattern_count": int,
            "overlap_with_existing": float
        }
    """
    if encoding is None and content:
        encoding = await bio_encode(content)["encoding"]

    network = get_attractor_network()
    result = network.store(pattern_id, torch.tensor(encoding))

    return {
        "stored": result.success,
        "capacity_usage": network.usage_ratio,
        "pattern_count": network.pattern_count,
        "overlap_with_existing": result.max_overlap
    }
```

#### `bio_attractor_retrieve` - Retrieve pattern with settling dynamics

```python
@mcp.tool()
async def bio_attractor_retrieve(
    cue: Union[List[float], str],
    max_steps: int = 100,
    noise_level: float = 0.0,
    return_trajectory: bool = False
) -> Dict[str, Any]:
    """
    Retrieve pattern from attractor network with optional noise.

    Args:
        cue: Partial/noisy pattern or content string
        max_steps: Maximum settling iterations
        noise_level: Gaussian noise to add to cue (for robustness testing)
        return_trajectory: Include energy trajectory

    Returns:
        {
            "retrieved_pattern": List[float],
            "matched_pattern_id": str,
            "confidence": float,
            "settling_steps": int,
            "final_energy": float,
            "trajectory": List[Dict] (if requested)
        }
    """
    if isinstance(cue, str):
        cue = await bio_encode(cue)["encoding"]

    cue_tensor = torch.tensor(cue)
    if noise_level > 0:
        cue_tensor += noise_level * torch.randn_like(cue_tensor)

    network = get_attractor_network()
    result = network.retrieve(cue_tensor, max_steps=max_steps, track=return_trajectory)

    return {
        "retrieved_pattern": result.pattern.tolist(),
        "matched_pattern_id": result.pattern_id,
        "confidence": result.confidence,
        "settling_steps": result.steps,
        "final_energy": result.energy,
        "trajectory": result.trajectory if return_trajectory else None
    }
```

#### `bio_attractor_analyze` - Analyze attractor network state

```python
@mcp.tool()
async def bio_attractor_analyze(
    include_energy_landscape: bool = False,
    sample_resolution: int = 100
) -> Dict[str, Any]:
    """
    Analyze current state of attractor network.

    Returns capacity analysis, pattern statistics, and optionally
    a sampled energy landscape for visualization.
    """
    network = get_attractor_network()

    analysis = {
        "pattern_count": network.pattern_count,
        "theoretical_capacity": int(0.14 * network.dim),
        "capacity_usage": network.pattern_count / (0.14 * network.dim),
        "average_pattern_overlap": network.compute_avg_overlap(),
        "min_basin_width": network.estimate_min_basin(),
        "weight_matrix_rank": network.weight_rank(),
        "spectral_gap": network.compute_spectral_gap()
    }

    if include_energy_landscape:
        analysis["energy_landscape"] = network.sample_landscape(sample_resolution)

    return analysis
```

### 2.3 Neuromodulator Control Tools

#### `bio_neuromod_set` - Set neuromodulator levels

```python
@mcp.tool()
async def bio_neuromod_set(
    dopamine: Optional[float] = None,
    norepinephrine: Optional[float] = None,
    acetylcholine: Optional[float] = None,
    serotonin: Optional[float] = None,
    gaba: Optional[float] = None,
    duration_ms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Manually set neuromodulator levels (for testing/debugging).

    Args:
        dopamine: DA level (0-1), affects reward learning
        norepinephrine: NE level (0-1), affects attention
        acetylcholine: ACh level (0-1), affects encoding
        serotonin: 5-HT level (0-1), affects mood/stability
        gaba: GABA level (0-1), affects inhibition
        duration_ms: How long to maintain levels (None = permanent)

    Returns:
        Previous and new levels, effective learning rate
    """
    orchestra = get_neuromodulator_orchestra()

    previous = orchestra.get_levels()

    if dopamine is not None:
        orchestra.set_dopamine(dopamine, duration_ms)
    if norepinephrine is not None:
        orchestra.set_norepinephrine(norepinephrine, duration_ms)
    # ... etc

    current = orchestra.get_levels()

    return {
        "previous": previous,
        "current": current,
        "effective_learning_rate": orchestra.compute_effective_lr(),
        "learning_rate_ratio": orchestra.compute_lr_ratio()
    }
```

#### `bio_neuromod_simulate` - Simulate neuromodulator response

```python
@mcp.tool()
async def bio_neuromod_simulate(
    event_type: Literal["reward", "punishment", "novelty", "stress", "rest"],
    intensity: float = 1.0,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simulate neuromodulator response to event type.

    Args:
        event_type: Type of event to simulate
        intensity: Event intensity (0-2)
        context: Additional context affecting response

    Returns:
        Neuromodulator trajectory over time
    """
    orchestra = get_neuromodulator_orchestra()

    trajectory = orchestra.simulate_response(
        event_type=event_type,
        intensity=intensity,
        context=context,
        duration_steps=100
    )

    return {
        "trajectory": trajectory,
        "peak_da": max(t["dopamine"] for t in trajectory),
        "peak_ne": max(t["norepinephrine"] for t in trajectory),
        "settling_time": find_settling_time(trajectory),
        "effective_lr_range": [
            min(t["effective_lr"] for t in trajectory),
            max(t["effective_lr"] for t in trajectory)
        ]
    }
```

### 2.4 Eligibility Trace Tools

#### `bio_trace_inspect` - Inspect current eligibility traces

```python
@mcp.tool()
async def bio_trace_inspect(
    memory_id: Optional[str] = None,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Inspect active eligibility traces.

    Args:
        memory_id: Specific memory to inspect (None = all active)
        threshold: Minimum trace magnitude to include

    Returns:
        Active traces with magnitudes and ages
    """
    tracer = get_eligibility_tracer()

    if memory_id:
        traces = [tracer.get_trace(memory_id)]
    else:
        traces = tracer.get_all_active(threshold=threshold)

    return {
        "active_count": len(traces),
        "traces": [
            {
                "memory_id": t.memory_id,
                "magnitude": t.magnitude,
                "age_steps": t.age,
                "decay_rate": t.decay,
                "estimated_remaining_steps": t.estimate_remaining(threshold)
            }
            for t in traces
        ],
        "total_trace_magnitude": sum(t.magnitude for t in traces)
    }
```

#### `bio_trace_assign_credit` - Manually assign credit

```python
@mcp.tool()
async def bio_trace_assign_credit(
    reward: float,
    decay_factor: Optional[float] = None
) -> Dict[str, Any]:
    """
    Assign credit to active eligibility traces.

    Args:
        reward: Reward signal to propagate
        decay_factor: Override default decay (None = use trace's decay)

    Returns:
        Credit assigned to each active trace
    """
    tracer = get_eligibility_tracer()

    assignments = tracer.assign_credit(reward, decay_factor)

    return {
        "total_credit_assigned": sum(a.credit for a in assignments),
        "assignments": [
            {
                "memory_id": a.memory_id,
                "trace_magnitude": a.trace_magnitude,
                "credit": a.credit,
                "weight_update": a.weight_delta
            }
            for a in assignments
        ]
    }
```

### 2.5 Fast Episodic Store Tools

#### `bio_episodic_store` - Store in fast episodic memory

```python
@mcp.tool()
async def bio_episodic_store(
    content: str,
    context: Optional[Dict[str, Any]] = None,
    importance: float = 1.0
) -> Dict[str, Any]:
    """
    Store episode in fast (hippocampal-like) memory.

    One-shot learning for rapid acquisition.

    Args:
        content: Episode content
        context: Contextual information
        importance: Importance weight (affects consolidation priority)

    Returns:
        Storage confirmation with capacity info
    """
    store = get_fast_episodic_store()

    episode = Episode(
        content=content,
        context=context or {},
        importance=importance,
        timestamp=time.time()
    )

    result = store.write(episode)

    return {
        "episode_id": result.episode_id,
        "stored": result.success,
        "encoding_sparsity": result.sparsity,
        "capacity_usage": store.usage_ratio,
        "consolidation_eligible": episode.importance >= store.consolidation_threshold
    }
```

#### `bio_episodic_consolidate` - Trigger memory consolidation

```python
@mcp.tool()
async def bio_episodic_consolidate(
    episode_ids: Optional[List[str]] = None,
    auto_select: bool = True,
    max_episodes: int = 10
) -> Dict[str, Any]:
    """
    Consolidate fast episodic memories to semantic store.

    Args:
        episode_ids: Specific episodes to consolidate (optional)
        auto_select: Auto-select based on replay frequency
        max_episodes: Maximum episodes to consolidate per call

    Returns:
        Consolidation results
    """
    store = get_fast_episodic_store()
    semantic = get_semantic_store()

    if episode_ids is None and auto_select:
        candidates = store.get_consolidation_candidates(max_episodes)
        episode_ids = [c.episode_id for c in candidates]

    results = []
    for eid in episode_ids:
        episode = store.get(eid)
        result = semantic.absorb(episode)
        if result.success:
            store.mark_consolidated(eid)
        results.append({
            "episode_id": eid,
            "consolidated": result.success,
            "semantic_location": result.node_id if result.success else None
        })

    return {
        "consolidated_count": sum(1 for r in results if r["consolidated"]),
        "results": results,
        "remaining_capacity": store.remaining_capacity
    }
```

### 2.6 Dendritic Inspection Tools

#### `bio_dendritic_inspect` - Inspect dendritic compartments

```python
@mcp.tool()
async def bio_dendritic_inspect(
    layer: Optional[str] = None,
    neuron_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Inspect dendritic neuron compartment states.

    Args:
        layer: Specific layer to inspect
        neuron_indices: Specific neurons (None = summary stats)

    Returns:
        Compartment states, coupling strengths, membrane potentials
    """
    network = get_dendritic_network()

    if layer:
        neurons = network.get_layer(layer)
    else:
        neurons = network.get_all_neurons()

    if neuron_indices:
        neurons = [neurons[i] for i in neuron_indices]

    return {
        "neuron_count": len(neurons),
        "compartment_stats": {
            "basal": {
                "mean_potential": np.mean([n.basal_potential for n in neurons]),
                "std_potential": np.std([n.basal_potential for n in neurons])
            },
            "apical": {
                "mean_potential": np.mean([n.apical_potential for n in neurons]),
                "std_potential": np.std([n.apical_potential for n in neurons])
            }
        },
        "coupling_strength": {
            "mean": np.mean([n.coupling for n in neurons]),
            "range": [min(n.coupling for n in neurons), max(n.coupling for n in neurons)]
        },
        "context_influence": compute_context_influence(neurons)
    }
```

### 2.7 Biological Validation Tools

#### `bio_validate` - Run biological validation suite

```python
@mcp.tool()
async def bio_validate(
    checks: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run biological plausibility validation checks.

    Args:
        checks: Specific checks to run (None = all)
        verbose: Include detailed results

    Returns:
        Validation results with pass/fail status
    """
    validator = BiologicalValidator()

    all_checks = [
        "sparsity_range",
        "learning_rate_separation",
        "attractor_capacity",
        "eligibility_decay",
        "neuromodulator_bounds",
        "pattern_orthogonality",
        "consolidation_threshold"
    ]

    checks_to_run = checks or all_checks

    results = {}
    for check in checks_to_run:
        result = validator.run_check(check)
        results[check] = {
            "passed": result.passed,
            "expected": result.expected_range,
            "actual": result.actual_value,
            "details": result.details if verbose else None
        }

    return {
        "overall_valid": all(r["passed"] for r in results.values()),
        "passed_count": sum(1 for r in results.values() if r["passed"]),
        "total_checks": len(results),
        "results": results
    }
```

---

## 3. Development & Debugging Tools

### 3.1 CLI Tools

```bash
# ww-bio-cli - Bioinspired debugging CLI

# Inspect sparse encoding
ww-bio encode "test content" --type hippocampal --visualize

# Analyze attractor network
ww-bio attractor analyze --energy-landscape

# Set neuromodulator state
ww-bio neuromod set --da 0.8 --ne 0.5 --duration 1000

# Run validation
ww-bio validate --checks all --verbose

# Trace inspection
ww-bio trace list --threshold 0.01

# Fast episodic management
ww-bio episodic consolidate --auto --max 10
```

### 3.2 Python Debug Utilities

```python
# src/bioinspired/debug.py

class BioinspiredDebugger:
    """Interactive debugging utilities for bioinspired components."""

    def __init__(self, system):
        self.system = system
        self.history = []

    def snapshot(self) -> Dict[str, Any]:
        """Capture full system state snapshot."""
        return {
            "timestamp": time.time(),
            "sparse_encoder": self.system.encoder.state_dict(),
            "attractor": self.system.attractor.state_dict(),
            "neuromod": self.system.neuromod.get_levels(),
            "traces": self.system.tracer.get_all_active(),
            "episodic_usage": self.system.fast_store.usage_ratio
        }

    def compare_snapshots(self, snap1: Dict, snap2: Dict) -> Dict[str, Any]:
        """Compare two snapshots for debugging state changes."""
        pass

    def inject_pattern(self, pattern: torch.Tensor, target: str):
        """Inject test pattern into component."""
        pass

    def step_attractor(self, cue: torch.Tensor, steps: int = 1):
        """Single-step attractor dynamics for inspection."""
        pass

    def plot_energy_trajectory(self, trajectory: List[float]):
        """Plot attractor energy over settling."""
        pass

    def visualize_sparse_pattern(self, encoding: torch.Tensor):
        """ASCII visualization of sparse pattern."""
        pass
```

### 3.3 Jupyter Integration

```python
# notebooks/bioinspired_debug.ipynb support

from ww.bioinspired.jupyter import (
    SparseEncodingWidget,
    AttractorVisualizerWidget,
    NeuromodulatorDashboard,
    EligibilityTraceTimeline
)

# Interactive sparse encoding exploration
sparse_widget = SparseEncodingWidget(encoder)
sparse_widget.display()

# Real-time attractor visualization
attractor_viz = AttractorVisualizerWidget(network)
attractor_viz.show_energy_landscape()
attractor_viz.animate_settling(cue)

# Neuromodulator control panel
neuromod_dash = NeuromodulatorDashboard(orchestra)
neuromod_dash.interactive()

# Timeline of eligibility traces
trace_timeline = EligibilityTraceTimeline(tracer)
trace_timeline.show(last_n_steps=1000)
```

---

## 4. Analysis Tools

### 4.1 Biological Metrics Analyzer

```python
# src/bioinspired/analysis.py

class BiologicalMetricsAnalyzer:
    """Analyze bioinspired system behavior against literature targets."""

    def sparsity_distribution(
        self,
        samples: int = 1000
    ) -> Dict[str, Any]:
        """Analyze sparsity distribution."""
        pass

    def learning_rate_dynamics(
        self,
        episode_count: int = 100
    ) -> Dict[str, Any]:
        """Analyze learning rate modulation over episodes."""
        pass

    def capacity_curve(
        self,
        max_patterns: int = 200
    ) -> Dict[str, Any]:
        """Plot retrieval accuracy vs pattern count."""
        pass

    def consolidation_timeline(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Analyze consolidation patterns over time."""
        pass

    def generate_report(self) -> str:
        """Generate comprehensive biological plausibility report."""
        pass
```

### 4.2 Comparison Tools

```python
# Compare bioinspired vs standard encoding
def compare_encoding_methods(
    content_samples: List[str],
    methods: List[str] = ["standard", "sparse", "dendritic"]
) -> pd.DataFrame:
    """Compare encoding methods on retrieval accuracy, speed, capacity."""
    pass

# A/B test learning configurations
def ab_test_learning(
    config_a: Dict,
    config_b: Dict,
    test_episodes: int = 1000
) -> Dict[str, Any]:
    """A/B test different bioinspired configurations."""
    pass
```

---

## 5. Tool Priority Matrix

| Tool | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| `bio_encode` | High | Low | Sparse encoder |
| `bio_decode` | Medium | Medium | Encoder + decoder |
| `bio_attractor_store` | High | Medium | Attractor network |
| `bio_attractor_retrieve` | High | Medium | Attractor network |
| `bio_attractor_analyze` | Medium | High | Attractor + analysis |
| `bio_neuromod_set` | High | Low | Neuromodulator |
| `bio_neuromod_simulate` | Medium | Medium | Neuromodulator |
| `bio_trace_inspect` | Medium | Low | Eligibility system |
| `bio_trace_assign_credit` | High | Medium | Eligibility + learning |
| `bio_episodic_store` | High | Medium | Fast store |
| `bio_episodic_consolidate` | High | High | Fast + semantic stores |
| `bio_dendritic_inspect` | Low | Medium | Dendritic neurons |
| `bio_validate` | High | Medium | All components |

---

## 6. Implementation Phases

### Phase 1: Core Tools (Week 1-2)
- `bio_encode`, `bio_decode`
- `bio_attractor_store`, `bio_attractor_retrieve`
- `bio_neuromod_set`
- `bio_validate`

### Phase 2: Management Tools (Week 3-4)
- `bio_episodic_store`, `bio_episodic_consolidate`
- `bio_trace_inspect`, `bio_trace_assign_credit`
- CLI utilities

### Phase 3: Analysis Tools (Week 5-6)
- `bio_attractor_analyze`
- `bio_neuromod_simulate`
- `bio_dendritic_inspect`
- Jupyter widgets
- Metrics analyzer

---

## Summary

Total new tools proposed: **15 MCP tools + CLI + debugging utilities + analysis tools**

Key gaps filled:
- Sparse encoding inspection and manipulation
- Attractor network debugging and analysis
- Neuromodulator manual control
- Eligibility trace visibility
- Fast episodic store management
- Biological validation automation
- Development-time debugging utilities

Implementation approach: Phased rollout with core operational tools first, then management, then analysis.
