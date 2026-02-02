# T4DM Visualization - Quick Start Guide

## 30-Second Overview

```python
from t4dm.visualization import (
    ActivationHeatmap,           # Memory activation heatmaps
    PlasticityTracer,            # LTP/LTD weight changes
    NeuromodulatorDashboard,     # DA/NE/ACh/5-HT/GABA traces
    PatternSeparationVisualizer, # DG orthogonalization effects
    ConsolidationVisualizer,     # SWR replay sequences
    EmbeddingProjector           # t-SNE/UMAP projections
)
```

## Common Patterns

### Pattern 1: Track and Visualize Activations

```python
tracker = ActivationHeatmap()

# During system operation
tracker.record_snapshot(
    episodic_activations={"mem_1": 0.8},
    semantic_activations={"entity_1": 0.6},
    neuromod_state={"dopamine": 0.2}
)

# Visualize
from t4dm.visualization import plot_activation_heatmap
plot_activation_heatmap(tracker, save_path="activations.png")
```

### Pattern 2: Analyze Plasticity Dynamics

```python
tracer = PlasticityTracer()

# Record weight updates
tracer.record_update(
    source_id="src_1",
    target_id="tgt_1",
    old_weight=0.5,
    new_weight=0.6,
    update_type="ltp",
    activation_level=0.8
)

# Visualize BCM curve
from t4dm.visualization import plot_bcm_curve
plot_bcm_curve(tracer, interactive=True)
```

### Pattern 3: Monitor Neuromodulators

```python
dashboard = NeuromodulatorDashboard()

# Record state from orchestra
dashboard.record_state(
    dopamine_rpe=state.dopamine_rpe,
    norepinephrine_gain=state.norepinephrine_gain,
    acetylcholine_mode=state.acetylcholine_mode,
    serotonin_mood=state.serotonin_mood,
    gaba_sparsity=state.inhibition_sparsity
)

# Visualize timeline
from t4dm.visualization import plot_neuromodulator_traces
plot_neuromodulator_traces(dashboard)
```

### Pattern 4: Evaluate Pattern Separation

```python
from t4dm.visualization import plot_separation_comparison

# Compare before/after
plot_separation_comparison(
    original_embeddings,
    separated_embeddings,
    save_path="separation.html",
    interactive=True
)
```

### Pattern 5: Analyze Consolidation

```python
visualizer = ConsolidationVisualizer()

# Record replay sequences
visualizer.record_replay_sequence(
    memory_ids=["mem_1", "mem_2"],
    priority_scores=[0.8, 0.6],
    phase="nrem"
)

# Visualize
from t4dm.visualization import plot_swr_sequence
plot_swr_sequence(visualizer, sequence_index=0)
```

### Pattern 6: Explore Embedding Space

```python
projector = EmbeddingProjector()

from t4dm.visualization import plot_tsne_projection
plot_tsne_projection(
    embeddings,
    labels=memory_ids,
    colors=importance_scores,
    projector=projector
)
```

## Plot Function Reference

All plot functions follow this signature:

```python
plot_function(
    data_source,              # Tracker/visualizer instance
    save_path=None,           # Optional: Path to save (PNG/HTML)
    interactive=False,        # True=plotly, False=matplotlib
    **kwargs                  # Function-specific options
)
```

## File Formats

- **Static plots**: Use `save_path="output.png"` (matplotlib)
- **Interactive plots**: Use `save_path="output.html"` (plotly)

## Integration Examples

### With DentateGyrus

```python
from t4dm.memory.pattern_separation import DentateGyrus
from t4dm.visualization import PatternSeparationVisualizer

dg = DentateGyrus(embedding_provider, vector_store)
vis = PatternSeparationVisualizer()

# Get separation history
history = dg.get_separation_history()

# Analyze
for result in history:
    stats = vis.analyze_separation(
        np.array([result.original_embedding]),
        np.array([result.separated_embedding])
    )
    print(f"Separation: {stats['mean_separation_magnitude']:.4f}")
```

### With NeuromodulatorOrchestra

```python
from t4dm.learning.neuromodulators import NeuromodulatorOrchestra
from t4dm.visualization import NeuromodulatorDashboard

orchestra = NeuromodulatorOrchestra()
dashboard = NeuromodulatorDashboard()

# Process and record
state = orchestra.process_query(query_embedding)
dashboard.record_state(
    state.dopamine_rpe,
    state.norepinephrine_gain,
    state.acetylcholine_mode,
    state.serotonin_mood,
    state.inhibition_sparsity
)
```

### With SleepConsolidation

```python
from t4dm.consolidation.sleep import SleepConsolidation
from t4dm.visualization import ConsolidationVisualizer

consolidation = SleepConsolidation(episodic, semantic, graph_store)
visualizer = ConsolidationVisualizer()

# Run sleep cycle
result = await consolidation.full_sleep_cycle("session_1")

# Get replay history
history = consolidation.get_replay_history()
for event in history:
    visualizer.record_replay_sequence(
        memory_ids=[str(event.episode_id)],
        priority_scores=[event.priority_score],
        phase="nrem"
    )
```

## Troubleshooting

### Import Error: plotly not found

```bash
pip install plotly
# Or use matplotlib mode: interactive=False
```

### Import Error: sklearn not found

```bash
pip install scikit-learn
# For UMAP: pip install umap-learn
```

### t-SNE fails with perplexity error

```python
# Reduce perplexity for small datasets
projector.project_tsne(embeddings, perplexity=min(30, len(embeddings) - 1))
```

### Empty plots

```python
# Check that data was recorded
print(f"Snapshots: {len(tracker._snapshots)}")
print(f"Updates: {len(tracer._updates)}")
```

## Performance Tips

1. **Limit window size** for real-time tracking:
   ```python
   tracker = ActivationHeatmap(window_size=100)  # Not 10000
   ```

2. **Cache projections** when exploring:
   ```python
   projector = EmbeddingProjector()  # Reuse same instance
   ```

3. **Use PCA for quick exploration**:
   ```python
   projection = projector.project_pca(embeddings)  # Faster than t-SNE
   ```

4. **Batch record updates**:
   ```python
   for update in batch_updates:
       tracer.record_update(...)  # Fast
   ```

## Next Steps

- Run the full demo: `python examples/visualization_demo.py`
- Read the detailed README: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/README.md`
- Explore integration with your components
- Create custom visualizations by subclassing base classes

## Quick Reference Table

| Visualization | Use Case | Key Method | Output |
|--------------|----------|------------|--------|
| ActivationHeatmap | Track memory activation over time | `record_snapshot()` | Heatmap + timeline |
| PlasticityTracer | Analyze weight changes | `record_update()` | BCM curve + distributions |
| NeuromodulatorDashboard | Monitor neuromodulator state | `record_state()` | Multi-panel traces + radar |
| PatternSeparationVisualizer | Evaluate DG separation | `analyze_separation()` | Similarity matrices |
| ConsolidationVisualizer | Analyze sleep replay | `record_replay_sequence()` | SWR sequences |
| EmbeddingProjector | Explore embedding space | `project_tsne()` | 2D/3D projections |

## Code Stats

- **Total lines**: 2,435 lines of code
- **Modules**: 7 Python files
- **Type hints**: Complete coverage
- **Documentation**: Comprehensive docstrings + examples
