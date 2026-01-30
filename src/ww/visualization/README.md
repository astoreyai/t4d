# World Weaver Visualization Modules

Comprehensive visualization tools for analyzing the neurocomputational dynamics of the World Weaver memory system.

## Overview

This package provides visualizations for understanding:

1. **Activation Patterns** - Memory activation heatmaps across time
2. **Synaptic Plasticity** - LTP/LTD dynamics and BCM learning curves
3. **Neuromodulator State** - DA/NE/ACh/5-HT/GABA timelines and dashboards
4. **Pattern Separation** - DG-style orthogonalization effects
5. **Consolidation Replay** - SWR sequences and sleep-based learning
6. **Embedding Projections** - t-SNE/UMAP visualizations of memory space
7. **Persistence State** - WAL segments, checkpoints, and durability metrics

## Modules

### 1. Activation Heatmap (`activation_heatmap.py`)

Visualizes activation patterns across memory types over time.

```python
from ww.visualization import ActivationHeatmap, plot_activation_heatmap

tracker = ActivationHeatmap(window_size=100)

# Record activations
tracker.record_snapshot(
    episodic_activations={"mem_1": 0.8, "mem_2": 0.5},
    semantic_activations={"entity_1": 0.6},
    neuromod_state={"dopamine": 0.2, "norepinephrine": 1.5}
)

# Visualize
plot_activation_heatmap(tracker, memory_type="episodic")
```

**Features:**
- Sliding window tracking of activations
- Separate visualization for episodic/semantic memories
- Integrated neuromodulator timeline
- Supports both matplotlib and plotly

### 2. Plasticity Traces (`plasticity_traces.py`)

Visualizes synaptic weight changes following BCM and homeostatic plasticity rules.

```python
from ww.visualization import PlasticityTracer, plot_bcm_curve

tracer = PlasticityTracer(max_updates=10000)

# Record weight update
tracer.record_update(
    source_id="neuron_1",
    target_id="neuron_2",
    old_weight=0.5,
    new_weight=0.6,
    update_type="ltp",
    activation_level=0.8
)

# Visualize BCM learning curve
plot_bcm_curve(tracer)

# Plot LTP/LTD distributions
plot_ltp_ltd_distribution(tracer)
```

**Features:**
- BCM learning curve visualization (activation vs weight change)
- LTP/LTD magnitude distributions
- Weight change timeline with stacked events
- Tracks homeostatic scaling events

### 3. Neuromodulator State (`neuromodulator_state.py`)

Comprehensive dashboard for neuromodulator dynamics.

```python
from ww.visualization import NeuromodulatorDashboard, plot_neuromodulator_traces

dashboard = NeuromodulatorDashboard(window_size=1000)

# Record state
dashboard.record_state(
    dopamine_rpe=0.3,
    norepinephrine_gain=1.2,
    acetylcholine_mode="encoding",
    serotonin_mood=0.6,
    gaba_sparsity=0.08
)

# Visualize traces
plot_neuromodulator_traces(dashboard)

# Radar chart of current state
plot_neuromodulator_radar(dashboard)
```

**Features:**
- Multi-panel timeline for all modulators
- Radar chart for current state snapshot
- Statistical summaries (mean, std, min, max)
- ACh mode distribution tracking

### 4. Pattern Separation (`pattern_separation.py`)

Visualizes the effects of dentate gyrus-style pattern separation.

```python
from ww.visualization import (
    PatternSeparationVisualizer,
    plot_separation_comparison
)

vis = PatternSeparationVisualizer()

# Analyze separation
stats = vis.analyze_separation(original_embeddings, separated_embeddings)

# Visualize before/after similarity matrices
plot_separation_comparison(original_embeddings, separated_embeddings)

# Show sparsity changes
plot_sparsity_distribution(original_embeddings, separated_embeddings)
```

**Features:**
- Before/after similarity matrix heatmaps
- Sparsity distribution comparisons
- Separation statistics (similarity reduction, sparsity increase)
- Quantifies orthogonalization effectiveness

### 5. Consolidation Replay (`consolidation_replay.py`)

Visualizes sharp-wave ripple sequences and consolidation dynamics.

```python
from ww.visualization import ConsolidationVisualizer, plot_swr_sequence

visualizer = ConsolidationVisualizer()

# Record replay sequence
visualizer.record_replay_sequence(
    memory_ids=["mem_1", "mem_2", "mem_3"],
    priority_scores=[0.8, 0.6, 0.7],
    phase="nrem"
)

# Visualize specific sequence
plot_swr_sequence(visualizer, sequence_index=0)

# Compare NREM vs REM priorities
plot_replay_priority(visualizer)
```

**Features:**
- SWR sequence visualization with priority scores
- NREM vs REM priority distributions
- Sequence length statistics
- Replay matrix heatmap

### 6. Embedding Projections (`embedding_projections.py`)

Projects high-dimensional embeddings to 2D/3D for visualization.

```python
from ww.visualization import EmbeddingProjector, plot_tsne_projection

projector = EmbeddingProjector()

# t-SNE projection
plot_tsne_projection(
    embeddings,
    labels=memory_ids,
    colors=priority_scores,
    projector=projector
)

# UMAP projection
plot_umap_projection(
    embeddings,
    labels=memory_ids,
    colors=cluster_ids,
    projector=projector
)
```

**Features:**
- t-SNE, UMAP, and PCA projections
- Automatic caching for fast re-plotting
- Color-coded by any continuous or categorical variable
- Interactive (plotly) and static (matplotlib) modes

## Installation

The visualization modules require:

```bash
# Core dependencies (always required)
pip install numpy

# For static plots (matplotlib)
pip install matplotlib

# For interactive plots (plotly)
pip install plotly

# For embedding projections
pip install scikit-learn  # For t-SNE and PCA
pip install umap-learn    # For UMAP (optional)
```

## Usage Patterns

### Interactive vs Static Plotting

All plot functions accept an `interactive` parameter:

```python
# Interactive (plotly) - better for exploration
plot_activation_heatmap(tracker, interactive=True)

# Static (matplotlib) - better for papers/reports
plot_activation_heatmap(tracker, interactive=False)
```

### Saving Figures

All plot functions accept a `save_path` parameter:

```python
from pathlib import Path

# Save as HTML (interactive)
plot_neuromodulator_traces(
    dashboard,
    save_path=Path("outputs/neuromod_traces.html"),
    interactive=True
)

# Save as PNG (static)
plot_bcm_curve(
    tracer,
    save_path=Path("outputs/bcm_curve.png"),
    interactive=False
)
```

### Integration with World Weaver Components

Example integration with pattern separation:

```python
from ww.memory.pattern_separation import DentateGyrus
from ww.visualization import PatternSeparationVisualizer

# Create DG
dg = DentateGyrus(embedding_provider, vector_store)

# Encode with separation
original_emb = await dg.embedding.embed_query("test content")
separated_emb = await dg.encode("test content", apply_separation=True)

# Visualize
vis = PatternSeparationVisualizer()
stats = vis.analyze_separation(
    np.array([original_emb]),
    np.array([separated_emb])
)
print(f"Separation magnitude: {stats['mean_separation_magnitude']:.4f}")
```

Example integration with neuromodulators:

```python
from ww.learning.neuromodulators import NeuromodulatorOrchestra
from ww.visualization import NeuromodulatorDashboard

orchestra = NeuromodulatorOrchestra()
dashboard = NeuromodulatorDashboard()

# Process query
state = orchestra.process_query(query_embedding, is_question=True)

# Record for visualization
dashboard.record_state(
    dopamine_rpe=state.dopamine_rpe,
    norepinephrine_gain=state.norepinephrine_gain,
    acetylcholine_mode=state.acetylcholine_mode,
    serotonin_mood=state.serotonin_mood,
    gaba_sparsity=state.inhibition_sparsity
)

# Visualize over time
plot_neuromodulator_traces(dashboard)
```

### 7. Persistence State (`persistence_state.py`)

Visualizes WAL, checkpoints, and durability metrics for the persistence layer.

```python
from ww.visualization import (
    PersistenceVisualizer,
    PersistenceMetrics,
    plot_wal_timeline,
    plot_durability_dashboard,
    plot_checkpoint_history,
)

visualizer = PersistenceVisualizer()

# Update metrics
visualizer.update_metrics(PersistenceMetrics(
    current_lsn=12345,
    checkpoint_lsn=12000,
    operations_since_checkpoint=345,
    wal_segment_count=3,
    wal_total_size_bytes=15_000_000,
    last_checkpoint_age_seconds=180.0,
    recovery_mode="warm_start",
))

# Visualize durability dashboard
plot_durability_dashboard(visualizer)

# Plot WAL timeline
plot_wal_timeline(visualizer)

# Plot checkpoint history
plot_checkpoint_history(visualizer)
```

**Features:**
- WAL segment size visualization
- LSN progression over time
- Checkpoint markers and history
- Durability dashboard with color-coded health indicators
- Real-time metrics tracking

## Example Application

See `/mnt/projects/ww/examples/visualization_demo.py` for a comprehensive demonstration of all visualization modules.

Run the demo:

```bash
cd /mnt/projects/ww
python examples/visualization_demo.py
```

## Design Principles

1. **Biological Fidelity**: Visualizations reflect actual neural mechanisms (BCM curves, SWR sequences, etc.)

2. **Dual Rendering**: All plots support both matplotlib (static, publication-ready) and plotly (interactive, exploratory)

3. **Async-Compatible**: Designed to work with async World Weaver components

4. **Export Options**: PNG/HTML export for integration into reports and dashboards

5. **Type Hints**: Complete type annotations for IDE support

6. **Minimal Dependencies**: Core functionality works with numpy alone; plotting libraries are optional

## Performance Notes

- **t-SNE**: O(N²) complexity, slow for N > 1000. Use perplexity < N.
- **UMAP**: Faster than t-SNE, scales to N > 10,000.
- **PCA**: Fastest (O(ND²)), good for quick exploration.
- **Caching**: EmbeddingProjector caches projections to avoid recomputation.

## Biological Basis

Each visualization module corresponds to actual neuroscience:

- **BCM Curve**: Bienenstock-Cooper-Munro theory of synaptic modification
- **SWR Sequences**: Sharp-wave ripples observed in hippocampus during sleep
- **Pattern Separation**: Dentate gyrus orthogonalization of similar inputs
- **Neuromodulator Traces**: Dopamine, norepinephrine, acetylcholine, serotonin dynamics
- **Homeostatic Scaling**: TNFα-mediated synaptic scaling for stability

## References

- Bienenstock et al. (1982) "Theory for the development of neuron selectivity"
- Turrigiano (2008) "The self-tuning neuron: Synaptic scaling of excitatory synapses"
- Buzsáki (2015) "Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning"
- Leutgeb et al. (2007) "Pattern separation in the dentate gyrus and CA3"
- Schultz (1998) "Predictive reward signal of dopamine neurons"

## License

Part of the World Weaver project. See main repository for license details.
