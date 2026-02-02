# T4DM Visualization Walkthrough

**Version**: 0.1.0
**Last Updated**: 2025-12-09

A comprehensive guide to all visualization modules in the T4DM memory system.

---

## Table of Contents

1. [Overview](#overview)
2. [Activation Heatmap](#2-activation-heatmap)
3. [Plasticity Traces](#3-plasticity-traces)
4. [Neuromodulator State](#4-neuromodulator-state)
5. [Pattern Separation](#5-pattern-separation)
6. [Consolidation Replay](#6-consolidation-replay)
7. [Embedding Projections](#7-embedding-projections)
8. [Persistence State](#8-persistence-state)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)

---

## 1. Overview

### Purpose of Visualizations

The T4DM visualization suite provides tools for understanding and analyzing the neurocomputational dynamics of the memory system. Each module corresponds to actual neuroscience mechanisms and offers insights into how memories are encoded, consolidated, and retrieved.

**Key Focus Areas**:
- Memory activation patterns across episodic and semantic systems
- Synaptic plasticity dynamics (LTP/LTD, BCM learning)
- Neuromodulator influences (DA/NE/ACh/5-HT/GABA)
- Pattern separation and orthogonalization
- Sleep-based consolidation and replay
- High-dimensional embedding spaces
- Persistence and durability metrics

### Static vs Interactive Modes

All visualization functions support two rendering modes:

#### Static Mode (matplotlib)
- **Use case**: Publication-ready figures, reports, papers
- **Format**: PNG, PDF, SVG
- **Pros**: High-quality, customizable, works offline
- **Cons**: No interactivity, fixed views

```python
plot_activation_heatmap(tracker, interactive=False)
```

#### Interactive Mode (plotly)
- **Use case**: Exploratory analysis, dashboards, presentations
- **Format**: HTML with embedded JavaScript
- **Pros**: Zoom, pan, hover tooltips, dynamic updates
- **Cons**: Larger file sizes, requires browser

```python
plot_activation_heatmap(tracker, interactive=True)
```

### Saving Figures

All plot functions accept a `save_path` parameter:

```python
from pathlib import Path

# Save interactive HTML
plot_activation_heatmap(
    tracker,
    save_path=Path("outputs/activation_heatmap.html"),
    interactive=True
)

# Save static PNG
plot_bcm_curve(
    tracer,
    save_path=Path("outputs/bcm_curve.png"),
    interactive=False
)
```

### Dependencies

```python
# Core (always required)
import numpy as np

# Static plots
import matplotlib.pyplot as plt

# Interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Embedding projections
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap  # Optional, falls back to t-SNE
```

---

## 2. Activation Heatmap

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/activation_heatmap.py`

Visualizes activation patterns across memory types (episodic, semantic) over time, integrated with neuromodulator dynamics.

### ActivationHeatmap Class

The `ActivationHeatmap` class tracks activation snapshots in a sliding window.

```python
from t4dm.visualization import ActivationHeatmap

tracker = ActivationHeatmap(
    window_size=100,           # Number of timesteps to track
    max_memories_tracked=50    # Maximum memories to show in heatmap
)
```

**Key Methods**:

#### record_snapshot()
Records activation state at current time:

```python
tracker.record_snapshot(
    episodic_activations={"mem_1": 0.8, "mem_2": 0.5, "mem_3": 0.3},
    semantic_activations={"entity_1": 0.6, "entity_2": 0.4},
    neuromod_state={
        "dopamine": 0.2,
        "norepinephrine": 1.5,
        "acetylcholine": 0.7,
        "serotonin": 0.5,
        "gaba": 0.08
    }
)
```

#### get_activation_matrix()
Returns activation matrix for visualization:

```python
matrix, memory_ids, timestamps = tracker.get_activation_matrix(
    memory_type="episodic"  # or "semantic"
)
# matrix shape: (time_steps, n_memories)
```

#### get_neuromod_timeline()
Returns neuromodulator timeline:

```python
matrix, mod_names, timestamps = tracker.get_neuromod_timeline()
# matrix shape: (time_steps, n_modulators)
```

### plot_activation_heatmap()

Visualizes memory activation patterns as a heatmap.

```python
from t4dm.visualization import plot_activation_heatmap

plot_activation_heatmap(
    tracker,
    memory_type="episodic",  # or "semantic"
    save_path=Path("outputs/activation_heatmap.png"),
    interactive=False
)
```

**Features**:
- Color-coded activation levels (viridis colormap)
- Time on x-axis, memory IDs on y-axis
- Automatic scaling for readability
- Truncated memory IDs for clarity

### plot_activation_timeline()

Plots neuromodulator levels over time.

```python
from t4dm.visualization import plot_activation_timeline

plot_activation_timeline(
    tracker,
    save_path=Path("outputs/neuromod_timeline.html"),
    interactive=True
)
```

**Features**:
- Multi-line plot with one trace per neuromodulator
- Color-coded by modulator type
- Unified hover mode (interactive)
- Reference lines at key thresholds

### Example Usage

```python
from t4dm.visualization import ActivationHeatmap, plot_activation_heatmap
from datetime import datetime
import time

# Initialize tracker
tracker = ActivationHeatmap(window_size=50)

# Simulate activation pattern over time
for t in range(50):
    # Simulate memories with varying activation
    episodic = {
        f"mem_{i}": 0.5 + 0.4 * np.sin(t / 5 + i)
        for i in range(10)
    }
    semantic = {
        f"entity_{i}": 0.3 + 0.3 * np.cos(t / 4 + i)
        for i in range(5)
    }

    # Simulate neuromodulator dynamics
    neuromod = {
        "dopamine": 0.2 * np.sin(t / 10),
        "norepinephrine": 1.0 + 0.5 * np.cos(t / 8),
        "acetylcholine": 0.6,
        "serotonin": 0.5,
        "gaba": 0.08
    }

    tracker.record_snapshot(episodic, semantic, neuromod)
    time.sleep(0.1)

# Visualize episodic activation patterns
plot_activation_heatmap(tracker, memory_type="episodic", interactive=True)

# Visualize neuromodulator timeline
plot_activation_timeline(tracker, interactive=True)
```

---

## 3. Plasticity Traces

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/plasticity_traces.py`

Visualizes synaptic weight changes following BCM (Bienenstock-Cooper-Munro) and homeostatic plasticity rules.

### PlasticityTracer Class

The `PlasticityTracer` class tracks weight updates from LTP/LTD and homeostatic mechanisms.

```python
from t4dm.visualization import PlasticityTracer

tracer = PlasticityTracer(max_updates=10000)
```

**Key Methods**:

#### record_update()
Records a synaptic weight change:

```python
tracer.record_update(
    source_id="neuron_1",
    target_id="neuron_2",
    old_weight=0.5,
    new_weight=0.6,
    update_type="ltp",  # or "ltd", "homeostatic"
    activation_level=0.8
)
```

#### get_bcm_curve_data()
Returns data for BCM learning curve:

```python
activations, changes = tracer.get_bcm_curve_data()
# activations: activation levels that triggered updates
# changes: corresponding weight changes
```

#### get_ltp_ltd_distribution()
Returns distributions of LTP and LTD magnitudes:

```python
ltp_mags, ltd_mags = tracer.get_ltp_ltd_distribution()
```

#### get_timeline_data()
Returns binned timeline of plasticity events:

```python
bin_times, type_counts = tracer.get_timeline_data(bin_size_minutes=5)
# type_counts: dict mapping update_type -> list of counts per bin
```

### plot_bcm_curve()

Visualizes the BCM learning rule: weight change as a function of activation level.

```python
from t4dm.visualization import plot_bcm_curve

plot_bcm_curve(
    tracer,
    save_path=Path("outputs/bcm_curve.png"),
    interactive=False
)
```

**Features**:
- Scatter plot of actual weight changes vs activation
- Theoretical BCM curve overlay (dashed line)
- Color-coded by weight change magnitude
- Modification threshold (θ_m) marker
- Zero-crossing reference lines

**Biological Basis**: The BCM rule states that synaptic modification depends on postsynaptic activity relative to a sliding threshold. Low activity causes LTD, high activity causes LTP.

### plot_weight_changes()

Timeline of weight change events, stacked by type.

```python
from t4dm.visualization import plot_weight_changes

plot_weight_changes(
    tracer,
    save_path=Path("outputs/weight_changes.html"),
    interactive=True
)
```

**Features**:
- Stacked bar chart showing LTP/LTD/homeostatic events
- Time-binned for clarity
- Color-coded by update type
- Hover tooltips with exact counts

### plot_ltp_ltd_distribution()

Distribution of LTP and LTD magnitudes.

```python
from t4dm.visualization import plot_ltp_ltd_distribution

plot_ltp_ltd_distribution(
    tracer,
    save_path=Path("outputs/ltp_ltd_dist.png"),
    interactive=False
)
```

**Features**:
- Side-by-side histograms
- LTP in green, LTD in red
- Mean lines with annotations
- Event counts in titles

### Example Usage

```python
from t4dm.visualization import PlasticityTracer, plot_bcm_curve, plot_ltp_ltd_distribution

tracer = PlasticityTracer(max_updates=5000)

# Simulate BCM-style learning
for _ in range(1000):
    activation = np.random.uniform(0, 1)
    old_weight = np.random.uniform(0, 1)

    # BCM rule: weight change depends on activation relative to threshold
    theta_m = 0.5
    delta_w = 0.01 * activation * (activation - theta_m)

    new_weight = np.clip(old_weight + delta_w, 0, 1)
    update_type = "ltp" if delta_w > 0 else "ltd"

    tracer.record_update(
        source_id="pre_1",
        target_id="post_1",
        old_weight=old_weight,
        new_weight=new_weight,
        update_type=update_type,
        activation_level=activation
    )

# Visualize BCM curve
plot_bcm_curve(tracer, interactive=True)

# Visualize distributions
plot_ltp_ltd_distribution(tracer, interactive=True)
```

---

## 4. Neuromodulator State

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/neuromodulator_state.py`

Comprehensive dashboard for neuromodulator dynamics across five systems:
- **Dopamine (DA)**: Reward prediction error
- **Norepinephrine (NE)**: Arousal and novelty gain
- **Acetylcholine (ACh)**: Encoding/retrieval mode
- **Serotonin (5-HT)**: Long-term mood/value
- **GABA**: Inhibitory sparsity

### NeuromodulatorDashboard Class

```python
from t4dm.visualization import NeuromodulatorDashboard

dashboard = NeuromodulatorDashboard(window_size=1000)
```

**Key Methods**:

#### record_state()
Records current neuromodulator state:

```python
dashboard.record_state(
    dopamine_rpe=0.3,              # [-1, 1]
    norepinephrine_gain=1.2,       # [0.5, 2.0]
    acetylcholine_mode="encoding", # "encoding", "balanced", "retrieval"
    serotonin_mood=0.6,            # [0, 1]
    gaba_sparsity=0.08             # [0, 1]
)
```

#### get_trace_data()
Returns time series for all modulators:

```python
trace_data = dashboard.get_trace_data()
# Returns: dict mapping modulator -> (timestamps, values)
```

#### get_mode_distribution()
Returns distribution of ACh modes:

```python
mode_dist = dashboard.get_mode_distribution()
# Example: {"encoding": 300, "balanced": 500, "retrieval": 200}
```

#### get_statistics()
Returns summary statistics:

```python
stats = dashboard.get_statistics()
# Example: {"dopamine": {"mean": 0.1, "std": 0.2, "min": -0.5, "max": 0.8}, ...}
```

### plot_neuromodulator_traces()

Multi-panel timeline of all neuromodulator levels.

```python
from t4dm.visualization import plot_neuromodulator_traces

plot_neuromodulator_traces(
    dashboard,
    save_path=Path("outputs/neuromod_traces.html"),
    interactive=True
)
```

**Features**:
- Four stacked subplots (one per modulator)
- Color-coded traces
- Reference lines at key thresholds
- Grid for easy reading

### plot_neuromodulator_radar()

Radar chart showing current neuromodulator state snapshot.

```python
from t4dm.visualization import plot_neuromodulator_radar

plot_neuromodulator_radar(
    dashboard,
    save_path=Path("outputs/neuromod_radar.png"),
    interactive=False
)
```

**Features**:
- Pentagonal radar chart
- Normalized values [0, 1]
- Filled area for quick gestalt reading
- ACh mode represented as binary encoding indicator

### Example Usage

```python
from t4dm.visualization import NeuromodulatorDashboard, plot_neuromodulator_traces, plot_neuromodulator_radar
import time

dashboard = NeuromodulatorDashboard(window_size=500)

# Simulate dynamic neuromodulator state
for t in range(200):
    # Dopamine: reward prediction errors
    rpe = 0.3 * np.sin(t / 20) + 0.1 * np.random.randn()

    # Norepinephrine: arousal varies with novelty
    ne_gain = 1.0 + 0.3 * np.exp(-t / 50)

    # Acetylcholine: switches between encoding/retrieval
    if t < 100:
        ach_mode = "encoding"
    elif t < 150:
        ach_mode = "balanced"
    else:
        ach_mode = "retrieval"

    # Serotonin: slowly varying mood
    serotonin = 0.5 + 0.2 * np.sin(t / 100)

    # GABA: stable sparsity
    gaba = 0.08 + 0.02 * np.random.randn()

    dashboard.record_state(rpe, ne_gain, ach_mode, serotonin, gaba)
    time.sleep(0.05)

# Plot traces
plot_neuromodulator_traces(dashboard, interactive=True)

# Plot current state radar
plot_neuromodulator_radar(dashboard, interactive=True)

# Get statistics
stats = dashboard.get_statistics()
print(f"Mean dopamine RPE: {stats['dopamine']['mean']:.3f}")
```

---

## 5. Pattern Separation

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/pattern_separation.py`

Visualizes the effects of dentate gyrus-style pattern separation: orthogonalization of similar inputs.

### PatternSeparationVisualizer Class

```python
from t4dm.visualization import PatternSeparationVisualizer

vis = PatternSeparationVisualizer()
```

**Key Methods**:

#### compute_similarity_matrix()
Computes pairwise similarity matrix:

```python
similarity_matrix = vis.compute_similarity_matrix(
    embeddings,
    normalize=True  # Use cosine similarity if True, else dot product
)
# Returns: (N x N) similarity matrix
```

#### compute_sparsity()
Computes sparsity of a single embedding:

```python
sparsity = vis.compute_sparsity(
    embedding,
    threshold=0.01  # Values below this are considered zero
)
# Returns: fraction of near-zero elements [0, 1]
```

#### analyze_separation()
Analyzes separation statistics:

```python
stats = vis.analyze_separation(
    original_embeddings,
    separated_embeddings
)
# Returns: dict with metrics like mean similarity reduction, sparsity increase
```

**Example output**:
```python
{
    "mean_similarity_before": 0.45,
    "mean_similarity_after": 0.18,
    "similarity_reduction": 0.27,
    "mean_separation_magnitude": 0.32,
    "mean_sparsity_before": 0.05,
    "mean_sparsity_after": 0.15,
    "sparsity_increase": 0.10
}
```

### plot_separation_comparison()

Side-by-side similarity matrices showing before/after separation.

```python
from t4dm.visualization import plot_separation_comparison

plot_separation_comparison(
    original_embeddings,
    separated_embeddings,
    save_path=Path("outputs/separation_comparison.png"),
    interactive=False
)
```

**Features**:
- Two heatmaps: before and after separation
- Same color scale for direct comparison
- Lower off-diagonal values indicate better separation

### plot_sparsity_distribution()

Distribution of sparsity before and after separation.

```python
from t4dm.visualization import plot_sparsity_distribution

plot_sparsity_distribution(
    original_embeddings,
    separated_embeddings,
    save_path=Path("outputs/sparsity_dist.html"),
    interactive=True
)
```

**Features**:
- Overlaid histograms
- Before in blue, after in red
- Mean lines with annotations
- Shows increased sparsity after separation

### Example Usage

```python
from t4dm.visualization import PatternSeparationVisualizer, plot_separation_comparison
from t4dm.memory.pattern_separation import DentateGyrus
import numpy as np

# Create sample embeddings (similar pairs)
n_samples = 20
dim = 128
original = np.random.randn(n_samples, dim)

# Simulate pattern separation (add noise, increase sparsity)
separated = original + 0.5 * np.random.randn(n_samples, dim)
separated[np.abs(separated) < 0.3] = 0  # Sparsify

# Analyze separation
vis = PatternSeparationVisualizer()
stats = vis.analyze_separation(original, separated)

print(f"Similarity reduction: {stats['similarity_reduction']:.3f}")
print(f"Sparsity increase: {stats['sparsity_increase']:.3f}")

# Visualize
plot_separation_comparison(original, separated, interactive=True)
plot_sparsity_distribution(original, separated, interactive=True)
```

**Integration with DentateGyrus**:

```python
from t4dm.memory.pattern_separation import DentateGyrus

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
```

---

## 6. Consolidation Replay

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/consolidation_replay.py`

Visualizes sleep-based memory consolidation: sharp-wave ripple (SWR) sequences and replay priorities.

### ConsolidationVisualizer Class

```python
from t4dm.visualization import ConsolidationVisualizer

visualizer = ConsolidationVisualizer()
```

**Key Methods**:

#### record_replay_sequence()
Records a replay sequence:

```python
visualizer.record_replay_sequence(
    memory_ids=["mem_1", "mem_2", "mem_3", "mem_4"],
    priority_scores=[0.8, 0.6, 0.7, 0.5],
    phase="nrem"  # or "rem"
)
```

#### get_priority_distribution()
Returns priority distributions by phase:

```python
nrem_priorities, rem_priorities = visualizer.get_priority_distribution()
```

#### get_sequence_lengths()
Returns sequence length distributions:

```python
nrem_lengths, rem_lengths = visualizer.get_sequence_lengths()
```

#### get_replay_matrix()
Returns replay matrix for heatmap:

```python
matrix, memory_ids, sequence_ids = visualizer.get_replay_matrix()
# matrix[i, j] = 1 if memory j was in sequence i
```

### plot_swr_sequence()

Visualizes a single SWR replay sequence.

```python
from t4dm.visualization import plot_swr_sequence

plot_swr_sequence(
    visualizer,
    sequence_index=0,
    save_path=Path("outputs/swr_sequence.png"),
    interactive=False
)
```

**Features**:
- Scatter plot with memories ordered by replay position
- Color-coded by priority score
- Connected with dashed lines to show sequence
- Labels with truncated memory IDs

### plot_replay_priority()

Distribution of replay priorities by sleep phase.

```python
from t4dm.visualization import plot_replay_priority

plot_replay_priority(
    visualizer,
    save_path=Path("outputs/replay_priority.html"),
    interactive=True
)
```

**Features**:
- Side-by-side histograms for NREM and REM
- NREM in blue, REM in green
- Mean lines with annotations
- Shows differential replay between phases

### Example Usage

```python
from t4dm.visualization import ConsolidationVisualizer, plot_swr_sequence, plot_replay_priority

visualizer = ConsolidationVisualizer()

# Simulate NREM replay (high-priority, sequential)
for _ in range(10):
    n_memories = np.random.randint(3, 8)
    memory_ids = [f"mem_{i}" for i in range(n_memories)]
    priorities = np.random.uniform(0.6, 1.0, size=n_memories)
    visualizer.record_replay_sequence(memory_ids, priorities.tolist(), "nrem")

# Simulate REM replay (lower-priority, creative)
for _ in range(10):
    n_memories = np.random.randint(2, 6)
    memory_ids = [f"mem_{i}" for i in range(n_memories)]
    priorities = np.random.uniform(0.3, 0.7, size=n_memories)
    visualizer.record_replay_sequence(memory_ids, priorities.tolist(), "rem")

# Visualize first sequence
plot_swr_sequence(visualizer, sequence_index=0, interactive=True)

# Compare NREM vs REM priorities
plot_replay_priority(visualizer, interactive=True)
```

**Biological Basis**: During sleep, the hippocampus replays recent experiences in sharp-wave ripples (SWRs). NREM sleep prioritizes high-value memories for consolidation, while REM sleep may support creative recombination.

---

## 7. Embedding Projections

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/embedding_projections.py`

Projects high-dimensional memory embeddings to 2D/3D using dimensionality reduction techniques.

### EmbeddingProjector Class

The `EmbeddingProjector` class handles projection with automatic caching.

```python
from t4dm.visualization import EmbeddingProjector

projector = EmbeddingProjector()
```

**Key Methods**:

#### project_tsne()
Projects embeddings using t-SNE:

```python
projection = projector.project_tsne(
    embeddings,
    n_components=2,
    perplexity=30.0,
    random_state=42
)
```

**Parameters**:
- `perplexity`: Balance between local and global structure (typically 5-50)
- `n_components`: 2 or 3
- Complexity: O(N²), slow for N > 1000

#### project_umap()
Projects embeddings using UMAP (faster than t-SNE):

```python
projection = projector.project_umap(
    embeddings,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
```

**Parameters**:
- `n_neighbors`: Local neighborhood size
- `min_dist`: Minimum distance between points in projection
- Scales better: handles N > 10,000

#### project_pca()
Projects embeddings using PCA (fastest):

```python
projection = projector.project_pca(
    embeddings,
    n_components=2
)
```

**Parameters**:
- Linear method, preserves global structure
- O(ND²) complexity
- Reports explained variance ratio

#### clear_cache()
Clears cached projections:

```python
projector.clear_cache()
```

### plot_tsne_projection()

Plots t-SNE projection of embeddings.

```python
from t4dm.visualization import plot_tsne_projection

plot_tsne_projection(
    embeddings,
    labels=memory_ids,
    colors=priority_scores,
    save_path=Path("outputs/tsne_projection.html"),
    interactive=True,
    projector=projector  # Optional: reuse projector for caching
)
```

**Features**:
- Color-coded by continuous values (priorities, timestamps)
- Hover labels showing memory IDs
- Automatic fallback to PCA if t-SNE unavailable

### plot_umap_projection()

Plots UMAP projection of embeddings.

```python
from t4dm.visualization import plot_umap_projection

plot_umap_projection(
    embeddings,
    labels=memory_ids,
    colors=cluster_ids,
    save_path=Path("outputs/umap_projection.png"),
    interactive=False,
    projector=projector
)
```

**Features**:
- Faster than t-SNE, better for large datasets
- Preserves both local and global structure
- Automatic fallback to t-SNE if UMAP unavailable

### Example Usage

```python
from t4dm.visualization import EmbeddingProjector, plot_tsne_projection, plot_umap_projection
import numpy as np

# Generate sample embeddings (e.g., from memory system)
n_memories = 200
embedding_dim = 768
embeddings = np.random.randn(n_memories, embedding_dim)

# Create memory IDs and priority scores
memory_ids = [f"mem_{i:03d}" for i in range(n_memories)]
priority_scores = np.random.uniform(0, 1, size=n_memories)

# Create projector (for caching)
projector = EmbeddingProjector()

# t-SNE projection
plot_tsne_projection(
    embeddings,
    labels=memory_ids,
    colors=priority_scores,
    projector=projector,
    interactive=True
)

# UMAP projection (reuses cached embeddings if parameters match)
plot_umap_projection(
    embeddings,
    labels=memory_ids,
    colors=priority_scores,
    projector=projector,
    interactive=True
)

# PCA for quick exploration
pca_proj = projector.project_pca(embeddings, n_components=2)
print(f"PCA projection shape: {pca_proj.shape}")
```

**Performance Tips**:
- Use PCA for quick exploration (fastest)
- Use UMAP for large datasets (N > 1000)
- Use t-SNE for smaller datasets with complex structure (N < 1000)
- Reuse projector instance to leverage caching
- Adjust perplexity (t-SNE) or n_neighbors (UMAP) based on dataset size

---

## 8. Persistence State

**Module**: `/mnt/projects/t4d/t4dm/src/t4dm/visualization/persistence_state.py`

Visualizes WAL (Write-Ahead Log), checkpoints, and durability metrics for the persistence layer.

### PersistenceVisualizer Class

```python
from t4dm.visualization import PersistenceVisualizer, PersistenceMetrics

visualizer = PersistenceVisualizer(max_history=1000)
```

**Key Data Structures**:

#### WALSegmentInfo
```python
from t4dm.visualization import WALSegmentInfo
from pathlib import Path
from datetime import datetime

segment = WALSegmentInfo(
    segment_number=1,
    path=Path("data/wal/segment_001.wal"),
    size_bytes=1024 * 1024,  # 1 MB
    min_lsn=1000,
    max_lsn=2000,
    created_at=datetime.now(),
    entry_count=1000
)
```

#### CheckpointInfo
```python
from t4dm.visualization import CheckpointInfo

checkpoint = CheckpointInfo(
    lsn=2000,
    timestamp=datetime.now(),
    size_bytes=5 * 1024 * 1024,  # 5 MB
    components=["episodic", "semantic", "graph"],
    duration_seconds=2.5
)
```

#### PersistenceMetrics
```python
metrics = PersistenceMetrics(
    current_lsn=12345,
    checkpoint_lsn=12000,
    operations_since_checkpoint=345,
    wal_segment_count=3,
    wal_total_size_bytes=15_000_000,
    checkpoint_count=10,
    last_checkpoint_age_seconds=180.0,
    recovery_mode="warm_start"  # or "cold_start", "unknown"
)
```

**Key Methods**:

#### record_wal_segment()
```python
visualizer.record_wal_segment(segment)
```

#### record_checkpoint()
```python
visualizer.record_checkpoint(checkpoint)
```

#### update_metrics()
```python
visualizer.update_metrics(metrics)
```

#### get_wal_size_over_time()
```python
timestamps, sizes = visualizer.get_wal_size_over_time()
```

#### get_lsn_over_time()
```python
timestamps, lsns = visualizer.get_lsn_over_time()
```

#### get_checkpoint_timeline()
```python
timestamps, lsns = visualizer.get_checkpoint_timeline()
```

### plot_wal_timeline()

Visualizes WAL segment sizes and LSN progression.

```python
from t4dm.visualization import plot_wal_timeline

plot_wal_timeline(
    visualizer,
    interactive=True,
    save_path=Path("outputs/wal_timeline.html")
)
```

**Features**:
- Two subplots: WAL segment sizes (bar chart) and LSN progression (line chart)
- Checkpoint markers as red diamonds on LSN plot
- Hover tooltips showing segment details (LSN ranges, sizes)
- Color-coded for easy reading

### plot_durability_dashboard()

Real-time durability metrics dashboard.

```python
from t4dm.visualization import plot_durability_dashboard

plot_durability_dashboard(
    visualizer,
    interactive=True,
    save_path=Path("outputs/durability_dashboard.html")
)
```

**Features**:
- Four metric panels:
  1. **Current LSN**: Latest log sequence number
  2. **Uncommitted Ops**: Operations since last checkpoint (color-coded health)
  3. **Checkpoint Age**: Time since last checkpoint (color-coded health)
  4. **WAL Size**: Total WAL size in MB
- Color indicators:
  - Green: Healthy
  - Orange: Warning
  - Red: Critical
- Recovery mode displayed in title

**Health Thresholds**:
- Uncommitted ops: < 1000 (green), < 5000 (orange), >= 5000 (red)
- Checkpoint age: < 300s (green), < 600s (orange), >= 600s (red)

### plot_checkpoint_history()

Checkpoint size and duration over time.

```python
from t4dm.visualization import plot_checkpoint_history

plot_checkpoint_history(
    visualizer,
    interactive=True,
    save_path=Path("outputs/checkpoint_history.html")
)
```

**Features**:
- Two subplots: checkpoint size (KB) and duration (seconds)
- Line charts with markers
- Identifies trends in checkpoint performance

### Example Usage

```python
from t4dm.visualization import (
    PersistenceVisualizer,
    WALSegmentInfo,
    CheckpointInfo,
    PersistenceMetrics,
    plot_wal_timeline,
    plot_durability_dashboard,
    plot_checkpoint_history
)
from pathlib import Path
from datetime import datetime, timedelta
import time

visualizer = PersistenceVisualizer()

# Simulate WAL segments
base_time = datetime.now()
current_lsn = 0

for i in range(10):
    segment_size = np.random.randint(500_000, 2_000_000)
    entry_count = segment_size // 1000

    segment = WALSegmentInfo(
        segment_number=i,
        path=Path(f"data/wal/segment_{i:03d}.wal"),
        size_bytes=segment_size,
        min_lsn=current_lsn,
        max_lsn=current_lsn + entry_count,
        created_at=base_time + timedelta(seconds=i * 60),
        entry_count=entry_count
    )

    visualizer.record_wal_segment(segment)
    current_lsn += entry_count

# Simulate checkpoints
for i in range(3):
    checkpoint = CheckpointInfo(
        lsn=current_lsn // 3 * (i + 1),
        timestamp=base_time + timedelta(seconds=(i + 1) * 300),
        size_bytes=np.random.randint(3_000_000, 8_000_000),
        components=["episodic", "semantic", "graph"],
        duration_seconds=np.random.uniform(1.5, 4.0)
    )

    visualizer.record_checkpoint(checkpoint)

# Update metrics over time
for t in range(50):
    metrics = PersistenceMetrics(
        current_lsn=current_lsn + t * 100,
        checkpoint_lsn=current_lsn,
        operations_since_checkpoint=t * 100,
        wal_segment_count=len(visualizer.wal_segments),
        wal_total_size_bytes=sum(s.size_bytes for s in visualizer.wal_segments),
        checkpoint_count=len(visualizer.checkpoints),
        last_checkpoint_age_seconds=t * 10,
        recovery_mode="warm_start"
    )

    visualizer.update_metrics(metrics)
    time.sleep(0.1)

# Visualize
plot_wal_timeline(visualizer, interactive=True)
plot_durability_dashboard(visualizer, interactive=True)
plot_checkpoint_history(visualizer, interactive=True)
```

---

## 9. Advanced Usage Patterns

### Integration with T4DM Components

#### Pattern Separation Integration

```python
from t4dm.memory.pattern_separation import DentateGyrus
from t4dm.visualization import PatternSeparationVisualizer, plot_separation_comparison

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

plot_separation_comparison(
    np.array([original_emb]),
    np.array([separated_emb]),
    interactive=True
)
```

#### Neuromodulator Integration

```python
from t4dm.learning.neuromodulators import NeuromodulatorOrchestra
from t4dm.visualization import NeuromodulatorDashboard, plot_neuromodulator_traces

orchestra = NeuromodulatorOrchestra()
dashboard = NeuromodulatorDashboard()

# Process queries and record state
for query in queries:
    query_embedding = await embedding_provider.embed_query(query)
    state = orchestra.process_query(query_embedding, is_question=True)

    dashboard.record_state(
        dopamine_rpe=state.dopamine_rpe,
        norepinephrine_gain=state.norepinephrine_gain,
        acetylcholine_mode=state.acetylcholine_mode,
        serotonin_mood=state.serotonin_mood,
        gaba_sparsity=state.inhibition_sparsity
    )

# Visualize dynamics
plot_neuromodulator_traces(dashboard, interactive=True)
```

#### Plasticity Tracking During Learning

```python
from t4dm.learning.synaptic_plasticity import BCMPlasticity
from t4dm.visualization import PlasticityTracer, plot_bcm_curve

plasticity = BCMPlasticity()
tracer = PlasticityTracer()

# Track weight updates during learning
for source, target, old_w, new_w, activation in training_updates:
    tracer.record_update(
        source_id=source,
        target_id=target,
        old_weight=old_w,
        new_weight=new_w,
        update_type="ltp" if new_w > old_w else "ltd",
        activation_level=activation
    )

# Visualize learning dynamics
plot_bcm_curve(tracer, interactive=True)
plot_ltp_ltd_distribution(tracer, interactive=True)
```

### Multi-System Analysis

Combine multiple visualizations to understand system-wide dynamics:

```python
from t4dm.visualization import *
from pathlib import Path

output_dir = Path("outputs/system_analysis")
output_dir.mkdir(exist_ok=True)

# Activation patterns
plot_activation_heatmap(
    activation_tracker,
    memory_type="episodic",
    save_path=output_dir / "activation_episodic.html",
    interactive=True
)

# Neuromodulator state
plot_neuromodulator_traces(
    neuromod_dashboard,
    save_path=output_dir / "neuromod_traces.html",
    interactive=True
)

# Plasticity dynamics
plot_bcm_curve(
    plasticity_tracer,
    save_path=output_dir / "bcm_curve.png",
    interactive=False
)

# Embedding space
plot_umap_projection(
    memory_embeddings,
    labels=memory_ids,
    colors=priority_scores,
    save_path=output_dir / "embedding_space.html",
    interactive=True
)

# Consolidation replay
plot_replay_priority(
    consolidation_visualizer,
    save_path=output_dir / "replay_priority.png",
    interactive=False
)

# Persistence metrics
plot_durability_dashboard(
    persistence_visualizer,
    save_path=output_dir / "durability_dashboard.html",
    interactive=True
)
```

### Real-Time Dashboard

Create a real-time monitoring dashboard:

```python
import asyncio
from datetime import datetime

class MemorySystemDashboard:
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.activation_tracker = ActivationHeatmap()
        self.neuromod_dashboard = NeuromodulatorDashboard()
        self.plasticity_tracer = PlasticityTracer()
        self.persistence_visualizer = PersistenceVisualizer()

    async def update(self):
        """Update all trackers with current system state."""
        # Record activation snapshot
        self.activation_tracker.record_snapshot(
            episodic_activations=await self.memory_system.get_episodic_activations(),
            semantic_activations=await self.memory_system.get_semantic_activations(),
            neuromod_state=self.memory_system.neuromodulators.get_state()
        )

        # Record neuromodulator state
        state = self.memory_system.neuromodulators.get_state()
        self.neuromod_dashboard.record_state(**state)

        # Record plasticity updates
        for update in self.memory_system.get_recent_plasticity_updates():
            self.plasticity_tracer.record_update(**update)

        # Update persistence metrics
        metrics = await self.memory_system.persistence.get_metrics()
        self.persistence_visualizer.update_metrics(metrics)

    def generate_report(self, output_dir: Path):
        """Generate comprehensive visualization report."""
        output_dir.mkdir(exist_ok=True)

        plot_activation_heatmap(
            self.activation_tracker,
            save_path=output_dir / "activation.html",
            interactive=True
        )

        plot_neuromodulator_traces(
            self.neuromod_dashboard,
            save_path=output_dir / "neuromod.html",
            interactive=True
        )

        plot_bcm_curve(
            self.plasticity_tracer,
            save_path=output_dir / "plasticity.html",
            interactive=True
        )

        plot_durability_dashboard(
            self.persistence_visualizer,
            save_path=output_dir / "durability.html",
            interactive=True
        )

# Usage
dashboard = MemorySystemDashboard(memory_system)

async def monitor():
    while True:
        await dashboard.update()
        await asyncio.sleep(1)

# Generate report every hour
async def periodic_report():
    while True:
        dashboard.generate_report(Path("outputs/reports") / datetime.now().strftime("%Y%m%d_%H%M%S"))
        await asyncio.sleep(3600)
```

### Publication-Ready Figures

Tips for creating publication-ready figures:

```python
# Use matplotlib for static, high-resolution figures
plot_bcm_curve(
    tracer,
    save_path=Path("figures/bcm_curve.pdf"),  # PDF for vector graphics
    interactive=False
)

# Customize matplotlib figures
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Then call plot functions
plot_ltp_ltd_distribution(tracer, save_path=Path("figures/ltp_ltd.pdf"))

# For multi-panel figures, use subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Manually create visualizations in each subplot
# (requires accessing internal visualization code)
```

### Performance Optimization

Tips for handling large datasets:

```python
# Use UMAP instead of t-SNE for large datasets
if len(embeddings) > 1000:
    plot_umap_projection(embeddings, labels, colors)
else:
    plot_tsne_projection(embeddings, labels, colors)

# Limit heatmap size
tracker = ActivationHeatmap(
    window_size=100,         # Keep window small
    max_memories_tracked=50  # Limit memories shown
)

# Use PCA for quick exploration before UMAP/t-SNE
projector = EmbeddingProjector()
pca_proj = projector.project_pca(embeddings)  # Fast preview
# If interesting, compute full UMAP
umap_proj = projector.project_umap(embeddings)

# Batch updates for better performance
updates = []
for source, target, weights in batch:
    updates.append({
        "source_id": source,
        "target_id": target,
        "old_weight": weights[0],
        "new_weight": weights[1],
        "update_type": "ltp",
        "activation_level": 0.8
    })

# Record in bulk
for update in updates:
    tracer.record_update(**update)
```

---

## Biological Basis Summary

Each visualization module corresponds to actual neuroscience mechanisms:

1. **Activation Heatmap**: Neural population activity patterns observed via fMRI, calcium imaging
2. **BCM Curve**: Bienenstock-Cooper-Munro theory of synaptic modification (1982)
3. **LTP/LTD**: Long-term potentiation and depression, basis of learning and memory
4. **Neuromodulators**: Dopamine (reward), norepinephrine (arousal), acetylcholine (attention), serotonin (mood), GABA (inhibition)
5. **Pattern Separation**: Dentate gyrus orthogonalization of similar inputs (Leutgeb et al., 2007)
6. **SWR Replay**: Sharp-wave ripples in hippocampus during sleep consolidation (Buzsáki, 2015)
7. **Homeostatic Scaling**: TNFα-mediated synaptic scaling for stability (Turrigiano, 2008)

---

## References

1. **Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982)**. Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. *Journal of Neuroscience, 2*(1), 32-48.

2. **Turrigiano, G. (2008)**. The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell, 135*(3), 422-435.

3. **Buzsáki, G. (2015)**. Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning. *Hippocampus, 25*(10), 1073-1188.

4. **Leutgeb, J. K., Leutgeb, S., Moser, M. B., & Moser, E. I. (2007)**. Pattern separation in the dentate gyrus and CA3 of the hippocampus. *Science, 315*(5814), 961-966.

5. **Schultz, W. (1998)**. Predictive reward signal of dopamine neurons. *Journal of Neurophysiology, 80*(1), 1-27.

---

## Troubleshooting

### Common Issues

**Issue**: ImportError for plotly or matplotlib
```python
# Solution: Install missing dependencies
pip install matplotlib plotly

# For UMAP
pip install umap-learn
```

**Issue**: t-SNE is very slow
```python
# Solution: Use UMAP or PCA for large datasets
if len(embeddings) > 1000:
    projection = projector.project_umap(embeddings)
else:
    projection = projector.project_tsne(embeddings)
```

**Issue**: Heatmap is too crowded
```python
# Solution: Reduce max_memories_tracked
tracker = ActivationHeatmap(
    window_size=100,
    max_memories_tracked=20  # Show fewer memories
)
```

**Issue**: Figures have memory leaks
```python
# Solution: Always close matplotlib figures
import matplotlib.pyplot as plt

plot_bcm_curve(tracer, interactive=False)
plt.close('all')  # Close all figures

# Or use context manager
with plt.rc_context():
    plot_bcm_curve(tracer)
# Figures auto-closed after block
```

**Issue**: Colors don't show variation
```python
# Solution: Normalize color values
colors = (colors - colors.min()) / (colors.max() - colors.min())
plot_tsne_projection(embeddings, colors=colors)
```

---

## Complete Example Script

See `/mnt/projects/t4d/t4dm/examples/visualization_demo.py` for a comprehensive demonstration of all visualization modules.

Run the demo:
```bash
cd /mnt/projects/ww
python examples/visualization_demo.py
```

---

## License

Part of the T4DM project. See main repository for license details.
