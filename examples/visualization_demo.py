#!/usr/bin/env python
"""
Demonstration of World Weaver visualization modules.

This script shows how to use each visualization module to analyze
and understand the neurocomputational dynamics of the memory system.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ww.visualization import (
    ActivationHeatmap,
    plot_activation_heatmap,
    plot_activation_timeline,
    PlasticityTracer,
    plot_bcm_curve,
    plot_weight_changes,
    plot_ltp_ltd_distribution,
    NeuromodulatorDashboard,
    plot_neuromodulator_traces,
    plot_neuromodulator_radar,
    PatternSeparationVisualizer,
    plot_separation_comparison,
    plot_sparsity_distribution,
    ConsolidationVisualizer,
    plot_swr_sequence,
    plot_replay_priority,
    EmbeddingProjector,
    plot_tsne_projection,
    plot_umap_projection,
)


def demo_activation_heatmap():
    """Demonstrate activation heatmap visualization."""
    print("\n=== Activation Heatmap Demo ===")

    tracker = ActivationHeatmap(window_size=50, max_memories_tracked=20)

    # Simulate activation patterns over time
    for t in range(50):
        # Create simulated activations
        n_episodic = np.random.randint(5, 15)
        n_semantic = np.random.randint(5, 10)

        episodic_acts = {
            f"ep_{i}": np.random.rand() for i in range(n_episodic)
        }
        semantic_acts = {
            f"sem_{i}": np.random.rand() for i in range(n_semantic)
        }

        neuromod_state = {
            "dopamine": np.random.randn() * 0.3,
            "norepinephrine": 0.5 + np.random.rand() * 1.5,
            "serotonin": np.random.rand(),
            "gaba": np.random.rand() * 0.2
        }

        tracker.record_snapshot(episodic_acts, semantic_acts, neuromod_state)

    # Plot
    print("Plotting activation heatmap...")
    plot_activation_heatmap(tracker, memory_type="episodic")

    print("Plotting neuromodulator timeline...")
    plot_activation_timeline(tracker)


def demo_plasticity_traces():
    """Demonstrate plasticity trace visualization."""
    print("\n=== Plasticity Traces Demo ===")

    tracer = PlasticityTracer(max_updates=1000)

    # Simulate weight updates following BCM-like dynamics
    for i in range(500):
        activation = np.random.rand()
        theta_m = 0.5  # Modification threshold

        # BCM rule: delta_w proportional to activation * (activation - theta_m)
        delta_w = activation * (activation - theta_m) * 0.1

        update_type = "ltp" if delta_w > 0 else "ltd"
        old_weight = 0.5 + np.random.randn() * 0.1
        new_weight = old_weight + delta_w

        tracer.record_update(
            source_id=f"src_{i % 20}",
            target_id=f"tgt_{i % 15}",
            old_weight=old_weight,
            new_weight=new_weight,
            update_type=update_type,
            activation_level=activation
        )

    # Add some homeostatic scaling events
    for i in range(100):
        tracer.record_update(
            source_id=f"src_{i % 20}",
            target_id=f"tgt_{i % 15}",
            old_weight=np.random.rand(),
            new_weight=np.random.rand() * 0.8,
            update_type="homeostatic",
            activation_level=0.0
        )

    # Plot BCM curve
    print("Plotting BCM learning curve...")
    plot_bcm_curve(tracer)

    # Plot weight change timeline
    print("Plotting weight changes over time...")
    plot_weight_changes(tracer)

    # Plot LTP/LTD distribution
    print("Plotting LTP/LTD distributions...")
    plot_ltp_ltd_distribution(tracer)


def demo_neuromodulator_state():
    """Demonstrate neuromodulator state visualization."""
    print("\n=== Neuromodulator State Demo ===")

    dashboard = NeuromodulatorDashboard(window_size=200)

    # Simulate neuromodulator dynamics
    modes = ["encoding", "balanced", "retrieval"]
    mode_idx = 0

    for t in range(200):
        # Dopamine: random walk with occasional spikes
        da_rpe = np.random.randn() * 0.3
        if np.random.rand() < 0.05:  # 5% chance of surprise
            da_rpe += np.random.choice([-1.0, 1.0])

        # Norepinephrine: arousal with novelty spikes
        ne_gain = 1.0 + np.random.rand() * 0.5
        if np.random.rand() < 0.1:  # 10% chance of novelty
            ne_gain += 0.5

        # Acetylcholine: mode switches
        if t % 50 == 0:
            mode_idx = (mode_idx + 1) % len(modes)
        ach_mode = modes[mode_idx]

        # Serotonin: slow drift with mean reversion
        ht_mood = 0.5 + np.random.randn() * 0.1

        # GABA: sparsity varies
        gaba_sparsity = 0.05 + np.random.rand() * 0.15

        dashboard.record_state(da_rpe, ne_gain, ach_mode, ht_mood, gaba_sparsity)

    # Plot traces
    print("Plotting neuromodulator traces...")
    plot_neuromodulator_traces(dashboard)

    # Plot radar chart
    print("Plotting current state radar...")
    plot_neuromodulator_radar(dashboard)

    # Print statistics
    stats = dashboard.get_statistics()
    print("\nNeuromodulator statistics:")
    for mod, values in stats.items():
        print(f"  {mod}: mean={values['mean']:.3f}, std={values['std']:.3f}")


def demo_pattern_separation():
    """Demonstrate pattern separation visualization."""
    print("\n=== Pattern Separation Demo ===")

    # Generate original embeddings (similar cluster)
    n_samples = 30
    dim = 128

    # Create a cluster of similar vectors
    center = np.random.randn(dim)
    center /= np.linalg.norm(center)

    original = []
    for _ in range(n_samples):
        # Add noise to center
        noisy = center + np.random.randn(dim) * 0.2
        noisy /= np.linalg.norm(noisy)
        original.append(noisy)

    original = np.array(original)

    # Simulate pattern separation
    separated = []
    for i, emb in enumerate(original):
        # Orthogonalize against cluster centroid
        centroid = original.mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        # Remove component in centroid direction
        projection = np.dot(emb, centroid) * centroid
        orthogonalized = emb - 0.3 * projection

        # Add random perturbation
        noise = np.random.randn(dim) * 0.05
        orthogonalized += noise

        # Sparsify
        threshold = np.percentile(np.abs(orthogonalized), 95)
        sparse = np.where(np.abs(orthogonalized) > threshold, orthogonalized, 0)

        # Normalize
        sparse /= np.linalg.norm(sparse) if np.linalg.norm(sparse) > 0 else 1

        separated.append(sparse)

    separated = np.array(separated)

    # Analyze
    vis = PatternSeparationVisualizer()
    stats = vis.analyze_separation(original, separated)

    print("\nPattern separation statistics:")
    print(f"  Mean similarity before: {stats['mean_similarity_before']:.3f}")
    print(f"  Mean similarity after: {stats['mean_similarity_after']:.3f}")
    print(f"  Similarity reduction: {stats['similarity_reduction']:.3f}")
    print(f"  Mean sparsity before: {stats['mean_sparsity_before']:.3f}")
    print(f"  Mean sparsity after: {stats['mean_sparsity_after']:.3f}")

    # Plot comparisons
    print("\nPlotting similarity matrix comparison...")
    plot_separation_comparison(original, separated)

    print("Plotting sparsity distributions...")
    plot_sparsity_distribution(original, separated)


def demo_consolidation_replay():
    """Demonstrate consolidation replay visualization."""
    print("\n=== Consolidation Replay Demo ===")

    visualizer = ConsolidationVisualizer()

    # Simulate NREM replay sequences
    for seq_num in range(10):
        seq_length = np.random.randint(3, 8)
        memory_ids = [f"mem_{np.random.randint(0, 50)}" for _ in range(seq_length)]
        priorities = np.random.beta(2, 5, seq_length)  # Skewed toward high priority

        visualizer.record_replay_sequence(memory_ids, list(priorities), phase="nrem")

    # Simulate REM replay sequences
    for seq_num in range(5):
        seq_length = np.random.randint(4, 6)
        memory_ids = [f"mem_{np.random.randint(0, 50)}" for _ in range(seq_length)]
        priorities = np.random.beta(1.5, 1.5, seq_length)  # More uniform

        visualizer.record_replay_sequence(memory_ids, list(priorities), phase="rem")

    # Plot first sequence
    print("Plotting first SWR sequence...")
    plot_swr_sequence(visualizer, sequence_index=0)

    # Plot priority distributions
    print("Plotting replay priority distributions...")
    plot_replay_priority(visualizer)


def demo_embedding_projections():
    """Demonstrate embedding projection visualization."""
    print("\n=== Embedding Projections Demo ===")

    # Generate high-dimensional embeddings with structure
    n_clusters = 3
    samples_per_cluster = 20
    dim = 1024

    embeddings = []
    labels = []
    colors = []

    for cluster in range(n_clusters):
        # Cluster center
        center = np.random.randn(dim)
        center /= np.linalg.norm(center)

        for sample in range(samples_per_cluster):
            # Add noise
            noisy = center + np.random.randn(dim) * 0.3
            noisy /= np.linalg.norm(noisy)

            embeddings.append(noisy)
            labels.append(f"c{cluster}_s{sample}")
            colors.append(cluster)

    embeddings = np.array(embeddings)
    colors = np.array(colors)

    projector = EmbeddingProjector()

    # Plot t-SNE
    print("Computing and plotting t-SNE projection...")
    plot_tsne_projection(
        embeddings,
        labels=labels,
        colors=colors,
        projector=projector
    )

    # Plot UMAP
    print("Computing and plotting UMAP projection...")
    plot_umap_projection(
        embeddings,
        labels=labels,
        colors=colors,
        projector=projector
    )


def main():
    """Run all visualization demos."""
    print("=" * 60)
    print("World Weaver Visualization Demo")
    print("=" * 60)

    # Run demos
    demo_activation_heatmap()
    demo_plasticity_traces()
    demo_neuromodulator_state()
    demo_pattern_separation()
    demo_consolidation_replay()
    demo_embedding_projections()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
