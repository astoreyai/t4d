#!/usr/bin/env python3
"""
NCA Module Demo - Neuro Cognitive Architecture for World Weaver

Demonstrates the KATIE-inspired neural dynamics integrated with WW's
memory system. Shows:
1. Neural field PDE dynamics (6-neurotransmitter system)
2. Learnable coupling matrix
3. Cognitive state attractors and transitions
4. Energy-based learning (Hopfield, Contrastive Divergence)
5. Memory-NCA bridge for augmented encoding/retrieval

Usage:
    python examples/nca_demo.py
"""

import numpy as np
from uuid import uuid4
import sys
sys.path.insert(0, 'src')

from ww.nca import (
    # Neural Field
    NeuralFieldConfig,
    NeuralFieldSolver,
    NeurotransmitterState,
    # Coupling
    LearnableCoupling,
    BiologicalBounds,
    CouplingConfig,
    # Attractors
    CognitiveState,
    AttractorBasin,
    StateTransitionManager,
    # Energy
    EnergyLandscape,
    HopfieldIntegration,
    EnergyConfig,
    EnergyBasedLearner,
    LearningPhase,
)
from ww.bridge import (
    MemoryNCABridge,
    BridgeConfig,
    EncodingContext,
)


def demo_neural_field():
    """Demonstrate neural field PDE dynamics."""
    print("\n" + "="*60)
    print("1. NEURAL FIELD PDE DYNAMICS")
    print("="*60)

    # Create solver with biologically-plausible parameters
    config = NeuralFieldConfig(
        dt=0.001,
        grid_size=16,
        spatial_dims=1,
        # Decay rates (1/timescale in seconds)
        alpha_da=10.0,    # Dopamine: ~100ms
        alpha_5ht=2.0,    # Serotonin: ~500ms
        alpha_ach=20.0,   # Acetylcholine: ~50ms
        alpha_ne=5.0,     # Norepinephrine: ~200ms
        alpha_gaba=100.0, # GABA: ~10ms (fast inhibition)
        alpha_glu=200.0,  # Glutamate: ~5ms (fast excitation)
        # Diffusion coefficients
        diffusion_da=0.05,
        diffusion_5ht=0.02,
    )
    solver = NeuralFieldSolver(config=config)

    print(f"Grid: {config.spatial_dims}D, size={config.grid_size}")
    print(f"Timestep: {config.dt}s")
    print(f"NT decay rates: DA={config.alpha_da}, 5-HT={config.alpha_5ht}, "
          f"ACh={config.alpha_ach}, NE={config.alpha_ne}")

    # Get initial state
    initial = solver.get_mean_state()
    print(f"\nInitial state (baseline): {initial}")

    # Inject dopamine stimulus (reward signal)
    from ww.nca.neural_field import NeurotransmitterType
    solver.inject_stimulus(NeurotransmitterType.DOPAMINE, magnitude=0.3)
    print("\nInjected dopamine stimulus (magnitude=0.3)")

    # Evolve the system
    print("\nEvolving dynamics for 100 steps...")
    for i in range(100):
        solver.step()
        if i % 25 == 0:
            state = solver.get_mean_state()
            print(f"  Step {i:3d}: DA={state.dopamine:.3f}, "
                  f"5-HT={state.serotonin:.3f}, NE={state.norepinephrine:.3f}")

    final = solver.get_mean_state()
    print(f"\nFinal state: DA={final.dopamine:.3f}, 5-HT={final.serotonin:.3f}")
    stats = solver.get_stats()
    print(f"Stats: {stats['step_count']} steps, {stats['time']:.4f}s simulated")


def demo_coupling():
    """Demonstrate learnable coupling matrix."""
    print("\n" + "="*60)
    print("2. LEARNABLE COUPLING MATRIX")
    print("="*60)

    # Create coupling with biological bounds
    config = CouplingConfig(
        learning_rate=0.01,
        enforce_ei_balance=True,
    )
    coupling = LearnableCoupling(config=config)

    print("Initial coupling matrix K:")
    K = coupling.K.copy()
    print(f"  Shape: {K.shape}")
    print(f"  DA->5-HT: {K[0,1]:.3f} (antagonistic)")
    print(f"  ACh->DA:  {K[2,0]:.3f} (modulatory)")
    print(f"  GABA->Glu: {K[4,5]:.3f} (inhibitory)")

    # Apply RPE-based update (positive reward prediction error)
    U = np.array([0.7, 0.4, 0.6, 0.5, 0.3, 0.6])  # Current NT state
    rpe = 0.5  # Positive surprise

    print(f"\nApplying RPE update (rpe={rpe})...")
    coupling.update_from_rpe(rpe, U)

    K_new = coupling.K
    print(f"  DA->5-HT: {K[0,1]:.3f} -> {K_new[0,1]:.3f}")
    print(f"  Coupling strength: {coupling.get_coupling_strength():.3f}")

    # Compute coupling effect
    effect = coupling.compute_coupling(U)
    print(f"\nCoupling effect on state: {effect}")


def demo_attractors():
    """Demonstrate cognitive state attractors."""
    print("\n" + "="*60)
    print("3. COGNITIVE STATE ATTRACTORS")
    print("="*60)

    # Create state manager with hysteresis
    manager = StateTransitionManager(hysteresis=0.15)

    print("Cognitive states and their NT signatures:")
    for state in CognitiveState:
        attractor = manager.attractors[state]
        print(f"  {state.name:12s}: center={attractor.center[:3]}..., "
              f"width={attractor.width:.2f}, stability={attractor.stability:.2f}")

    print(f"\nInitial state: {manager.get_current_state().name}")

    # Simulate transition to FOCUS
    print("\nSimulating shift toward FOCUS state...")
    focus_center = manager.attractors[CognitiveState.FOCUS].center

    # Gradual approach
    current = manager.attractors[CognitiveState.REST].center.copy()
    for i in range(10):
        # Move toward FOCUS
        current = current * 0.8 + focus_center * 0.2
        nt_state = NeurotransmitterState.from_array(current)
        transition = manager.update(nt_state, dt=0.1)

        if transition:
            print(f"  Step {i}: TRANSITION {transition.from_state.name} -> "
                  f"{transition.to_state.name}")
        else:
            classified, dist = manager.classify_state(nt_state)
            print(f"  Step {i}: state={manager.get_current_state().name}, "
                  f"nearest={classified.name}, dist={dist:.3f}")


def demo_energy():
    """Demonstrate energy-based learning."""
    print("\n" + "="*60)
    print("4. ENERGY-BASED LEARNING")
    print("="*60)

    # Create energy landscape
    coupling = LearnableCoupling()
    manager = StateTransitionManager()
    landscape = EnergyLandscape(
        coupling=coupling,
        state_manager=manager,
        config=EnergyConfig(temperature=1.0)
    )

    # Compute energies at different states
    print("Energy at cognitive state centers:")
    for state in [CognitiveState.REST, CognitiveState.FOCUS, CognitiveState.ALERT]:
        center = manager.attractors[state].center
        energy = landscape.compute_total_energy(center)
        print(f"  {state.name:8s}: E = {energy:.3f}")

    # Modern Hopfield network
    print("\nModern Hopfield Network (exponential capacity):")
    hopfield = HopfieldIntegration(dim=64, num_patterns=100, beta=2.0)

    # Store patterns
    patterns = []
    for i in range(5):
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        nt = np.random.rand(6).astype(np.float32)
        hopfield.store_pattern(emb, nt)
        patterns.append((emb, nt))

    print(f"  Stored {len(patterns)} patterns")

    # Retrieve with noisy query
    query = patterns[0][0] + np.random.randn(64).astype(np.float32) * 0.2
    query = query / np.linalg.norm(query)

    retrieved, nt_retrieved, similarity = hopfield.retrieve(query, np.zeros(6))
    print(f"  Query similarity to pattern[0]: {similarity:.3f}")
    print(f"  Retrieved NT state: {nt_retrieved[:3]}...")


def demo_bridge():
    """Demonstrate Memory-NCA bridge."""
    print("\n" + "="*60)
    print("5. MEMORY-NCA BRIDGE")
    print("="*60)

    # Create full system
    field = NeuralFieldSolver()
    coupling = LearnableCoupling()
    manager = StateTransitionManager()
    energy = EnergyLandscape(coupling=coupling, state_manager=manager)

    bridge = MemoryNCABridge(
        neural_field=field,
        coupling=coupling,
        state_manager=manager,
        energy_landscape=energy,
        config=BridgeConfig(
            encoding_nt_weight=0.3,
            focus_boost=1.5,
            retrieval_state_matching=True,
        )
    )

    print("Bridge configured with NT-augmented encoding")
    print(f"  Encoding NT weight: {bridge.config.encoding_nt_weight}")
    print(f"  Focus boost: {bridge.config.focus_boost}x")

    # Encode memory in FOCUS state
    manager.force_state(CognitiveState.FOCUS)
    print(f"\nEncoding memory in {manager.get_current_state().name} state...")

    embedding = np.random.randn(768).astype(np.float32)
    memory_id = uuid4()

    augmented, context = bridge.augment_encoding(embedding, memory_id)
    print(f"  Original embedding: {len(embedding)} dims")
    print(f"  Augmented embedding: {len(augmented)} dims")
    print(f"  Encoding cognitive state: {context.cognitive_state.name}")
    print(f"  Encoding NT state: DA={context.nt_state[0]:.2f}, "
          f"ACh={context.nt_state[2]:.2f}")

    # Simulate time passing
    print("\nSimulating time passage (state transitions)...")
    for _ in range(50):
        bridge.step(dt=0.01)

    # Switch to REST and retrieve
    manager.force_state(CognitiveState.REST)
    print(f"\nRetrieval in {manager.get_current_state().name} state...")

    # Create some candidate memories
    candidates = [context]  # Include the one we encoded
    for i in range(4):
        ctx = EncodingContext(
            memory_id=uuid4(),
            embedding=np.random.randn(768),
            nt_state=np.random.rand(6),
            cognitive_state=list(CognitiveState)[i % 5]
        )
        candidates.append(ctx)

    query = embedding + np.random.randn(768).astype(np.float32) * 0.1
    ranked, ret_ctx = bridge.modulate_retrieval(query, candidates, top_k=3)

    print(f"  Retrieved top 3 (state-aware ranking):")
    for i, mem_id in enumerate(ranked):
        sim = ret_ctx.state_similarities[i] if i < len(ret_ctx.state_similarities) else 0
        print(f"    {i+1}. {str(mem_id)[:8]}... (state_sim={sim:.3f})")

    # Learning signal
    print("\nComputing learning signal from retrieval outcome...")
    signal = bridge.compute_learning_signal(memory_id, outcome=0.9)
    print(f"  Outcome: {signal['outcome']}")
    print(f"  RPE: {signal['rpe']:.3f}")
    print(f"  Effective LR: {signal['effective_lr']:.4f}")


def demo_full_cycle():
    """Demonstrate complete encode-consolidate-retrieve cycle."""
    print("\n" + "="*60)
    print("6. FULL CYCLE: ENCODE -> CONSOLIDATE -> RETRIEVE")
    print("="*60)

    # Initialize system
    field = NeuralFieldSolver()
    coupling = LearnableCoupling()
    manager = StateTransitionManager()
    energy = EnergyLandscape(coupling=coupling, state_manager=manager)
    bridge = MemoryNCABridge(
        neural_field=field,
        coupling=coupling,
        state_manager=manager,
        energy_landscape=energy,
    )

    # Phase 1: Encoding (FOCUS state, high ACh)
    print("\n[PHASE 1: ENCODING]")
    manager.force_state(CognitiveState.FOCUS)
    from ww.nca.neural_field import NeurotransmitterType
    field.inject_stimulus(NeurotransmitterType.ACETYLCHOLINE, 0.3)

    memories = []
    for i in range(5):
        emb = np.random.randn(768).astype(np.float32)
        mem_id = uuid4()
        aug, ctx = bridge.augment_encoding(emb, mem_id)
        memories.append(ctx)
        print(f"  Encoded memory {i+1}: {str(mem_id)[:8]}...")

    # Phase 2: Consolidation (CONSOLIDATE state, replay)
    print("\n[PHASE 2: CONSOLIDATION]")
    manager.force_state(CognitiveState.CONSOLIDATE)

    # Simulate consolidation dynamics
    for step in range(100):
        bridge.step(dt=0.01)
        if step % 25 == 0:
            state = field.get_mean_state()
            print(f"  Step {step}: DA={state.dopamine:.2f}, "
                  f"GABA={state.gaba:.2f}, state={manager.get_current_state().name}")

    # Trigger replay
    replay_ids = bridge.trigger_consolidation()
    print(f"  Consolidated {len(replay_ids)} memories for replay")

    # Phase 3: Retrieval (REST state)
    print("\n[PHASE 3: RETRIEVAL]")
    manager.force_state(CognitiveState.REST)

    # Query similar to first memory
    query = memories[0].embedding + np.random.randn(768).astype(np.float32) * 0.15
    ranked, ret_ctx = bridge.modulate_retrieval(query, memories, top_k=3)

    print(f"  Query in {manager.get_current_state().name} state")
    print(f"  Top match: {str(ranked[0])[:8]}...")
    print(f"  Original:  {str(memories[0].memory_id)[:8]}...")
    print(f"  Match: {ranked[0] == memories[0].memory_id}")

    # Learning from outcome
    signal = bridge.compute_learning_signal(ranked[0], outcome=1.0)
    print(f"\n  Learning: RPE={signal['rpe']:.3f}, "
          f"coupling updated={signal.get('coupling_updated', False)}")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("  WORLD WEAVER NCA MODULE DEMO")
    print("  KATIE-Inspired Neural Dynamics")
    print("="*60)

    demo_neural_field()
    demo_coupling()
    demo_attractors()
    demo_energy()
    demo_bridge()
    demo_full_cycle()
