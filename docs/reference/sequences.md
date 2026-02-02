# Sequence & State Diagrams

Detailed behavioral diagrams for T4DM operations.

## State Machines

### Memory Gate State Machine

The learned memory gate controls what enters long-term memory.

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> EVALUATING: new_content
    EVALUATING --> PENDING: need_more_evidence
    EVALUATING --> ACCEPTED: evidence > θ
    EVALUATING --> REJECTED: evidence < -θ

    PENDING --> EVALUATING: co_retrieval_event
    PENDING --> DECAYED: decay_timeout
    PENDING --> ACCEPTED: accumulated_evidence > θ

    ACCEPTED --> STORING: begin_store
    STORING --> STORED: store_success
    STORING --> FAILED: store_error

    REJECTED --> IDLE: reset
    DECAYED --> IDLE: cleanup
    STORED --> IDLE: complete
    FAILED --> PENDING: retry

    note right of EVALUATING
        Bayesian LR inference
        Thompson sampling
        Evidence accumulation
    end note

    note right of PENDING
        Buffer storage
        Awaiting co-retrieval
        Max 24h retention
    end note

    note right of ACCEPTED
        Evidence threshold met
        Ready for LTM
    end note
```

### Neuromodulator State Machine

Tracks the state of each neuromodulator in the NCA system.

```mermaid
stateDiagram-v2
    [*] --> BASELINE

    state Dopamine {
        BASELINE --> BURST: positive_RPE
        BASELINE --> DIP: negative_RPE
        BURST --> BASELINE: decay(τ=100ms)
        DIP --> BASELINE: decay(τ=100ms)

        note right of BURST
            Reward > Expected
            Reinforcement signal
        end note

        note right of DIP
            Reward < Expected
            Extinction signal
        end note
    }

    state Norepinephrine {
        [*] --> DROWSY
        DROWSY --> ALERT: arousal_signal
        ALERT --> VIGILANT: threat_signal
        VIGILANT --> HYPERAROUSED: extreme_threat
        HYPERAROUSED --> ALERT: calm_signal
        ALERT --> DROWSY: fatigue
    }

    state Acetylcholine {
        [*] --> BALANCED
        BALANCED --> ENCODING: novel_input
        BALANCED --> RETRIEVAL: query_input
        ENCODING --> BALANCED: encode_complete
        RETRIEVAL --> BALANCED: recall_complete
    }

    state Serotonin {
        [*] --> NEUTRAL
        NEUTRAL --> ELEVATED: long_term_reward
        NEUTRAL --> DEPLETED: chronic_stress
        ELEVATED --> NEUTRAL: decay(τ=500ms)
        DEPLETED --> NEUTRAL: recovery
    }
```

### Consolidation State Machine

Sleep-based memory consolidation phases with delta oscillations and sleep spindles.

```mermaid
stateDiagram-v2
    [*] --> AWAKE

    AWAKE --> NREM1: consolidate_trigger
    AWAKE --> AWAKE: normal_operation

    state NREM1 {
        [*] --> LightSleep
        LightSleep: Alpha→Theta transition
    }

    NREM1 --> NREM2: deepening

    state NREM2 {
        [*] --> SpindleMonitor
        SpindleMonitor --> SpindleBurst: delta_up_state
        SpindleBurst --> SpindleMonitor: spindle_complete

        note right of SpindleBurst
            11-16 Hz bursts
            500-2000ms duration
            Gates HPC→CTX transfer
        end note
    }

    NREM2 --> NREM3: delta_onset

    state NREM3 {
        [*] --> DeltaDominant
        DeltaDominant --> UpState: phase_transition
        UpState --> DownState: ~500ms
        DownState --> DeltaDominant: cycle

        note right of UpState
            0.5-4 Hz delta
            Active consolidation
            Spindle trigger window
        end note

        note right of DownState
            Synaptic downscaling
            Homeostatic reset
        end note
    }

    NREM3 --> NREM2: lightening
    NREM2 --> REM: ~90min_cycle

    state REM {
        [*] --> Clustering
        Clustering --> Abstraction
        Abstraction --> Integration
        Integration --> [*]

        note right of Clustering
            HDBSCAN on semantic
            Find concept clusters
        end note

        note right of Abstraction
            LLM-based extraction
            Common patterns
        end note
    }

    REM --> PRUNING: phase_transition

    state PRUNING {
        [*] --> IdentifyWeak
        IdentifyWeak --> Scale
        Scale --> Remove
        Remove --> [*]

        note right of Scale
            Homeostatic scaling
            Target 3% activity
        end note
    }

    PRUNING --> AWAKE: consolidation_complete
    PRUNING --> NREM2: another_cycle
```

### Delta-Spindle Coordination

Detailed timing of delta oscillations and sleep spindles during NREM.

```mermaid
sequenceDiagram
    participant ADN as Adenosine
    participant Delta as DeltaOscillator
    participant TRN as Thalamic Reticular
    participant Spindle as SleepSpindleGenerator
    participant HPC as Hippocampus
    participant CTX as Cortex

    Note over ADN,CTX: Sleep onset (adenosine > 0.5)

    ADN->>Delta: High sleep pressure
    Delta->>Delta: Activate (sleep_depth > 0.3)

    loop Delta cycle (0.5-4 Hz)
        Delta->>Delta: DOWN-state (~500ms)
        Note right of Delta: Synaptic downscaling active

        Delta->>Delta: UP-state transition
        Delta->>TRN: Up-state onset signal
        TRN->>Spindle: Trigger spindle

        Spindle->>Spindle: RISING phase (~100ms)
        Spindle->>Spindle: PLATEAU phase (500-2000ms)

        par Memory transfer window
            Spindle->>HPC: Consolidation gate OPEN
            HPC->>CTX: Replay compressed memories
            CTX->>CTX: Integrate with semantic
        end

        Spindle->>Spindle: FALLING phase (~100ms)
        Spindle->>Spindle: REFRACTORY (~200ms)
    end

    Note over ADN,CTX: Adenosine cleared → wake transition
```

### Circuit Breaker State Machine

Protection against backend failures (also in storage-resilience.md).

```mermaid
stateDiagram-v2
    [*] --> CLOSED

    CLOSED --> CLOSED: success
    CLOSED --> CLOSED: failure (count < 5)
    CLOSED --> OPEN: failure (count >= 5)

    OPEN --> OPEN: request (reject)
    OPEN --> HALF_OPEN: timeout (60s)

    HALF_OPEN --> CLOSED: success (count >= 2)
    HALF_OPEN --> OPEN: failure

    state CLOSED {
        [*] --> Tracking
        Tracking: failure_count = 0..4
        Tracking: Normal operation
    }

    state OPEN {
        [*] --> Rejecting
        Rejecting: All requests fail fast
        Rejecting: Timer running
    }

    state HALF_OPEN {
        [*] --> Probing
        Probing: Limited requests allowed
        Probing: Testing recovery
    }
```

## Full Sequence Diagrams

### Complete Store Flow

End-to-end memory storage with all components.

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant PreHooks as Pre Hooks
    participant Gate as Memory Gate
    participant Embed as Embedding
    participant Buffer as Working Memory
    participant NCA as NCA Dynamics
    participant Saga as Saga Coordinator
    participant Neo4j
    participant Qdrant
    participant PostHooks as Post Hooks
    participant Learning

    Client->>API: store(content, importance)

    rect rgb(255, 243, 224)
        Note over API,PreHooks: Pre-Processing
        API->>PreHooks: pre_store(content)
        PreHooks->>PreHooks: CachingHook.check()
        PreHooks->>PreHooks: ValidationHook.validate()
        PreHooks-->>API: validated_content
    end

    rect rgb(227, 242, 253)
        Note over API,Embed: Embedding Generation
        API->>Embed: generate_embedding(content)
        Embed->>Embed: L1 cache check
        Embed->>Embed: L2 cache check
        Embed->>Embed: BGE-M3 inference
        Embed-->>API: embedding[1024]
    end

    rect rgb(232, 245, 233)
        Note over API,Gate: Gate Evaluation
        API->>Gate: evaluate(embedding, content)
        Gate->>Gate: compute_features()
        Gate->>Gate: bayesian_inference()
        Gate->>Gate: thompson_sample()

        alt Evidence > θ (Accept)
            Gate-->>API: GateDecision.ACCEPT
        else Evidence < -θ (Reject)
            Gate-->>API: GateDecision.REJECT
            API-->>Client: Rejected (not stored)
        else Need more evidence
            Gate->>Buffer: add_to_pending()
            Gate-->>API: GateDecision.PENDING
        end
    end

    rect rgb(255, 235, 238)
        Note over API,NCA: NCA State Update
        API->>NCA: process_store_event()
        NCA->>NCA: update_acetylcholine(ENCODING)
        NCA->>NCA: update_norepinephrine(novelty)
        NCA->>NCA: theta_gamma_coupling()
        NCA-->>API: nca_state
    end

    rect rgb(232, 245, 233)
        Note over API,Qdrant: Distributed Storage
        API->>Saga: begin_transaction()

        par Store to backends
            Saga->>Qdrant: upsert_vector(embedding)
            Saga->>Neo4j: create_node(episode)
        end

        Qdrant-->>Saga: point_id
        Neo4j-->>Saga: node_id

        Saga->>Saga: commit_transaction()
        Saga-->>API: episode_id
    end

    rect rgb(227, 242, 253)
        Note over API,Learning: Post-Processing & Learning
        API->>PostHooks: post_store(episode)
        PostHooks->>PostHooks: AuditHook.log()
        PostHooks->>PostHooks: MetricsHook.record()

        API->>Learning: update_on_store(episode)
        Learning->>Learning: dopamine_signal()
        Learning->>Learning: update_gate_weights()
    end

    API-->>Client: StoreResult(episode_id)
```

### Complete Retrieve Flow

End-to-end memory retrieval with all components.

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant PreHooks as Pre Hooks
    participant Embed as Embedding
    participant Pattern as Pattern Completion
    participant NCA as NCA Dynamics
    participant Buffer as Working Memory
    participant Qdrant
    participant Neo4j
    participant Scorer as Retrieval Scorer
    participant PostHooks as Post Hooks
    participant Learning

    Client->>API: recall(query, limit=5)

    rect rgb(255, 243, 224)
        Note over API,PreHooks: Pre-Processing
        API->>PreHooks: pre_recall(query)
        PreHooks->>PreHooks: CachingHook.check()
        PreHooks-->>API: cache_miss
    end

    rect rgb(227, 242, 253)
        Note over API,Embed: Query Embedding
        API->>Embed: generate_embedding(query)
        Embed-->>API: query_embedding[1024]
    end

    rect rgb(232, 245, 233)
        Note over API,Pattern: Pattern Completion
        API->>Pattern: complete(query_embedding)
        Pattern->>Pattern: attractor_dynamics()
        Pattern->>Pattern: hopfield_update()
        Pattern-->>API: completed_embedding
    end

    rect rgb(255, 235, 238)
        Note over API,NCA: NCA State Update
        API->>NCA: process_recall_event()
        NCA->>NCA: update_acetylcholine(RETRIEVAL)
        NCA->>NCA: spatial_cell_activation()
        NCA->>NCA: theta_phase_precession()
        NCA-->>API: retrieval_context
    end

    rect rgb(227, 242, 253)
        Note over API,Buffer: Buffer Probe
        API->>Buffer: probe(query_embedding)
        Buffer->>Buffer: similarity_search()
        Buffer-->>API: buffer_matches[]
    end

    rect rgb(232, 245, 233)
        Note over API,Neo4j: Multi-Source Search
        par Parallel search
            API->>Qdrant: search(embedding, limit=20)
            API->>Neo4j: graph_traverse(query)
        end

        Qdrant-->>API: vector_results[]
        Neo4j-->>API: graph_results[]
    end

    rect rgb(255, 243, 224)
        Note over API,Scorer: Score Fusion & Reranking
        API->>Scorer: fuse_and_rerank(vector, graph, buffer)
        Scorer->>Scorer: semantic_score(0.4)
        Scorer->>Scorer: recency_score(0.25)
        Scorer->>Scorer: outcome_score(0.2)
        Scorer->>Scorer: importance_score(0.15)
        Scorer->>Scorer: neural_rerank()
        Scorer-->>API: ranked_results[limit]
    end

    rect rgb(227, 242, 253)
        Note over API,Learning: Post-Processing & Learning
        API->>PostHooks: post_recall(results)
        PostHooks->>PostHooks: MetricsHook.record()

        API->>Learning: hebbian_update(co_retrieved)
        Learning->>Learning: strengthen_connections()
        Learning->>Learning: update_access_counts()
    end

    API-->>Client: RecallResult(episodes[])
```

### Complete Consolidation Flow

Full consolidation cycle with all phases.

```mermaid
sequenceDiagram
    participant Trigger as Consolidation Trigger
    participant Coord as Consolidation Coordinator
    participant Buffer as Working Memory
    participant Episodic as Episodic Memory
    participant Semantic as Semantic Memory
    participant NCA as NCA Dynamics
    participant Learning as Learning System
    participant Neo4j
    participant Qdrant

    Trigger->>Coord: start_consolidation()

    rect rgb(255, 243, 224)
        Note over Coord,Buffer: NREM Phase 1: Episode Selection
        Coord->>Buffer: get_pending_episodes()
        Buffer-->>Coord: pending_episodes[]

        Coord->>Episodic: get_recent_episodes(24h)
        Episodic-->>Coord: recent_episodes[]

        Coord->>Coord: priority_score(episodes)
        Note right of Coord: 0.3×outcome + 0.25×importance<br/>+ 0.25×recency + 0.2×novelty
        Coord->>Coord: select_top_k(100)
    end

    rect rgb(232, 245, 233)
        Note over Coord,Learning: NREM Phase 2: SWR Replay
        loop For each high-priority episode
            Coord->>NCA: trigger_swr_burst()
            NCA->>NCA: temporal_compress(10-20x)

            Coord->>Learning: replay_episode(episode)
            Learning->>Learning: eligibility_trace_update()
            Learning->>Learning: stdp_update()
        end
    end

    rect rgb(227, 242, 253)
        Note over Coord,Neo4j: NREM Phase 3: Entity Extraction
        Coord->>Coord: find_recurring_patterns(3+)

        loop For each pattern
            Coord->>Semantic: extract_entity(pattern)
            Semantic->>Neo4j: create_entity_node()
            Neo4j-->>Semantic: entity_id

            Coord->>Neo4j: link_episodes_to_entity()
        end
    end

    rect rgb(255, 235, 238)
        Note over Coord,Semantic: REM Phase: Clustering & Abstraction
        Coord->>Semantic: get_recent_entities()
        Semantic-->>Coord: entities[]

        Coord->>Coord: hdbscan_cluster(min_size=3)

        loop For each cluster
            Coord->>Coord: llm_abstract(cluster)
            Coord->>Semantic: create_concept(abstraction)
            Semantic->>Neo4j: create_concept_node()
        end

        Coord->>Coord: cross_cluster_integration()
    end

    rect rgb(255, 243, 224)
        Note over Coord,Qdrant: Pruning Phase: Homeostatic Scaling
        Coord->>Neo4j: get_connection_weights()
        Neo4j-->>Coord: weights[]

        Coord->>Coord: identify_weak(threshold=0.1)
        Coord->>Coord: scale_to_target(activity=0.03)

        par Parallel updates
            Coord->>Neo4j: prune_weak_connections()
            Coord->>Neo4j: scale_remaining_weights()
            Coord->>Qdrant: update_importance_scores()
        end
    end

    rect rgb(232, 245, 233)
        Note over Coord,Learning: Finalization
        Coord->>Learning: consolidation_complete_signal()
        Learning->>Learning: reset_eligibility_traces()

        Coord->>Buffer: clear_consolidated()
    end

    Coord-->>Trigger: ConsolidationResult(stats)
```

### Dreaming Sequence

Dream trajectory generation for creative exploration.

```mermaid
sequenceDiagram
    participant Trigger as Dream Trigger
    participant Dream as Dreaming System
    participant Pred as Hierarchical Predictor
    participant Causal as Causal Discovery
    participant Semantic as Semantic Memory
    participant NCA as NCA Dynamics

    Trigger->>Dream: start_dream(seed_memories)

    rect rgb(227, 242, 253)
        Note over Dream,NCA: Initialize Dream State
        Dream->>NCA: set_dream_mode()
        NCA->>NCA: reduce_norepinephrine()
        NCA->>NCA: increase_acetylcholine()
        NCA->>NCA: enable_theta_modulation()
    end

    loop Dream Steps (10-50)
        rect rgb(232, 245, 233)
            Note over Dream,Pred: Prediction Phase
            Dream->>Pred: predict_next(context)
            Pred->>Pred: fast_head(1-step)
            Pred->>Pred: medium_head(5-step)
            Pred->>Pred: slow_head(15-step)
            Pred-->>Dream: predictions[]
        end

        rect rgb(255, 243, 224)
            Note over Dream,Causal: Causal Exploration
            Dream->>Causal: explore_counterfactual()
            Causal->>Causal: intervene(variable)
            Causal->>Causal: predict_effect()
            Causal-->>Dream: alternative_trajectory
        end

        rect rgb(227, 242, 253)
            Note over Dream,Semantic: Semantic Integration
            Dream->>Semantic: find_related_concepts()
            Semantic-->>Dream: concepts[]
            Dream->>Dream: integrate_concepts()
        end

        Dream->>Dream: record_trajectory_step()
    end

    Dream->>Dream: compile_insights()
    Dream-->>Trigger: DreamResult(trajectories, insights)
```

## Component Interaction Diagrams

### Hook Execution Order

```mermaid
sequenceDiagram
    participant Op as Operation
    participant Reg as Hook Registry
    participant H1 as CachingHook (p=10)
    participant H2 as ValidationHook (p=50)
    participant H3 as GateHook (p=100)
    participant H4 as AuditHook (p=200)

    Op->>Reg: execute_pre_hooks()

    rect rgb(232, 245, 233)
        Note over Reg,H3: Pre Hooks (sorted by priority)
        Reg->>H1: pre_store()
        H1-->>Reg: continue
        Reg->>H2: pre_store()
        H2-->>Reg: continue
        Reg->>H3: pre_store()
        H3-->>Reg: continue
    end

    Reg-->>Op: pre_hooks_complete

    Op->>Op: execute_operation()

    Op->>Reg: execute_post_hooks()

    rect rgb(227, 242, 253)
        Note over Reg,H4: Post Hooks
        Reg->>H4: post_store()
        H4-->>Reg: complete
    end

    Reg-->>Op: post_hooks_complete
```

### Learning Signal Propagation

```mermaid
sequenceDiagram
    participant Event as Memory Event
    participant DA as Dopamine System
    participant Elig as Eligibility Traces
    participant STDP as STDP Updater
    participant Hebb as Hebbian Learning
    participant Weights as Connection Weights

    Event->>DA: compute_rpe(actual, expected)
    DA->>DA: δ = actual - expected

    alt δ > 0 (Positive surprise)
        DA->>Elig: tag_active_synapses()
        DA->>STDP: ltp_signal()
    else δ < 0 (Negative surprise)
        DA->>STDP: ltd_signal()
    else δ = 0 (As expected)
        DA->>DA: baseline_signal()
    end

    STDP->>Elig: get_tagged_synapses()
    Elig-->>STDP: eligible_synapses[]

    loop For each eligible synapse
        STDP->>STDP: Δw = A × exp(-Δt/τ) × δ
        STDP->>Weights: update(synapse, Δw)
    end

    Event->>Hebb: co_activation_event(i, j)
    Hebb->>Hebb: Δw = η × a_i × a_j
    Hebb->>Weights: update(i_j, Δw)

    Weights->>Weights: apply_homeostatic_scaling()
```
