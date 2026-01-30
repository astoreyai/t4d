# Memory Lifecycle

Complete flow from encoding to consolidation to forgetting.

## Full Memory Lifecycle

```mermaid
flowchart TB
    subgraph Encoding["1. Encoding"]
        INPUT[User Input]
        GATE[Learned Gate]
        WM[Working Memory<br/>~4 items]
        FES[Fast Episodic Store<br/>~10K capacity]
        BUFFER[Buffer Manager]
    end

    subgraph Storage["2. Storage"]
        EPIS[Episodic Memory]
        SEM[Semantic Memory]
        PROC[Procedural Memory]
    end

    subgraph Retrieval["3. Retrieval"]
        QUERY[Query]
        CLUSTER[Cluster Index]
        SPARSE[Sparse Addressing]
        PATTERN[Pattern Completion]
        RANK[Learned Ranking]
    end

    subgraph Consolidation["4. Consolidation"]
        NREM[NREM Replay]
        REM[REM Abstraction]
        PRUNE[Synaptic Pruning]
    end

    subgraph Forgetting["5. Forgetting"]
        DECAY[Time Decay]
        INTERF[Interference]
        VALUE[Value Threshold]
        ARCHIVE[Cold Storage]
    end

    %% Encoding flow
    INPUT --> GATE
    GATE -->|"pass"| WM
    GATE -->|"pass"| FES
    WM --> BUFFER
    FES --> BUFFER
    BUFFER -->|"promote"| EPIS

    %% Storage connections
    EPIS <--> SEM
    EPIS --> PROC
    SEM --> PROC

    %% Retrieval flow
    QUERY --> CLUSTER
    CLUSTER --> SPARSE
    SPARSE --> PATTERN
    PATTERN --> RANK
    RANK --> EPIS
    RANK --> SEM
    RANK --> PROC

    %% Consolidation flow
    EPIS --> NREM
    NREM --> REM
    REM --> SEM
    REM --> PRUNE
    PRUNE --> EPIS

    %% Forgetting flow
    EPIS --> DECAY
    EPIS --> INTERF
    EPIS --> VALUE
    DECAY --> ARCHIVE
    VALUE --> ARCHIVE
    ARCHIVE -.->|"retrieve"| EPIS

    %% Styling
    classDef encode fill:#e1f5fe,stroke:#0288d1
    classDef store fill:#f3e5f5,stroke:#7b1fa2
    classDef retrieve fill:#e8f5e9,stroke:#388e3c
    classDef consol fill:#fff3e0,stroke:#f57c00
    classDef forget fill:#ffebee,stroke:#c62828

    class INPUT,GATE,WM,FES,BUFFER encode
    class EPIS,SEM,PROC store
    class QUERY,CLUSTER,SPARSE,PATTERN,RANK retrieve
    class NREM,REM,PRUNE consol
    class DECAY,INTERF,VALUE,ARCHIVE forget
```

## Encoding Phase Detail

```mermaid
flowchart LR
    subgraph Input
        CONTENT[Content]
        EMBED[BGE-M3 Embedding]
    end

    subgraph Gate["Learned Gate (Thompson Sampling)"]
        NOVEL[Novelty Score]
        IMPORT[Importance Score]
        CONTEXT[Context Score]
        GATE_SCORE[Combined Score]
        THRESH[Threshold: 0.6]
    end

    subgraph Output
        PASS[Pass → Store]
        REJECT[Reject → Drop]
    end

    CONTENT --> EMBED
    EMBED --> NOVEL
    EMBED --> IMPORT
    EMBED --> CONTEXT

    NOVEL --> GATE_SCORE
    IMPORT --> GATE_SCORE
    CONTEXT --> GATE_SCORE

    GATE_SCORE -->|"≥ 0.6"| PASS
    GATE_SCORE -->|"< 0.6"| REJECT
```

## Retrieval Phase Detail

```mermaid
flowchart TB
    subgraph Query
        Q[Query Text]
        Q_EMB[Query Embedding]
    end

    subgraph Stage1["Stage 1: Cluster Selection"]
        CLUSTERS[K Clusters]
        SELECT[Top-k Selection]
        NE_MOD[NE Modulation<br/>High → Broad]
    end

    subgraph Stage2["Stage 2: Within-Cluster Search"]
        KNN[k-NN Search]
        SPARSE[Learned Sparse<br/>Addressing]
    end

    subgraph Stage3["Stage 3: Pattern Completion"]
        HOPFIELD[Modern Hopfield]
        DG[Dentate Gyrus<br/>Separation]
    end

    subgraph Stage4["Stage 4: Ranking"]
        SCORER[Learned Scorer]
        RERANK[Reranker]
        INHIB[GABA Inhibition]
    end

    Q --> Q_EMB
    Q_EMB --> CLUSTERS
    CLUSTERS --> SELECT
    NE_MOD --> SELECT
    SELECT --> KNN
    KNN --> SPARSE
    SPARSE --> HOPFIELD
    HOPFIELD --> DG
    DG --> SCORER
    SCORER --> RERANK
    RERANK --> INHIB
    INHIB --> RESULTS[Ranked Results]
```

## Consolidation Phase Detail

```mermaid
flowchart TB
    subgraph NREM["NREM Phase (4 cycles)"]
        PRIOR[Priority Sort<br/>PE + Importance + Recency]
        SWR[Sharp-Wave Ripple<br/>10x Compression]
        REPLAY[Sequence Replay<br/>500ms intervals]
        STDP[STDP Updates]
        TAGS[Synaptic Tags]
    end

    subgraph REM["REM Phase"]
        CLUSTER[HDBSCAN Clustering]
        CONCEPT[Concept Creation]
        ABSTRACT[ABSTRACTS Relations]
    end

    subgraph PRUNE["Prune Phase"]
        WEAK[Find weight < 0.05]
        DELETE[Delete Weak]
        HOMEO[Homeostatic Scaling]
    end

    EPIS[Episodes] --> PRIOR
    PRIOR --> SWR
    SWR --> REPLAY
    REPLAY --> STDP
    STDP --> TAGS

    TAGS --> CLUSTER
    CLUSTER --> CONCEPT
    CONCEPT --> ABSTRACT
    ABSTRACT --> SEM[Semantic]

    ABSTRACT --> WEAK
    WEAK --> DELETE
    DELETE --> HOMEO
```

## State Transitions

```mermaid
stateDiagram-v2
    [*] --> BufferPending: Input received

    BufferPending --> GateCheck: Evidence accumulated
    GateCheck --> Rejected: Score < threshold
    GateCheck --> WorkingMemory: Score ≥ threshold

    Rejected --> [*]

    WorkingMemory --> Decaying: No rehearsal
    WorkingMemory --> Active: Rehearsal
    Decaying --> Evicted: Decay complete
    Evicted --> [*]

    Active --> EpisodicStore: Consolidate

    EpisodicStore --> NREMReplay: Sleep cycle
    NREMReplay --> REMAbstract: High replay count
    REMAbstract --> SemanticStore: Create concept

    EpisodicStore --> Decaying: FSRS decay
    Decaying --> Archived: Below threshold
    Archived --> ColdStorage: Age > 90 days

    ColdStorage --> EpisodicStore: Retrieval request
```

## Timing Parameters

| Phase | Parameter | Value |
|-------|-----------|-------|
| Gate | Threshold | 0.6 |
| Working Memory | Capacity | 4 items |
| Working Memory | Decay | Exponential |
| FES | Capacity | 10,000 |
| FES | Learning rate | 100x faster |
| Buffer | Max residence | 300s |
| Buffer | Promotion threshold | 0.65 |
| NREM | Replay delay | 500ms |
| NREM | Cycles | 4 |
| SWR | Compression | 10x |
| Prune | Weight threshold | 0.05 |
| Archive | Age threshold | 90 days |
