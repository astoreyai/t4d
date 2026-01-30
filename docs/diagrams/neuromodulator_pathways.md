# Neuromodulator Pathways

Complete neural pathway map showing neuromodulator flow and interactions.

## Full Neuromodulator System

```mermaid
flowchart TB
    subgraph Input["Input Signals"]
        QUERY[Query Embedding]
        OUTCOME[Task Outcome]
        NOVELTY[Novelty Detection]
        CONTEXT[Context State]
    end

    subgraph NT["Neuromodulator Systems"]
        subgraph DA["Dopamine (VTA)"]
            DA_TONIC[Tonic: 4-5 Hz]
            DA_PHASIC[Phasic: 20-40 Hz]
            DA_RPE[RPE: δ = actual - expected]
        end

        subgraph NE["Norepinephrine (LC)"]
            NE_TONIC[Tonic Baseline]
            NE_PHASIC[Phasic Burst]
            NE_GAIN[Gain: 0.5-2.0x]
        end

        subgraph ACH["Acetylcholine (NBM)"]
            ACH_ENC[Encoding Mode: 2.0x]
            ACH_RET[Retrieval Mode: 1.2x]
            ACH_BAL[Balanced: 1.0x]
        end

        subgraph SHT["Serotonin (Raphe)"]
            SHT_MOOD[Mood State]
            SHT_ELIG[Long Eligibility τ=60s]
            SHT_VAL[Temporal Value]
        end

        subgraph GABA["GABA/Glutamate"]
            GABA_INH[Lateral Inhibition]
            GABA_WTA[Winner-Take-All]
            GABA_SPARSE[Sparsity: 20%]
        end
    end

    subgraph Output["Learning Signals"]
        THREE[Three-Factor Rule]
        ELIG[Eligibility Traces]
        RECON[Reconsolidation]
        WEIGHT[Weight Updates]
    end

    %% Input to NT connections
    OUTCOME --> DA_RPE
    NOVELTY --> NE_PHASIC
    NOVELTY --> DA_PHASIC
    CONTEXT --> ACH_ENC
    CONTEXT --> ACH_RET
    QUERY --> NE_TONIC

    %% Internal NT connections
    DA_TONIC --> DA_RPE
    DA_PHASIC --> DA_RPE
    NE_TONIC --> NE_GAIN
    NE_PHASIC --> NE_GAIN
    ACH_ENC --> ACH_BAL
    ACH_RET --> ACH_BAL
    SHT_MOOD --> SHT_VAL
    SHT_ELIG --> SHT_VAL

    %% NT to Output connections
    DA_RPE --> THREE
    NE_GAIN --> THREE
    ACH_BAL --> THREE
    SHT_VAL --> THREE
    GABA_SPARSE --> THREE

    THREE --> ELIG
    THREE --> RECON
    ELIG --> WEIGHT
    RECON --> WEIGHT

    %% Styling
    classDef da fill:#ffd700,stroke:#333
    classDef ne fill:#ff6b6b,stroke:#333
    classDef ach fill:#4ecdc4,stroke:#333
    classDef sht fill:#9b59b6,stroke:#333
    classDef gaba fill:#3498db,stroke:#333

    class DA_TONIC,DA_PHASIC,DA_RPE da
    class NE_TONIC,NE_PHASIC,NE_GAIN ne
    class ACH_ENC,ACH_RET,ACH_BAL ach
    class SHT_MOOD,SHT_ELIG,SHT_VAL sht
    class GABA_INH,GABA_WTA,GABA_SPARSE gaba
```

## Three-Factor Integration

```mermaid
flowchart LR
    subgraph Factor1["Factor 1: Eligibility"]
        E_FAST[Fast τ=5s]
        E_SLOW[Slow τ=60s]
        E_DECAY[Exponential Decay]
    end

    subgraph Factor2["Factor 2: Neuromod Gate"]
        NE_G[NE × Arousal]
        ACH_G[ACh × Mode]
        SHT_G[5-HT × Mood]
        GATE[Gate = NE × ACh × 5-HT]
    end

    subgraph Factor3["Factor 3: DA Surprise"]
        RPE[|RPE| Magnitude]
        SIGN[Signed PE]
    end

    subgraph Output["Effective LR"]
        LR[base_lr × elig × gate × |RPE|]
    end

    E_FAST --> E_DECAY
    E_SLOW --> E_DECAY
    E_DECAY --> LR

    NE_G --> GATE
    ACH_G --> GATE
    SHT_G --> GATE
    GATE --> LR

    RPE --> LR

    LR --> |"Weight Update"| WEIGHT[Δw]
```

## Signal Flow During Query-Retrieval-Outcome

```mermaid
sequenceDiagram
    participant Q as Query
    participant NE as Norepinephrine
    participant ACh as Acetylcholine
    participant Ret as Retrieval
    participant 5HT as Serotonin
    participant DA as Dopamine
    participant TF as Three-Factor

    Q->>NE: Novelty detection
    NE-->>NE: Set arousal level
    Q->>ACh: Context assessment
    ACh-->>ACh: Set encoding/retrieval mode

    NE->>Ret: Arousal modulates breadth
    ACh->>Ret: Mode affects pattern completion

    Ret->>5HT: Add eligibility traces
    Note over 5HT: τ = 60s decay

    Ret-->>Q: Return memories

    Q->>DA: Receive outcome
    DA-->>DA: Compute RPE: δ = actual - expected

    DA->>TF: RPE magnitude |δ|
    NE->>TF: Arousal gain
    ACh->>TF: Mode boost
    5HT->>TF: Eligibility traces

    TF->>TF: effective_lr = base × elig × gate × |δ|
    TF-->>Ret: Update embeddings
```

## Biological Parameters

| System | Parameter | Value | Source |
|--------|-----------|-------|--------|
| DA Tonic | Firing rate | 4-5 Hz | Schultz 1998 |
| DA Phasic | Burst rate | 20-40 Hz | Schultz 1998 |
| NE Baseline | Arousal | 0.5 | Aston-Jones 2005 |
| NE Gain | Range | 0.5-2.0x | Yerkes-Dodson |
| ACh Encoding | Boost | 2.0x | Hasselmo 2006 |
| ACh Retrieval | Boost | 1.2x | BIO-002 fix |
| 5-HT Baseline | Mood | 0.5 | Doya 2002 |
| 5-HT Eligibility | τ | 60s | Temporal credit |
| GABA Sparsity | Target | 20% | Competitive |
