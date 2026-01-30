# Learning Signal Flow

Complete learning pathway from events to weight updates.

## Full Learning Signal Flow

```mermaid
flowchart TB
    subgraph Events["Event Capture"]
        RET_EVENT[Retrieval Event]
        OUT_EVENT[Outcome Event]
        EXP[Experience]
    end

    subgraph Eligibility["Eligibility Traces"]
        FAST[Fast Layer τ=5s]
        SLOW[Slow Layer τ=60s]
        DECAY[Exponential Decay]
        CREDIT[Credit Assignment]
    end

    subgraph Dopamine["Dopamine System"]
        VALUE[Value Estimator<br/>V(s)]
        RPE[RPE: δ = r - V(s)]
        UPDATE[Update V(s)]
        TD[TD(λ) Distribution]
    end

    subgraph Orchestra["Neuromod Orchestra"]
        NE[NE Gain]
        ACH[ACh Mode]
        SHT[5-HT Mood]
        GATE[Neuromod Gate]
    end

    subgraph ThreeFactor["Three-Factor Rule"]
        ELIG_F[Factor 1: Eligibility]
        GATE_F[Factor 2: Gate]
        DA_F[Factor 3: |RPE|]
        COMBINE[LR = base × elig × gate × |δ|]
    end

    subgraph Output["Weight Updates"]
        RECON[Reconsolidation]
        SCORER[Scorer Update]
        GATE_W[Gate Weights]
        EMB[Embedding Shift]
    end

    %% Event flow
    RET_EVENT --> FAST
    RET_EVENT --> SLOW
    RET_EVENT --> EXP
    OUT_EVENT --> RPE
    OUT_EVENT --> EXP

    %% Eligibility flow
    FAST --> DECAY
    SLOW --> DECAY
    DECAY --> CREDIT
    CREDIT --> ELIG_F

    %% Dopamine flow
    VALUE --> RPE
    OUT_EVENT --> UPDATE
    UPDATE --> VALUE
    RPE --> TD
    TD --> DA_F

    %% Orchestra flow
    RET_EVENT --> NE
    RET_EVENT --> ACH
    SHT --> GATE
    NE --> GATE
    ACH --> GATE
    GATE --> GATE_F

    %% Three-factor combination
    ELIG_F --> COMBINE
    GATE_F --> COMBINE
    DA_F --> COMBINE

    %% Output
    COMBINE --> RECON
    COMBINE --> SCORER
    COMBINE --> GATE_W
    RECON --> EMB

    %% Styling
    classDef event fill:#e3f2fd,stroke:#1565c0
    classDef elig fill:#f3e5f5,stroke:#7b1fa2
    classDef da fill:#fff8e1,stroke:#f9a825
    classDef orch fill:#e8f5e9,stroke:#2e7d32
    classDef tf fill:#fce4ec,stroke:#c2185b
    classDef out fill:#ffebee,stroke:#c62828

    class RET_EVENT,OUT_EVENT,EXP event
    class FAST,SLOW,DECAY,CREDIT elig
    class VALUE,RPE,UPDATE,TD da
    class NE,ACH,SHT,GATE orch
    class ELIG_F,GATE_F,DA_F,COMBINE tf
    class RECON,SCORER,GATE_W,EMB out
```

## Learned Components

```mermaid
flowchart LR
    subgraph Input["Input Features"]
        SIM[Similarity]
        REC[Recency]
        IMP[Importance]
        OUT[Outcome History]
    end

    subgraph Scorer["Learned Scorer"]
        FC1[FC1: 4→32]
        RELU1[ReLU + LayerNorm]
        FC2[FC2: 32→32<br/>+ Residual]
        RELU2[ReLU + LayerNorm]
        FC3[FC3: 32→1]
        SIG[Sigmoid]
    end

    subgraph Training["Training"]
        LISTMLE[ListMLE Loss]
        REPLAY[Priority Replay]
        GRAD[Gradient Update]
    end

    SIM --> FC1
    REC --> FC1
    IMP --> FC1
    OUT --> FC1

    FC1 --> RELU1
    RELU1 --> FC2
    FC2 --> RELU2
    RELU2 --> FC3
    FC3 --> SIG
    SIG --> SCORE[Score]

    SCORE --> LISTMLE
    LISTMLE --> GRAD
    REPLAY --> GRAD
```

## Fusion Weight Learning

```mermaid
flowchart TB
    subgraph Query["Query Processing"]
        Q[Query Embedding]
        COMPRESS[Compress 1024→16]
    end

    subgraph Fusion["Learned Fusion Weights"]
        MLP[2-Layer MLP]
        W_SEM[w_semantic]
        W_REC[w_recency]
        W_OUT[w_outcome]
        W_IMP[w_importance]
    end

    subgraph Scoring["Score Computation"]
        S_SEM[s_semantic]
        S_REC[s_recency]
        S_OUT[s_outcome]
        S_IMP[s_importance]
        COMBINE[Σ w_i × s_i]
    end

    subgraph Learn["Online Learning"]
        OUTCOME[Outcome]
        LOSS[MSE Loss]
        GRAD[Gradient Descent]
    end

    Q --> COMPRESS
    COMPRESS --> MLP
    MLP --> W_SEM
    MLP --> W_REC
    MLP --> W_OUT
    MLP --> W_IMP

    W_SEM --> COMBINE
    W_REC --> COMBINE
    W_OUT --> COMBINE
    W_IMP --> COMBINE

    S_SEM --> COMBINE
    S_REC --> COMBINE
    S_OUT --> COMBINE
    S_IMP --> COMBINE

    COMBINE --> PREDICTED[Predicted Utility]
    PREDICTED --> LOSS
    OUTCOME --> LOSS
    LOSS --> GRAD
    GRAD --> MLP
```

## Reconsolidation Engine

```mermaid
flowchart LR
    subgraph Input
        MEM[Memory Embedding]
        QUERY[Query Embedding]
        OUT[Outcome]
    end

    subgraph Direction["Update Direction"]
        POS[Positive → Move toward query]
        NEG[Negative → Move away]
        ADV[Advantage = |outcome - 0.5|]
    end

    subgraph Modulation["Learning Rate"]
        BASE[Base LR: 0.01]
        THREE[Three-Factor]
        EFF[Effective LR]
    end

    subgraph Update
        SHIFT[Embedding Shift]
        BOUND[Bounded Update]
        NEW[New Embedding]
    end

    MEM --> SHIFT
    QUERY --> SHIFT
    OUT --> POS
    OUT --> NEG
    OUT --> ADV

    POS --> SHIFT
    NEG --> SHIFT
    ADV --> EFF

    BASE --> THREE
    THREE --> EFF
    EFF --> BOUND
    SHIFT --> BOUND
    BOUND --> NEW
```

## STDP Weight Updates

```mermaid
sequenceDiagram
    participant Pre as Pre-Synaptic
    participant Post as Post-Synaptic
    participant STDP as STDP Learner
    participant Weight as Synapse

    Pre->>STDP: Spike at t=100ms
    Note over STDP: Record spike time

    Post->>STDP: Spike at t=105ms
    Note over STDP: Δt = +5ms (causal)

    STDP->>STDP: Compute: A+ × exp(-Δt/τ+)
    STDP->>Weight: LTP: Δw = +0.005

    Pre->>STDP: Spike at t=200ms
    Post->>STDP: Spike at t=195ms
    Note over STDP: Δt = -5ms (anti-causal)

    STDP->>STDP: Compute: -A- × exp(Δt/τ-)
    STDP->>Weight: LTD: Δw = -0.00525
```

## Learning Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Eligibility Fast | τ | 5s |
| Eligibility Slow | τ | 60s |
| Dopamine | Value LR | 0.1 |
| Dopamine | Surprise threshold | 0.05 |
| Reconsolidation | Base LR | 0.01 |
| Scorer | Hidden dim | 32 |
| Scorer | Dropout | 0.1 |
| STDP | A+ | 0.005 |
| STDP | A- | 0.00525 |
| STDP | τ+ | 20ms |
| STDP | τ- | 20ms |
