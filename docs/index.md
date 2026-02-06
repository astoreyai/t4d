# T4DM

**Biologically-inspired memory for AI**

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/astoreyai/t4d/releases)
[![Tests](https://img.shields.io/badge/tests-6%2C540%2B%20passing-green)](https://github.com/astoreyai/ww)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green)](https://github.com/astoreyai/ww)

T4DM is a modular framework implementing tripartite neural memory (episodic, semantic, procedural) with cognitive consolidation, neuromodulator dynamics, FSRS-based spaced repetition, and world model prediction.

## Features

<div class="grid cards" markdown>

-   :brain: **Tripartite Memory**

    ---

    Episodic, Semantic, and Procedural memory subsystems inspired by cognitive neuroscience

-   :dna: **Biologically Plausible**

    ---

    6-neurotransmitter PDE system with theta-gamma coupling and place/grid cells

-   :crystal_ball: **World Model Prediction**

    ---

    JEPA-style latent prediction with hierarchical multi-timescale horizons

-   :zap: **Production Ready**

    ---

    REST API, Python SDK, CLI, and comprehensive test coverage

</div>

## Quick Start

=== "Python API"

    ```python
    from ww import memory

    # Store content
    await memory.store("User prefers dark mode interfaces")

    # Recall similar memories
    results = await memory.recall("interface preferences")
    for r in results:
        print(f"{r.memory_type}: {r.content}")
    ```

=== "CLI"

    ```bash
    # Store a memory
    ww store "Learned about decorators in Python"

    # Recall memories
    ww recall "Python decorators"

    # Show system status
    ww status
    ```

=== "REST API"

    ```bash
    # Create episode
    curl -X POST http://localhost:8765/api/v1/episodes \
      -H "Content-Type: application/json" \
      -d '{"content": "Learning about T4DM"}'

    # Search memories
    curl -X POST http://localhost:8765/api/v1/episodes/recall \
      -H "Content-Type: application/json" \
      -d '{"query": "T4DM", "limit": 5}'
    ```

## Architecture

```mermaid
graph TB
    subgraph API["API Layer"]
        CLI[CLI]
        REST[REST API]
        SDK[Python SDK]
    end

    subgraph Memory["Memory Subsystems"]
        EP[Episodic<br/>FSRS]
        SEM[Semantic<br/>ACT-R]
        PROC[Procedural<br/>Skills]
    end

    subgraph Prediction["Prediction & Dreaming"]
        HIER[Hierarchical<br/>Predictor]
        DREAM[Dreaming<br/>System]
        CAUSAL[Causal<br/>Discovery]
    end

    subgraph NCA["NCA Dynamics"]
        NT[Neural Field<br/>6-NT PDE]
        TG[Theta-Gamma<br/>Coupling]
        SPACE[Place/Grid<br/>Cells]
    end

    subgraph Storage["T4DX Storage Engine"]
        T4DX[(T4DX<br/>Embedded LSM)]
        HNSW[HNSW<br/>Vector Index]
        CSR[CSR<br/>Graph]
    end

    API --> Memory
    Memory --> Prediction
    Memory --> NCA
    Prediction --> NCA
    Memory --> Storage
```

## Version 0.4.0 Highlights

### Advanced Neuroscience Integration

- **Hierarchical Prediction**: Fast (1-step), medium (5-step), slow (15-step) timescales
- **Causal Discovery**: Graph-based counterfactual learning
- **Place/Grid Cells**: Nobel Prize 2014 mechanisms (O'Keefe, Moser)
- **Theta-Gamma Coupling**: Working memory slots (7±2 from Miller's Law)

### Previous Releases

| Version | Highlights |
|---------|------------|
| 0.3.0 | JEPA-style prediction, 15-step dreaming, prioritized replay |
| 0.2.0 | Simplified API, CLI tool, YAML configuration |
| 0.1.0 | Core memory systems, hooks, learning algorithms |

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started/index.md) | Installation, quick start, configuration |
| [Concepts](concepts/index.md) | Architecture, memory types, NCA dynamics |
| [Guides](guides/index.md) | Hook development, performance tuning |
| [API Reference](reference/index.md) | REST API, SDK, CLI documentation |
| [Science](science/index.md) | Biology audit, learning theory, algorithms |
| [Operations](operations/index.md) | Deployment, monitoring |

## License

MIT License - see [LICENSE](https://github.com/astoreyai/t4d/blob/master/LICENSE) for details.
