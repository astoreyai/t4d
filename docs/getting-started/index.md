# Getting Started

Welcome to World Weaver! This section will help you get up and running quickly.

## Installation Options

| Method | Best For | Documentation |
|--------|----------|---------------|
| **pip** | Most users | [Installation Guide](installation.md) |
| **Docker** | Production deployments | [Installation Guide](installation.md#docker) |
| **Development** | Contributors | [Installation Guide](installation.md#development) |

## Quick Links

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    ---

    Install World Weaver and its dependencies

-   :material-rocket-launch: **[Quick Start](quickstart.md)**

    ---

    Your first memory operations in 5 minutes

-   :material-cog: **[Configuration](configuration.md)**

    ---

    YAML config, environment variables, and settings

</div>

## Prerequisites

- **Python**: 3.10 or higher
- **Infrastructure** (optional):
    - Neo4j 5.x for graph storage
    - Qdrant 1.7+ for vector storage
- **GPU** (optional): CUDA-compatible GPU for embedding acceleration

## Choosing Your Setup

```mermaid
flowchart TD
    A[Start] --> B{Production or<br/>Development?}
    B -->|Production| C{Infrastructure<br/>available?}
    B -->|Development| D[pip install -e .]
    C -->|Yes| E[Docker Compose]
    C -->|No| F[In-Memory Mode]
    D --> G[Ready!]
    E --> G
    F --> G
```
