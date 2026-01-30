# Installation

## Quick Install

```bash
pip install world-weaver
```

## Installation Options

### Standard Installation

The basic installation includes all core functionality:

```bash
pip install world-weaver
```

### With API Server

To run the REST API server:

```bash
pip install world-weaver[api]
```

### With Embeddings

For local BGE-M3 embeddings (requires ~1.3GB GPU memory):

```bash
pip install world-weaver[embedding]
```

### Full Installation

All optional dependencies:

```bash
pip install world-weaver[api,embedding,dev]
```

## Docker

### Infrastructure Only

Start Neo4j and Qdrant for storage:

```bash
git clone https://github.com/astoreyai/ww
cd ww

# Generate secure passwords
./scripts/setup-env.sh

# Start infrastructure
docker-compose up -d
```

### Full Stack

Run the complete system including API server:

```bash
docker-compose -f docker-compose.full.yml up -d
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8765/api/v1/health

# Access API documentation
open http://localhost:8765/docs
```

## Development Setup {#development}

### Clone and Install

```bash
git clone https://github.com/astoreyai/ww
cd ww

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[api,dev]"
```

### Start Infrastructure

```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Verify containers are running
docker-compose ps
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/ww --cov-report=term-missing
```

## Verification

After installation, verify everything works:

=== "Python"

    ```python
    from ww import memory

    # Test store and recall
    async def test():
        await memory.store("Test memory")
        results = await memory.recall("test")
        print(f"Found {len(results)} results")

    import asyncio
    asyncio.run(test())
    ```

=== "CLI"

    ```bash
    # Check status
    ww status

    # Store and recall
    ww store "Test memory"
    ww recall "test"
    ```

=== "API"

    ```bash
    # Start server
    ww serve &

    # Test health endpoint
    curl http://localhost:8765/api/v1/health
    ```

## Troubleshooting

### Common Issues

??? question "Neo4j connection refused"

    Ensure Neo4j is running and the URI is correct:
    ```bash
    docker-compose ps
    docker-compose logs neo4j
    ```

??? question "Qdrant connection timeout"

    Check Qdrant container and port:
    ```bash
    docker-compose ps
    curl http://localhost:6333/health
    ```

??? question "GPU memory error with embeddings"

    The BGE-M3 model requires ~1.3GB GPU memory. Options:

    - Use CPU mode: Set `WW_EMBEDDING_DEVICE=cpu`
    - Use smaller batch size: Set `WW_EMBEDDING_BATCH_SIZE=8`

### Getting Help

- [GitHub Issues](https://github.com/astoreyai/ww/issues)
- [API Documentation](http://localhost:8765/docs) (when running)
