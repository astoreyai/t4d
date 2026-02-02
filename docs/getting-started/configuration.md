# Configuration

T4DM supports YAML configuration files with environment variable overrides.

## Configuration Files

T4DM searches for configuration in this order:

1. `./t4dm.yaml` (current directory)
2. `~/.t4dm/config.yaml` (user home)
3. `/etc/t4dm/config.yaml` (system-wide)

## Basic Configuration

```yaml
# t4dm.yaml
session_id: my-project
environment: development

# Storage
qdrant_host: localhost
qdrant_port: 6333
neo4j_uri: bolt://localhost:7687
neo4j_user: neo4j
neo4j_password: your-password

# Embedding
embedding_model: bge-m3
embedding_dim: 1024

# API
api_host: 0.0.0.0
api_port: 8765
```

## Environment Variables

Environment variables override YAML settings. Use the `T4DM_` prefix:

```bash
export T4DM_SESSION_ID=my-session
export T4DM_QDRANT_HOST=localhost
export T4DM_NEO4J_PASSWORD=secret
```

## Configuration Reference

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `session_id` | str | `"default"` | Session identifier for isolation |
| `environment` | str | `"development"` | Environment mode |

### Storage Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `qdrant_host` | str | `"localhost"` | Qdrant server host |
| `qdrant_port` | int | `6333` | Qdrant server port |
| `neo4j_uri` | str | `"bolt://localhost:7687"` | Neo4j connection URI |
| `neo4j_user` | str | `"neo4j"` | Neo4j username |
| `neo4j_password` | str | `""` | Neo4j password |

### Embedding Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `embedding_model` | str | `"bge-m3"` | Embedding model name |
| `embedding_dim` | int | `1024` | Embedding dimension |
| `embedding_device` | str | `"cuda"` | Device for embeddings |
| `embedding_batch_size` | int | `32` | Batch size for embedding |
| `embedding_cache_size` | int | `1000` | LRU cache size |

### API Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `api_host` | str | `"0.0.0.0"` | API server bind host |
| `api_port` | int | `8765` | API server port |
| `api_workers` | int | `1` | Number of workers |
| `api_key` | str | `""` | API key for authentication |
| `admin_key` | str | `""` | Admin key for sensitive ops |

### Learning Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `dopamine_enabled` | bool | `true` | Enable dopamine RPE |
| `serotonin_enabled` | bool | `true` | Enable serotonin credit |
| `hebbian_enabled` | bool | `true` | Enable Hebbian learning |
| `fsrs_enabled` | bool | `true` | Enable FSRS scheduling |

### Bioinspired Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `procedural_dopamine_enabled` | bool | `true` | Dopamine for procedural |
| `neuromodulator_da_enabled` | bool | `true` | Dopamine system |
| `neuromodulator_5ht_enabled` | bool | `true` | Serotonin system |
| `plasticity_ltd_enabled` | bool | `true` | Long-term depression |

## Example Configurations

### Development

```yaml
# ww-dev.yaml
session_id: dev-session
environment: development

qdrant_host: localhost
qdrant_port: 6333
neo4j_uri: bolt://localhost:7687
neo4j_user: neo4j
neo4j_password: dev-password

embedding_device: cpu
embedding_cache_size: 100

api_port: 8765
```

### Production

```yaml
# ww-prod.yaml
session_id: prod-session
environment: production

qdrant_host: qdrant.internal
qdrant_port: 6333
neo4j_uri: bolt://neo4j.internal:7687
neo4j_user: ww_service
# neo4j_password via environment variable

embedding_device: cuda
embedding_batch_size: 64
embedding_cache_size: 10000

api_host: 0.0.0.0
api_port: 8765
api_workers: 4
# api_key via environment variable
# admin_key via environment variable
```

### Testing

```yaml
# ww-test.yaml
session_id: test-session
environment: testing

# Use in-memory mocks
qdrant_host: ""
neo4j_uri: ""

embedding_device: cpu
embedding_batch_size: 4

dopamine_enabled: false
serotonin_enabled: false
```

## Programmatic Configuration

```python
from t4dm.core.config import get_settings, reset_settings

# Get current settings
settings = get_settings()
print(f"Session: {settings.session_id}")
print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")

# Reset settings cache (reloads from files/env)
reset_settings()
```

## CLI Configuration

```bash
# Show current configuration
t4dm config --show

# Initialize default config file
t4dm config --init

# Show config file path
t4dm config --path
```
