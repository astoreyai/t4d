# T4DM Full Setup Guide

**Version**: 2.0.0
**Last Updated**: 2026-02-05
**Purpose**: Complete installation, configuration, and deployment guide for T4DM

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation Options](#2-installation-options)
3. [Configuration](#3-configuration)
4. [Hardware Profiles](#4-hardware-profiles)
5. [Component Setup](#5-component-setup)
6. [Verification](#6-verification)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### System Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **CPU** | 4 cores | 8+ cores | 16+ cores |
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **GPU VRAM** | 4 GB | 8 GB | 16+ GB |
| **Disk** | 20 GB SSD | 100 GB NVMe | 500+ GB NVMe |
| **Python** | 3.11 | 3.11-3.12 | 3.11 |

### Software Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential git curl

# macOS
brew install python@3.11 git

# Windows (PowerShell as Admin)
winget install Python.Python.3.11 Git.Git
```

### GPU Setup (Optional but Recommended)

```bash
# NVIDIA CUDA (Linux)
# Check: https://developer.nvidia.com/cuda-downloads

# Verify CUDA
nvidia-smi
nvcc --version

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Installation Options

### Option A: Quick Install (PyPI)

```bash
# Basic installation
pip install t4dm

# With API server
pip install t4dm[api]

# Full installation (all extras)
pip install t4dm[api,consolidation,observability,cache]

# With flash attention (requires compatible GPU)
pip install t4dm[api,flash]
```

### Option B: Development Install

```bash
# Clone repository
git clone https://github.com/astoreyai/t4dm.git
cd t4dm

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev,api,consolidation]"

# Verify installation
python -c "import t4dm; print(t4dm.__version__)"
```

### Option C: Docker

```bash
# Pull image
docker pull ghcr.io/astoreyai/t4dm:latest

# Run with persistent storage
docker run -d \
    --name t4dm \
    -p 8765:8765 \
    -v t4dm-data:/data \
    -e T4DM_STORAGE_PATH=/data \
    ghcr.io/astoreyai/t4dm:latest

# With GPU support
docker run -d \
    --name t4dm \
    --gpus all \
    -p 8765:8765 \
    -v t4dm-data:/data \
    ghcr.io/astoreyai/t4dm:latest
```

### Option D: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  t4dm:
    image: ghcr.io/astoreyai/t4dm:latest
    ports:
      - "8765:8765"
    volumes:
      - t4dm-data:/data
      - ./config:/etc/t4dm
    environment:
      - T4DM_STORAGE_PATH=/data
      - T4DM_CONFIG_PATH=/etc/t4dm/config.yaml
      - T4DM_LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  t4dm-data:
```

```bash
docker-compose up -d
```

---

## 3. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `T4DM_STORAGE_PATH` | `./data` | T4DX data directory |
| `T4DM_HOST` | `127.0.0.1` | API server bind host |
| `T4DM_PORT` | `8765` | API server port |
| `T4DM_SESSION_ID` | `default` | Default session identifier |
| `T4DM_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `T4DM_EMBEDDING_DIM` | `1024` | Embedding dimension |
| `T4DM_LOG_LEVEL` | `INFO` | Logging level |
| `T4DM_CHECKPOINT_INTERVAL` | `300` | Checkpoint interval (seconds) |
| `T4DM_CONSOLIDATION_INTERVAL` | `3600` | Auto-consolidation interval |
| `T4DM_DEVICE` | `auto` | PyTorch device (auto/cpu/cuda/mps) |
| `T4DM_DTYPE` | `float16` | Tensor dtype (float16/float32/bfloat16) |

### Configuration File

Create `config.yaml`:

```yaml
# T4DM Configuration
version: "2.0"

# Storage settings
storage:
  path: /var/lib/t4dm/data
  wal_enabled: true
  wal_sync_mode: fsync  # fsync | async | none
  memtable_size_mb: 64
  segment_size_mb: 256
  max_segments: 100

# Embedding model
embedding:
  model: BAAI/bge-m3
  dimension: 1024
  batch_size: 32
  device: auto  # auto | cpu | cuda | cuda:0 | mps
  dtype: float16

# Qwen LLM backbone
qwen:
  enabled: true
  model: Qwen/Qwen2.5-3B-Instruct
  quantization: nf4  # nf4 | int8 | none
  qlora:
    enabled: true
    rank: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]

# Spiking cortical blocks
spiking:
  enabled: true
  num_blocks: 6
  hidden_dim: 1024
  lif:
    threshold: 1.0
    decay: 0.9
    reset: soft
  neuromodulation:
    dopamine_baseline: 0.5
    norepinephrine_baseline: 0.3
    acetylcholine_baseline: 0.4
    serotonin_baseline: 0.5
  oscillators:
    theta_freq: 6.0
    gamma_freq: 40.0
    delta_freq: 2.0

# Consolidation settings
consolidation:
  auto_enabled: true
  interval_seconds: 3600
  nrem:
    replay_count: 3
    kappa_increment: 0.05
  rem:
    clustering_enabled: true
    min_cluster_size: 5
  prune:
    min_kappa: 0.01
    min_importance: 0.05
    max_age_days: 365

# HNSW index settings
hnsw:
  m: 16
  ef_construction: 200
  ef_search: 50
  max_elements: 1000000

# API settings
api:
  host: 0.0.0.0
  port: 8765
  workers: 4
  cors_origins: ["*"]
  rate_limit:
    enabled: true
    requests_per_minute: 1000

# Observability
observability:
  metrics_enabled: true
  tracing_enabled: false
  otlp_endpoint: ""
  log_level: INFO
  log_format: json  # json | text

# Cache (optional Redis)
cache:
  enabled: false
  redis_url: redis://localhost:6379/0
  ttl_seconds: 3600
```

Load configuration:

```bash
# Via environment variable
export T4DM_CONFIG_PATH=/path/to/config.yaml
t4dm serve

# Via CLI flag
t4dm serve --config /path/to/config.yaml
```

---

## 4. Hardware Profiles

### Profile: Laptop (CPU-only)

```yaml
# laptop-config.yaml
embedding:
  model: BAAI/bge-small-en-v1.5
  dimension: 384
  device: cpu
  dtype: float32

qwen:
  enabled: false  # Too heavy for CPU

spiking:
  enabled: false  # Use simplified mode

storage:
  memtable_size_mb: 32
  segment_size_mb: 128
```

### Profile: Desktop (Single GPU, 8GB VRAM)

```yaml
# desktop-config.yaml
embedding:
  model: BAAI/bge-m3
  dimension: 1024
  device: cuda
  dtype: float16

qwen:
  enabled: true
  quantization: nf4  # 4-bit = ~2GB VRAM

spiking:
  enabled: true
  num_blocks: 4  # Reduced from 6
```

### Profile: Workstation (24GB VRAM) - Recommended

```yaml
# workstation-config.yaml (DEFAULT)
embedding:
  model: BAAI/bge-m3
  dimension: 1024
  device: cuda
  dtype: float16

qwen:
  enabled: true
  quantization: nf4
  qlora:
    enabled: true
    rank: 16

spiking:
  enabled: true
  num_blocks: 6
```

### Profile: Server (Multi-GPU)

```yaml
# server-config.yaml
embedding:
  device: cuda:1  # Dedicated GPU for embeddings

qwen:
  enabled: true
  quantization: int8  # Better quality than nf4
  device_map: auto  # Distribute across GPUs

spiking:
  enabled: true
  num_blocks: 6
  device: cuda:0

api:
  workers: 8

storage:
  memtable_size_mb: 256
  segment_size_mb: 1024
```

---

## 5. Component Setup

### 5.1 T4DX Storage Engine

T4DX is embedded - no external setup required.

```python
# Verify T4DX
from t4dm.storage.t4dx import T4DXEngine

engine = T4DXEngine(storage_path="./data")
print(f"Segments: {engine.segment_count}")
print(f"Items: {engine.item_count}")
print(f"WAL size: {engine.wal_size_bytes}")
```

### 5.2 Embedding Model

```python
# Pre-download embedding model
from t4dm.core.config import get_settings
from sentence_transformers import SentenceTransformer

settings = get_settings()
model = SentenceTransformer(settings.embedding_model)
print(f"Model loaded: {settings.embedding_model}")
print(f"Dimension: {model.get_sentence_embedding_dimension()}")
```

### 5.3 Qwen LLM (Optional)

```python
# Pre-download Qwen model
from t4dm.qwen.loader import load_qwen_model

model, tokenizer = load_qwen_model(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    quantization="nf4",
    device="cuda"
)
print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
```

### 5.4 MCP Server (Claude Integration)

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "t4dm": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.server"],
      "env": {
        "T4DM_STORAGE_PATH": "/path/to/data",
        "T4DM_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Or for Claude Code (`.claude/settings.json`):

```json
{
  "mcpServers": {
    "t4dm": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.server"],
      "cwd": "/path/to/t4dm"
    }
  }
}
```

### 5.5 Systemd Service (Production Linux)

Create `/etc/systemd/system/t4dm.service`:

```ini
[Unit]
Description=T4DM Memory Server
After=network.target

[Service]
Type=simple
User=t4dm
Group=t4dm
WorkingDirectory=/opt/t4dm
ExecStart=/opt/t4dm/venv/bin/t4dm serve --config /etc/t4dm/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="T4DM_STORAGE_PATH=/var/lib/t4dm/data"
Environment="T4DM_LOG_LEVEL=INFO"

# Resource limits
LimitNOFILE=65535
MemoryMax=16G

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable t4dm
sudo systemctl start t4dm
sudo systemctl status t4dm
```

---

## 6. Verification

### 6.1 Health Check

```bash
# CLI health check
t4dm health

# API health check
curl http://localhost:8765/api/v1/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "storage": {"status": "ok", "items": 0, "segments": 0},
    "embedding": {"status": "ok", "model": "BAAI/bge-m3"},
    "qwen": {"status": "ok", "model": "Qwen/Qwen2.5-3B-Instruct"},
    "spiking": {"status": "ok", "blocks": 6}
  },
  "memory": {"rss_mb": 1024, "vram_mb": 8192}
}
```

### 6.2 Smoke Test

```python
from t4dm import T4DM

# Initialize
mem = T4DM()

# Store
mem.add("Test memory for verification", metadata={"test": True})

# Search
results = mem.search("verification test")
assert len(results) > 0
print(f"Smoke test passed: {results[0].content}")

# Cleanup
mem.close()
```

### 6.3 Run Test Suite

```bash
# Quick tests
pytest tests/unit -x --tb=short

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/t4dm --cov-report=html

# Performance tests
pytest tests/performance -v

# Integration tests
pytest tests/integration -v
```

---

## 7. Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size
export T4DM_EMBEDDING_BATCH_SIZE=8

# Use CPU for embeddings
export T4DM_DEVICE=cpu

# Reduce spiking blocks
# In config.yaml: spiking.num_blocks: 4
```

#### Slow Startup

```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

#### WAL Corruption Recovery

```bash
# Rebuild from segments (discards uncommitted WAL)
t4dm recover --storage-path /path/to/data --discard-wal

# Verify integrity
t4dm verify --storage-path /path/to/data
```

#### Port Already in Use

```bash
# Find process
lsof -i :8765

# Use different port
t4dm serve --port 8766
```

#### Permission Denied

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER /var/lib/t4dm
chmod 750 /var/lib/t4dm
```

### Logs

```bash
# View logs (systemd)
journalctl -u t4dm -f

# Debug logging
export T4DM_LOG_LEVEL=DEBUG
t4dm serve
```

### Support

- GitHub Issues: https://github.com/astoreyai/t4dm/issues
- Documentation: https://github.com/astoreyai/t4dm/docs

---

*Generated 2026-02-05*
