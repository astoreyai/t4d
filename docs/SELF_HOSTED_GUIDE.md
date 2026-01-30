# World Weaver - Self-Hosted Deployment Guide

**Version**: 0.2.0
**Last Updated**: 2025-12-06
**Target Audience**: DevOps engineers, system administrators, self-hosters

---

## Table of Contents

1. [Quick Start (Docker)](#1-quick-start-docker)
2. [Manual Installation](#2-manual-installation)
3. [Configuration](#3-configuration)
4. [MCP Server Integration](#4-mcp-server-integration)
5. [REST API Deployment](#5-rest-api-deployment)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Security Hardening](#7-security-hardening)
8. [Backup & Recovery](#8-backup--recovery)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start (Docker)

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 16GB RAM minimum (32GB recommended)
- 50GB disk space
- (Optional) NVIDIA GPU with CUDA 11.8+ for embedding acceleration

### Full Stack Deployment

```bash
# Clone repository
git clone https://github.com/astoreyai/world-weaver.git
cd world-weaver

# Create environment file
cp .env.example .env

# Generate secure passwords
./scripts/setup-env.sh

# Start all services (Neo4j + Qdrant + API)
docker-compose -f docker-compose.full.yml up -d

# Verify health
curl http://localhost:8765/api/v1/health

# Access API documentation
open http://localhost:8765/docs
```

### Infrastructure Only (No API)

```bash
# Start just Neo4j and Qdrant
docker-compose up -d

# Verify services
docker-compose ps
curl http://localhost:6333/readyz  # Qdrant
curl http://localhost:7474         # Neo4j browser
```

---

## 2. Manual Installation

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16GB
- Disk: 50GB SSD
- OS: Ubuntu 22.04 LTS, Debian 12, or macOS 13+

**Recommended**:
- CPU: 8+ cores
- RAM: 32GB
- Disk: 100GB NVMe SSD
- GPU: NVIDIA RTX 3060+ (12GB VRAM) for embedding acceleration

### Step 1: Install Dependencies

#### Ubuntu/Debian
```bash
# System packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git curl

# Neo4j (Community Edition)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j=1:5.15.0

# Qdrant (binary release)
wget https://github.com/qdrant/qdrant/releases/download/v1.12.1/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
sudo mv qdrant /usr/local/bin/
```

#### macOS
```bash
# Homebrew
brew install python@3.11 neo4j

# Qdrant
brew install qdrant
```

### Step 2: Configure Neo4j

```bash
# Edit /etc/neo4j/neo4j.conf (or $(brew --prefix)/etc/neo4j/neo4j.conf on macOS)
sudo nano /etc/neo4j/neo4j.conf
```

Key settings:
```conf
# Memory allocation
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G

# Security
dbms.security.auth_enabled=true

# APOC plugin
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Network (localhost only for security)
server.bolt.listen_address=127.0.0.1:7687
server.http.listen_address=127.0.0.1:7474
```

Download APOC plugin:
```bash
cd /var/lib/neo4j/plugins  # or $(brew --prefix)/var/neo4j/plugins
wget https://github.com/neo4j/apoc/releases/download/5.15.0/apoc-5.15.0-core.jar
```

Start Neo4j:
```bash
sudo systemctl enable neo4j
sudo systemctl start neo4j

# Set initial password
cypher-shell -u neo4j -p neo4j
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'YourSecurePassword123!';
```

### Step 3: Configure Qdrant

Create config file `/etc/qdrant/config.yaml`:
```yaml
service:
  grpc_port: 6334
  http_port: 6333
  enable_tls: false  # Enable in production with certificates

storage:
  storage_path: /var/lib/qdrant/storage
  snapshots_path: /var/lib/qdrant/snapshots
  on_disk_payload: true

log_level: INFO
```

Create directories:
```bash
sudo mkdir -p /var/lib/qdrant/{storage,snapshots}
sudo chown -R $USER:$USER /var/lib/qdrant
```

Start Qdrant:
```bash
# Using systemd
sudo tee /etc/systemd/system/qdrant.service << EOF
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=simple
User=qdrant
Group=qdrant
ExecStart=/usr/local/bin/qdrant --config-path /etc/qdrant/config.yaml
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable qdrant
sudo systemctl start qdrant
```

### Step 4: Install World Weaver

```bash
# Clone repository
cd /opt
sudo git clone https://github.com/astoreyai/world-weaver.git
cd world-weaver

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with all features
pip install -e ".[api,consolidation,dev]"

# Download embedding model (4GB download)
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"
```

### Step 5: Configure Environment

```bash
cp .env.example .env
nano .env
```

Critical settings:
```bash
# Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=YourSecurePassword123!
NEO4J_URI=bolt://localhost:7687

# Qdrant
QDRANT_URL=http://localhost:6333

# Embedding (use 'cuda:0' if you have GPU)
WW_EMBEDDING_DEVICE=cpu
WW_EMBEDDING_USE_FP16=true
```

### Step 6: Initialize Database

```bash
# Create indexes
python -m ww.scripts.init_database

# Verify
python -c "
from ww.storage.neo4j_store import get_neo4j_store
from ww.storage.qdrant_store import get_qdrant_store
import asyncio

async def verify():
    neo4j = get_neo4j_store()
    qdrant = get_qdrant_store()
    await neo4j.initialize()
    await qdrant.initialize()
    print('✓ Databases initialized')

asyncio.run(verify())
"
```

---

## 3. Configuration

### Environment Variables Reference

#### Session Management
```bash
WW_SESSION_ID=default               # Instance namespace
```

#### Neo4j Connection
```bash
WW_NEO4J_URI=bolt://localhost:7687
WW_NEO4J_USER=neo4j
WW_NEO4J_PASSWORD=<secure-password>
WW_NEO4J_DATABASE=neo4j
WW_NEO4J_MAX_CONNECTION_LIFETIME=3600  # seconds
WW_NEO4J_MAX_CONNECTION_POOL_SIZE=50
WW_NEO4J_CONNECTION_TIMEOUT=30
```

#### Qdrant Connection
```bash
WW_QDRANT_URL=http://localhost:6333
WW_QDRANT_API_KEY=                  # Optional, for production
WW_QDRANT_TIMEOUT=60
WW_QDRANT_GRPC_PORT=6334
```

#### Embedding Configuration
```bash
WW_EMBEDDING_MODEL=BAAI/bge-m3      # Or other sentence-transformers model
WW_EMBEDDING_DIMENSION=1024         # Must match model output
WW_EMBEDDING_DEVICE=cuda:0          # cpu, cuda:0, cuda:1, etc.
WW_EMBEDDING_USE_FP16=true          # Faster, uses less VRAM
WW_EMBEDDING_BATCH_SIZE=32
WW_EMBEDDING_MAX_LENGTH=512
WW_EMBEDDING_CACHE_DIR=/opt/models
```

#### Memory Parameters
```bash
# FSRS decay
WW_FSRS_DEFAULT_STABILITY=1.0       # Initial stability (days)
WW_FSRS_RETENTION_TARGET=0.9        # Target retrievability

# Hebbian learning
WW_HEBBIAN_LEARNING_RATE=0.1        # η in w' = w + η(1-w)
WW_HEBBIAN_INITIAL_WEIGHT=0.1
WW_HEBBIAN_DECAY_RATE=0.01
WW_HEBBIAN_MIN_WEIGHT=0.01
WW_HEBBIAN_STALE_DAYS=180

# ACT-R activation
WW_ACTR_DECAY=0.5                   # d in t^(-d)
WW_ACTR_THRESHOLD=0.0
WW_ACTR_NOISE=0.5                   # σ in N(0, σ²)
WW_ACTR_SPREADING_STRENGTH=1.6
```

#### Retrieval Weights
```bash
WW_RETRIEVAL_SEMANTIC_WEIGHT=0.4
WW_RETRIEVAL_RECENCY_WEIGHT=0.25
WW_RETRIEVAL_OUTCOME_WEIGHT=0.2
WW_RETRIEVAL_IMPORTANCE_WEIGHT=0.15
```

#### Consolidation
```bash
WW_CONSOLIDATION_MIN_SIMILARITY=0.75
WW_CONSOLIDATION_MIN_OCCURRENCES=3
WW_CONSOLIDATION_SKILL_SIMILARITY=0.85
```

#### API Server
```bash
WW_API_HOST=0.0.0.0
WW_API_PORT=8765
WW_API_WORKERS=4                    # Uvicorn workers
WW_API_CORS_ORIGINS=*               # Restrict in production!
```

#### Observability
```bash
WW_OTEL_ENABLED=false
WW_OTEL_ENDPOINT=http://localhost:4317
WW_OTEL_INSECURE=true               # Set false in production
WW_LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
```

---

## 4. MCP Server Integration

### Claude Desktop Configuration

Edit `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ww-memory": {
      "command": "python",
      "args": ["-m", "ww.mcp.server"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "${NEO4J_PASSWORD}",
        "QDRANT_URL": "http://localhost:6333",
        "WW_SESSION_ID": "claude-desktop-${USER}",
        "WW_EMBEDDING_DEVICE": "cuda:0",
        "PYTHONPATH": "/opt/world-weaver"
      }
    }
  }
}
```

**Important**: Use absolute path to Python if not in PATH:
```json
"command": "/opt/world-weaver/.venv/bin/python"
```

### Claude Code CLI Configuration

Install as skill:
```bash
# Clone to skills directory
cd ~/.claude/skills
ln -s /opt/world-weaver ww-memory

# Add to skills manifest
cat >> ~/.claude/skills/manifest.json << EOF
{
  "ww-memory": {
    "description": "Tripartite neural memory system",
    "command": "python -m ww.mcp.server",
    "working_directory": "/opt/world-weaver"
  }
}
EOF
```

### Verify MCP Connection

```bash
# Test MCP server standalone
cd /opt/world-weaver
python -m ww.mcp.server

# Should output JSON-RPC initialization
# Press Ctrl+D to send EOF and trigger response
```

---

## 5. REST API Deployment

### Development Server

```bash
# Direct invocation
cd /opt/world-weaver
source .venv/bin/activate
python -m ww.api.server

# Or using script entry point
ww-api
```

### Production Server (Systemd)

Create `/etc/systemd/system/ww-api.service`:

```ini
[Unit]
Description=World Weaver REST API
After=network.target neo4j.service qdrant.service

[Service]
Type=notify
User=ww
Group=ww
WorkingDirectory=/opt/world-weaver
Environment="PATH=/opt/world-weaver/.venv/bin"
EnvironmentFile=/opt/world-weaver/.env
ExecStart=/opt/world-weaver/.venv/bin/uvicorn ww.api.server:app \
  --host 0.0.0.0 \
  --port 8765 \
  --workers 4 \
  --log-level info \
  --access-log
Restart=on-failure
RestartSec=5s

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/world-weaver/models

[Install]
WantedBy=multi-user.target
```

Create dedicated user:
```bash
sudo useradd -r -s /bin/false -d /opt/world-weaver ww
sudo chown -R ww:ww /opt/world-weaver
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ww-api
sudo systemctl start ww-api
sudo systemctl status ww-api
```

### NGINX Reverse Proxy

```nginx
server {
    listen 80;
    server_name memory.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name memory.example.com;

    ssl_certificate /etc/letsencrypt/live/memory.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.example.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy to WW API
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://127.0.0.1:8765;
    }
}
```

Install and start:
```bash
sudo ln -s /etc/nginx/sites-available/ww-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6. Monitoring & Observability

### Prometheus Metrics (Available)

World Weaver exposes OpenTelemetry metrics convertible to Prometheus format:

```bash
# Enable in .env
WW_OTEL_ENABLED=true
WW_OTEL_ENDPOINT=http://localhost:4317
```

**Available Metrics**:
- `ww_episodic_create_duration_seconds` - Episode creation latency
- `ww_semantic_recall_duration_seconds` - Semantic retrieval latency
- `ww_embedding_generation_duration_seconds` - BGE-M3 encoding time
- `ww_neo4j_query_duration_seconds` - Graph query latency
- `ww_qdrant_search_duration_seconds` - Vector search latency
- `ww_consolidation_episode_count` - Episodes processed in consolidation

### Prometheus Configuration

`/etc/prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'world-weaver'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8765']
    metrics_path: '/metrics'

  - job_name: 'neo4j'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:7474']
    metrics_path: '/metrics'

  - job_name: 'qdrant'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:6333']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import template from `/mnt/projects/ww/monitoring/grafana-dashboard.json` (if exists), or create custom panels:

**Key Panels**:
1. **API Throughput**: Requests per second by endpoint
2. **Latency Percentiles**: p50, p95, p99 for all operations
3. **Memory Usage**: Episode/entity/skill counts over time
4. **Error Rate**: Failed operations per minute
5. **Database Health**: Neo4j/Qdrant connection pool stats

### Health Checks

```bash
# API health
curl http://localhost:8765/api/v1/health

# Neo4j
echo "RETURN 1;" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD"

# Qdrant
curl http://localhost:6333/readyz
```

### Logging

Logs to stdout by default (Docker/systemd friendly):

```bash
# View API logs
sudo journalctl -u ww-api -f

# In Docker
docker logs -f ww-api

# Adjust log level
WW_LOG_LEVEL=DEBUG python -m ww.api.server
```

Log format (JSON structured):
```json
{
  "timestamp": "2025-12-06T10:30:00Z",
  "level": "INFO",
  "logger": "ww.memory.episodic",
  "message": "Created episode",
  "session_id": "default",
  "episode_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_ms": 45.2
}
```

---

## 7. Security Hardening

### Network Security

1. **Bind to localhost only** (edit configs):
   - Neo4j: `server.bolt.listen_address=127.0.0.1:7687`
   - Qdrant: `service.host=127.0.0.1`
   - API: `WW_API_HOST=127.0.0.1` (use reverse proxy for external access)

2. **Firewall rules**:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS (NGINX)
sudo ufw enable
```

### Authentication

#### API Key Authentication (Production)

Add to `/opt/world-weaver/ww/api/middleware.py`:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("WW_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

Set in `.env`:
```bash
WW_API_KEY=$(openssl rand -hex 32)
```

#### Neo4j Encryption

Enable TLS in `/etc/neo4j/neo4j.conf`:
```conf
dbms.ssl.policy.bolt.enabled=true
dbms.ssl.policy.bolt.base_directory=/etc/neo4j/ssl
dbms.ssl.policy.bolt.private_key=private.key
dbms.ssl.policy.bolt.public_certificate=public.crt
```

Update URI:
```bash
WW_NEO4J_URI=bolt+s://localhost:7687
```

### Secrets Management

**Never commit `.env` to git!**

Use environment-specific secrets:

```bash
# Production secrets in systemd
sudo systemctl edit ww-api

# Add:
[Service]
Environment="NEO4J_PASSWORD=<from-vault>"
Environment="WW_API_KEY=<from-vault>"
```

Or use external secrets manager:
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault

### File Permissions

```bash
cd /opt/world-weaver
sudo chown -R ww:ww .
sudo chmod 600 .env
sudo chmod 700 scripts/*.sh
```

---

## 8. Backup & Recovery

### Neo4j Backup

```bash
# Online backup (Enterprise only)
neo4j-admin backup --backup-dir=/backups/neo4j --name=ww-graph

# Community edition: Stop service and copy data
sudo systemctl stop neo4j
sudo tar czf /backups/neo4j-$(date +%Y%m%d).tar.gz /var/lib/neo4j/data
sudo systemctl start neo4j
```

### Qdrant Backup

```bash
# Create snapshot
curl -X POST http://localhost:6333/snapshots/create

# List snapshots
curl http://localhost:6333/snapshots

# Copy snapshot files
sudo cp -r /var/lib/qdrant/snapshots /backups/qdrant-$(date +%Y%m%d)
```

### Automated Backup Script

`/opt/world-weaver/scripts/backup.sh`:
```bash
#!/bin/bash
set -euo pipefail

BACKUP_DIR=/backups/world-weaver
DATE=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS=30

mkdir -p "$BACKUP_DIR"

# Backup Neo4j
echo "Backing up Neo4j..."
sudo systemctl stop neo4j
sudo tar czf "$BACKUP_DIR/neo4j-$DATE.tar.gz" /var/lib/neo4j/data
sudo systemctl start neo4j

# Backup Qdrant (snapshot)
echo "Backing up Qdrant..."
curl -X POST http://localhost:6333/snapshots/create
sleep 5
sudo cp -r /var/lib/qdrant/snapshots "$BACKUP_DIR/qdrant-$DATE"

# Backup config
echo "Backing up configuration..."
sudo cp /opt/world-weaver/.env "$BACKUP_DIR/env-$DATE"

# Cleanup old backups
echo "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $BACKUP_DIR/*-$DATE"
```

Add to crontab:
```bash
sudo crontab -e

# Daily backup at 2 AM
0 2 * * * /opt/world-weaver/scripts/backup.sh >> /var/log/ww-backup.log 2>&1
```

### Recovery

```bash
# Stop services
sudo systemctl stop ww-api neo4j qdrant

# Restore Neo4j
sudo rm -rf /var/lib/neo4j/data/*
sudo tar xzf /backups/world-weaver/neo4j-20251206-020000.tar.gz -C /
sudo chown -R neo4j:neo4j /var/lib/neo4j/data

# Restore Qdrant
sudo rm -rf /var/lib/qdrant/storage/*
sudo cp -r /backups/world-weaver/qdrant-20251206-020000/* /var/lib/qdrant/snapshots/
curl -X POST http://localhost:6333/snapshots/recover?snapshot_name=<snapshot-name>

# Restore config
sudo cp /backups/world-weaver/env-20251206-020000 /opt/world-weaver/.env

# Restart services
sudo systemctl start neo4j qdrant ww-api
```

---

## 9. Troubleshooting

### API Won't Start

**Symptom**: `ww-api` service fails to start

**Diagnosis**:
```bash
sudo journalctl -u ww-api -n 50
```

**Common causes**:
1. **Neo4j not running**: `sudo systemctl start neo4j`
2. **Qdrant not running**: `sudo systemctl start qdrant`
3. **Port already in use**: Check with `sudo lsof -i :8765`
4. **Missing environment variables**: Verify `.env` file exists and is readable
5. **Python dependencies**: Re-run `pip install -e ".[api]"`

---

### Embedding Generation Slow

**Symptom**: Episode creation takes >10 seconds

**Diagnosis**:
```bash
# Check if GPU is being used
nvidia-smi

# Test embedding speed
python -c "
from ww.embedding.bge_m3 import get_embedding_provider
import time

emb = get_embedding_provider()
start = time.time()
vec = emb.embed_query('test sentence')
print(f'Embedding time: {time.time() - start:.2f}s')
print(f'Device: {emb.device}')
"
```

**Solutions**:
1. **Enable GPU**: Set `WW_EMBEDDING_DEVICE=cuda:0` in `.env`
2. **Enable FP16**: Set `WW_EMBEDDING_USE_FP16=true` (2x speedup)
3. **Reduce max_length**: Set `WW_EMBEDDING_MAX_LENGTH=256` if long texts not needed
4. **Batch requests**: Use SDK batch methods instead of individual calls

---

### Neo4j Connection Timeout

**Symptom**: `Neo4jError: Failed to establish connection`

**Diagnosis**:
```bash
# Check Neo4j status
sudo systemctl status neo4j

# Test connection manually
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" -a bolt://localhost:7687
```

**Solutions**:
1. **Increase timeout**: Set `WW_NEO4J_CONNECTION_TIMEOUT=60` in `.env`
2. **Check password**: Verify `NEO4J_PASSWORD` matches database
3. **Check network**: Ensure `127.0.0.1:7687` is accessible
4. **Review logs**: `sudo journalctl -u neo4j -f`

---

### Qdrant Collection Not Found

**Symptom**: `QdrantException: Collection 'ww-episodes-default' not found`

**Diagnosis**:
```bash
# List collections
curl http://localhost:6333/collections

# Check specific collection
curl http://localhost:6333/collections/ww-episodes-default
```

**Solutions**:
1. **Initialize database**: `python -m ww.scripts.init_database`
2. **Verify session ID**: Ensure `WW_SESSION_ID` matches collection name
3. **Check Qdrant logs**: `sudo journalctl -u qdrant -f`

---

### High Memory Usage

**Symptom**: System OOM killer terminates processes

**Diagnosis**:
```bash
# Check memory usage
free -h
docker stats  # If using Docker

# Check which component is consuming memory
sudo ps aux --sort=-%mem | head -20
```

**Solutions**:
1. **Reduce Neo4j heap**: Set `dbms.memory.heap.max_size=1G` (down from 2G)
2. **Limit API workers**: Set `WW_API_WORKERS=2` (down from 4)
3. **Use CPU instead of GPU**: Set `WW_EMBEDDING_DEVICE=cpu` (frees VRAM)
4. **Enable consolidation**: Periodically run `curl -X POST http://localhost:8765/api/v1/consolidate`

---

### Disk Space Full

**Symptom**: `No space left on device`

**Diagnosis**:
```bash
df -h
du -sh /var/lib/neo4j/data
du -sh /var/lib/qdrant/storage
```

**Solutions**:
1. **Run consolidation**: Merges duplicate episodes
2. **Delete old snapshots**: `rm -rf /var/lib/qdrant/snapshots/old-*`
3. **Clean Neo4j logs**: `sudo truncate -s 0 /var/lib/neo4j/logs/*.log`
4. **Archive old data**: Export and delete old sessions

---

### MCP Server Not Responding

**Symptom**: Claude Desktop shows "MCP server disconnected"

**Diagnosis**:
```bash
# Test MCP server manually
cd /opt/world-weaver
python -m ww.mcp.server
# Type: {"method": "initialize", "id": 1}
# Press Ctrl+D
```

**Solutions**:
1. **Check Python path**: Use absolute path in config: `/opt/world-weaver/.venv/bin/python`
2. **Verify environment**: Ensure all env vars are set in config
3. **Check logs**: Look in Claude Desktop logs (`~/Library/Logs/Claude/` on macOS)
4. **Test dependencies**: `pip list | grep -E "mcp|fastmcp"`

---

## Performance Tuning

### Neo4j Optimization

```conf
# /etc/neo4j/neo4j.conf

# Increase page cache for large graphs
dbms.memory.pagecache.size=4G

# Enable query logging for slow queries
dbms.logs.query.enabled=true
dbms.logs.query.threshold=1s

# Parallel query execution
dbms.query.parallel_runtime_support=true
```

### Qdrant Optimization

```yaml
# /etc/qdrant/config.yaml

# Use HNSW for faster search
hnsw:
  m: 16                # Number of edges per node (higher = slower indexing, faster search)
  ef_construct: 100    # Construction parameter (higher = better quality, slower indexing)

# Enable payload on disk for lower memory
storage:
  on_disk_payload: true
```

### API Optimization

```bash
# Increase worker count (1 per CPU core)
WW_API_WORKERS=8

# Enable connection pooling
WW_NEO4J_MAX_CONNECTION_POOL_SIZE=100

# Reduce batch size for lower latency
WW_EMBEDDING_BATCH_SIZE=16
```

---

## Support Resources

- **Documentation**: `/mnt/projects/ww/docs/`
- **GitHub Issues**: https://github.com/astoreyai/world-weaver/issues
- **Neo4j Docs**: https://neo4j.com/docs/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **MCP Spec**: https://spec.modelcontextprotocol.io/

---

**Document Status**: Complete ✓
**Tested On**: Ubuntu 22.04, Debian 12, macOS 14
**Last Updated**: 2025-12-06
