# World Weaver Deployment Guide

**Version**: 1.0.0 | **Last Updated**: 2025-11-27

## Overview

World Weaver is a tripartite memory system for Claude Code instances, providing episodic, semantic, and procedural memory via MCP (Model Context Protocol).

**Architecture**:
- **Qdrant**: Vector embeddings for semantic search (BGE-M3)
- **Neo4j**: Graph database for entities, relationships, and temporal queries
- **Python 3.11+**: FastMCP server with async/await
- **MCP Protocol**: stdio-based communication with Claude Code

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB | 50 GB SSD |
| GPU | None | CUDA-capable (for embeddings) |

### Software Dependencies

- **Python**: 3.11 or higher
- **Docker**: 20.10+ with Docker Compose
- **Git**: For cloning repository
- **CUDA**: Optional, for GPU acceleration of embeddings

### Supported Platforms

- Linux (Debian, Ubuntu, Fedora, Arch)
- macOS 12+ (Intel/Apple Silicon)
- Windows 10/11 with WSL2

---

## Quick Start (Docker Compose)

### 1. Clone Repository

```bash
git clone https://github.com/astoreyai/world-weaver.git
cd world-weaver
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit configuration
nano .env
```

**Required Variables**:

| Variable | Description | Example |
|----------|-------------|---------|
| `WW_SESSION_ID` | Session identifier | `default` |
| `WW_NEO4J_URI` | Neo4j connection string | `bolt://localhost:7687` |
| `WW_NEO4J_USER` | Neo4j username | `neo4j` |
| `WW_NEO4J_PASSWORD` | Neo4j password | `your-secure-password` |
| `WW_QDRANT_URL` | Qdrant REST API URL | `http://localhost:6333` |

**Optional Variables** (see `.env` for defaults):

```bash
# Embedding Configuration
WW_EMBEDDING_MODEL=BAAI/bge-m3
WW_EMBEDDING_DIMENSION=1024
WW_EMBEDDING_DEVICE=cuda:0  # or cpu
WW_EMBEDDING_USE_FP16=true
WW_EMBEDDING_BATCH_SIZE=32

# Memory Parameters
WW_FSRS_DEFAULT_STABILITY=1.0
WW_FSRS_RETENTION_TARGET=0.9
WW_HEBBIAN_LEARNING_RATE=0.1

# Consolidation Thresholds
WW_CONSOLIDATION_MIN_SIMILARITY=0.75
WW_CONSOLIDATION_MIN_OCCURRENCES=3
```

### 3. Start Infrastructure

```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Verify services are running
docker-compose ps
```

**Expected Output**:
```
NAME                COMMAND                  SERVICE             STATUS              PORTS
ww-neo4j            "/startup/docker-ent…"   neo4j               running             0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
ww-qdrant           "./qdrant"               qdrant              running             0.0.0.0:6333->6333/tcp, 0.0.0.0:6334->6334/tcp
```

### 4. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 5. Verify Installation

```bash
# Check health of services
curl http://localhost:6333/health  # Qdrant
curl http://localhost:7474         # Neo4j browser

# Run tests
pytest tests/ -v

# Start MCP server (test mode)
python -m ww.mcp.memory_gateway
```

---

## Docker Compose Configuration

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5-community
    container_name: ww-neo4j
    ports:
      - "7474:7474"  # HTTP browser
      - "7687:7687"  # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/your-secure-password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=1G
    volumes:
      - ww_neo4j_data:/data
      - ww_neo4j_logs:/logs
      - ww_neo4j_plugins:/plugins
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: ww-qdrant
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - ww_qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  ww_neo4j_data:
    driver: local
  ww_neo4j_logs:
    driver: local
  ww_neo4j_plugins:
    driver: local
  ww_qdrant_storage:
    driver: local

networks:
  default:
    name: ww-network
```

---

## Manual Installation (Without Docker)

### Install Neo4j

**Debian/Ubuntu**:
```bash
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j
sudo systemctl enable neo4j
sudo systemctl start neo4j
```

**macOS** (Homebrew):
```bash
brew install neo4j
brew services start neo4j
```

**Configuration** (`/etc/neo4j/neo4j.conf`):
```
dbms.default_listen_address=0.0.0.0
dbms.connector.bolt.listen_address=:7687
dbms.connector.http.listen_address=:7474
dbms.security.procedures.unrestricted=apoc.*
```

Set password:
```bash
neo4j-admin set-initial-password your-secure-password
```

### Install Qdrant

**Linux** (binary):
```bash
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
./qdrant &
```

**macOS** (Homebrew):
```bash
brew install qdrant
brew services start qdrant
```

**Configuration** (`config/config.yaml`):
```yaml
service:
  http_port: 6333
  grpc_port: 6334
storage:
  storage_path: ./storage
log_level: INFO
```

---

## Production Deployment

### Security Hardening

#### 1. Strong Passwords

```bash
# Generate secure password
openssl rand -base64 32

# Update .env
WW_NEO4J_PASSWORD=<generated-password>
```

#### 2. TLS/SSL for Neo4j

**Generate certificates**:
```bash
mkdir -p /etc/neo4j/certificates/bolt
cd /etc/neo4j/certificates/bolt

# Generate private key
openssl genrsa -out private.key 2048

# Generate certificate
openssl req -new -x509 -key private.key -out public.crt -days 365
```

**Configure** (`neo4j.conf`):
```
dbms.ssl.policy.bolt.enabled=true
dbms.ssl.policy.bolt.base_directory=/etc/neo4j/certificates/bolt
dbms.ssl.policy.bolt.private_key=private.key
dbms.ssl.policy.bolt.public_certificate=public.crt
```

**Update connection**:
```bash
WW_NEO4J_URI=bolt+s://localhost:7687
```

#### 3. File Permissions

```bash
# Secure .env file
chmod 600 .env

# Secure configuration directory
chmod 700 config/
chmod 600 config/*.yaml
```

#### 4. Network Isolation

**Restrict to localhost** (docker-compose.yml):
```yaml
services:
  neo4j:
    ports:
      - "127.0.0.1:7474:7474"
      - "127.0.0.1:7687:7687"

  qdrant:
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
```

#### 5. Rate Limiting

Configure in `.env`:
```bash
WW_RATE_LIMIT_MAX_REQUESTS=100
WW_RATE_LIMIT_WINDOW_SECONDS=60
```

### Resource Optimization

#### Neo4j Tuning

**For 16GB RAM** (`neo4j.conf`):
```
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=4G
dbms.jvm.additional=-XX:+UseG1GC
```

**Connection pooling** (already configured in World Weaver):
```python
# src/ww/storage/neo4j_store.py
max_connection_pool_size=100
connection_acquisition_timeout=60.0
connection_timeout=30.0
```

#### Qdrant Tuning

**For production** (`config.yaml`):
```yaml
service:
  max_request_size_mb: 32
  max_workers: 0  # Auto-detect CPU cores

storage:
  on_disk_payload: true  # Save RAM
  optimizers:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
```

### Monitoring & Logging

#### 1. Enable Structured Logging

```bash
# In .env
WW_LOG_LEVEL=INFO
WW_LOG_FORMAT=json
WW_LOG_FILE=/var/log/ww/memory_gateway.log
```

#### 2. Health Checks

**Script**: `scripts/health_check.sh`
```bash
#!/bin/bash
set -e

# Check Qdrant
curl -f http://localhost:6333/health || exit 1

# Check Neo4j
curl -f http://localhost:7474 || exit 1

# Check MCP server
python -c "from ww.observability.health import check_health; check_health()" || exit 1

echo "All services healthy"
```

**Cron job**:
```bash
# Run every 5 minutes
*/5 * * * * /opt/ww/scripts/health_check.sh >> /var/log/ww/health.log 2>&1
```

#### 3. Metrics Export

World Weaver exposes Prometheus-compatible metrics via the `ww.observability.metrics` module.

**Configuration**:
```bash
# In .env
WW_METRICS_ENABLED=true
WW_METRICS_PORT=9090
```

**Prometheus scrape config**:
```yaml
scrape_configs:
  - job_name: 'world-weaver'
    static_configs:
      - targets: ['localhost:9090']
```

### Backup & Recovery

#### Automated Backups

**Script**: `scripts/backup.sh`
```bash
#!/bin/bash
BACKUP_DIR="/var/backups/ww/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Backup Neo4j
docker exec ww-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j.dump
docker cp ww-neo4j:/backups/neo4j.dump "$BACKUP_DIR/"

# Backup Qdrant
docker exec ww-qdrant tar czf /tmp/qdrant-backup.tar.gz /qdrant/storage
docker cp ww-qdrant:/tmp/qdrant-backup.tar.gz "$BACKUP_DIR/"

# Backup .env and configs
cp .env "$BACKUP_DIR/"
cp -r config/ "$BACKUP_DIR/"

echo "Backup completed: $BACKUP_DIR"
```

**Cron job** (daily at 2 AM):
```bash
0 2 * * * /opt/ww/scripts/backup.sh >> /var/log/ww/backup.log 2>&1
```

#### Recovery

```bash
# Stop services
docker-compose down

# Restore Neo4j
docker run --rm -v ww_neo4j_data:/data -v $(pwd)/backup:/backup neo4j:5 \
  neo4j-admin load --from=/backup/neo4j.dump --database=neo4j --force

# Restore Qdrant
docker run --rm -v ww_qdrant_storage:/qdrant/storage -v $(pwd)/backup:/backup alpine \
  tar xzf /backup/qdrant-backup.tar.gz -C /

# Restart services
docker-compose up -d
```

---

## Claude Code Integration

### MCP Server Configuration

**File**: `~/.config/claude-code/mcp_servers.json`

```json
{
  "mcpServers": {
    "world-weaver": {
      "command": "python",
      "args": ["-m", "ww.mcp.memory_gateway"],
      "cwd": "/mnt/projects/ww",
      "env": {
        "WW_SESSION_ID": "my-session",
        "WW_NEO4J_URI": "bolt://localhost:7687",
        "WW_NEO4J_USER": "neo4j",
        "WW_NEO4J_PASSWORD": "your-password",
        "WW_QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Verify**:
```bash
# Test MCP server manually
python -m ww.mcp.memory_gateway
# Should start and listen for stdio MCP messages
```

### Using World Weaver in Claude Code

Once configured, Claude Code can access 17 memory tools:

**Example conversation**:
```
User: "Remember this: FastAPI is great for async web APIs"

Claude: [Uses create_episode to store the conversation]
        [Uses create_entity to extract FastAPI concept]

User: "What did we discuss about web frameworks?"

Claude: [Uses recall_episodes and semantic_recall to retrieve context]
```

---

## Troubleshooting

### Common Issues

#### 1. Neo4j Connection Refused

**Symptom**:
```
neo4j.exceptions.ServiceUnavailable: Unable to connect to bolt://localhost:7687
```

**Solution**:
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs ww-neo4j

# Restart
docker-compose restart neo4j
```

#### 2. Qdrant Collection Not Found

**Symptom**:
```
qdrant_client.http.exceptions.UnexpectedResponse: Collection ww_episodes not found
```

**Solution**:
```bash
# Initialize collections
python -c "
from ww.memory.episodic import get_episodic_memory
import asyncio
async def init():
    mem = get_episodic_memory('default')
    await mem.initialize()
asyncio.run(init())
"
```

#### 3. CUDA Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solution** (use CPU for embeddings):
```bash
# In .env
WW_EMBEDDING_DEVICE=cpu
WW_EMBEDDING_USE_FP16=false
WW_EMBEDDING_BATCH_SIZE=8  # Reduce batch size
```

#### 4. Rate Limit Issues

**Symptom**:
```json
{"error": "rate_limited", "retry_after": 15.3}
```

**Solution**:
```bash
# Increase rate limit in .env
WW_RATE_LIMIT_MAX_REQUESTS=500
WW_RATE_LIMIT_WINDOW_SECONDS=60

# Or reset rate limiter
python -c "
from ww.mcp.memory_gateway import _rate_limiter
_rate_limiter.reset()
"
```

### Debug Mode

**Enable verbose logging**:
```bash
# In .env
WW_LOG_LEVEL=DEBUG

# Run server with debug output
python -m ww.mcp.memory_gateway 2>&1 | tee debug.log
```

### Performance Profiling

```bash
# Profile memory usage
python -m memory_profiler -m ww.mcp.memory_gateway

# Profile CPU usage
python -m cProfile -o profile.stats -m ww.mcp.memory_gateway

# Analyze profile
python -m pstats profile.stats
```

---

## Production Checklist

Before deploying to production:

- [ ] Set strong `WW_NEO4J_PASSWORD` (32+ characters)
- [ ] Enable TLS/SSL for Neo4j
- [ ] Configure rate limiting (default: 100/min)
- [ ] Restrict network access (bind to 127.0.0.1)
- [ ] Set up automated backups (daily)
- [ ] Configure log rotation
- [ ] Enable health check monitoring
- [ ] Test recovery procedure
- [ ] Document custom configuration
- [ ] Review file permissions (chmod 600 for .env)
- [ ] Set resource limits (Neo4j heap, Qdrant workers)
- [ ] Configure firewall rules
- [ ] Enable metrics export (Prometheus)
- [ ] Set up alerting (disk space, memory, errors)
- [ ] Test MCP integration with Claude Code

---

## Resource Requirements by Scale

### Small (1-10 users)

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| World Weaver | 2 cores | 2GB | 1GB |
| Qdrant | 2 cores | 4GB | 10GB |
| Neo4j | 2 cores | 4GB | 10GB |
| **Total** | 6 cores | 10GB | 21GB |

### Medium (10-100 users)

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| World Weaver | 4 cores | 8GB | 5GB |
| Qdrant | 4 cores | 16GB | 50GB |
| Neo4j | 4 cores | 16GB | 50GB |
| **Total** | 12 cores | 40GB | 105GB |

### Large (100+ users)

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| World Weaver | 8 cores | 16GB | 20GB |
| Qdrant | 8 cores | 32GB | 200GB SSD |
| Neo4j | 8 cores | 32GB | 200GB SSD |
| **Total** | 24 cores | 80GB | 420GB |

**Notes**:
- Disk requirements grow with memory size
- GPU significantly accelerates embedding generation
- Neo4j benefits from SSD for graph traversals
- Qdrant can use on-disk payloads to reduce RAM

---

## Migration & Upgrades

### Upgrading World Weaver

```bash
# Backup data first
./scripts/backup.sh

# Pull latest code
git pull origin main

# Upgrade dependencies
pip install -e ".[dev]" --upgrade

# Run migrations (if any)
python -m ww.migrations.run

# Restart services
docker-compose restart
```

### Data Migration from v0.x to v1.x

**Script**: `scripts/migrate_v0_to_v1.py`
```python
# See migration documentation in docs/migrations/
```

---

## Support & Resources

- **Documentation**: `/mnt/projects/ww/docs/`
- **Issues**: https://github.com/astoreyai/world-weaver/issues
- **API Reference**: `/mnt/projects/ww/docs/api.md`
- **Architecture**: `/mnt/projects/ww/ARCHITECTURE.md`
- **Memory System**: `/mnt/projects/ww/MEMORY_ARCHITECTURE.md`

---

## License

World Weaver is released under the MIT License. See `LICENSE` file for details.
