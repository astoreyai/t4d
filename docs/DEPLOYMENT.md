# World Weaver Deployment Guide

## Prerequisites

- Docker 20.10+ and Docker Compose
- 4GB+ RAM (8GB recommended for embedding model)
- 10GB+ disk space

## Quick Start (Development)

1. **Clone and configure**
```bash
git clone https://github.com/astoreyai/world-weaver
cd world-weaver
cp .env.example .env
./scripts/setup-env.sh  # Generates secure passwords
```

2. **Start infrastructure only**
```bash
docker-compose up -d
```

3. **Run API locally** (for development)
```bash
pip install -e ".[api]"
python -m ww.api.server
```

4. **Access API**
- API: http://localhost:8765
- Docs: http://localhost:8765/docs
- Neo4j Browser: http://localhost:7474

## Full Stack Deployment

1. **Configure environment**
```bash
cp .env.example .env
# Edit .env with production values
```

2. **Build and start all services**
```bash
docker-compose -f docker-compose.full.yml up -d --build
```

3. **Verify deployment**
```bash
# Check health
curl http://localhost:8765/api/v1/health

# Check logs
docker-compose -f docker-compose.full.yml logs -f ww-api
```

## Environment Configuration

### Required Variables

```bash
# Neo4j credentials
NEO4J_USER=neo4j
NEO4J_PASSWORD=<secure-password>

# Must match Neo4j
WW_NEO4J_URI=bolt://localhost:7687
WW_NEO4J_USER=neo4j
WW_NEO4J_PASSWORD=<same-password>
```

### Optional Variables

```bash
# Session isolation
WW_SESSION_ID=production

# Qdrant (vector storage)
WW_QDRANT_URL=http://localhost:6333

# Embedding model
WW_EMBEDDING_MODEL=BAAI/bge-m3
WW_EMBEDDING_DIMENSION=1024
WW_EMBEDDING_DEVICE=cuda:0  # or 'cpu'

# API settings
WW_API_HOST=0.0.0.0
WW_API_PORT=8765
WW_API_WORKERS=4

# Memory parameters
WW_FSRS_DEFAULT_STABILITY=1.0
WW_FSRS_RETENTION_TARGET=0.9
WW_HEBBIAN_LEARNING_RATE=0.1

# Observability
WW_OTEL_ENABLED=true
WW_OTEL_ENDPOINT=http://jaeger:4317
```

## Production Checklist

### Security

- [ ] Strong passwords in `.env` (min 16 chars)
- [ ] Change Neo4j default credentials
- [ ] Enable Qdrant API key: `QDRANT__SERVICE__API_KEY=<key>`
- [ ] **Enable API key authentication** (required for production):
  ```bash
  WW_API_KEY=$(openssl rand -hex 32)
  # API key auto-required when WW_ENVIRONMENT=production and WW_API_KEY is set
  # Pass via X-API-Key header on all requests
  ```
  - Exempt endpoints: `/`, `/api/v1/health`, `/docs`, `/redoc`, `/openapi.json`
- [ ] Configure CORS: `WW_API_CORS_ORIGINS=https://yourdomain.com`
  - **Note**: Wildcards (`*`) are rejected in production (`WW_ENVIRONMENT=production`)
  - Allowed headers: `Authorization`, `Content-Type`, `X-Session-ID`, `X-Request-ID`, `X-API-Key`, `X-Admin-Key`
- [ ] Run behind reverse proxy (nginx/traefik) with TLS
- [ ] Bind ports to 127.0.0.1 (not 0.0.0.0) when using reverse proxy
- [ ] Set `WW_ENVIRONMENT=production` to enable security validators

### Performance

- [ ] Use GPU for embeddings: `WW_EMBEDDING_DEVICE=cuda:0`
- [ ] Configure API workers appropriately (see Worker Configuration section below)
- [ ] Tune Neo4j memory in docker-compose:
  ```yaml
  NEO4J_dbms_memory_heap_max__size=4G
  NEO4J_dbms_memory_pagecache_size=2G
  ```

### Persistence

- [ ] Use named volumes (default configuration)
- [ ] Set up volume backups:
  ```bash
  # Neo4j backup
  docker exec ww-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j.dump

  # Qdrant snapshot
  curl -X POST http://localhost:6333/snapshots
  ```

### Monitoring

- [ ] Enable OpenTelemetry: `WW_OTEL_ENABLED=true`
- [ ] Configure health check alerts
- [ ] Set up log aggregation

## Architecture

```
                    ┌─────────────────┐
                    │   Client Apps   │
                    │  (SDK / HTTP)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   WW REST API   │
                    │  (FastAPI)      │
                    │  Port: 8765     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│  Neo4j Graph    │ │  Qdrant Vector  │ │  BGE-M3 Model   │
│  (Entities,     │ │  (Embeddings)   │ │  (Embeddings)   │
│   Relations)    │ │  Port: 6333     │ │                 │
│  Port: 7687     │ └─────────────────┘ └─────────────────┘
└─────────────────┘
```

## Worker Configuration

### Overview

The `WW_API_WORKERS` setting controls how many Uvicorn worker processes handle requests.
Each worker is a separate Python process with its own memory space.

**Default**: 1 worker (recommended for development and small deployments)
**Maximum**: 32 workers

### Recommended Settings

| Deployment | CPU Cores | Memory | Workers | Notes |
|------------|-----------|--------|---------|-------|
| Development | Any | 4GB | 1 | Simpler debugging |
| Small Production | 2-4 | 8GB | 2-4 | `workers = cores - 1` |
| Medium Production | 4-8 | 16GB | 4-8 | `workers = cores` |
| Large Production | 8+ | 32GB+ | 8-16 | `workers = 2 * cores` |

### Formula

```bash
# For CPU-bound workloads (embedding generation)
WW_API_WORKERS=$(($(nproc) - 1))

# For I/O-bound workloads (database queries)
WW_API_WORKERS=$(($(nproc) * 2))
```

### Memory Considerations

Each worker allocates:
- ~500MB base Python runtime
- ~2GB for BGE-M3 embedding model (shared via copy-on-write if using same model)
- Variable heap for request processing

**Example calculation for 16GB RAM**:
```
Available: 16GB
- Neo4j heap: 4GB
- Neo4j pagecache: 2GB
- Qdrant: 2GB
- OS/buffers: 2GB
- Remaining: 6GB for API workers
- Max workers: 6GB / 1GB per worker = 6 workers
```

### Rate Limiting Considerations

**IMPORTANT**: The built-in rate limiter is **per-worker, not distributed**.

With `WW_API_WORKERS=4` and rate limit of 100 req/min:
- Each worker allows 100 req/min independently
- Effective limit: 400 req/min (4 × 100)

**For strict rate limiting in multi-worker deployments**:

1. **Option A: Keep single worker** (simplest)
   ```bash
   WW_API_WORKERS=1  # Built-in rate limiter works correctly
   ```

2. **Option B: Use nginx rate limiting** (recommended for production)
   ```nginx
   # nginx.conf
   limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;

   location /api/ {
       limit_req zone=api burst=20 nodelay;
       proxy_pass http://ww-api:8765;
   }
   ```

3. **Option C: Use Redis-based rate limiter** (for distributed deployments)
   ```bash
   # Future enhancement - not yet implemented
   WW_RATE_LIMITER_BACKEND=redis
   WW_REDIS_URL=redis://localhost:6379
   ```

### Configuration Example

```bash
# .env for 8-core, 32GB production server
WW_API_WORKERS=8
WW_API_HOST=127.0.0.1  # Behind reverse proxy
WW_API_PORT=8765

# Neo4j tuning for 8 workers
WW_NEO4J_POOL_SIZE=100  # 100 connections shared across workers

# Embedding cache (per-worker)
WW_EMBEDDING_CACHE_SIZE=500  # Reduce if memory constrained
```

## Scaling

### Horizontal Scaling

For high availability, run multiple API instances behind a load balancer:

```yaml
# docker-compose.prod.yml
services:
  ww-api:
    deploy:
      replicas: 3
    # ... rest of config
```

### Resource Limits

```yaml
services:
  ww-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Troubleshooting

### API Won't Start

1. Check storage services are healthy:
```bash
docker-compose ps
curl http://localhost:6333/readyz  # Qdrant
curl http://localhost:7474         # Neo4j
```

2. Check logs:
```bash
docker-compose logs ww-api
```

3. Verify environment:
```bash
docker exec ww-api env | grep WW_
```

### Slow Embedding

1. Use GPU if available: `WW_EMBEDDING_DEVICE=cuda:0`
2. Reduce batch size: `WW_EMBEDDING_BATCH_SIZE=16`
3. First request loads model (~30s), subsequent requests are fast

### Memory Issues

1. Reduce Neo4j heap: `NEO4J_dbms_memory_heap_max__size=1G`
2. Use FP16 embeddings: `WW_EMBEDDING_USE_FP16=true`
3. Limit concurrent requests via API workers

### Connection Refused

1. Check if services are on same Docker network
2. Use internal hostnames in docker-compose (e.g., `bolt://neo4j:7687`)
3. Verify firewall rules

## Maintenance

### Backup

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d)

# Stop services
docker-compose stop ww-api

# Neo4j dump
docker exec ww-neo4j neo4j-admin database dump neo4j --to-path=/backups
docker cp ww-neo4j:/backups/neo4j.dump ./backups/neo4j-$DATE.dump

# Qdrant snapshot
curl -X POST http://localhost:6333/snapshots
docker cp ww-qdrant:/qdrant/snapshots ./backups/qdrant-$DATE/

# Restart
docker-compose start ww-api
```

### Restore

```bash
# Neo4j
docker cp ./backups/neo4j-$DATE.dump ww-neo4j:/backups/
docker exec ww-neo4j neo4j-admin database load neo4j --from-path=/backups --overwrite-destination

# Qdrant - copy snapshot to storage directory
```

### Upgrade

```bash
# Pull latest
git pull origin main

# Rebuild
docker-compose -f docker-compose.full.yml build

# Rolling restart
docker-compose -f docker-compose.full.yml up -d --no-deps ww-api
```

---

## Phase 9: Production Infrastructure

### Kubernetes Deployment

**Kustomize overlays** for dev/staging/prod:

```bash
# Development (single replica, reduced resources)
kubectl apply -k deploy/kubernetes/overlays/dev

# Staging (moderate replicas)
kubectl apply -k deploy/kubernetes/overlays/staging

# Production (HA with PDB)
kubectl apply -k deploy/kubernetes/overlays/prod
```

### Helm Chart

```bash
# Add dependencies
helm dependency update deploy/helm/world-weaver

# Install with secrets
helm install ww deploy/helm/world-weaver \
  --namespace world-weaver \
  --create-namespace \
  --set secrets.jwtSecret="$(openssl rand -base64 32)" \
  --set secrets.databasePassword="$(openssl rand -base64 16)" \
  --set secrets.neo4jPassword="$(openssl rand -base64 16)"

# Upgrade
helm upgrade ww deploy/helm/world-weaver --reuse-values
```

### Control Plane API (Phase 9)

Admin endpoints for runtime control:

**Feature Flags**:
```bash
# List all flags
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8765/api/v1/control/flags

# Toggle a flag
curl -X PATCH \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"enabled": false}' \
  http://localhost:8765/api/v1/control/flags/telemetry
```

**Emergency Controls**:
```bash
# Trigger panic mode (LIMITED = read-only)
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"level": "LIMITED", "reason": "Maintenance"}' \
  http://localhost:8765/api/v1/control/emergency/panic

# Recover
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"level": "NONE"}' \
  http://localhost:8765/api/v1/control/emergency/recover
```

**Circuit Breakers**:
```bash
# List all circuit breakers
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8765/api/v1/control/circuits

# Reset a tripped circuit
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"action": "reset"}' \
  http://localhost:8765/api/v1/control/circuits/database
```

### Secrets Management (Phase 9)

The system auto-detects the secrets backend:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `auto` | Auto-detect (default) | All environments |
| `env` | Environment variables | Development |
| `file` | File-based (`/run/secrets/`) | Docker/K8s secrets |
| `chained` | File fallback to env | Production |

Override with `WW_SECRETS_BACKEND=<backend>`.

### Feature Flags (Phase 9)

Control subsystems at runtime without restart:

| Flag | Description | Default |
|------|-------------|---------|
| `ff_encoder` | Learnable FF encoding | `true` |
| `capsule_encoding` | Capsule representations | `true` |
| `lability_window` | Protein synthesis gate | `true` |
| `three_factor_learning` | Neuromodulated learning | `true` |
| `api_rate_limiting` | Request rate limits | `true` |
| `read_only_mode` | Block all writes | `false` |
| `maintenance_mode` | Block all requests | `false` |

Set via environment: `WW_FLAG_<NAME>=true|false`
