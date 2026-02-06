# T4DM Deployment Guide

**Version**: 2.0.0
**Last Updated**: 2026-02-05

> **Note**: T4DM 2.0 uses an embedded T4DX storage engine. No external databases required.

## Prerequisites

- Python 3.11+
- 4GB+ RAM (8GB recommended for embedding model + Qwen)
- 10GB+ disk space
- CUDA-capable GPU (optional, for faster inference)

## Quick Start (Development)

1. **Clone and install**
```bash
git clone https://github.com/astoreyai/t4dm
cd t4dm
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

2. **Run API server**
```bash
t4dm serve
# Or: python -m t4dm.api.server
```

3. **Access API**
- API: http://localhost:8765
- Docs: http://localhost:8765/docs
- Health: http://localhost:8765/api/v1/health

## Production Deployment

### Single Binary (Recommended)

T4DM runs as a self-contained service with embedded storage:

```bash
# Install
pip install t4dm

# Configure
export T4DM_STORAGE_PATH=/var/lib/t4dm/data
export T4DM_SESSION_ID=production
export T4DM_HOST=0.0.0.0
export T4DM_PORT=8765

# Run
t4dm serve --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

ENV T4DM_STORAGE_PATH=/data
ENV T4DM_HOST=0.0.0.0

EXPOSE 8765
VOLUME /data

CMD ["t4dm", "serve"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  t4dm:
    build: .
    ports:
      - "8765:8765"
    volumes:
      - t4dm_data:/data
    environment:
      - T4DM_SESSION_ID=production
      - T4DM_EMBEDDING_DEVICE=cpu

volumes:
  t4dm_data:
```

### Systemd Service

```ini
# /etc/systemd/system/t4dm.service
[Unit]
Description=T4DM Memory Service
After=network.target

[Service]
Type=simple
User=t4dm
Group=t4dm
WorkingDirectory=/opt/t4dm
Environment="T4DM_STORAGE_PATH=/var/lib/t4dm/data"
Environment="T4DM_SESSION_ID=production"
ExecStart=/opt/t4dm/venv/bin/t4dm serve --host 0.0.0.0 --port 8765
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Environment Configuration

### Required Variables

```bash
# Storage location (must be writable)
T4DM_STORAGE_PATH=/var/lib/t4dm/data

# Session isolation
T4DM_SESSION_ID=production
```

### Optional Variables

```bash
# Server settings
T4DM_HOST=0.0.0.0
T4DM_PORT=8765
T4DM_WORKERS=4

# Embedding model
T4DM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
T4DM_EMBEDDING_DEVICE=cuda:0  # or "cpu"

# Rate limiting
T4DM_RATE_LIMIT=100  # requests per minute per session

# Logging
T4DM_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# WAL settings
T4DM_WAL_FSYNC_INTERVAL=1.0  # seconds
T4DM_SEGMENT_SIZE=67108864   # 64MB default
```

## Resource Requirements

### Minimum (Development)
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB

### Recommended (Production)
- CPU: 4+ cores
- RAM: 16GB (8GB for Qwen + 8GB for T4DX)
- Disk: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (optional)

### Storage Sizing

| Items | T4DX Size | Notes |
|-------|-----------|-------|
| 10K | ~50MB | Small deployment |
| 100K | ~500MB | Medium deployment |
| 1M | ~5GB | Large deployment |
| 10M | ~50GB | Very large deployment |

## Monitoring

### Health Check

```bash
curl http://localhost:8765/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-05T12:00:00Z",
  "version": "2.0.0",
  "storage": {
    "segments": 5,
    "total_items": 10000,
    "memtable_size": 1024
  }
}
```

### Prometheus Metrics

Metrics available at `/metrics`:

- `t4dm_requests_total` - Total API requests
- `t4dm_request_duration_seconds` - Request latency histogram
- `t4dm_storage_items_total` - Total stored items
- `t4dm_consolidation_runs_total` - Consolidation executions
- `t4dm_kappa_distribution` - κ value distribution

### Visualization Endpoints

Access live metrics via REST:
- `/api/v1/viz/realtime/metrics` - Aggregated metrics
- `/api/v1/viz/kappa/distribution` - κ distribution
- `/api/v1/viz/t4dx/storage` - Storage statistics

## Backup & Recovery

### Backup

```bash
# Stop writes (optional, for consistency)
curl -X POST http://localhost:8765/api/v1/control/pause

# Backup storage directory
tar -czf t4dm-backup-$(date +%Y%m%d).tar.gz /var/lib/t4dm/data

# Resume writes
curl -X POST http://localhost:8765/api/v1/control/resume
```

### Recovery

```bash
# Stop service
systemctl stop t4dm

# Restore backup
rm -rf /var/lib/t4dm/data
tar -xzf t4dm-backup-20260205.tar.gz -C /

# Start service (WAL replay happens automatically)
systemctl start t4dm
```

### Point-in-Time Recovery

T4DX uses write-ahead logging (WAL) for crash recovery:
1. WAL is replayed on startup
2. Uncommitted transactions are recovered
3. Corrupted segments are skipped with warnings

## Security

### API Key Authentication

```bash
# Set API key
export T4DM_API_KEY=your-secret-key

# Clients must provide header
curl -H "X-API-Key: your-secret-key" http://localhost:8765/api/v1/health
```

### Rate Limiting

Default: 100 requests/minute per session

```bash
# Override
T4DM_RATE_LIMIT=500
```

### Network Security

- Run behind reverse proxy (nginx, Caddy) in production
- Use TLS termination at proxy level
- Restrict access to trusted networks

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow startup | Large WAL replay | Normal on first start after crash |
| OOM errors | Insufficient RAM | Increase RAM or reduce batch sizes |
| 429 errors | Rate limiting | Increase T4DM_RATE_LIMIT |
| Connection refused | Service not running | Check `systemctl status t4dm` |

### Debug Mode

```bash
T4DM_LOG_LEVEL=DEBUG t4dm serve
```

### Log Locations

- stdout/stderr (Docker, systemd)
- `/var/log/t4dm/` (if configured)

## Upgrading

### Minor Version (2.0.x → 2.0.y)

```bash
pip install --upgrade t4dm
systemctl restart t4dm
```

### Major Version (1.x → 2.x)

See migration guide in release notes. Major versions may require:
1. Data export from old version
2. Schema migration
3. Data import to new version

---

For additional support, see [GitHub Issues](https://github.com/astoreyai/t4dm/issues).
