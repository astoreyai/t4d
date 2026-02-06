# T4DM - Self-Hosted Deployment Guide

**Version**: 2.0.0
**Last Updated**: 2026-02-05
**Target Audience**: DevOps engineers, system administrators, self-hosters

> **Note**: T4DM 2.0 uses an embedded T4DX storage engine. No external databases required.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Manual Installation](#2-manual-installation)
3. [Configuration](#3-configuration)
4. [MCP Server Integration](#4-mcp-server-integration)
5. [REST API Deployment](#5-rest-api-deployment)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Security Hardening](#7-security-hardening)
8. [Backup & Recovery](#8-backup--recovery)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

### Prerequisites

- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- 20GB disk space
- (Optional) NVIDIA GPU with CUDA 12+ for faster inference

### Installation

```bash
# Clone repository
git clone https://github.com/astoreyai/t4dm.git
cd t4dm

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install with all extras
pip install -e ".[dev,api]"

# Verify installation
t4dm --version
pytest tests/unit/ -x -q
```

### Run Server

```bash
# Start API server
t4dm serve

# Or with options
t4dm serve --host 0.0.0.0 --port 8765 --workers 4

# Verify health
curl http://localhost:8765/api/v1/health
```

### Docker Deployment

```bash
# Build image
docker build -t t4dm:latest .

# Run container
docker run -d \
  -p 8765:8765 \
  -v t4dm_data:/data \
  -e T4DM_STORAGE_PATH=/data \
  -e T4DM_SESSION_ID=production \
  t4dm:latest

# Check logs
docker logs -f $(docker ps -q -f ancestor=t4dm:latest)
```

---

## 2. Manual Installation

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB SSD
- OS: Ubuntu 22.04 LTS, Debian 12, macOS 14+, Windows 11

**Recommended**:
- CPU: 8+ cores
- RAM: 24GB
- Disk: 100GB NVMe SSD
- GPU: NVIDIA RTX 3060+ (12GB VRAM)

### Step 1: Install Python

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# macOS
brew install python@3.11

# Windows (use Python installer from python.org)
```

### Step 2: Install T4DM

```bash
# Create directory
mkdir -p /opt/t4dm
cd /opt/t4dm

# Clone and setup
git clone https://github.com/astoreyai/t4dm.git .
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[api]"
```

### Step 3: Configure Storage

```bash
# Create data directory
sudo mkdir -p /var/lib/t4dm/data
sudo chown $USER:$USER /var/lib/t4dm/data

# Set environment
export T4DM_STORAGE_PATH=/var/lib/t4dm/data
```

---

## 3. Configuration

### Environment Variables

```bash
# Required
T4DM_STORAGE_PATH=/var/lib/t4dm/data    # Storage location
T4DM_SESSION_ID=production               # Session namespace

# Server
T4DM_HOST=0.0.0.0                        # Bind address
T4DM_PORT=8765                           # Port
T4DM_WORKERS=4                           # Uvicorn workers

# Embedding
T4DM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
T4DM_EMBEDDING_DEVICE=cuda:0             # or "cpu"

# Rate Limiting
T4DM_RATE_LIMIT=100                      # Requests/minute/session

# Logging
T4DM_LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR

# WAL Settings
T4DM_WAL_FSYNC_INTERVAL=1.0             # Fsync interval (seconds)
T4DM_SEGMENT_SIZE=67108864               # 64MB segment size
```

### Configuration File

Create `/etc/t4dm/config.yaml`:

```yaml
storage:
  path: /var/lib/t4dm/data
  segment_size: 67108864
  wal_fsync_interval: 1.0

server:
  host: 0.0.0.0
  port: 8765
  workers: 4

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cuda:0
  batch_size: 32

rate_limit:
  requests_per_minute: 100
  burst: 20

logging:
  level: INFO
  format: json
```

---

## 4. MCP Server Integration

### Claude Desktop

Edit `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "t4dm": {
      "command": "/opt/t4dm/venv/bin/t4dm",
      "args": ["mcp", "server"],
      "env": {
        "T4DM_SESSION_ID": "claude-desktop",
        "T4DM_STORAGE_PATH": "/var/lib/t4dm/data"
      }
    }
  }
}
```

### Claude Code

Edit `~/.claude/mcp_servers.json`:

```json
{
  "t4dm": {
    "command": "t4dm",
    "args": ["mcp", "server"],
    "env": {
      "T4DM_SESSION_ID": "${INSTANCE_ID}"
    }
  }
}
```

### Verify MCP

```bash
# Test MCP server directly
echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | t4dm mcp server
```

---

## 5. REST API Deployment

### Systemd Service

Create `/etc/systemd/system/t4dm.service`:

```ini
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
Environment="T4DM_LOG_LEVEL=INFO"
ExecStart=/opt/t4dm/venv/bin/t4dm serve --host 0.0.0.0 --port 8765 --workers 4
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable t4dm
sudo systemctl start t4dm

# Check status
sudo systemctl status t4dm
sudo journalctl -u t4dm -f
```

### Nginx Reverse Proxy

```nginx
upstream t4dm {
    server 127.0.0.1:8765;
}

server {
    listen 443 ssl http2;
    server_name memory.example.com;

    ssl_certificate /etc/letsencrypt/live/memory.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.example.com/privkey.pem;

    location / {
        proxy_pass http://t4dm;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 6. Monitoring & Observability

### Prometheus Metrics

Metrics available at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 't4dm'
    static_configs:
      - targets: ['localhost:8765']
    metrics_path: /metrics
```

Key metrics:
- `t4dm_requests_total` - Total API requests
- `t4dm_request_duration_seconds` - Latency histogram
- `t4dm_storage_items_total` - Stored items
- `t4dm_consolidation_runs_total` - Consolidation count
- `t4dm_kappa_distribution` - κ value distribution

### Health Checks

```bash
# Basic health
curl http://localhost:8765/api/v1/health

# Detailed metrics
curl http://localhost:8765/api/v1/viz/realtime/metrics

# Storage stats
curl http://localhost:8765/api/v1/viz/t4dx/storage
```

### Logging

```bash
# View logs
sudo journalctl -u t4dm -f

# Debug logging
T4DM_LOG_LEVEL=DEBUG t4dm serve
```

---

## 7. Security Hardening

### API Key Authentication

```bash
# Set API key
export T4DM_API_KEY=$(openssl rand -hex 32)

# Clients must provide header
curl -H "X-API-Key: $T4DM_API_KEY" http://localhost:8765/api/v1/health
```

### Firewall Rules

```bash
# UFW (Ubuntu)
sudo ufw allow from 10.0.0.0/8 to any port 8765
sudo ufw deny 8765

# iptables
iptables -A INPUT -p tcp --dport 8765 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8765 -j DROP
```

### TLS Configuration

Use nginx or Caddy for TLS termination (see reverse proxy section).

### File Permissions

```bash
# Secure storage directory
sudo chown -R t4dm:t4dm /var/lib/t4dm
sudo chmod 700 /var/lib/t4dm/data
```

---

## 8. Backup & Recovery

### Backup Procedure

```bash
# Option 1: Live backup (recommended)
curl -X POST http://localhost:8765/api/v1/control/checkpoint
tar -czf t4dm-backup-$(date +%Y%m%d).tar.gz /var/lib/t4dm/data

# Option 2: Stop and backup
sudo systemctl stop t4dm
tar -czf t4dm-backup-$(date +%Y%m%d).tar.gz /var/lib/t4dm/data
sudo systemctl start t4dm
```

### Restore Procedure

```bash
# Stop service
sudo systemctl stop t4dm

# Restore backup
rm -rf /var/lib/t4dm/data
tar -xzf t4dm-backup-20260205.tar.gz -C /

# Start service (WAL replay automatic)
sudo systemctl start t4dm

# Verify
curl http://localhost:8765/api/v1/health
```

### Automated Backups

```bash
# /etc/cron.daily/t4dm-backup
#!/bin/bash
BACKUP_DIR=/var/backups/t4dm
mkdir -p $BACKUP_DIR
curl -X POST http://localhost:8765/api/v1/control/checkpoint
tar -czf $BACKUP_DIR/t4dm-$(date +%Y%m%d).tar.gz /var/lib/t4dm/data
find $BACKUP_DIR -name "t4dm-*.tar.gz" -mtime +7 -delete
```

---

## 9. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow startup | Large WAL replay | Normal on first start after crash |
| OOM errors | Insufficient RAM | Increase RAM or reduce batch sizes |
| 429 errors | Rate limiting | Increase T4DM_RATE_LIMIT |
| Connection refused | Service not running | `systemctl status t4dm` |
| Permission denied | Wrong file ownership | `chown -R t4dm:t4dm /var/lib/t4dm` |
| CUDA not found | Missing drivers | Install NVIDIA drivers + CUDA |

### Debug Mode

```bash
# Run with debug logging
T4DM_LOG_LEVEL=DEBUG t4dm serve

# Check storage integrity
t4dm storage verify /var/lib/t4dm/data

# Force checkpoint
curl -X POST http://localhost:8765/api/v1/control/checkpoint
```

### Performance Tuning

```bash
# Increase file descriptors
ulimit -n 65535

# Tune kernel parameters
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "vm.swappiness = 10" >> /etc/sysctl.conf
sysctl -p
```

### Getting Help

- GitHub Issues: https://github.com/astoreyai/t4dm/issues
- Documentation: https://github.com/astoreyai/t4dm/tree/main/docs

---

## Appendix: Docker Compose

```yaml
version: '3.8'

services:
  t4dm:
    build: .
    ports:
      - "8765:8765"
    volumes:
      - t4dm_data:/data
    environment:
      - T4DM_STORAGE_PATH=/data
      - T4DM_SESSION_ID=production
      - T4DM_LOG_LEVEL=INFO
      - T4DM_WORKERS=4
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  t4dm_data:
```
