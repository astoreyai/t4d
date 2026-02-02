# World Weaver Operational Scripts

Production-ready scripts for managing World Weaver infrastructure.

## Security Scripts

### setup-env.sh
Generates secure environment configuration with strong passwords.

**Usage**:
```bash
./setup-env.sh
```

**Features**:
- Creates .env from .env.example template
- Generates cryptographically secure passwords (20 chars)
- Optional Qdrant API key generation for production
- Sets secure file permissions (600)
- Backs up existing .env before overwriting

**Example**:
```bash
# Initial setup
./setup-env.sh

# Review generated configuration
cat .env

# Validate before starting services
./validate-env.sh
```

### validate-env.sh
Validates environment configuration for security and completeness.

**Usage**:
```bash
./validate-env.sh
# Exit 0 = valid, 1 = errors found
```

**Checks**:
- Required variables (NEO4J_PASSWORD, WW_NEO4J_PASSWORD, etc.)
- Password strength (min 8 chars, no common passwords)
- Password synchronization (NEO4J_PASSWORD == WW_NEO4J_PASSWORD)
- File permissions (.env should be 600)
- Production security (Qdrant API key, OTEL settings)
- Docker configuration (no hardcoded passwords)

**Example**:
```bash
# Validate configuration
./validate-env.sh

# Use in CI/CD pipeline
./validate-env.sh && docker-compose up -d

# Pre-deployment check
./validate-env.sh || exit 1
```

## Operational Scripts

### backup.sh
Creates authenticated backups of Neo4j and Qdrant data.

**Usage**:
```bash
./backup.sh [backup_dir]
# Default: /var/backups/ww
```

**Features**:
- Neo4j database dump via `neo4j-admin` with authentication
- Qdrant collection snapshots with optional API key authentication
- Automatic compression of backups older than 1 day
- Automatic cleanup of backups older than 30 days
- Timestamped logs and backup files
- Auto-discovers collections from Qdrant API
- Validates credentials before backup

**Example**:
```bash
# Backup to default location
./backup.sh

# Backup to custom directory
./backup.sh /opt/backups/ww

# Set up daily backups (cron)
0 2 * * * /mnt/projects/t4d/t4dm/scripts/backup.sh >> /var/log/t4dm/backup.log 2>&1
```

### restore.sh
Restores from a specific backup.

**Usage**:
```bash
./restore.sh <backup_date> [backup_dir]
# Example: ./restore.sh 20251127_143022
```

**Features**:
- Automatic service shutdown before restore
- Handles compressed (.gz) backups
- Restores both Neo4j and Qdrant data
- Automatic service restart after restore

**Example**:
```bash
# List available backups
ls -lh /var/backups/t4dm/

# Restore from specific date
./restore.sh 20251127_143022

# Restore from custom backup directory
./restore.sh 20251127_143022 /opt/backups/ww
```

### health_check.sh
Checks health of all services with authentication.

**Usage**:
```bash
./health_check.sh
# Exit 0 = healthy, 1 = unhealthy
```

**Checks**:
- Neo4j HTTP and Bolt connections with authentication
- Neo4j database stats (node count) via Cypher
- Qdrant health and readiness endpoints with optional API key
- Qdrant collections (ww_episodes, ww_entities, ww_procedures) with point counts
- Docker container status (ww-neo4j, ww-qdrant)
- Disk space usage (warns at 80%, critical at 90%)
- Memory usage (warns at 80%, critical at 90%)

**Example**:
```bash
# Manual health check
./health_check.sh

# Set up monitoring (cron)
*/5 * * * * /mnt/projects/t4d/t4dm/scripts/health_check.sh >> /var/log/t4dm/health.log 2>&1

# Use in Docker healthchecks
HEALTHCHECK CMD /scripts/health_check.sh || exit 1
```

### migrate.sh
Applies database migrations.

**Usage**:
```bash
./migrate.sh [version]
```

**Features**:
- Tracks schema version in Neo4j
- Applies migrations incrementally
- Idempotent (safe to run multiple times)

**Example**:
```bash
# Apply all pending migrations
./migrate.sh

# Run before deployment
./migrate.sh && docker-compose restart
```

## Environment Variables

All scripts support the following environment variables from `.env`:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NEO4J_USER` | Neo4j username (Docker) | Yes | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password (Docker) | Yes | - |
| `WW_NEO4J_PASSWORD` | Neo4j password (App) | Yes | - |
| `QDRANT_API_KEY` | Qdrant API key | Production | - |
| `QDRANT_READ_ONLY_KEY` | Qdrant read-only API key | No | - |
| `QDRANT_HOST` | Qdrant hostname | No | `localhost` |
| `QDRANT_PORT` | Qdrant port | No | `6333` |
| `NEO4J_HOST` | Neo4j hostname | No | `localhost` |
| `NEO4J_HTTP_PORT` | Neo4j HTTP port | No | `7474` |
| `NEO4J_BOLT_PORT` | Neo4j Bolt port | No | `7687` |
| `BACKUP_RETENTION_DAYS` | Days to keep backups | No | `30` |

**Security Requirements**:
- `NEO4J_PASSWORD` must match `WW_NEO4J_PASSWORD`
- Passwords must be min 8 characters with complexity (2 of: upper, lower, digit, special)
- No common passwords (password, admin, neo4j, etc.)
- `.env` file must have 600 permissions
- **Production**: Enable `QDRANT_API_KEY` for authentication
- **Production**: Use `QDRANT_READ_ONLY_KEY` for monitoring/read-only access
- **Production**: Consider enabling TLS for both Neo4j and Qdrant

**Setup**:
```bash
# Generate secure configuration
./setup-env.sh

# Validate configuration
./validate-env.sh

# Override for remote services
export QDRANT_HOST=qdrant.example.com
export NEO4J_HOST=neo4j.example.com
export NEO4J_PASSWORD=$(cat .env | grep NEO4J_PASSWORD= | cut -d= -f2)

./backup.sh
```

## Production Setup

### 1. Configure Environment (REQUIRED)

```bash
# Generate secure configuration
cd /mnt/projects/ww
./scripts/setup-env.sh

# Validate configuration
./scripts/validate-env.sh
```

### 2. Make Scripts Executable

```bash
chmod +x scripts/*.sh
```

### 3. Set Up Automated Backups

```bash
# Add to crontab (crontab -e)
0 2 * * * /mnt/projects/t4d/t4dm/scripts/backup.sh /var/backups/ww >> /var/log/t4dm/backup.log 2>&1
```

### 4. Set Up Health Monitoring

```bash
# Add to crontab (crontab -e)
*/5 * * * * /mnt/projects/t4d/t4dm/scripts/health_check.sh >> /var/log/t4dm/health.log 2>&1
```

### 5. Test Restore Procedure

```bash
# Create test backup
./backup.sh /tmp/test-backup

# Stop services
docker-compose down

# Restore from backup
./restore.sh $(ls -1 /tmp/test-backup/ | grep neo4j | cut -d_ -f2-3 | cut -d. -f1 | head -1) /tmp/test-backup

# Verify
./health_check.sh
```

## Dependencies

**Required**:
- Docker & Docker Compose
- `curl` (for health checks and Qdrant API)
- `jq` (for parsing Qdrant JSON responses)
- `nc` (netcat, for port checks)

**Optional**:
- `gzip` (for backup compression)
- `find` (for cleanup)

**Install on Debian/Ubuntu**:
```bash
sudo apt-get install curl jq netcat-openbsd gzip findutils
```

## Troubleshooting

### Backup Fails: "jq: command not found"
```bash
sudo apt-get install jq
```

### Restore Fails: "Container not running"
```bash
# Start containers first
docker-compose up -d
# Then run restore
./restore.sh <backup_date>
```

### Health Check Fails: "nc: command not found"
```bash
sudo apt-get install netcat-openbsd
```

### Migration Fails: "NEO4J_PASSWORD not set"
```bash
export NEO4J_PASSWORD=your-secure-password
./migrate.sh
```

## Best Practices

1. **Use setup-env.sh** - Always generate passwords via script, never hardcode
2. **Validate before deploy** - Run `validate-env.sh` in CI/CD pipelines
3. **Test backups regularly** - Run restore to a test environment monthly
4. **Monitor disk space** - Ensure backup directory has adequate space
5. **Rotate logs** - Set up logrotate for backup and health check logs
6. **Secure .env** - Ensure 600 permissions, never commit to git
7. **Version migrations** - Name migration files: `001_initial.cypher`, `002_add_index.cypher`, etc.
8. **Production hardening**:
   - Enable Qdrant API key authentication (`QDRANT_API_KEY`)
   - Use separate read-only key for monitoring (`QDRANT_READ_ONLY_KEY`)
   - Bind database ports to localhost only (already configured in docker-compose.yml)
   - Enable Neo4j authentication (already configured)
   - Consider enabling TLS for production deployments
   - Disable CSV import from file URLs in Neo4j (already configured)
9. **Authentication**: All scripts auto-detect and use credentials from .env
10. **Credentials Management**: Never log passwords, use masked values in logs

## Integration with Monitoring

### Prometheus/Grafana

```bash
# Export health check metrics
./health_check.sh && echo "ww_health_status 1" || echo "ww_health_status 0" | curl --data-binary @- http://pushgateway:9091/metrics/job/ww
```

### Nagios/Icinga

```bash
# Use health_check.sh as Nagios plugin
command[check_ww]=/mnt/projects/t4d/t4dm/scripts/health_check.sh
```

### Slack/Email Alerts

```bash
#!/bin/bash
# In cron job
./health_check.sh || echo "World Weaver health check failed!" | mail -s "WW Alert" admin@example.com
```

## See Also

- [Deployment Guide](../docs/deployment.md)
- [Architecture](../ARCHITECTURE.md)
- [Docker Compose Configuration](../docker-compose.yml)
