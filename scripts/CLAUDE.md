# Scripts
**Path**: `/mnt/projects/t4d/t4dm/scripts/`

## What
Production operational scripts for managing T4DM infrastructure: environment setup, backups, restores, health checks, and database migrations.

## How
- Bash scripts that read credentials from `.env`
- Auto-detect Neo4j and Qdrant authentication
- Designed for cron scheduling and CI/CD integration

## Why
Provides secure, repeatable operations for managing the dual-store backend (Neo4j + Qdrant) in production, including automated backup rotation and health monitoring.

## Key Files
| File | Purpose |
|------|---------|
| `setup-env.sh` | Generate secure `.env` with cryptographic passwords |
| `validate-env.sh` | Validate env config for security and completeness |
| `backup.sh` | Neo4j dump + Qdrant snapshots with auto-compression |
| `restore.sh` | Restore from timestamped backup |
| `health_check.sh` | Check Neo4j, Qdrant, Docker, disk, memory health |
| `migrate.sh` | Apply incremental Cypher migrations to Neo4j |

## Data Flow
```
.env (credentials) → scripts → Neo4j + Qdrant (backends)
                              → /var/backups/ww (backup storage)
                              → stdout/logs (health status)
```

## Integration Points
- **Cron**: Automated backups (daily) and health checks (every 5 min)
- **Docker Compose**: Scripts manage container lifecycle for restores
- **Prometheus**: Health check output exportable to pushgateway
- **CI/CD**: `validate-env.sh` as pre-deployment gate
