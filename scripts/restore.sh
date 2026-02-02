#!/bin/bash
# T4DM Restore Script
# Usage: ./restore.sh <backup_date> [backup_dir]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_date> [backup_dir]"
    echo "Example: $0 20251127_143022"
    exit 1
fi

BACKUP_DATE="$1"
BACKUP_DIR="${2:-/var/backups/t4dm}"
T4DX_DATA_DIR="${T4DX_DATA_DIR:-/app/data/t4dx}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting restore from $BACKUP_DATE"

# Stop services
log "Stopping T4DM services..."
docker compose -f docker-compose.yml stop || true

# Restore T4DX
T4DX_BACKUP="$BACKUP_DIR/t4dx_$BACKUP_DATE.tar.gz"
if [ -f "$T4DX_BACKUP" ] || [ -f "$T4DX_BACKUP.gz" ]; then
    [ -f "$T4DX_BACKUP.gz" ] && gunzip -k "$T4DX_BACKUP.gz"
    log "Restoring T4DX from $T4DX_BACKUP..."

    # Restore to container if running
    if docker exec t4dm-api test -d "$T4DX_DATA_DIR" 2>/dev/null; then
        docker cp "$T4DX_BACKUP" t4dm-api:/tmp/t4dx_restore.tar.gz
        docker exec t4dm-api sh -c "rm -rf $T4DX_DATA_DIR/* && tar xzf /tmp/t4dx_restore.tar.gz -C $T4DX_DATA_DIR"
        docker exec t4dm-api rm -f /tmp/t4dx_restore.tar.gz
        log "T4DX restored to container"
    else
        # Restore to local directory
        mkdir -p "$PROJECT_DIR/data/t4dx"
        tar xzf "$T4DX_BACKUP" -C "$PROJECT_DIR/data/t4dx"
        log "T4DX restored to local directory"
    fi
else
    log "WARNING: T4DX backup not found"
fi

# Restart services
log "Starting services..."
docker compose -f docker-compose.yml start

log "Restore complete!"
