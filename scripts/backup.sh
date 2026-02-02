#!/bin/bash
# T4DM Backup Script
# Usage: ./backup.sh [backup_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if available
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

BACKUP_DIR="${1:-${BACKUP_DIR:-/var/backups/t4dm}}"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BACKUP_DIR/backup_$DATE.log"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# T4DX data directory
T4DX_DATA_DIR="${T4DX_DATA_DIR:-/app/data/t4dx}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

mkdir -p "$BACKUP_DIR"
log "Starting T4DM backup to $BACKUP_DIR"

# T4DX data backup
log "Backing up T4DX data..."

# Check if T4DX data directory exists (in container or local)
if docker exec t4dm-api test -d "$T4DX_DATA_DIR" 2>/dev/null; then
    # Backup from container
    docker exec t4dm-api tar czf /tmp/t4dx_backup_$DATE.tar.gz -C "$T4DX_DATA_DIR" . 2>&1 | tee -a "$LOG_FILE"
    docker cp t4dm-api:/tmp/t4dx_backup_$DATE.tar.gz "$BACKUP_DIR/t4dx_$DATE.tar.gz" 2>/dev/null && \
        log "T4DX backup complete: t4dx_$DATE.tar.gz" || \
        log "WARNING: Failed to copy T4DX backup"
    docker exec t4dm-api rm -f /tmp/t4dx_backup_$DATE.tar.gz 2>/dev/null || true
elif [[ -d "$PROJECT_DIR/data/t4dx" ]]; then
    # Backup from local directory
    tar czf "$BACKUP_DIR/t4dx_$DATE.tar.gz" -C "$PROJECT_DIR/data/t4dx" . 2>&1 | tee -a "$LOG_FILE" && \
        log "T4DX backup complete: t4dx_$DATE.tar.gz" || \
        log "WARNING: Failed to create T4DX backup"
else
    log "WARNING: T4DX data directory not found, skipping"
fi

# Compress backups older than 1 day
log "Compressing old backups..."
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +1 -exec gzip {} \; 2>/dev/null || true

# Remove backups older than retention period
log "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

# Summary
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "unknown")
log "Backup complete!"
log "Location: $BACKUP_DIR"
log "Total size: $BACKUP_SIZE"

# List created backups
log "Backups created:"
ls -lh "$BACKUP_DIR"/*_$DATE.* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' | tee -a "$LOG_FILE" || log "  (no new backups created)"
