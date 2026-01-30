#!/bin/bash
# World Weaver Backup Script with Authentication
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

BACKUP_DIR="${1:-${BACKUP_DIR:-/var/backups/ww}}"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BACKUP_DIR/backup_$DATE.log"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Database connection settings
NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Validate required credentials
if [[ -z "$NEO4J_PASSWORD" ]]; then
    echo "ERROR: NEO4J_PASSWORD not set. Load .env or set environment variable."
    exit 1
fi

mkdir -p "$BACKUP_DIR"
log "Starting World Weaver backup to $BACKUP_DIR"

# Neo4j backup
log "Backing up Neo4j..."

# Build Neo4j auth header (Basic authentication)
NEO4J_AUTH_HEADER="Authorization: Basic $(echo -n "$NEO4J_USER:$NEO4J_PASSWORD" | base64)"

# Check Neo4j is accessible with authentication
if curl -sf -H "$NEO4J_AUTH_HEADER" "http://$NEO4J_HOST:$NEO4J_HTTP_PORT" > /dev/null 2>&1; then
    # Create backups directory in container
    docker exec ww-neo4j mkdir -p /backups 2>/dev/null || true

    # Use neo4j-admin for offline dump if available
    if docker exec ww-neo4j neo4j-admin database dump neo4j --to-path=/backups --overwrite-destination=true 2>&1 | tee -a "$LOG_FILE"; then
        docker cp ww-neo4j:/backups/neo4j.dump "$BACKUP_DIR/neo4j_$DATE.dump" 2>/dev/null && \
            log "Neo4j backup complete: neo4j_$DATE.dump" || \
            log "WARNING: Failed to copy Neo4j dump"
    else
        log "WARNING: neo4j-admin dump failed, trying Cypher export"
        # Fallback to Cypher export (requires APOC)
        # Note: This would require more complex implementation
    fi
else
    log "WARNING: Neo4j not accessible with credentials, skipping"
fi

# Qdrant backup (create snapshots)
log "Backing up Qdrant..."

# Build Qdrant auth header if API key is set
QDRANT_AUTH_ARGS=""
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
    QDRANT_AUTH_ARGS="-H \"api-key: $QDRANT_API_KEY\""
    log "Using Qdrant API key authentication"
fi

# Get collection names from Qdrant
COLLECTIONS=$(eval curl -sf $QDRANT_AUTH_ARGS "http://$QDRANT_HOST:$QDRANT_PORT/collections" 2>/dev/null | jq -r '.result.collections[].name // empty' 2>/dev/null || echo "")

if [[ -z "$COLLECTIONS" ]]; then
    log "WARNING: No Qdrant collections found or authentication failed"
    # Try default collections
    COLLECTIONS="ww_episodes ww_entities ww_procedures"
fi

for collection in $COLLECTIONS; do
    log "Creating snapshot for collection: $collection"

    # Create snapshot
    SNAPSHOT=$(eval curl -sf $QDRANT_AUTH_ARGS -X POST \
        "http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection/snapshots" 2>/dev/null \
        | jq -r '.result.name // empty' 2>/dev/null)

    if [[ -n "$SNAPSHOT" ]]; then
        # Download snapshot
        eval curl -sf $QDRANT_AUTH_ARGS \
            "http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection/snapshots/$SNAPSHOT" \
            -o "$BACKUP_DIR/qdrant_${collection}_$DATE.snapshot" 2>/dev/null && \
            log "  Saved: qdrant_${collection}_$DATE.snapshot" || \
            log "  WARNING: Failed to download snapshot for $collection"
    else
        log "  WARNING: Failed to create snapshot for $collection"
    fi
done

# Compress backups older than 1 day
log "Compressing old backups..."
find "$BACKUP_DIR" -name "*.dump" -mtime +1 -exec gzip {} \; 2>/dev/null || true
find "$BACKUP_DIR" -name "*.snapshot" -mtime +1 -exec gzip {} \; 2>/dev/null || true

# Remove backups older than retention period
log "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "*.dump" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "*.snapshot" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

# Summary
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "unknown")
log "Backup complete!"
log "Location: $BACKUP_DIR"
log "Total size: $BACKUP_SIZE"

# List created backups
log "Backups created:"
ls -lh "$BACKUP_DIR"/*_$DATE.* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' | tee -a "$LOG_FILE" || log "  (no new backups created)"
