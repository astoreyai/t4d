#!/bin/bash
# World Weaver Restore Script
# Usage: ./restore.sh <backup_date> [backup_dir]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_date> [backup_dir]"
    echo "Example: $0 20251127_143022"
    exit 1
fi

BACKUP_DATE="$1"
BACKUP_DIR="${2:-/var/backups/ww}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting restore from $BACKUP_DATE"

# Stop services
log "Stopping World Weaver services..."
docker compose -f docker-compose.yml stop || true

# Restore Neo4j
NEO4J_BACKUP="$BACKUP_DIR/neo4j_$BACKUP_DATE.dump"
if [ -f "$NEO4J_BACKUP" ] || [ -f "$NEO4J_BACKUP.gz" ]; then
    [ -f "$NEO4J_BACKUP.gz" ] && gunzip -k "$NEO4J_BACKUP.gz"
    log "Restoring Neo4j from $NEO4J_BACKUP..."
    docker cp "$NEO4J_BACKUP" ww-neo4j:/backups/restore.dump
    docker exec ww-neo4j neo4j-admin database load neo4j --from-path=/backups/restore.dump --overwrite-destination
    log "Neo4j restored"
else
    log "WARNING: Neo4j backup not found"
fi

# Restore Qdrant
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

for collection in episodes entities procedures; do
    SNAPSHOT="$BACKUP_DIR/qdrant_${collection}_$BACKUP_DATE.snapshot"
    if [ -f "$SNAPSHOT" ] || [ -f "$SNAPSHOT.gz" ]; then
        [ -f "$SNAPSHOT.gz" ] && gunzip -k "$SNAPSHOT.gz"
        log "Restoring Qdrant $collection..."
        curl -X POST "http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection/snapshots/upload" \
            -H "Content-Type: multipart/form-data" \
            -F "snapshot=@$SNAPSHOT"
        log "Qdrant $collection restored"
    fi
done

# Restart services
log "Starting services..."
docker compose -f docker-compose.yml start

log "Restore complete!"
