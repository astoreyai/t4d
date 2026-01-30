#!/bin/bash
# World Weaver Migration Script
# Usage: ./migrate.sh [version]

set -euo pipefail

VERSION="${1:-latest}"
MIGRATIONS_DIR="$(dirname "$0")/../migrations"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

if [ ! -d "$MIGRATIONS_DIR" ]; then
    log "No migrations directory found at $MIGRATIONS_DIR"
    exit 0
fi

# Get current version from Neo4j
CURRENT=$(docker exec ww-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
    "MATCH (v:SchemaVersion) RETURN v.version ORDER BY v.applied DESC LIMIT 1" 2>/dev/null | tail -1 || echo "0")

log "Current schema version: $CURRENT"

# Apply migrations
for migration in "$MIGRATIONS_DIR"/*.cypher; do
    [ -f "$migration" ] || continue

    MIGRATION_VERSION=$(basename "$migration" .cypher | cut -d_ -f1)

    if [ "$MIGRATION_VERSION" -gt "$CURRENT" ]; then
        log "Applying migration: $migration"
        docker exec -i ww-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" < "$migration"

        # Record migration
        docker exec ww-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
            "CREATE (v:SchemaVersion {version: '$MIGRATION_VERSION', applied: datetime()})"

        log "Migration $MIGRATION_VERSION applied"
    fi
done

log "Migrations complete"
