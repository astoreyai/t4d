#!/bin/bash
# World Weaver Health Check Script with Authentication
# Usage: ./health_check.sh
# Exit codes: 0 = healthy, 1 = unhealthy

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if available
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Configuration
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

STATUS=0

check() {
    local name="$1"
    local cmd="$2"

    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name"
        return 0
    else
        echo -e "${RED}✗${NC} $name"
        STATUS=1
        return 1
    fi
}

echo -e "${BLUE}World Weaver Health Check${NC}"
echo "========================="
echo ""

# Build authentication headers
NEO4J_AUTH_HEADER=""
if [[ -n "${NEO4J_PASSWORD:-}" ]]; then
    NEO4J_AUTH_HEADER="-H \"Authorization: Basic $(echo -n "$NEO4J_USER:$NEO4J_PASSWORD" | base64)\""
fi

QDRANT_AUTH_ARGS=""
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
    QDRANT_AUTH_ARGS="-H \"api-key: $QDRANT_API_KEY\""
fi

# Check Neo4j
echo -e "${BLUE}Neo4j Graph Database${NC}"
if [[ -n "$NEO4J_AUTH_HEADER" ]]; then
    check "Neo4j HTTP" "curl -sf $NEO4J_AUTH_HEADER http://$NEO4J_HOST:$NEO4J_HTTP_PORT"
else
    echo -e "${YELLOW}⚠${NC} Neo4j HTTP (no credentials - check may fail)"
    check "Neo4j HTTP" "curl -sf http://$NEO4J_HOST:$NEO4J_HTTP_PORT"
fi

check "Neo4j Bolt" "nc -z $NEO4J_HOST $NEO4J_BOLT_PORT"

# Get Neo4j version and database stats
if [[ -n "$NEO4J_AUTH_HEADER" ]]; then
    NEO4J_INFO=$(eval curl -sf $NEO4J_AUTH_HEADER http://$NEO4J_HOST:$NEO4J_HTTP_PORT/db/neo4j/tx/commit \
        -H \"Content-Type: application/json\" \
        -d '{\"statements\":[{\"statement\":\"MATCH (n) RETURN count(n) as count\"}]}' 2>/dev/null \
        | jq -r '.results[0].data[0].row[0] // "N/A"' 2>/dev/null)

    if [[ "$NEO4J_INFO" != "N/A" ]]; then
        echo -e "  ${GREEN}→${NC} Total nodes: $NEO4J_INFO"
    fi
fi

echo ""

# Check Qdrant
echo -e "${BLUE}Qdrant Vector Database${NC}"
check "Qdrant Health" "eval curl -sf $QDRANT_AUTH_ARGS http://$QDRANT_HOST:$QDRANT_PORT/health"
check "Qdrant Ready" "eval curl -sf $QDRANT_AUTH_ARGS http://$QDRANT_HOST:$QDRANT_PORT/readyz"

# Check Qdrant collections with point counts
echo ""
echo -e "${BLUE}Qdrant Collections${NC}"
for collection in ww_episodes ww_entities ww_procedures; do
    if check "Collection: $collection" "eval curl -sf $QDRANT_AUTH_ARGS http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection"; then
        # Get point count
        POINT_COUNT=$(eval curl -sf $QDRANT_AUTH_ARGS \
            "http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection" 2>/dev/null \
            | jq -r '.result.points_count // "N/A"' 2>/dev/null)

        if [[ "$POINT_COUNT" != "N/A" ]]; then
            echo -e "  ${GREEN}→${NC} Points: $POINT_COUNT"
        fi
    fi
done

echo ""

# Check system resources
echo -e "${BLUE}System Resources${NC}"

# Check disk space (warn if >80% used)
DISK_FREE=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$DISK_FREE" -gt 90 ]; then
    echo -e "${RED}✗${NC} Disk space: ${DISK_FREE}% used (critical)"
    STATUS=1
elif [ "$DISK_FREE" -gt 80 ]; then
    echo -e "${YELLOW}⚠${NC} Disk space: ${DISK_FREE}% used (warning)"
else
    echo -e "${GREEN}✓${NC} Disk space: ${DISK_FREE}% used"
fi

# Check memory
MEM_FREE=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
if [ "$MEM_FREE" -gt 90 ]; then
    echo -e "${RED}✗${NC} Memory: ${MEM_FREE}% used (critical)"
    STATUS=1
elif [ "$MEM_FREE" -gt 80 ]; then
    echo -e "${YELLOW}⚠${NC} Memory: ${MEM_FREE}% used (warning)"
else
    echo -e "${GREEN}✓${NC} Memory: ${MEM_FREE}% used"
fi

# Check Docker containers
echo ""
echo -e "${BLUE}Docker Containers${NC}"
check "ww-neo4j" "docker ps --filter 'name=ww-neo4j' --filter 'status=running' | grep -q ww-neo4j"
check "ww-qdrant" "docker ps --filter 'name=ww-qdrant' --filter 'status=running' | grep -q ww-qdrant"

# Summary
echo ""
echo "========================="
if [ $STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed${NC}"
else
    echo -e "${RED}✗ Some checks failed${NC}"
fi

exit $STATUS
