#!/bin/bash
# T4DM Health Check Script
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
T4DM_API_HOST="${T4DM_API_HOST:-localhost}"
T4DM_API_PORT="${T4DM_API_PORT:-8765}"

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

echo -e "${BLUE}T4DM Health Check${NC}"
echo "========================="
echo ""

# Check T4DM API
echo -e "${BLUE}T4DM API Server${NC}"
check "API Health Endpoint" "curl -sf http://$T4DM_API_HOST:$T4DM_API_PORT/api/v1/health"

# Get API info
API_INFO=$(curl -sf "http://$T4DM_API_HOST:$T4DM_API_PORT/api/v1/health" 2>/dev/null | jq -r '.status // "N/A"' 2>/dev/null)
if [[ "$API_INFO" != "N/A" ]]; then
    echo -e "  ${GREEN}→${NC} Status: $API_INFO"
fi

echo ""

# Check T4DX storage
echo -e "${BLUE}T4DX Storage Engine${NC}"
T4DX_DATA_DIR="${T4DX_DATA_DIR:-/app/data/t4dx}"

if docker exec t4dm-api test -d "$T4DX_DATA_DIR" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} T4DX data directory exists"
    # Get data size
    DATA_SIZE=$(docker exec t4dm-api du -sh "$T4DX_DATA_DIR" 2>/dev/null | cut -f1 || echo "N/A")
    echo -e "  ${GREEN}→${NC} Data size: $DATA_SIZE"
else
    echo -e "${YELLOW}⚠${NC} T4DX data directory not found (may not be initialized)"
fi

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
check "t4dm-api" "docker ps --filter 'name=t4dm-api' --filter 'status=running' | grep -q t4dm-api"

# Summary
echo ""
echo "========================="
if [ $STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed${NC}"
else
    echo -e "${RED}✗ Some checks failed${NC}"
fi

exit $STATUS
