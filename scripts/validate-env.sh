#!/bin/bash
# Validate environment configuration before starting T4DM
# Checks required variables, password strength, and security settings
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

errors=0
warnings=0

echo -e "${BLUE}T4DM Environment Validation${NC}"
echo "===================================="
echo

# Load .env if exists
if [[ -f "$PROJECT_DIR/.env" ]]; then
    # Use set -a to export all variables, then source
    set -a
    source "$PROJECT_DIR/.env"
    set +a
    echo -e "${GREEN}✓ .env file found${NC}"
else
    echo -e "${RED}ERROR: .env file not found${NC}"
    echo "Run: ./scripts/setup-env.sh to create it"
    exit 1
fi

echo

# Check required variables
check_required() {
    local var_name=$1
    local var_value=${!var_name}

    if [[ -z "$var_value" ]]; then
        echo -e "${RED}✗ ERROR: $var_name is required but not set${NC}"
        ((errors++))
        return 1
    fi
    return 0
}

# Check password strength
check_password() {
    local var_name=$1
    local var_value=${!var_name}
    local weak_passwords=("password" "Password" "PASSWORD" "neo4j" "admin" "root" "test" "123456" "qwerty" "wwpassword")

    if [[ -z "$var_value" ]]; then
        echo -e "${RED}✗ ERROR: $var_name is not set${NC}"
        ((errors++))
        return 1
    fi

    if [[ ${#var_value} -lt 8 ]]; then
        echo -e "${RED}✗ ERROR: $var_name must be at least 8 characters (current: ${#var_value})${NC}"
        ((errors++))
        return 1
    fi

    for weak in "${weak_passwords[@]}"; do
        if [[ "${var_value,,}" == "${weak,,}" ]]; then
            echo -e "${RED}✗ ERROR: $var_name is too weak (common/default password)${NC}"
            ((errors++))
            return 1
        fi
    done

    echo -e "${GREEN}✓ $var_name: Secure (${#var_value} chars)${NC}"
    return 0
}

# Check file permissions
check_permissions() {
    local file=$1
    local expected=$2

    # Works on both Linux (stat -c) and macOS (stat -f)
    local perms=$(stat -c %a "$file" 2>/dev/null || stat -f %Lp "$file" 2>/dev/null)

    if [[ "$perms" != "$expected" ]]; then
        echo -e "${YELLOW}⚠ WARNING: $file has permissive permissions ($perms, expected $expected)${NC}"
        echo "  Fix with: chmod $expected $file"
        ((warnings++))
        return 1
    else
        echo -e "${GREEN}✓ $file: Secure permissions ($perms)${NC}"
    fi
    return 0
}

echo "Application Configuration:"
echo "--------------------------"

# Database password check
if check_required "T4DM_DATABASE_PASSWORD"; then
    check_password "T4DM_DATABASE_PASSWORD"
fi

# Check required app variables
echo
if check_required "T4DM_SESSION_ID"; then
    echo -e "${GREEN}✓ T4DM_SESSION_ID: $T4DM_SESSION_ID${NC}"
fi

# Check T4DX data directory setting
if [[ -n "${T4DX_DATA_DIR:-}" ]]; then
    echo -e "${GREEN}✓ T4DX_DATA_DIR: $T4DX_DATA_DIR${NC}"
else
    echo -e "${YELLOW}⚠ WARNING: T4DX_DATA_DIR not set, using default${NC}"
    ((warnings++))
fi

echo
echo "Security Settings:"
echo "------------------"

# Check .env permissions
check_permissions "$PROJECT_DIR/.env" "600"

# Production environment checks
if [[ "$T4DM_ENVIRONMENT" == "production" ]]; then
    echo
    echo -e "${YELLOW}Production Environment Detected${NC}"
    echo "-------------------------------"

    if [[ "$T4DM_OTEL_INSECURE" == "true" ]]; then
        echo -e "${YELLOW}⚠ WARNING: OTEL insecure mode enabled in production${NC}"
        echo "  Recommended: Set T4DM_OTEL_INSECURE=false"
        ((warnings++))
    fi

    # Check if ports are bound to localhost in docker-compose
    if grep -q '"8765:8765"' "$PROJECT_DIR/docker-compose.yml" 2>/dev/null; then
        echo -e "${YELLOW}⚠ WARNING: Docker ports not bound to localhost${NC}"
        echo "  Security risk: Services exposed to network"
        echo "  Recommended: Use '127.0.0.1:8765:8765' in docker-compose.yml"
        ((warnings++))
    fi
fi

# Check docker-compose.yml exists
echo
if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
    echo -e "${GREEN}✓ docker-compose.yml found${NC}"
else
    echo -e "${YELLOW}⚠ WARNING: docker-compose.yml not found${NC}"
    ((warnings++))
fi

echo
echo "===================================="

if [[ $errors -gt 0 ]]; then
    echo -e "${RED}Validation FAILED${NC}"
    echo "  Errors:   $errors"
    echo "  Warnings: $warnings"
    echo
    echo "Fix errors before starting services."
    echo "Run: ./scripts/setup-env.sh to reconfigure"
    exit 1
elif [[ $warnings -gt 0 ]]; then
    echo -e "${YELLOW}Validation PASSED with warnings${NC}"
    echo "  Errors:   $errors"
    echo "  Warnings: $warnings"
    echo
    echo "Services can start, but review warnings for security improvements."
    exit 0
else
    echo -e "${GREEN}Validation PASSED${NC}"
    echo "  All checks passed!"
    echo
    echo "Ready to start services:"
    echo "  docker-compose up -d"
    exit 0
fi
