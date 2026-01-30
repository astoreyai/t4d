#!/bin/bash
# Validate environment configuration before starting World Weaver
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

echo -e "${BLUE}World Weaver Environment Validation${NC}"
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

echo "Docker Service Configuration:"
echo "------------------------------"

# Neo4j password checks
check_required "NEO4J_PASSWORD" && check_password "NEO4J_PASSWORD"

echo
echo "Application Configuration:"
echo "--------------------------"

# Application password checks
check_required "WW_NEO4J_PASSWORD" && check_password "WW_NEO4J_PASSWORD"

# Check passwords match
if [[ -n "$NEO4J_PASSWORD" && -n "$WW_NEO4J_PASSWORD" ]]; then
    if [[ "$NEO4J_PASSWORD" != "$WW_NEO4J_PASSWORD" ]]; then
        echo -e "${RED}✗ ERROR: NEO4J_PASSWORD and WW_NEO4J_PASSWORD must match${NC}"
        ((errors++))
    else
        echo -e "${GREEN}✓ Neo4j passwords match${NC}"
    fi
fi

# Check required app variables
echo
if check_required "WW_SESSION_ID"; then
    echo -e "${GREEN}✓ WW_SESSION_ID: $WW_SESSION_ID${NC}"
fi

if check_required "WW_NEO4J_URI"; then
    echo -e "${GREEN}✓ WW_NEO4J_URI: $WW_NEO4J_URI${NC}"
fi

if check_required "WW_QDRANT_URL"; then
    echo -e "${GREEN}✓ WW_QDRANT_URL: $WW_QDRANT_URL${NC}"
fi

echo
echo "Security Settings:"
echo "------------------"

# Check .env permissions
check_permissions "$PROJECT_DIR/.env" "600"

# Production environment checks
if [[ "$WW_ENVIRONMENT" == "production" ]]; then
    echo
    echo -e "${YELLOW}Production Environment Detected${NC}"
    echo "-------------------------------"

    if [[ -z "$QDRANT_API_KEY" ]]; then
        echo -e "${YELLOW}⚠ WARNING: QDRANT_API_KEY not set in production${NC}"
        echo "  Recommended: Set QDRANT_API_KEY for production deployments"
        ((warnings++))
    else
        echo -e "${GREEN}✓ Qdrant API key configured${NC}"
    fi

    if [[ "$WW_OTEL_INSECURE" == "true" ]]; then
        echo -e "${YELLOW}⚠ WARNING: OTEL insecure mode enabled in production${NC}"
        echo "  Recommended: Set WW_OTEL_INSECURE=false"
        ((warnings++))
    fi

    # Check if ports are bound to localhost in docker-compose
    if grep -q '"7474:7474"' "$PROJECT_DIR/docker-compose.yml" 2>/dev/null; then
        echo -e "${YELLOW}⚠ WARNING: Docker ports not bound to localhost${NC}"
        echo "  Security risk: Services exposed to network"
        echo "  Recommended: Use '127.0.0.1:7474:7474' in docker-compose.yml"
        ((warnings++))
    fi
fi

# Check docker-compose.yml exists
echo
if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
    echo -e "${GREEN}✓ docker-compose.yml found${NC}"

    # Verify no hardcoded passwords
    if grep -q "NEO4J_AUTH=neo4j/[^$]" "$PROJECT_DIR/docker-compose.yml" 2>/dev/null; then
        echo -e "${RED}✗ ERROR: Hardcoded password found in docker-compose.yml${NC}"
        echo "  Security risk: Password should use environment variables"
        ((errors++))
    else
        echo -e "${GREEN}✓ No hardcoded passwords in docker-compose.yml${NC}"
    fi
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
