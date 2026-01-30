#!/bin/bash
# Setup script for World Weaver environment
# Generates secure passwords and creates .env from template
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"

echo "World Weaver Environment Setup"
echo "=============================="
echo

# Check if .env already exists
if [[ -f "$ENV_FILE" ]]; then
    echo "WARNING: .env already exists"
    read -p "Overwrite existing .env? This will REPLACE your current configuration (y/N): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file."
        echo "To manually update, edit: $ENV_FILE"
        exit 0
    fi
    # Backup existing .env
    backup_file="$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$ENV_FILE" "$backup_file"
    echo "Backed up existing .env to: $backup_file"
    echo
fi

# Copy example
if [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "Created .env from template"
else
    echo "Error: .env.example not found at $ENV_EXAMPLE"
    exit 1
fi

# Generate secure password
generate_password() {
    # Generate 20-char alphanumeric password
    openssl rand -base64 24 | tr -d '/+=' | head -c 20
}

echo
echo "Generating secure Neo4j password..."
NEO4J_PASS=$(generate_password)

# Update .env with generated password (both locations)
sed -i "s/^NEO4J_PASSWORD=.*/NEO4J_PASSWORD=$NEO4J_PASS/" "$ENV_FILE"
sed -i "s/^WW_NEO4J_PASSWORD=.*/WW_NEO4J_PASSWORD=$NEO4J_PASS/" "$ENV_FILE"

# Optional: Generate Qdrant API key for production
read -p "Generate Qdrant API key for production? (y/N): " gen_qdrant
if [[ "$gen_qdrant" =~ ^[Yy]$ ]]; then
    QDRANT_KEY=$(generate_password)
    sed -i "s/^# QDRANT_API_KEY=.*/QDRANT_API_KEY=$QDRANT_KEY/" "$ENV_FILE"
    sed -i "s/^# WW_QDRANT_API_KEY=.*/WW_QDRANT_API_KEY=$QDRANT_KEY/" "$ENV_FILE"
    echo "Qdrant API key: $QDRANT_KEY"
fi

# Set secure permissions (owner read/write only)
chmod 600 "$ENV_FILE"

echo
echo "=================================="
echo "Environment configured successfully!"
echo "=================================="
echo
echo "Neo4j Credentials:"
echo "  User:     neo4j"
echo "  Password: $NEO4J_PASS"
echo
echo "IMPORTANT: Save these credentials securely!"
echo "File permissions set to 600 (owner-only read/write)"
echo
echo "Next steps:"
echo "  1. Review configuration: nano $ENV_FILE"
echo "  2. Validate environment:  ./scripts/validate-env.sh"
echo "  3. Start services:        docker-compose up -d"
echo
echo "To verify services are running:"
echo "  docker-compose ps"
echo "  docker-compose logs -f"
echo
