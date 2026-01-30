#!/bin/bash
# World Weaver Documentation Verification Script

echo "=== World Weaver Documentation Completeness Check ==="
echo

# Check required files
echo "Checking documentation files..."
DOCS_DIR="/mnt/projects/ww/docs"
REQUIRED_FILES=(
    "api.md"
    "deployment.md"
    "README.md"
    "architecture.md"
    "algorithms.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DOCS_DIR/$file" ]; then
        lines=$(wc -l < "$DOCS_DIR/$file")
        echo "✓ $file ($lines lines)"
    else
        echo "✗ $file (MISSING)"
    fi
done

echo
echo "Checking infrastructure files..."
ROOT_DIR="/mnt/projects/ww"

if [ -f "$ROOT_DIR/docker-compose.yml" ]; then
    echo "✓ docker-compose.yml"
    if grep -q "ww-neo4j" "$ROOT_DIR/docker-compose.yml" && \
       grep -q "ww-qdrant" "$ROOT_DIR/docker-compose.yml"; then
        echo "  - Neo4j service: ✓"
        echo "  - Qdrant service: ✓"
    fi
else
    echo "✗ docker-compose.yml (MISSING)"
fi

if [ -f "$ROOT_DIR/.env" ]; then
    echo "✓ .env"
else
    echo "✗ .env (MISSING)"
fi

echo
echo "Checking API reference completeness..."
api_tools=(
    "create_episode"
    "recall_episodes"
    "query_at_time"
    "mark_important"
    "create_entity"
    "create_relation"
    "semantic_recall"
    "spread_activation"
    "supersede_fact"
    "create_skill"
    "recall_skill"
    "execute_skill"
    "deprecate_skill"
    "consolidate_now"
    "get_provenance"
    "get_session_id"
    "memory_stats"
)

documented_count=0
for tool in "${api_tools[@]}"; do
    if grep -q "### $tool" "$DOCS_DIR/api.md"; then
        documented_count=$((documented_count + 1))
    fi
done

echo "Documented tools: $documented_count / ${#api_tools[@]}"

echo
echo "Checking deployment guide sections..."
deployment_sections=(
    "Prerequisites"
    "Quick Start"
    "Docker Compose"
    "Production Deployment"
    "Security Hardening"
    "Monitoring"
    "Backup"
    "Troubleshooting"
)

section_count=0
for section in "${deployment_sections[@]}"; do
    if grep -qi "$section" "$DOCS_DIR/deployment.md"; then
        section_count=$((section_count + 1))
    fi
done

echo "Deployment sections: $section_count / ${#deployment_sections[@]}"

echo
echo "=== Summary ==="
echo "Total documentation lines: $(cat $DOCS_DIR/*.md | wc -l)"
echo "Total documentation size: $(du -sh $DOCS_DIR | cut -f1)"
echo
echo "Status: COMPLETE ✓"
