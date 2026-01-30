#!/bin/bash
# World Weaver Environment Bootstrap
# Generated: 2025-11-27

set -e

echo "=== World Weaver Environment Setup ==="

# Check working directory
if [[ ! -f "ww-features.json" ]]; then
    echo "Error: Must run from World Weaver project root"
    exit 1
fi

# Check Python version
echo "Checking Python..."
python3 --version || { echo "Python 3 required"; exit 1; }

# Create/activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Verify project structure
echo "Verifying project structure..."
REQUIRED_DIRS=(
    "skills/orchestration"
    "skills/memory"
    "skills/knowledge"
    "skills/domain"
    "skills/workflow"
    ".claude/agents"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir (missing)"
    fi
done

# Check core files
echo "Checking core files..."
REQUIRED_FILES=(
    "ww-features.json"
    "ww-progress.json"
    "ARCHITECTURE.md"
    "AGENTS_AND_SKILLS.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
    fi
done

# Run basic verification
echo ""
echo "Running verification..."

# Check feature file is valid JSON
if python3 -c "import json; json.load(open('ww-features.json'))" 2>/dev/null; then
    echo "  ✓ ww-features.json is valid JSON"
else
    echo "  ✗ ww-features.json is invalid"
fi

# Check progress file is valid JSON
if python3 -c "import json; json.load(open('ww-progress.json'))" 2>/dev/null; then
    echo "  ✓ ww-progress.json is valid JSON"
else
    echo "  ✗ ww-progress.json is invalid"
fi

# Show feature status
echo ""
echo "Feature Status:"
python3 -c "
import json
with open('ww-features.json') as f:
    data = json.load(f)
features = data.get('features', [])
passed = sum(1 for f in features if f.get('passes', False))
total = len(features)
print(f'  {passed}/{total} features complete ({100*passed/total:.1f}%)')
"

# Show session count
echo ""
echo "Progress Status:"
python3 -c "
import json
with open('ww-progress.json') as f:
    data = json.load(f)
sessions = data.get('sessions', [])
print(f'  {len(sessions)} sessions logged')
focus = data.get('current_focus')
if focus:
    print(f'  Current focus: {focus}')
"

echo ""
echo "=== Environment Ready ==="
echo ""
echo "To start a session:"
echo "  1. Load ww-progress.json to see recent work"
echo "  2. Check ww-features.json for pending features"
echo "  3. Work on highest priority incomplete feature"
echo "  4. Update progress and commit before ending"
