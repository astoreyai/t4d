# T4DM Makefile
# System startup, common commands, and graceful shutdown

.PHONY: help start stop restart health logs test lint clean deps api mcp docker-up docker-down install dev-install \
        frontend frontend-bg frontend-build frontend-install frontend-logs frontend-stop \
        up up-all down down-all status stats dev

# Default target
help:
	@echo "T4DM Memory System - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev-install   Install with development dependencies"
	@echo "  make deps          Start Docker services (Neo4j, Qdrant)"
	@echo ""
	@echo "Services:"
	@echo "  make start         Start API server (foreground)"
	@echo "  make start-bg      Start API server (background)"
	@echo "  make mcp           Start MCP server (foreground)"
	@echo "  make mcp-bg        Start MCP server (background)"
	@echo "  make stop          Stop all WW servers gracefully"
	@echo "  make restart       Restart API server"
	@echo ""
	@echo "Monitoring:"
	@echo "  make health        Check API health"
	@echo "  make stats         Get memory statistics"
	@echo "  make logs          Tail API server logs"
	@echo "  make status        Show running WW processes"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run tests without slow markers"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run linters (ruff, mypy)"
	@echo "  make format        Format code with black"
	@echo "  make clean         Remove cache and build artifacts"
	@echo ""
	@echo "Frontend:"
	@echo "  make frontend-install  Install frontend dependencies"
	@echo "  make frontend          Start frontend dev server (foreground)"
	@echo "  make frontend-bg       Start frontend dev server (background)"
	@echo "  make frontend-build    Build frontend for production"
	@echo "  make frontend-logs     Tail frontend logs"
	@echo "  make frontend-stop     Stop frontend server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     Start Neo4j and Qdrant containers"
	@echo "  make docker-down   Stop Docker containers"
	@echo "  make docker-logs   View Docker container logs"
	@echo ""
	@echo "Full Stack:"
	@echo "  make up-all        Start everything (Docker + API + Frontend)"
	@echo "  make down-all      Stop everything"

# Configuration
PYTHON := python
VENV := .venv
API_PORT := 8765
API_HOST := 0.0.0.0
FRONTEND_PORT := 3000
DATA_DIR := .data
FRONTEND_DIR := frontend
PID_FILE := .t4dm-api.pid
MCP_PID_FILE := .t4dm-mcp.pid
FRONTEND_PID_FILE := .t4dm-frontend.pid

# Ensure data directory exists
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Installation
install:
	$(PYTHON) -m pip install -e .

dev-install:
	$(PYTHON) -m pip install -e ".[dev,api,interfaces,consolidation]"

# Docker services (T4DX is embedded, no external services needed)
docker-up:
	@echo "T4DX is an embedded engine — no external services required."

docker-down:
	@echo "No external services to stop."

deps:

# Frontend
frontend-install:
	@echo "Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && npm install

frontend:
	@echo "Starting frontend dev server on port $(FRONTEND_PORT)..."
	cd $(FRONTEND_DIR) && npm run dev

frontend-bg: $(DATA_DIR)
	@echo "Starting frontend dev server in background..."
	@cd $(FRONTEND_DIR) && nohup npm run dev > ../$(DATA_DIR)/frontend.log 2>&1 & echo $$! > ../$(FRONTEND_PID_FILE)
	@sleep 3
	@if [ -f $(FRONTEND_PID_FILE) ] && kill -0 $$(cat $(FRONTEND_PID_FILE)) 2>/dev/null; then \
		echo "Frontend started (PID: $$(cat $(FRONTEND_PID_FILE)))"; \
		echo "URL: http://localhost:$(FRONTEND_PORT)"; \
		echo "Logs: tail -f $(DATA_DIR)/frontend.log"; \
	else \
		echo "Failed to start frontend. Check $(DATA_DIR)/frontend.log"; \
		exit 1; \
	fi

frontend-build:
	@echo "Building frontend for production..."
	cd $(FRONTEND_DIR) && npm run build

frontend-logs:
	@if [ -f $(DATA_DIR)/frontend.log ]; then \
		tail -f $(DATA_DIR)/frontend.log; \
	else \
		echo "No frontend log file found. Server may not be running in background mode."; \
	fi

frontend-stop:
	@if [ -f $(FRONTEND_PID_FILE) ]; then \
		PID=$$(cat $(FRONTEND_PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping frontend (PID: $$PID)..."; \
			kill -TERM $$PID; \
			sleep 2; \
			if kill -0 $$PID 2>/dev/null; then \
				echo "Force killing frontend..."; \
				kill -9 $$PID 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(FRONTEND_PID_FILE); \
	fi
	@pkill -f "npm run dev" 2>/dev/null || true
	@echo "Frontend stopped."

# API Server
start: $(DATA_DIR)
	@echo "Starting T4DM API Server on $(API_HOST):$(API_PORT)..."
	T4DM_DATA_DIR=$(DATA_DIR) $(PYTHON) -m t4dm.api.server

start-bg: $(DATA_DIR)
	@echo "Starting T4DM API Server in background..."
	@T4DM_DATA_DIR=$(DATA_DIR) nohup $(PYTHON) -m t4dm.api.server > $(DATA_DIR)/api.log 2>&1 & echo $$! > $(PID_FILE)
	@sleep 2
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "API Server started (PID: $$(cat $(PID_FILE)))"; \
		echo "Logs: tail -f $(DATA_DIR)/api.log"; \
	else \
		echo "Failed to start API server. Check $(DATA_DIR)/api.log"; \
		exit 1; \
	fi

api: start

# MCP Server
mcp:
	@echo "Starting T4DM MCP Server..."
	$(PYTHON) -m t4dm.mcp.server

mcp-bg:
	@echo "Starting T4DM MCP Server in background..."
	@nohup $(PYTHON) -m t4dm.mcp.server > $(DATA_DIR)/mcp.log 2>&1 & echo $$! > $(MCP_PID_FILE)
	@sleep 2
	@if [ -f $(MCP_PID_FILE) ] && kill -0 $$(cat $(MCP_PID_FILE)) 2>/dev/null; then \
		echo "MCP Server started (PID: $$(cat $(MCP_PID_FILE)))"; \
	else \
		echo "Failed to start MCP server. Check $(DATA_DIR)/mcp.log"; \
		exit 1; \
	fi

# Stop servers gracefully
stop:
	@echo "Stopping T4DM servers..."
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping API server (PID: $$PID)..."; \
			kill -TERM $$PID; \
			sleep 2; \
			if kill -0 $$PID 2>/dev/null; then \
				echo "Force killing API server..."; \
				kill -9 $$PID 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(PID_FILE); \
	fi
	@if [ -f $(MCP_PID_FILE) ]; then \
		PID=$$(cat $(MCP_PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping MCP server (PID: $$PID)..."; \
			kill -TERM $$PID; \
			sleep 2; \
			if kill -0 $$PID 2>/dev/null; then \
				echo "Force killing MCP server..."; \
				kill -9 $$PID 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(MCP_PID_FILE); \
	fi
	@if [ -f $(FRONTEND_PID_FILE) ]; then \
		PID=$$(cat $(FRONTEND_PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping frontend (PID: $$PID)..."; \
			kill -TERM $$PID; \
			sleep 2; \
			if kill -0 $$PID 2>/dev/null; then \
				echo "Force killing frontend..."; \
				kill -9 $$PID 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(FRONTEND_PID_FILE); \
	fi
	@# Also kill any orphaned processes
	@pkill -f "python -m t4dm.api.server" 2>/dev/null || true
	@pkill -f "python -m t4dm.mcp.server" 2>/dev/null || true
	@pkill -f "npm run dev" 2>/dev/null || true
	@echo "Servers stopped."

restart: stop start-bg

# Monitoring
health:
	@curl -s http://localhost:$(API_PORT)/api/v1/health | python -m json.tool 2>/dev/null || echo "API server not responding"

stats:
	@curl -s http://localhost:$(API_PORT)/api/v1/stats | python -m json.tool 2>/dev/null || echo "API server not responding"

status:
	@echo "T4DM Process Status:"
	@echo "----------------------------"
	@echo "Backend:"
	@ps aux | grep -E "(ww\.(api|mcp)\.server)" | grep -v grep || echo "  No backend processes running"
	@echo ""
	@echo "Frontend:"
	@ps aux | grep -E "npm run dev|vite" | grep -v grep || echo "  No frontend processes running"
	@echo ""
	@echo "PID Files:"
	@if [ -f $(PID_FILE) ]; then echo "  API: $$(cat $(PID_FILE))"; fi
	@if [ -f $(MCP_PID_FILE) ]; then echo "  MCP: $$(cat $(MCP_PID_FILE))"; fi
	@if [ -f $(FRONTEND_PID_FILE) ]; then echo "  Frontend: $$(cat $(FRONTEND_PID_FILE))"; fi

logs:
	@if [ -f $(DATA_DIR)/api.log ]; then \
		tail -f $(DATA_DIR)/api.log; \
	else \
		echo "No log file found. Server may not be running in background mode."; \
	fi

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow and not benchmark"

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src/t4dm --cov-report=term-missing --cov-report=html

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

# Linting and formatting
lint:
	$(PYTHON) -m ruff check src/t4dm tests/
	$(PYTHON) -m mypy src/t4dm --ignore-missing-imports

format:
	$(PYTHON) -m black src/t4dm tests/
	$(PYTHON) -m ruff check --fix src/t4dm tests/

# Cleanup
clean:
	rm -rf .pytest_cache __pycache__ .mypy_cache .ruff_cache
	rm -rf src/t4dm/__pycache__ tests/__pycache__
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-data:
	@echo "WARNING: This will delete all persisted data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(DATA_DIR)/*

# Quick development workflow
dev: docker-up dev-install start

# All-in-one startup (backend only)
up: docker-up start-bg
	@echo ""
	@echo "T4DM Backend is running!"
	@echo "  API:    http://localhost:$(API_PORT)"
	@echo "  Docs:   http://localhost:$(API_PORT)/docs"
	@echo "  Health: http://localhost:$(API_PORT)/api/v1/health"
	@echo ""
	@echo "Run 'make frontend' for UI, 'make logs' for logs, 'make stop' to shutdown"

# Full stack startup (Docker + API + Frontend)
up-all: docker-up start-bg frontend-bg
	@echo ""
	@echo "T4DM Full Stack is running!"
	@echo "  Frontend: http://localhost:$(FRONTEND_PORT)"
	@echo "  API:      http://localhost:$(API_PORT)"
	@echo "  Docs:     http://localhost:$(API_PORT)/docs"
	@echo "  Health:   http://localhost:$(API_PORT)/api/v1/health"
	@echo ""
	@echo "Run 'make status' to check, 'make stop' to shutdown"

# All-in-one shutdown (backend only)
down: stop docker-down
	@echo "T4DM backend fully stopped."

# Full stack shutdown
down-all: stop docker-down
	@echo "T4DM fully stopped."
