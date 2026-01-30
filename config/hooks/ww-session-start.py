#!/usr/bin/env python3
"""
World Weaver Session Start Hook.

Initializes memory context for new Claude Code sessions.
"""

import os
import sys
from datetime import datetime


def main():
    """Display World Weaver session initialization info."""
    session_id = os.environ.get("WW_SESSION_ID", "unknown")
    neo4j_uri = os.environ.get("WW_NEO4J_URI", "bolt://localhost:7687")
    qdrant_url = os.environ.get("WW_QDRANT_URL", "http://localhost:6333")

    print()
    print("=" * 50)
    print("World Weaver Memory System Initialized")
    print("=" * 50)
    print(f"Session ID: {session_id}")
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print(f"Neo4j:      {neo4j_uri}")
    print(f"Qdrant:     {qdrant_url}")
    print()
    print("Available MCP Tools:")
    print("  Episodic:   create_episode, recall_episodes, query_at_time, mark_important")
    print("  Semantic:   create_entity, create_relation, semantic_recall, spread_activation")
    print("  Procedural: build_skill, how_to, execute_skill, deprecate_skill")
    print("  Utility:    consolidate_now, get_provenance, get_session_id, memory_stats")
    print()
    print("=" * 50)


if __name__ == "__main__":
    main()
