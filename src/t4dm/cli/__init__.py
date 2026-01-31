"""
World Weaver CLI.

Command-line interface for interacting with World Weaver memory.

Usage:
    ww store "content" --tags tag1,tag2 --importance 0.8
    ww recall "query" --k 5 --type episodic
    ww consolidate --full
    ww status
    ww serve --port 8765
"""

from t4dm.cli.main import app

__all__ = ["app"]
