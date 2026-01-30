"""
World Weaver REST API.

FastAPI-based REST API for programmatic access to the tripartite memory system.
"""

from ww.api.server import app, main

__all__ = ["app", "main"]
