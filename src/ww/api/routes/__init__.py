"""
World Weaver REST API Routes.

Route modules for each memory subsystem, control plane, and Phase 11 demo APIs.
"""

# Phase 10: Agent SDK integration
from ww.api.routes.agents import router as agents_router
from ww.api.routes.config import router as config_router

# Phase 9: Control plane
from ww.api.routes.control import router as control_router
from ww.api.routes.dream_viewer import dream_router
from ww.api.routes.entities import router as entities_router
from ww.api.routes.episodes import router as episodes_router

# Diagram visualization
from ww.api.routes.diagrams import router as diagrams_router

# Phase 11: Demo APIs
from ww.api.routes.explorer import explorer_router
from ww.api.routes.learning_trace import learning_router
from ww.api.routes.nt_dashboard import nt_router
from ww.api.routes.persistence import router as persistence_router
from ww.api.routes.skills import router as skills_router
from ww.api.routes.system import router as system_router
from ww.api.routes.visualization import router as visualization_router

__all__ = [
    "agents_router",
    "config_router",
    "control_router",
    "entities_router",
    "episodes_router",
    "persistence_router",
    "skills_router",
    "system_router",
    "visualization_router",
    "diagrams_router",
    # Phase 11 demos
    "explorer_router",
    "dream_router",
    "nt_router",
    "learning_router",
]
