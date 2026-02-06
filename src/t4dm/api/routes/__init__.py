"""
T4DM REST API Routes.

Route modules for each memory subsystem, control plane, and Phase 11 demo APIs.
"""

# Phase 10: Agent SDK integration
from t4dm.api.routes.agents import router as agents_router
from t4dm.api.routes.config import router as config_router

# Phase 9: Control plane
from t4dm.api.routes.control import router as control_router
from t4dm.api.routes.dream_viewer import dream_router
from t4dm.api.routes.entities import router as entities_router
from t4dm.api.routes.episodes import router as episodes_router

# Diagram visualization
from t4dm.api.routes.diagrams import router as diagrams_router

# Phase 11: Demo APIs
from t4dm.api.routes.explorer import explorer_router
from t4dm.api.routes.learning_trace import learning_router
from t4dm.api.routes.nt_dashboard import nt_router
from t4dm.api.routes.persistence import router as persistence_router
from t4dm.api.routes.skills import router as skills_router
from t4dm.api.routes.system import router as system_router
from t4dm.api.routes.visualization import router as visualization_router

# Mem0-compatible API
from t4dm.api.routes.compat import router as compat_router

# Phase 5: Visualization module routes
from t4dm.api.routes.viz_modules import router as viz_modules_router

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
    "compat_router",
    # Phase 5: Visualization modules
    "viz_modules_router",
]
