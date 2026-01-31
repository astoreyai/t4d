"""
World Weaver Integration Adapters.

Provides integration with external agent frameworks and neural components:

ccapi (llm_agents):
  - WWMemory: Memory protocol adapter for conversation history
  - WWObserver: Observer for outcome feedback and learning
  - create_ww_router: FastAPI router for WW memory endpoints

Phase 1B: STDP-VTA Bridge:
  - STDPVTABridge: Dopamine modulation of STDP learning rates
  - Connects VTA dopamine signals to synaptic plasticity

Usage:
    from t4dm.integration import create_ww_memory, create_ww_observer, create_ww_router

    # Create memory adapter
    memory = create_ww_memory(session_id="agent-001")

    # Create observer for learning
    observer = create_ww_observer(session_id="agent-001")

    # Mount WW routes in FastAPI app
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(create_ww_router(), prefix="/memory")

    # Phase 1B: STDP-VTA bridge
    from t4dm.integration import STDPVTABridge
    from t4dm.learning.stdp import get_stdp_learner
    from t4dm.nca.vta import VTACircuit

    stdp = get_stdp_learner()
    vta = VTACircuit()
    bridge = STDPVTABridge(stdp, vta)

    # Get DA-modulated learning rates
    a_plus, a_minus = bridge.get_da_modulated_rates()
"""

from t4dm.integration.ccapi_memory import Message, WWMemory, create_ww_memory
from t4dm.integration.ccapi_observer import (
    Event,
    EventType,
    Span,
    WWObserver,
    create_ww_observer,
)
from t4dm.integration.ccapi_routes import (
    ContextResponse,
    LearningStatsResponse,
    MemoryResult,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    OutcomeRequest,
    OutcomeResponse,
    create_ww_router,
    get_ww_router,
)
from t4dm.integration.stdp_vta_bridge import (
    STDPVTABridge,
    STDPVTAConfig,
    get_stdp_vta_bridge,
    reset_stdp_vta_bridge,
)

__all__ = [
    # ccapi Memory
    "WWMemory",
    "Message",
    "create_ww_memory",
    # ccapi Observer
    "WWObserver",
    "Event",
    "Span",
    "EventType",
    "create_ww_observer",
    # FastAPI Router
    "create_ww_router",
    "get_ww_router",
    "MemoryStoreRequest",
    "MemoryStoreResponse",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemoryResult",
    "ContextResponse",
    "OutcomeRequest",
    "OutcomeResponse",
    "LearningStatsResponse",
    # Phase 1B: STDP-VTA Bridge
    "STDPVTABridge",
    "STDPVTAConfig",
    "get_stdp_vta_bridge",
    "reset_stdp_vta_bridge",
]
