# Integration
**Path**: `/mnt/projects/t4d/t4dm/src/ww/integration/`

## What
Adapters for connecting WW to external agent frameworks (ccapi/llm_agents) and bridging neural subsystems (STDP-VTA dopamine modulation).

## How
- **WWMemory** (`ccapi_memory.py`): Implements the llm_agents Memory protocol. Stores messages in an in-memory buffer with async commit to WW episodic memory. Provides sync/async search with fallback to simple text matching.
- **WWObserver** (`ccapi_observer.py`): Implements the llm_agents Observer protocol. Captures events (spans, outcomes) and feeds them into WW's learning system.
- **create_ww_router** (`ccapi_routes.py`): FastAPI router exposing `/store`, `/search`, `/outcome`, `/context`, `/entity` endpoints for REST API access.
- **STDPVTABridge** (`stdp_vta_bridge.py`): Bridges VTA dopamine circuit to STDP learning rate modulation. High DA enhances LTP and reduces LTD (reward learning); low DA does the inverse. Implements DA-based learning gating.

## Why
Enables WW to serve as the memory backend for any agent built on ccapi/llm_agents, and bridges biological neural subsystems (dopamine -> synaptic plasticity) for neuromodulated learning.

## Key Files
| File | Purpose |
|------|---------|
| `ccapi_memory.py` | `WWMemory` adapter, message buffering, lazy WW initialization |
| `ccapi_observer.py` | `WWObserver`, event/span capture, outcome feedback |
| `ccapi_routes.py` | FastAPI router with Pydantic request/response models |
| `stdp_vta_bridge.py` | `STDPVTABridge`, DA-modulated STDP rates, learning gating |

## Data Flow
```
Agent (ccapi) -> WWMemory.add(message) -> buffer + async episodic store
Agent query   -> WWMemory.search() -> WW episodic recall -> learning events
Agent outcome -> WWObserver -> learning collector -> reconsolidation

VTA dopamine level -> STDPVTABridge -> modulated A+/A- -> STDP weight updates
```

## Integration Points
- **memory/episodic.py**: WWMemory commits messages as episodes
- **learning/**: Observer emits outcome events; STDP bridge modulates learning rates
- **nca/vta.py**: VTA circuit provides dopamine levels to the STDP bridge
- **learning/stdp.py**: STDPLearner receives DA-modulated rates
