# Interfaces Module

**8 files | ~3,000 lines | Centrality: 6**

The interfaces module provides terminal UI components for exploring, managing, and monitoring T4DM memories and neural dynamics using the Rich library.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTERFACE LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │CRUDManager  │  │MemoryExplor│  │ TraceViewer │  │ SystemDashboard │ │
│  │Create/Read  │  │Browse/Search│  │Access Trace │  │  Health/Metrics │ │
│  │Update/Delete│  │View Details │  │Decay Curves │  │  Live Refresh   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ExportUtility│  │ NCAExplorer │  │     LearningInspector           │  │
│  │JSON/CSV/    │  │ NT State    │  │  Neuromodulator state           │  │
│  │GraphML      │  │ Energy/Caps │  │  Eligibility/FSRS               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                         MEMORY LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  EpisodicMemory     SemanticMemory     ProceduralMemory             ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│                         STORAGE LAYER                                    │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │   Qdrant (vectors)         │  │   Neo4j (graph)                    │ │
│  └────────────────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `crud_manager.py` | ~550 | Create, Read, Update, Delete operations |
| `memory_explorer.py` | ~550 | Interactive memory browsing |
| `trace_viewer.py` | ~450 | Memory access patterns and decay curves |
| `dashboard.py` | ~370 | Real-time system health monitoring |
| `export_utils.py` | ~600 | Export to JSON, CSV, GraphML formats |
| `nca_explorer.py` | ~800 | NCA state visualization |
| `learning_inspector.py` | ~800 | Learning dynamics monitoring |
| `__init__.py` | ~50 | Public API exports |

## Components

### CRUDManager

Full CRUD operations for all memory types:

```python
from t4dm.interfaces import CRUDManager

crud = CRUDManager(session_id="my-session")
await crud.initialize()

# Create
episode = await crud.create_episode(
    content="User asked about memory",
    outcome=0.8,
    emotional_valence=0.5,
    context={"source": "chat"}
)

entity = await crud.create_entity(
    name="Memory System",
    entity_type="concept",
    summary="A system for storing and retrieving memories"
)

skill = await crud.create_skill(
    name="search_memory",
    domain="retrieval",
    steps=["parse query", "embed", "search", "rank"],
    trigger_pattern="find|search|recall"
)

# Read
episode = await crud.get_episode(episode_id)
entity = await crud.get_entity(entity_id)
skill = await crud.get_skill(skill_id)

# Update
await crud.update_entity(entity_id, new_summary="Updated summary")
await crud.update_skill_performance(skill_id, success=True)

# Delete (with confirmation)
await crud.delete_episode(episode_id, confirm=True)
await crud.delete_entity(entity_id, cascade_relationships=True)

# Batch operations
episodes = await crud.batch_create_episodes([...])
count = await crud.batch_delete_episodes(ids, confirm=True)
```

### MemoryExplorer

Interactive terminal UI for memory browsing:

```python
from t4dm.interfaces import MemoryExplorer

explorer = MemoryExplorer(session_id="my-session")
await explorer.initialize()

# List memories
await explorer.list_episodes(limit=20, outcome_filter=">0.5")
await explorer.list_entities(limit=20, entity_type="person")
await explorer.list_skills(limit=20, domain="retrieval")

# View details
await explorer.view_episode(episode_id)
await explorer.view_entity_graph(entity_id, depth=2)

# Search across all memory types
await explorer.search("memory consolidation", limit=10)

# Interactive mode (menu-driven)
await explorer.interactive()
```

### TraceViewer

Memory access patterns and decay visualization:

```python
from t4dm.interfaces import TraceViewer

viewer = TraceViewer(session_id="my-session")
await viewer.initialize()

# Access timeline
await viewer.show_access_timeline(hours=24, memory_type="episode")

# FSRS decay curves (ASCII visualization)
await viewer.show_decay_curves(sample_size=10)
# Output:
# Episode abc123 [########..] 82% retrievable

# Consolidation events
await viewer.show_consolidation_events(limit=20)

# Entity activation history
await viewer.show_activation_history(entity_id, limit=50)

# Time-bucketed heatmap
await viewer.show_access_heatmap(hours=24, bucket_minutes=60)
```

**FSRS Formula**: R(t,S) = (1 + 0.9*t/S)^(-0.5)

### SystemDashboard

Real-time system health monitoring:

```python
from t4dm.interfaces import SystemDashboard

dashboard = SystemDashboard(session_id="my-session")
await dashboard.initialize()

# Get metrics
counts = await dashboard.get_memory_counts()
# {'episodes': 1234, 'entities': 567, 'skills': 89}

health = await dashboard.get_storage_health()
# {'qdrant': 'healthy', 'neo4j': 'healthy', 'circuit_breakers': {...}}

# Render dashboard (Rich Layout)
layout = await dashboard.render_dashboard()

# Live auto-refresh mode
await dashboard.show(refresh_interval=5)

# Detailed health report
await dashboard.show_detailed_health()
```

### ExportUtility

Multi-format memory export with security:

```python
from t4dm.interfaces import ExportUtility

export = ExportUtility(session_id="my-session")
await export.initialize()

# JSON export
count = await export.export_episodes_json(
    output_path="~/ww_exports/episodes.json",
    limit=1000,
    include_embeddings=False
)

# CSV export
count = await export.export_entities_csv(
    output_path="~/ww_exports/entities.csv",
    limit=None  # All entities
)

# GraphML for graph visualization tools
count = await export.export_graph_graphml(
    output_path="~/ww_exports/knowledge_graph.graphml",
    limit=500
)

# Full session backup
stats = await export.backup_session(output_dir="~/ww_exports/backup/")
# {'episodes': 1234, 'entities': 567, 'skills': 89}
```

**Security (SEC-003)**:
- Path traversal prevention via `Path.resolve()`
- Whitelist of allowed directories
- XML character escaping for GraphML

### NCAExplorer

Neural Cognitive Architecture visualization:

```python
from t4dm.interfaces import NCAExplorer

nca = NCAExplorer()

# Neurotransmitter state
nca.show_nt_state()
# Shows bar chart + radar for DA, 5-HT, ACh, NE, GABA, Glu

# Energy landscape
nca.show_energy_landscape()
# Hopfield + boundary + attractor energy components

# Coupling matrix
nca.show_coupling_matrix()
# 6x6 K matrix with spectral analysis

# Stability analysis
nca.show_stability()
# Eigenvalue analysis, stability classification

# Oscillations
nca.show_oscillations()
# Delta, Theta, Alpha, Beta, Gamma bands

# Advanced components
nca.show_forward_forward()   # FF network layers
nca.show_capsules()          # Capsule routing
nca.show_glymphatic()        # Waste clearance

# Dynamics simulation
nca.set_nt_state(nt_idx=0, value=0.8)  # Set DA
nca.step_dynamics(dt=0.01)
nca.relax_to_equilibrium(max_steps=100)

# Interactive menu
nca.interactive()
```

### LearningInspector

Learning dynamics and FSRS monitoring:

```python
from t4dm.interfaces import LearningInspector

inspector = LearningInspector()

# Neuromodulator state
inspector.show_neuromodulators()
# DA RPE, 5-HT credit, NE arousal, ACh mode

# Eligibility traces
inspector.show_eligibility()
# Layer-wise heatmap of trace values

# Three-factor learning
inspector.show_three_factor()
# effective_lr = eligibility x gate x |surprise|

# FSRS scheduling
inspector.show_fsrs()
# Due now, due today, stability, difficulty

# Causal attribution
inspector.show_causal()
# Tree view of causal relationships

# Recent learning events
inspector.show_events()

# Simulate outcome
results = inspector.simulate_outcome(
    outcome_type="success",
    n_memories=10
)

# Interactive menu
inspector.interactive()
```

## CLI Entry Points

| Module | Command | Description |
|--------|---------|-------------|
| `memory_explorer.py` | `ww-explore` | Interactive memory browsing |
| `trace_viewer.py` | `ww-trace` | Memory access timeline |
| `dashboard.py` | `ww-dashboard` | System health monitoring |
| `export_utils.py` | `ww-export` | Session backup |
| `nca_explorer.py` | `python -m t4dm.interfaces.nca_explorer` | NCA exploration |
| `learning_inspector.py` | `python -m t4dm.interfaces.learning_inspector` | Learning dynamics |

## Design Patterns

### Lazy Initialization

```python
async def initialize(self) -> None:
    if self._initialized:
        return
    # Initialize storage backends...
    self._initialized = True
```

### Optional Rich Dependency

```python
try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if not RICH_AVAILABLE:
    raise ImportError("rich library required")
```

### Session Isolation

All interfaces respect session boundaries:
- Optional `session_id` parameter
- Filtering at storage layer
- No cross-session data leakage

### Progress Tracking

```python
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    console=self.console,
) as progress:
    task = progress.add_task("Processing...", total=len(items))
    for item in items:
        # Process
        progress.advance(task)
```

## Installation

```bash
# Include interfaces extras
pip install -e ".[interfaces]"

# Or install Rich directly
pip install rich
```

## Testing

```bash
# Run interface tests
pytest tests/interfaces/ -v

# With coverage
pytest tests/interfaces/ --cov=t4dm.interfaces
```

## Public API

```python
__all__ = [
    "CRUDManager",
    "ExportUtility",
    "LearningInspector",
    "MemoryExplorer",
    "NCAExplorer",
    "SystemDashboard",
    "TraceViewer",
]
```

## Security Considerations

- **SEC-003 (Path Traversal)**: Export paths validated against whitelist
- **Session Isolation**: Queries filtered by session_id
- **Confirmation Prompts**: Destructive operations require confirmation
