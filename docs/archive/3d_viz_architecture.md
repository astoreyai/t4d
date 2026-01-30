# World Weaver 3D Memory Visualization Architecture

## Recommended Stack

```
React Three Fiber (R3F)     <- Component wrapper + rendering
D3-force-3d                 <- Physics engine (Barnes-Hut, Verlet)
Jotai                       <- State management
WebGL/Three.js              <- GPU acceleration
```

## Why This Stack?

| Criterion | React Three Fiber | Force-Graph 3D | Sigma.js |
|-----------|-------------------|----------------|----------|
| 10K+ nodes at 60fps | Excellent (instancing) | Good | Good |
| Physics layout | External (d3-force-3d) | Built-in | External |
| React ecosystem | Perfect fit | Via wrapper | Wrapper |
| Memory footprint | ~72MB | ~155MB | ~123MB |

## Performance Strategies

### GPU Instancing (Critical)
```typescript
<instancedMesh args={[geometry, material, 10000]}>
  // Single draw call for all nodes
</instancedMesh>
```

### Level of Detail
- < 100 nodes: Full detail with glow
- 100-1000: Simplified geometry
- > 1000: Instanced rendering, no glow

## Key Components

### Nodes (Memory Types)
- **Episodic**: Blue sphere (#3498db)
- **Semantic**: Green icosahedron (#2ecc71)
- **Procedural**: Orange octahedron (#e67e22)

### Edges
- Line thickness = Hebbian weight
- Animated particles for active edges
- Opacity decay for stale relationships

### UI Panels
- CRUD Operations (create/edit/delete)
- Filter/Search by type, time
- Timeline scrubber for replay
- Pruning interface for bulk delete

## API Endpoints

```
GET  /api/v1/viz/graph       <- Full graph with 3D positions
GET  /api/v1/viz/embeddings  <- Raw embeddings for projection
GET  /api/v1/viz/timeline    <- Temporal data for animation
GET  /api/v1/viz/activity    <- Recent activity metrics
POST /api/v1/viz/export      <- Export to PNG/SVG/GLTF
WS   /api/v1/viz/stream      <- Real-time updates
```

## Implementation Phases

1. **Phase 1** (2-3 weeks): R3F scaffold, basic rendering, physics
2. **Phase 2** (1-2 weeks): Selection, CRUD, search, WebSocket
3. **Phase 3** (2-3 weeks): Timeline, traces, pruning, export
4. **Phase 4** (1 week): Optimization, LOD, profiling

## Directory Structure

```
frontend/ww-viz/
├── src/
│   ├── components/
│   │   ├── Scene/          <- R3F Canvas, Camera, Lighting
│   │   ├── Graph/          <- Node, Edge, Cluster, Trace
│   │   ├── UI/             <- Sidebar, Inspector, Timeline
│   │   └── Effects/        <- Glow, Flow, Decay
│   ├── store/              <- Zustand/Jotai state
│   └── hooks/              <- useGraphData, useWebSocket
└── public/assets/shaders/  <- Custom GLSL
```
