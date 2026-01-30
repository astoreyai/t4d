# Memory Graph 3D - Architecture Diagram

## Component Hierarchy

```
MemoryGraphView (Main Container)
│
├─ ControlPanel (Left Sidebar)
│  ├─ Search Input
│  ├─ Memory Type Filters (checkboxes)
│  ├─ Edge Type Filters (checkboxes)
│  ├─ Activity Threshold Slider
│  ├─ Importance Threshold Slider
│  ├─ Layout Algorithm Selector
│  ├─ Node Scale Slider
│  ├─ Edge Opacity Slider
│  ├─ Toggle Switches (labels, edges, auto-rotate)
│  └─ Action Buttons (reset camera, reset filters)
│
├─ MemoryGraph3D (Center - 3D Canvas)
│  │
│  ├─ Canvas (React Three Fiber)
│  │  ├─ PerspectiveCamera
│  │  ├─ OrbitControls
│  │  ├─ Lighting (ambient, point lights)
│  │  ├─ Grid (spatial reference)
│  │  │
│  │  ├─ EdgeRenderer
│  │  │  └─ For each edge:
│  │  │     ├─ Line (with color, width, animation)
│  │  │     └─ ArrowHead (cone mesh)
│  │  │
│  │  └─ NodeRenderer
│  │     └─ For each node:
│  │        ├─ Main Sphere (with color, size)
│  │        ├─ Glow Shell (if activated)
│  │        ├─ Selection Ring (if selected/highlighted)
│  │        └─ Label (Text3D, on hover)
│  │
│  └─ TimelineSlider (Bottom Overlay)
│     ├─ Time Display
│     ├─ Progress Bar
│     ├─ Slider Control
│     ├─ Playback Buttons (start, play/pause, end)
│     └─ Speed Selector
│
├─ InspectorPanel (Right Sidebar - default mode)
│  ├─ Header (type badge, close button)
│  ├─ Content Section
│  ├─ Metadata Section (created, accessed, importance, tags)
│  ├─ Incoming Connections List
│  ├─ Outgoing Connections List
│  └─ Action Buttons (edit, delete)
│
└─ PruneMode (Right Sidebar - when mode === 'prune')
   ├─ Statistics (total, selected, memory)
   ├─ Selection Controls (select all, clear)
   ├─ Selected Nodes List (scrollable)
   ├─ Delete Button
   ├─ Cancel Button
   └─ Confirmation Dialog
```

## State Flow

```
User Interaction
      │
      ├─ Click Node ────────────► selectNodeAtom
      │                                 │
      │                                 ├─► selectionStateAtom
      │                                 │        │
      │                                 │        └─► selectedNodeAtom
      │                                 │                  │
      │                                 │                  └─► InspectorPanel
      │                                 │
      │                                 └─► connectedNodesAtom
      │                                          │
      │                                          └─► InspectorPanel (connections)
      │
      ├─ Change Filter ─────────► updateFiltersAtom
      │                                 │
      │                                 └─► filterStateAtom
      │                                          │
      │                                          ├─► filteredNodesAtom
      │                                          │        │
      │                                          │        └─► NodeRenderer
      │                                          │
      │                                          └─► filteredEdgesAtom
      │                                                   │
      │                                                   └─► EdgeRenderer
      │
      ├─ Adjust Control ────────► updateControlsAtom
      │                                 │
      │                                 └─► controlStateAtom
      │                                          │
      │                                          ├─► layoutAlgorithm ──► useForceLayout
      │                                          │                              │
      │                                          │                              └─► Node positions
      │                                          │
      │                                          ├─► nodeScale ──────► useNodeVisualConfigs
      │                                          │                              │
      │                                          │                              └─► Node sizes
      │                                          │
      │                                          └─► edgeOpacity ────► EdgeRenderer
      │
      ├─ Timeline Playback ──────► toggleTimelinePlaybackAtom
      │                                 │
      │                                 └─► timelineStateAtom
      │                                          │
      │                                          ├─► useTimelinePlayback (auto-advance)
      │                                          │
      │                                          └─► filteredNodesAtom (time filter)
      │
      └─ Enter Prune Mode ───────► setInteractionModeAtom('prune')
                                        │
                                        └─► interactionModeAtom
                                                 │
                                                 ├─► useNodeClickHandler (toggle multi-select)
                                                 │
                                                 └─► PruneMode component renders
```

## Data Flow

```
Backend (Go)
    │
    │ RPC: GetMemoryNodes()
    │ RPC: GetMemoryEdges()
    │
    ▼
nodesAtom, edgesAtom (raw data)
    │
    │ Apply filters
    │
    ▼
filteredNodesAtom, filteredEdgesAtom
    │
    ├─► useActivationMetrics
    │       │
    │       └─► Compute recency + frequency scores
    │
    ├─► useNodeVisualConfigs
    │       │
    │       └─► Compute size, color, glow
    │
    ├─► useEdgeVisualConfigs
    │       │
    │       └─► Compute width, color, animation
    │
    └─► useForceLayout
            │
            └─► Compute 3D positions
                    │
                    ▼
            NodeRenderer, EdgeRenderer
                    │
                    ▼
            Three.js Scene
                    │
                    ▼
            Canvas (rendered to screen)
```

## Hook Dependencies

```
useMemoryGraphData()
    └─► RpcApi.GetMemoryNodes(), RpcApi.GetMemoryEdges()
            │
            └─► Updates nodesAtom, edgesAtom

useActivationMetrics(nodes)
    └─► Depends on: timelineStateAtom.currentTime
    └─► Returns: Map<nodeId, { activation, recency, frequency }>

useNodeVisualConfigs(nodes, activationMetrics)
    └─► Depends on: controlStateAtom (nodeScale, showLabels)
    └─► Returns: Map<nodeId, NodeVisualConfig>

useEdgeVisualConfigs(edges, activationMetrics)
    └─► Depends on: controlStateAtom (edgeOpacity), timelineStateAtom
    └─► Returns: Map<edgeId, EdgeVisualConfig>

useForceLayout(nodes, edges, algorithm)
    └─► Depends on: controlStateAtom.layoutAlgorithm
    └─► Returns: Map<nodeId, Vector3>

useTimelinePlayback()
    └─► Depends on: timelineStateAtom.isPlaying
    └─► Side effect: Updates timelineStateAtom.currentTime every 100ms

useNodeSearch(nodes, query)
    └─► Depends on: filterStateAtom.searchQuery
    └─► Returns: { results, highlightIds }

useFocusNode()
    └─► Depends on: useForceLayout positions
    └─► Returns: (nodeId, cameraRef) => void (animate camera)

useNodeClickHandler(mode, onSelect, onToggleMultiSelect)
    └─► Depends on: interactionModeAtom.mode
    └─► Returns: (nodeId) => void (mode-aware click handler)
```

## Atom Dependency Graph

```
                    nodesAtom (primitive)
                         │
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
filterStateAtom   selectionStateAtom   timelineStateAtom
(primitive)        (primitive)         (primitive)
        │                │                │
        │                │                │
        └────────┬───────┴────────────────┘
                 │
                 ▼
        filteredNodesAtom (computed)
                 │
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    selectedNode  connected   graphMetrics
    Atom (comp)   NodesAtom   Atom (comp)
                  (comp)


                    edgesAtom (primitive)
                         │
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
filterStateAtom   selectionStateAtom   filteredNodesAtom
        │                │                │
        │                │                │
        └────────┬───────┴────────────────┘
                 │
                 ▼
        filteredEdgesAtom (computed)
                 │
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    incomingEdges     outgoingEdges
    Atom (comp)       Atom (comp)
```

## Interaction Modes

```
Mode: explore (default)
├─ Node Click: Select node → InspectorPanel shows details
├─ Node Hover: Show label
├─ Pan/Zoom: OrbitControls enabled
└─ UI: ControlPanel + InspectorPanel

Mode: inspect
├─ Same as explore
└─ UI: ControlPanel + InspectorPanel (enhanced)

Mode: timeline
├─ Node Click: Select node
├─ Nodes Filtered: By timelineState.currentTime
├─ Auto Playback: useTimelinePlayback() advances time
└─ UI: ControlPanel + InspectorPanel + TimelineSlider

Mode: prune
├─ Node Click: Toggle multi-select (red ring)
├─ Selection Shown: PruneMode panel
├─ Delete Action: Batch delete selected nodes
└─ UI: ControlPanel + PruneMode
```

## Rendering Pipeline

```
1. Fetch Data
   └─► useMemoryGraphData() → nodesAtom, edgesAtom

2. Apply Filters
   └─► filteredNodesAtom, filteredEdgesAtom

3. Compute Metrics
   └─► useActivationMetrics → Map<nodeId, metrics>

4. Compute Visual Configs
   ├─► useNodeVisualConfigs → Map<nodeId, { size, color, glow, label }>
   └─► useEdgeVisualConfigs → Map<edgeId, { width, color, animated }>

5. Compute Layout
   └─► useForceLayout → Map<nodeId, Vector3 position>

6. Render 3D Scene
   ├─► NodeRenderer
   │   └─► For each node:
   │       ├─ Position from useForceLayout
   │       ├─ Visual config from useNodeVisualConfigs
   │       └─ Selection state from selectionStateAtom
   │
   └─► EdgeRenderer
       └─► For each edge:
           ├─ Source/target positions from nodePositions map
           └─ Visual config from useEdgeVisualConfigs

7. Update UI Panels
   ├─► ControlPanel (filters, controls)
   ├─► InspectorPanel (selected node details)
   ├─► TimelineSlider (playback controls)
   └─► PruneMode (multi-select UI)
```

## Event Flow Examples

### Example 1: User selects a node

```
1. User clicks node in 3D canvas
2. NodeRenderer onNodeClick(nodeId)
3. useNodeClickHandler checks interactionModeAtom
4. Mode is 'explore' → calls selectNodeAtom(nodeId)
5. selectNodeAtom updates selectionStateAtom.selectedNode
6. selectedNodeAtom recomputes (finds node in filteredNodesAtom)
7. connectedNodesAtom recomputes (finds neighbors)
8. incomingEdgesAtom recomputes
9. outgoingEdgesAtom recomputes
10. InspectorPanel re-renders with new data
```

### Example 2: User changes filter

```
1. User moves activityThreshold slider in ControlPanel
2. Slider onChange calls updateFiltersAtom({ activityThreshold: 0.7 })
3. updateFiltersAtom updates filterStateAtom
4. filteredNodesAtom recomputes (filters by activation score)
5. filteredEdgesAtom recomputes (only edges between visible nodes)
6. useActivationMetrics recomputes for new node set
7. useNodeVisualConfigs recomputes
8. useForceLayout recomputes positions
9. NodeRenderer re-renders with new positions and configs
10. EdgeRenderer re-renders with new edge set
11. graphMetricsAtom updates
12. UI panels show updated counts
```

### Example 3: Timeline playback

```
1. User clicks Play in TimelineSlider
2. toggleTimelinePlaybackAtom updates timelineStateAtom.isPlaying = true
3. useTimelinePlayback hook starts interval (every 100ms):
   a. Increment currentTime by playbackSpeed
   b. Update timelineStateAtom.currentTime
4. filteredNodesAtom recomputes (filters nodes by creation time)
5. Nodes gradually appear in 3D canvas as time advances
6. When currentTime reaches endTime:
   a. Set isPlaying = false
   b. Stop interval
7. User sees "memory formation" replay
```

### Example 4: Prune mode

```
1. User clicks "Prune Mode" button
2. setInteractionModeAtom('prune') updates interactionModeAtom.mode
3. PruneMode component renders (replaces InspectorPanel)
4. User clicks node in canvas
5. useNodeClickHandler checks mode → 'prune'
6. Calls toggleMultiSelectAtom(nodeId)
7. toggleMultiSelectAtom adds/removes from selectionStateAtom.multiSelect
8. NodeRenderer shows red selection ring for multi-selected nodes
9. PruneMode panel updates selected count
10. User clicks "Delete N Nodes"
11. Confirmation dialog appears
12. User confirms
13. RpcApi.DeleteMemoryNodes(nodeIds)
14. Backend deletes nodes
15. Frontend reloads data
16. setInteractionModeAtom('explore') exits prune mode
```

## Performance Optimization Points

```
1. Atom Granularity
   ├─ Split large state objects into small atoms
   ├─ Components only subscribe to atoms they need
   └─ Computed atoms memoize results

2. Component Memoization
   ├─ React.memo() for pure components
   ├─ useMemo() for expensive calculations
   └─ useCallback() for event handlers

3. Rendering Optimization
   ├─ Throttle layout recalculations (useThrottle)
   ├─ Debounce search queries (useDebounce)
   ├─ Use instancing for many similar nodes
   └─ Level-of-detail (LOD) for distant nodes

4. Data Filtering
   ├─ Filter at atom level (filteredNodesAtom)
   ├─ Don't render filtered-out nodes
   └─ Early exit in visual config hooks

5. Animation
   ├─ Use requestAnimationFrame (via useFrame)
   ├─ Only animate visible objects
   └─ Pause animations when tab not visible
```

## File Size Summary

```
memorygraph-types.ts         ~900 lines    Type definitions, constants
memorygraph-state.ts         ~450 lines    Jotai atoms, actions
memorygraph-hooks.ts         ~550 lines    Custom React hooks
MemoryGraph3D.tsx           ~180 lines    Main 3D container
NodeRenderer.tsx            ~200 lines    Node rendering
EdgeRenderer.tsx            ~170 lines    Edge rendering
ControlPanel.tsx            ~280 lines    Filter/control UI
InspectorPanel.tsx          ~300 lines    Node details panel
TimelineSlider.tsx          ~200 lines    Timeline UI
PruneMode.tsx               ~250 lines    Prune mode UI
*.scss (5 files)            ~600 lines    Styles
index.ts                     ~80 lines    Barrel exports
───────────────────────────────────────────────────────
TOTAL:                     ~4,160 lines   Complete system
```

## Technology Stack

```
Core:
├─ React 18              Component framework
├─ TypeScript 5          Type safety
└─ Jotai                State management

3D Graphics:
├─ Three.js              WebGL library
├─ React Three Fiber     React renderer for Three.js
└─ @react-three/drei     Three.js helpers

UI Components:
├─ shadcn/ui             Component library
└─ Radix UI              Accessible primitives

Styling:
├─ SCSS                  CSS preprocessor
└─ CSS Variables         Theme system

Build:
└─ Vite                  Fast bundler with HMR
```

## Next Steps Checklist

```
Backend Integration:
☐ Implement GetMemoryNodes RPC endpoint
☐ Implement GetMemoryEdges RPC endpoint
☐ Implement UpdateMemoryNode RPC endpoint
☐ Implement DeleteMemoryNodes RPC endpoint
☐ Implement CreateMemoryEdge RPC endpoint

Frontend:
☐ Connect useMemoryGraphData to real RPC calls
☐ Add node edit dialog (form for metadata)
☐ Add edge creation UI (drag between nodes)
☐ Implement d3-force-3d for better layout
☐ Add export/import functionality

Testing:
☐ Unit tests for atoms and actions
☐ Component tests for UI
☐ Integration tests for interactions
☐ Performance tests with large graphs

Documentation:
☐ User guide (how to use the graph)
☐ Developer guide (how to extend)
☐ API documentation (RPC contracts)
```
