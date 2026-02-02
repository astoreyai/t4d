# Memory Graph 3D - Quick Start Guide

## Installation

```bash
cd /mnt/projects/t4d/t4dm/frontend

# Install dependencies
npm install three @react-three/fiber @react-three/drei jotai

# Install shadcn/ui components (if not already installed)
npx shadcn-ui@latest add card
npx shadcn-ui@latest add button
npx shadcn-ui@latest add slider
npx shadcn-ui@latest add switch
npx shadcn-ui@latest add checkbox
npx shadcn-ui@latest add select
npx shadcn-ui@latest add input
npx shadcn-ui@latest add label
npx shadcn-ui@latest add badge
npx shadcn-ui@latest add separator
npx shadcn-ui@latest add scroll-area
npx shadcn-ui@latest add alert-dialog
```

## Basic Usage

### 1. Import Components

```typescript
import {
    MemoryGraph3D,
    ControlPanel,
    InspectorPanel,
    TimelineSlider,
    useMemoryGraphData,
} from './view/memorygraph';
```

### 2. Minimal Example

```typescript
export const MemoryGraphView = () => {
    const { loadData } = useMemoryGraphData();

    useEffect(() => {
        loadData();
    }, []);

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            <ControlPanel />
            <MemoryGraph3D nodes={[]} edges={[]} />
            <InspectorPanel />
        </div>
    );
};
```

### 3. Full Layout Example

```typescript
import { useAtomValue } from 'jotai';
import {
    MemoryGraph3D,
    ControlPanel,
    InspectorPanel,
    TimelineSlider,
    PruneMode,
    nodesAtom,
    edgesAtom,
    interactionModeAtom,
} from './view/memorygraph';

export const MemoryGraphView = () => {
    const nodes = useAtomValue(nodesAtom);
    const edges = useAtomValue(edgesAtom);
    const mode = useAtomValue(interactionModeAtom);

    return (
        <div className="memory-graph-layout">
            {/* Left Sidebar: Controls */}
            <aside className="left-sidebar">
                <ControlPanel />
            </aside>

            {/* Center: 3D Graph + Timeline */}
            <main className="graph-container">
                <div className="canvas-wrapper">
                    <MemoryGraph3D
                        nodes={nodes}
                        edges={edges}
                        onNodeSelect={(id) => console.log('Selected:', id)}
                        onNodeDelete={(ids) => console.log('Delete:', ids)}
                    />
                </div>

                <div className="timeline-wrapper">
                    <TimelineSlider />
                </div>
            </main>

            {/* Right Sidebar: Inspector or Prune Mode */}
            <aside className="right-sidebar">
                {mode.mode === 'prune' ? (
                    <PruneMode onPrune={() => console.log('Pruning...')} />
                ) : (
                    <InspectorPanel
                        onEdit={(id) => console.log('Edit:', id)}
                        onDelete={(id) => console.log('Delete:', id)}
                        onNavigate={(id) => console.log('Navigate to:', id)}
                        onClose={() => console.log('Close inspector')}
                    />
                )}
            </aside>
        </div>
    );
};
```

### 4. Corresponding Styles

```scss
.memory-graph-layout {
    display: flex;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
    background: #0f172a;

    .left-sidebar {
        width: 320px;
        height: 100%;
        overflow-y: auto;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .graph-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        height: 100%;
        overflow: hidden;

        .canvas-wrapper {
            flex: 1;
            position: relative;
        }

        .timeline-wrapper {
            flex: 0 0 auto;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
    }

    .right-sidebar {
        width: 380px;
        height: 100%;
        overflow-y: auto;
        border-left: 1px solid rgba(255, 255, 255, 0.1);
    }
}
```

## Common Operations

### Load Data from Backend

```typescript
import { RpcApi } from '@/app/store/wshclientapi';
import { TabRpcClient } from '@/app/store/wshrpcutil';
import { useSetAtom } from 'jotai';
import { nodesAtom, edgesAtom, loadingAtom, errorAtom } from './view/memorygraph';

export const useLoadMemoryGraph = () => {
    const setNodes = useSetAtom(nodesAtom);
    const setEdges = useSetAtom(edgesAtom);
    const setLoading = useSetAtom(loadingAtom);
    const setError = useSetAtom(errorAtom);

    const loadGraph = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            // Fetch nodes
            const nodesResponse = await RpcApi.GetMemoryNodes(TabRpcClient, {});
            setNodes(nodesResponse.nodes);

            // Fetch edges
            const edgesResponse = await RpcApi.GetMemoryEdges(TabRpcClient, {});
            setEdges(edgesResponse.edges);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load graph');
        } finally {
            setLoading(false);
        }
    }, [setNodes, setEdges, setLoading, setError]);

    return { loadGraph };
};
```

### Select Node Programmatically

```typescript
import { useSetAtom } from 'jotai';
import { selectNodeAtom } from './view/memorygraph';

const selectNode = useSetAtom(selectNodeAtom);

// Select a node
selectNode('node-123');

// Clear selection
selectNode(null);
```

### Change Interaction Mode

```typescript
import { useSetAtom } from 'jotai';
import { setInteractionModeAtom } from './view/memorygraph';

const setMode = useSetAtom(setInteractionModeAtom);

// Enter prune mode
setMode('prune');

// Back to explore mode
setMode('explore');

// Timeline mode
setMode('timeline');
```

### Update Filters

```typescript
import { useSetAtom } from 'jotai';
import { updateFiltersAtom } from './view/memorygraph';

const updateFilters = useSetAtom(updateFiltersAtom);

// Filter by memory type
updateFilters({
    memoryTypes: new Set(['episodic', 'semantic'])
});

// Set activity threshold
updateFilters({
    activityThreshold: 0.5
});

// Search
updateFilters({
    searchQuery: 'important memory'
});
```

### Control Timeline

```typescript
import { useSetAtom } from 'jotai';
import { updateTimelineAtom, toggleTimelinePlaybackAtom } from './view/memorygraph';

const updateTimeline = useSetAtom(updateTimelineAtom);
const togglePlayback = useSetAtom(toggleTimelinePlaybackAtom);

// Set time range
updateTimeline({
    startTime: Date.now() - 30 * 24 * 60 * 60 * 1000, // 30 days ago
    endTime: Date.now(),
    currentTime: Date.now() - 30 * 24 * 60 * 60 * 1000
});

// Start playback
togglePlayback();

// Change speed
updateTimeline({
    playbackSpeed: 5.0 // 5x speed
});
```

### Delete Nodes

```typescript
import { RpcApi } from '@/app/store/wshclientapi';
import { TabRpcClient } from '@/app/store/wshrpcutil';
import { useAtomValue, useSetAtom } from 'jotai';
import { selectionStateAtom, clearMultiSelectAtom, nodesAtom } from './view/memorygraph';

const handleDelete = async () => {
    const selection = useAtomValue(selectionStateAtom);
    const clearSelection = useSetAtom(clearMultiSelectAtom);
    const setNodes = useSetAtom(nodesAtom);

    // Get node IDs to delete
    const nodeIds = Array.from(selection.multiSelect);

    if (nodeIds.length === 0) return;

    try {
        // Delete via RPC
        await RpcApi.DeleteMemoryNodes(TabRpcClient, {
            node_ids: nodeIds
        });

        // Update local state
        const currentNodes = useAtomValue(nodesAtom);
        const remainingNodes = currentNodes.filter(n => !nodeIds.includes(n.id));
        setNodes(remainingNodes);

        // Clear selection
        clearSelection();
    } catch (err) {
        console.error('Delete failed:', err);
    }
};
```

### Export Graph

```typescript
import { useExportGraph } from './view/memorygraph';

const exportGraph = useExportGraph();

// Click button to export
<Button onClick={exportGraph}>
    Export Graph
</Button>
```

### Get Graph Metrics

```typescript
import { useAtomValue } from 'jotai';
import { graphMetricsAtom } from './view/memorygraph';

const MyMetricsDisplay = () => {
    const metrics = useAtomValue(graphMetricsAtom);

    return (
        <div>
            <p>Total Nodes: {metrics.totalNodes}</p>
            <p>Visible Nodes: {metrics.visibleNodes}</p>
            <p>Total Edges: {metrics.totalEdges}</p>
            <p>Average Degree: {metrics.averageDegree.toFixed(2)}</p>
            <p>Density: {(metrics.density * 100).toFixed(2)}%</p>
        </div>
    );
};
```

## Customization Examples

### Custom Node Colors

```typescript
// In memorygraph-types.ts
export const MEMORY_TYPE_COLORS: Record<MemoryType, string> = {
    episodic: "#ff0000",    // Red instead of blue
    semantic: "#00ff00",    // Bright green
    procedural: "#0000ff",  // Blue instead of orange
};
```

### Custom Layout Algorithm

```typescript
// Add to useForceLayout hook in memorygraph-hooks.ts
if (layoutAlgorithm === "custom-grid") {
    const gridSize = Math.ceil(Math.sqrt(nodes.length));
    nodes.forEach((node, index) => {
        const row = Math.floor(index / gridSize);
        const col = index % gridSize;
        positions.set(
            node.id,
            new Vector3(col * 10, row * 10, 0)
        );
    });
}
```

### Custom Activation Metric

```typescript
// In useActivationMetrics hook
const activation = 0.3 * recency + 0.3 * frequency + 0.4 * importance;
```

### Add New Edge Type

```typescript
// In memorygraph-types.ts
export type EdgeType =
    | "CAUSED"
    | "SIMILAR_TO"
    | "PREREQUISITE"
    | "CONTRADICTS"
    | "REFERENCES"
    | "DERIVED_FROM"
    | "MY_CUSTOM_TYPE";  // Add new type

export const EDGE_TYPE_COLORS: Record<EdgeType, string> = {
    // ... existing colors ...
    MY_CUSTOM_TYPE: "#00ffff",
};
```

## Debugging Tips

### Enable Atom Debugging

```typescript
import { useAtomValue } from 'jotai';
import { useEffect } from 'react';

// Debug any atom
const DebugAtom = ({ atom, name }) => {
    const value = useAtomValue(atom);

    useEffect(() => {
        console.log(`[${name}]`, value);
    }, [value, name]);

    return null;
};

// Usage
<DebugAtom atom={filteredNodesAtom} name="Filtered Nodes" />
<DebugAtom atom={selectionStateAtom} name="Selection" />
```

### Inspect 3D Scene

```typescript
// Access Three.js scene directly
import { useThree } from '@react-three/fiber';

const SceneInspector = () => {
    const { scene, camera } = useThree();

    useEffect(() => {
        console.log('Scene:', scene);
        console.log('Camera:', camera);
        console.log('Objects:', scene.children);
    }, [scene, camera]);

    return null;
};

// Add to MemoryGraph3D
<SceneInspector />
```

### Performance Monitoring

```typescript
import { useFrame } from '@react-three/fiber';
import { useState } from 'react';

const FPSMonitor = () => {
    const [fps, setFps] = useState(0);
    let lastTime = performance.now();
    let frames = 0;

    useFrame(() => {
        frames++;
        const now = performance.now();
        if (now >= lastTime + 1000) {
            setFps(Math.round((frames * 1000) / (now - lastTime)));
            frames = 0;
            lastTime = now;
        }
    });

    return (
        <div style={{ position: 'absolute', top: 10, left: 10, color: 'white' }}>
            FPS: {fps}
        </div>
    );
};
```

## Troubleshooting

### Nodes not appearing

1. Check if data is loaded:
   ```typescript
   const nodes = useAtomValue(nodesAtom);
   console.log('Nodes:', nodes);
   ```

2. Check if filters are too restrictive:
   ```typescript
   const filtered = useAtomValue(filteredNodesAtom);
   console.log('Filtered nodes:', filtered);
   ```

3. Verify camera position:
   ```typescript
   const { camera } = useThree();
   console.log('Camera position:', camera.position);
   ```

### Poor performance

1. Reduce node count:
   ```typescript
   updateFilters({ importanceThreshold: 0.5 });
   ```

2. Disable labels:
   ```typescript
   updateControls({ showLabels: false });
   ```

3. Switch layout:
   ```typescript
   updateControls({ layoutAlgorithm: 'circular' });
   ```

### State not updating

1. Use hooks in components (not globalStore.get):
   ```typescript
   // ❌ Wrong
   const value = globalStore.get(myAtom);

   // ✅ Correct
   const value = useAtomValue(myAtom);
   ```

2. Check atom dependencies:
   ```typescript
   // Computed atom should list dependencies
   const myAtom = atom((get) => {
       const dep1 = get(dependency1);
       const dep2 = get(dependency2);
       return compute(dep1, dep2);
   });
   ```

## Reference Links

- **Files**: `/mnt/projects/t4d/t4dm/frontend/app/view/memorygraph/`
- **Integration Guide**: `/mnt/projects/t4d/t4dm/MEMORYGRAPH_INTEGRATION.md`
- **Architecture**: `/mnt/projects/t4d/t4dm/MEMORYGRAPH_ARCHITECTURE.md`
- **Three.js Docs**: https://threejs.org/docs/
- **React Three Fiber**: https://docs.pmnd.rs/react-three-fiber/
- **Jotai Docs**: https://jotai.org/
- **shadcn/ui**: https://ui.shadcn.com/

## Quick Command Reference

```bash
# Start development server
cd /mnt/projects/t4d/t4dm/frontend && npm run dev

# Run tests
npm test

# Build for production
npm run build

# Type check
npm run type-check
```

## Next Steps

1. **Connect to backend**: Implement RPC endpoints in Go
2. **Test with real data**: Load actual memory nodes/edges
3. **Customize appearance**: Adjust colors, sizes, effects
4. **Add features**: Node editing, edge creation, export/import
5. **Optimize**: Profile performance, add LOD, implement clustering

---

**You're ready to visualize memory graphs in 3D!**
