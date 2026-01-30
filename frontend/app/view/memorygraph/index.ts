/**
 * Memory Graph 3D Visualization
 *
 * Export barrel for all components, types, and utilities
 */

// Components
export { MemoryGraphPage } from "./components/MemoryGraphPage";
export { MemoryGraph3D } from "./components/MemoryGraph3D";
export { NodeRenderer } from "./components/NodeRenderer";
export { EdgeRenderer } from "./components/EdgeRenderer";
export { ControlPanel } from "./components/ControlPanel";
export { InspectorPanel } from "./components/InspectorPanel";
export { TimelineSlider } from "./components/TimelineSlider";
export { PruneMode } from "./components/PruneMode";

// Types
export type {
    MemoryNode,
    MemoryEdge,
    MemoryType,
    EdgeType,
    NodeVisualConfig,
    EdgeVisualConfig,
    FilterState,
    ControlState,
    SelectionState,
    InteractionMode,
    TimelineState,
    CameraState,
    GraphMetrics,
    MemoryGraph3DProps,
    NodeRendererProps,
    EdgeRendererProps,
    ControlPanelProps,
    InspectorPanelProps,
    TimelineSliderProps,
    PruneModeProps,
} from "./memorygraph-types";

export {
    MEMORY_TYPE_COLORS,
    EDGE_TYPE_COLORS,
    DEFAULT_NODE_SIZE,
    DEFAULT_EDGE_WIDTH,
    MAX_NODE_SIZE,
    MIN_NODE_SIZE,
} from "./memorygraph-types";

// State atoms
export {
    nodesAtom,
    edgesAtom,
    filterStateAtom,
    controlStateAtom,
    selectionStateAtom,
    interactionModeAtom,
    timelineStateAtom,
    cameraStateAtom,
    loadingAtom,
    errorAtom,
    filteredNodesAtom,
    filteredEdgesAtom,
    selectedNodeAtom,
    connectedNodesAtom,
    graphMetricsAtom,
    selectNodeAtom,
    hoverNodeAtom,
    toggleMultiSelectAtom,
    updateFiltersAtom,
    updateControlsAtom,
    resetFiltersAtom,
    resetCameraAtom,
    updateTimelineAtom,
    toggleTimelinePlaybackAtom,
    setInteractionModeAtom,
} from "./memorygraph-state";

// Hooks
export {
    useMemoryGraphData,
    useBiologicalMechanisms,
    useFSRSStates,
    useHebbianWeights,
    useActivationSpreading,
    useSleepConsolidation,
    useWorkingMemory,
    useActivationMetrics,
    useNodeVisualConfigs,
    useEdgeVisualConfigs,
    useForceLayout,
    useTimelinePlayback,
    useNodeSearch,
    useFocusNode,
    useThrottle,
    useDebounce,
    useNodeClickHandler,
    useExportGraph,
} from "./memorygraph-hooks";

// API Client
export { wwApi } from "./api";
export type {
    FSRSState,
    HebbianWeight,
    ActivationSpread,
    SleepConsolidationState,
    WorkingMemoryState,
    BiologicalMechanismsResponse,
    TimelineEvent,
    TimelineResponse,
    ActivityMetrics,
    ActivityResponse,
} from "./api";
