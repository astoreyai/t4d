/**
 * Jotai state management for 3D Memory Graph
 */

import { atom } from "jotai";
import { atomWithStorage, createJSONStorage } from "jotai/utils";
import {
    MemoryNode,
    MemoryEdge,
    FilterState,
    ControlState,
    SelectionState,
    InteractionMode,
    TimelineState,
    CameraState,
    GraphMetrics,
    MemoryType,
    EdgeType,
} from "./memorygraph-types";

// ============================================================================
// Custom storage for Sets (JSON doesn't serialize Sets properly)
// ============================================================================

interface SerializedFilterState {
    memoryTypes: MemoryType[];
    edgeTypes: EdgeType[];
    timeRange: { start: number; end: number };
    activityThreshold: number;
    importanceThreshold: number;
    searchQuery: string;
    highlightedNodes: string[];
}

const filterStorage = createJSONStorage<FilterState>(() => localStorage);

// Override getItem and setItem to handle Set serialization
const filterStorageWithSets = {
    ...filterStorage,
    getItem: (key: string, initialValue: FilterState): FilterState => {
        try {
            const stored = localStorage.getItem(key);
            if (!stored) return initialValue;

            const parsed: SerializedFilterState = JSON.parse(stored);
            return {
                memoryTypes: new Set(parsed.memoryTypes || ["episodic", "semantic", "procedural"]),
                edgeTypes: new Set(parsed.edgeTypes || ["CAUSED", "SIMILAR_TO", "PREREQUISITE", "CONTRADICTS", "REFERENCES", "DERIVED_FROM"]),
                timeRange: parsed.timeRange || { start: 0, end: Date.now() },
                activityThreshold: parsed.activityThreshold ?? 0,
                importanceThreshold: parsed.importanceThreshold ?? 0,
                searchQuery: parsed.searchQuery || "",
                highlightedNodes: new Set(parsed.highlightedNodes || []),
            };
        } catch {
            return initialValue;
        }
    },
    setItem: (key: string, value: FilterState): void => {
        const serialized: SerializedFilterState = {
            memoryTypes: Array.from(value.memoryTypes),
            edgeTypes: Array.from(value.edgeTypes),
            timeRange: value.timeRange,
            activityThreshold: value.activityThreshold,
            importanceThreshold: value.importanceThreshold,
            searchQuery: value.searchQuery,
            highlightedNodes: Array.from(value.highlightedNodes),
        };
        localStorage.setItem(key, JSON.stringify(serialized));
    },
    removeItem: (key: string): void => {
        localStorage.removeItem(key);
    },
};

// ============================================================================
// Data Atoms
// ============================================================================

export const nodesAtom = atom<MemoryNode[]>([]);
export const edgesAtom = atom<MemoryEdge[]>([]);

// ============================================================================
// Filter State
// ============================================================================

export const filterStateAtom = atomWithStorage<FilterState>(
    "memorygraph-filters",
    {
        memoryTypes: new Set<MemoryType>(["episodic", "semantic", "procedural"]),
        edgeTypes: new Set<EdgeType>([
            "CAUSED",
            "SIMILAR_TO",
            "PREREQUISITE",
            "CONTRADICTS",
            "REFERENCES",
            "DERIVED_FROM",
        ]),
        timeRange: {
            start: 0,
            end: Date.now(),
        },
        activityThreshold: 0,
        importanceThreshold: 0,
        searchQuery: "",
        highlightedNodes: new Set<string>(),
    },
    filterStorageWithSets
);

// ============================================================================
// Control State
// ============================================================================

export const controlStateAtom = atomWithStorage<ControlState>("memorygraph-controls", {
    autoRotate: false,
    showLabels: true,
    showEdges: true,
    edgeOpacity: 0.6,
    nodeScale: 1.0,
    animationSpeed: 1.0,
    layoutAlgorithm: "force-directed",
});

// ============================================================================
// Selection State
// ============================================================================

export const selectionStateAtom = atom<SelectionState>({
    selectedNode: null,
    hoveredNode: null,
    selectedEdge: null,
    hoveredEdge: null,
    multiSelect: new Set<string>(),
});

// ============================================================================
// Interaction Mode
// ============================================================================

export const interactionModeAtom = atom<InteractionMode>({
    mode: "explore",
    isPanning: false,
    isRotating: false,
});

// ============================================================================
// Timeline State
// ============================================================================

export const timelineStateAtom = atom<TimelineState>({
    currentTime: Date.now(),
    isPlaying: false,
    playbackSpeed: 1.0,
    startTime: 0,
    endTime: Date.now(),
});

// ============================================================================
// Camera State
// ============================================================================

export const cameraStateAtom = atom<CameraState>({
    position: [0, 0, 50],
    target: [0, 0, 0],
    zoom: 1,
});

// ============================================================================
// Loading & Error State
// ============================================================================

export const loadingAtom = atom<boolean>(false);
export const errorAtom = atom<string | null>(null);

// ============================================================================
// Computed Atoms
// ============================================================================

/**
 * Filtered nodes based on current filter state
 */
export const filteredNodesAtom = atom((get) => {
    const nodes = get(nodesAtom);
    const filters = get(filterStateAtom);
    const timeline = get(timelineStateAtom);

    // Defensive: ensure memoryTypes is a Set (localStorage corruption protection)
    const memoryTypes = filters.memoryTypes instanceof Set
        ? filters.memoryTypes
        : new Set<MemoryType>(["episodic", "semantic", "procedural"]);

    return nodes.filter((node) => {
        // Filter by type
        if (!memoryTypes.has(node.type)) {
            return false;
        }

        // Filter by time range
        const createdAt = node.metadata.created_at;
        if (createdAt < filters.timeRange.start || createdAt > filters.timeRange.end) {
            return false;
        }

        // Filter by timeline (if playing)
        if (timeline.isPlaying && createdAt > timeline.currentTime) {
            return false;
        }

        // Filter by importance
        if (node.metadata.importance < filters.importanceThreshold) {
            return false;
        }

        // Filter by activity (recency-based activation)
        const now = timeline.currentTime;
        const lastAccessed = node.metadata.last_accessed;
        const daysSinceAccess = (now - lastAccessed) / (1000 * 60 * 60 * 24);
        const recencyScore = Math.exp(-daysSinceAccess / 30); // Decay over 30 days

        if (recencyScore < filters.activityThreshold) {
            return false;
        }

        // Filter by search query
        if (filters.searchQuery) {
            const query = filters.searchQuery.toLowerCase();
            const contentMatch = node.content.toLowerCase().includes(query);
            const tagMatch = node.metadata.tags?.some((tag) => tag.toLowerCase().includes(query));
            if (!contentMatch && !tagMatch) {
                return false;
            }
        }

        return true;
    });
});

/**
 * Filtered edges based on filtered nodes and filter state
 */
export const filteredEdgesAtom = atom((get) => {
    const edges = get(edgesAtom);
    const filteredNodes = get(filteredNodesAtom);
    const filters = get(filterStateAtom);

    const visibleNodeIds = new Set(filteredNodes.map((n) => n.id));

    // Defensive: ensure edgeTypes is a Set (localStorage corruption protection)
    const edgeTypes = filters.edgeTypes instanceof Set
        ? filters.edgeTypes
        : new Set<EdgeType>(["CAUSED", "SIMILAR_TO", "PREREQUISITE", "CONTRADICTS", "REFERENCES", "DERIVED_FROM"]);

    return edges.filter((edge) => {
        // Only show edges between visible nodes
        if (!visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target)) {
            return false;
        }

        // Filter by edge type
        if (!edgeTypes.has(edge.type)) {
            return false;
        }

        return true;
    });
});

/**
 * Selected node with full details
 */
export const selectedNodeAtom = atom((get) => {
    const selection = get(selectionStateAtom);
    const nodes = get(filteredNodesAtom);

    if (!selection.selectedNode) {
        return null;
    }

    return nodes.find((n) => n.id === selection.selectedNode) || null;
});

/**
 * Connected nodes for selected node
 */
export const connectedNodesAtom = atom((get) => {
    const selection = get(selectionStateAtom);
    const nodes = get(filteredNodesAtom);
    const edges = get(filteredEdgesAtom);

    if (!selection.selectedNode) {
        return [];
    }

    const connectedIds = new Set<string>();
    edges.forEach((edge) => {
        if (edge.source === selection.selectedNode) {
            connectedIds.add(edge.target);
        }
        if (edge.target === selection.selectedNode) {
            connectedIds.add(edge.source);
        }
    });

    return nodes.filter((n) => connectedIds.has(n.id));
});

/**
 * Incoming edges for selected node
 */
export const incomingEdgesAtom = atom((get) => {
    const selection = get(selectionStateAtom);
    const edges = get(filteredEdgesAtom);

    if (!selection.selectedNode) {
        return [];
    }

    return edges.filter((e) => e.target === selection.selectedNode);
});

/**
 * Outgoing edges for selected node
 */
export const outgoingEdgesAtom = atom((get) => {
    const selection = get(selectionStateAtom);
    const edges = get(filteredEdgesAtom);

    if (!selection.selectedNode) {
        return [];
    }

    return edges.filter((e) => e.source === selection.selectedNode);
});

/**
 * Graph metrics for current view
 */
export const graphMetricsAtom = atom<GraphMetrics>((get) => {
    const allNodes = get(nodesAtom);
    const allEdges = get(edgesAtom);
    const visibleNodes = get(filteredNodesAtom);
    const visibleEdges = get(filteredEdgesAtom);
    const selection = get(selectionStateAtom);

    const visibleNodeCount = visibleNodes.length;
    const visibleEdgeCount = visibleEdges.length;

    const averageDegree =
        visibleNodeCount > 0 ? (visibleEdgeCount * 2) / visibleNodeCount : 0;

    const maxPossibleEdges = (visibleNodeCount * (visibleNodeCount - 1)) / 2;
    const density = maxPossibleEdges > 0 ? visibleEdgeCount / maxPossibleEdges : 0;

    return {
        totalNodes: allNodes.length,
        totalEdges: allEdges.length,
        visibleNodes: visibleNodeCount,
        visibleEdges: visibleEdgeCount,
        selectedCount: selection.multiSelect.size,
        averageDegree,
        density,
    };
});

/**
 * Highlighted nodes (from search or other interactions)
 */
export const highlightedNodesAtom = atom((get) => {
    const filters = get(filterStateAtom);
    return filters.highlightedNodes;
});

// ============================================================================
// Action Atoms (Writable)
// ============================================================================

/**
 * Select a node
 */
export const selectNodeAtom = atom(null, (get, set, nodeId: string | null) => {
    const selection = get(selectionStateAtom);
    set(selectionStateAtom, {
        ...selection,
        selectedNode: nodeId,
    });
});

/**
 * Hover a node
 */
export const hoverNodeAtom = atom(null, (get, set, nodeId: string | null) => {
    const selection = get(selectionStateAtom);
    set(selectionStateAtom, {
        ...selection,
        hoveredNode: nodeId,
    });
});

/**
 * Toggle node in multi-select (for prune mode)
 */
export const toggleMultiSelectAtom = atom(null, (get, set, nodeId: string) => {
    const selection = get(selectionStateAtom);
    const newMultiSelect = new Set(selection.multiSelect);

    if (newMultiSelect.has(nodeId)) {
        newMultiSelect.delete(nodeId);
    } else {
        newMultiSelect.add(nodeId);
    }

    set(selectionStateAtom, {
        ...selection,
        multiSelect: newMultiSelect,
    });
});

/**
 * Clear multi-select
 */
export const clearMultiSelectAtom = atom(null, (get, set) => {
    const selection = get(selectionStateAtom);
    set(selectionStateAtom, {
        ...selection,
        multiSelect: new Set(),
    });
});

/**
 * Update filter state
 */
export const updateFiltersAtom = atom(null, (get, set, updates: Partial<FilterState>) => {
    const current = get(filterStateAtom);
    set(filterStateAtom, {
        ...current,
        ...updates,
    });
});

/**
 * Update control state
 */
export const updateControlsAtom = atom(null, (get, set, updates: Partial<ControlState>) => {
    const current = get(controlStateAtom);
    set(controlStateAtom, {
        ...current,
        ...updates,
    });
});

/**
 * Reset filters to defaults
 */
export const resetFiltersAtom = atom(null, (get, set) => {
    set(filterStateAtom, {
        memoryTypes: new Set<MemoryType>(["episodic", "semantic", "procedural"]),
        edgeTypes: new Set<EdgeType>([
            "CAUSED",
            "SIMILAR_TO",
            "PREREQUISITE",
            "CONTRADICTS",
            "REFERENCES",
            "DERIVED_FROM",
        ]),
        timeRange: {
            start: 0,
            end: Date.now(),
        },
        activityThreshold: 0,
        importanceThreshold: 0,
        searchQuery: "",
        highlightedNodes: new Set(),
    });
});

/**
 * Reset camera to default position
 */
export const resetCameraAtom = atom(null, (get, set) => {
    set(cameraStateAtom, {
        position: [0, 0, 50],
        target: [0, 0, 0],
        zoom: 1,
    });
});

/**
 * Update timeline state
 */
export const updateTimelineAtom = atom(null, (get, set, updates: Partial<TimelineState>) => {
    const current = get(timelineStateAtom);
    set(timelineStateAtom, {
        ...current,
        ...updates,
    });
});

/**
 * Toggle timeline playback
 */
export const toggleTimelinePlaybackAtom = atom(null, (get, set) => {
    const timeline = get(timelineStateAtom);
    set(timelineStateAtom, {
        ...timeline,
        isPlaying: !timeline.isPlaying,
    });
});

/**
 * Change interaction mode
 */
export const setInteractionModeAtom = atom(
    null,
    (get, set, mode: InteractionMode["mode"]) => {
        const current = get(interactionModeAtom);
        set(interactionModeAtom, {
            ...current,
            mode,
        });

        // Clear multi-select when leaving prune mode
        if (mode !== "prune") {
            set(clearMultiSelectAtom);
        }
    }
);
