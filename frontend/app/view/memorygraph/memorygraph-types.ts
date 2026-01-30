/**
 * Type definitions for 3D Memory Graph visualization
 */

import { Vector3 } from "three";

// ============================================================================
// Memory Node Types
// ============================================================================

export type MemoryType = "episodic" | "semantic" | "procedural";

export interface MemoryNode {
    id: string;
    type: MemoryType;
    content: string;
    metadata: {
        created_at: number;
        last_accessed: number;
        access_count: number;
        importance: number; // 0-1
        tags?: string[];
        source?: string;
    };
    position?: Vector3; // Computed by layout algorithm
}

export interface MemoryEdge {
    id: string;
    source: string;
    target: string;
    type: EdgeType;
    weight: number; // 0-1
    metadata?: {
        created_at: number;
        last_activated?: number;
    };
}

export type EdgeType =
    | "CAUSED"
    | "SIMILAR_TO"
    | "PREREQUISITE"
    | "CONTRADICTS"
    | "REFERENCES"
    | "DERIVED_FROM";

// ============================================================================
// Visual Configuration
// ============================================================================

export interface NodeVisualConfig {
    baseSize: number;
    sizeMultiplier: number; // Based on importance
    color: string; // Base color for type
    glowIntensity: number; // 0-1, based on activation
    opacity: number;
    label?: string;
}

export interface EdgeVisualConfig {
    width: number; // Based on weight
    color: string; // Based on type
    opacity: number;
    animated: boolean; // For active traces
    dashSize?: number;
    gapSize?: number;
}

// ============================================================================
// Filter & Control State
// ============================================================================

export interface FilterState {
    memoryTypes: Set<MemoryType>;
    edgeTypes: Set<EdgeType>;
    timeRange: {
        start: number;
        end: number;
    };
    activityThreshold: number; // 0-1
    importanceThreshold: number; // 0-1
    searchQuery: string;
    highlightedNodes: Set<string>;
}

export interface CameraState {
    position: [number, number, number];
    target: [number, number, number];
    zoom: number;
}

export interface ControlState {
    autoRotate: boolean;
    showLabels: boolean;
    showEdges: boolean;
    edgeOpacity: number;
    nodeScale: number;
    animationSpeed: number;
    layoutAlgorithm: "force-directed" | "hierarchical" | "circular";
}

// ============================================================================
// Interaction State
// ============================================================================

export interface SelectionState {
    selectedNode: string | null;
    hoveredNode: string | null;
    selectedEdge: string | null;
    hoveredEdge: string | null;
    multiSelect: Set<string>; // For prune mode
}

export interface InteractionMode {
    mode: "explore" | "inspect" | "prune" | "timeline";
    isPanning: boolean;
    isRotating: boolean;
}

// ============================================================================
// Timeline State
// ============================================================================

export interface TimelineState {
    currentTime: number; // Unix timestamp
    isPlaying: boolean;
    playbackSpeed: number;
    startTime: number;
    endTime: number;
}

// ============================================================================
// Graph Layout
// ============================================================================

export interface LayoutNode {
    id: string;
    x: number;
    y: number;
    z: number;
    vx?: number;
    vy?: number;
    vz?: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
}

export interface LayoutEdge {
    source: string;
    target: string;
    weight: number;
}

export interface ForceSimulationConfig {
    iterations: number;
    alphaDecay: number;
    linkStrength: number;
    chargeStrength: number;
    centerStrength: number;
}

// ============================================================================
// Component Props
// ============================================================================

export interface MemoryGraph3DProps {
    nodes: MemoryNode[];
    edges: MemoryEdge[];
    onNodeSelect?: (nodeId: string | null) => void;
    onNodeEdit?: (nodeId: string) => void;
    onNodeDelete?: (nodeIds: string[]) => void;
    onEdgeCreate?: (source: string, target: string, type: EdgeType) => void;
    className?: string;
}

export interface NodeRendererProps {
    nodes: MemoryNode[];
    visualConfigs: Map<string, NodeVisualConfig>;
    selectedNode: string | null;
    hoveredNode: string | null;
    highlightedNodes: Set<string>;
    multiSelectNodes: Set<string>;
    showLabels: boolean;
    onNodeClick: (nodeId: string) => void;
    onNodeHover: (nodeId: string | null) => void;
}

export interface EdgeRendererProps {
    edges: MemoryEdge[];
    nodes: MemoryNode[];
    visualConfigs: Map<string, EdgeVisualConfig>;
    selectedEdge: string | null;
    hoveredEdge: string | null;
    showEdges: boolean;
    edgeOpacity: number;
    onEdgeClick?: (edgeId: string) => void;
    onEdgeHover?: (edgeId: string | null) => void;
}

export interface ControlPanelProps {
    filterState: FilterState;
    controlState: ControlState;
    onFilterChange: (filters: Partial<FilterState>) => void;
    onControlChange: (controls: Partial<ControlState>) => void;
    onResetCamera: () => void;
    onResetFilters: () => void;
}

export interface InspectorPanelProps {
    node: MemoryNode | null;
    connectedNodes: MemoryNode[];
    incomingEdges: MemoryEdge[];
    outgoingEdges: MemoryEdge[];
    onEdit: (nodeId: string) => void;
    onDelete: (nodeId: string) => void;
    onNavigate: (nodeId: string) => void;
    onClose: () => void;
}

export interface TimelineSliderProps {
    timelineState: TimelineState;
    onTimeChange: (time: number) => void;
    onPlayPause: () => void;
    onSpeedChange: (speed: number) => void;
}

export interface PruneModeProps {
    selectedNodes: Set<string>;
    nodes: MemoryNode[];
    onSelectionChange: (nodeIds: Set<string>) => void;
    onPrune: () => void;
    onCancel: () => void;
}

// ============================================================================
// Utility Types
// ============================================================================

export interface ActivationMetrics {
    nodeId: string;
    activation: number; // 0-1
    recency: number; // 0-1
    frequency: number; // 0-1
}

export interface GraphMetrics {
    totalNodes: number;
    totalEdges: number;
    visibleNodes: number;
    visibleEdges: number;
    selectedCount: number;
    averageDegree: number;
    density: number;
}

// ============================================================================
// Color Schemes
// ============================================================================

export const MEMORY_TYPE_COLORS: Record<MemoryType, string> = {
    episodic: "#3b82f6", // blue
    semantic: "#10b981", // green
    procedural: "#f97316", // orange
};

export const EDGE_TYPE_COLORS: Record<EdgeType, string> = {
    CAUSED: "#ef4444", // red
    SIMILAR_TO: "#8b5cf6", // purple
    PREREQUISITE: "#eab308", // yellow
    CONTRADICTS: "#dc2626", // dark red
    REFERENCES: "#06b6d4", // cyan
    DERIVED_FROM: "#ec4899", // pink
};

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_NODE_SIZE = 0.5;
export const DEFAULT_EDGE_WIDTH = 0.1;
export const MAX_NODE_SIZE = 2.0;
export const MIN_NODE_SIZE = 0.2;
export const GLOW_THRESHOLD = 0.5; // Activation threshold for glow
export const LABEL_DISTANCE = 15; // Distance at which labels appear
export const ANIMATION_SPEED = 1.0;
export const FORCE_SIMULATION_ITERATIONS = 300;
