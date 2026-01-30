/**
 * Custom hooks for 3D Memory Graph
 */

import { useEffect, useMemo, useCallback, useRef, useState } from "react";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { Vector3 } from "three";
import {
    MemoryNode,
    MemoryEdge,
    NodeVisualConfig,
    EdgeVisualConfig,
    ActivationMetrics,
    LayoutNode,
    MEMORY_TYPE_COLORS,
    EDGE_TYPE_COLORS,
    DEFAULT_NODE_SIZE,
    DEFAULT_EDGE_WIDTH,
    MAX_NODE_SIZE,
    MIN_NODE_SIZE,
    GLOW_THRESHOLD,
} from "./memorygraph-types";
import {
    nodesAtom,
    edgesAtom,
    filteredNodesAtom,
    filteredEdgesAtom,
    timelineStateAtom,
    controlStateAtom,
    updateTimelineAtom,
    loadingAtom,
    errorAtom,
} from "./memorygraph-state";
import {
    wwApi,
    type FSRSState,
    type HebbianWeight,
    type ActivationSpread,
    type SleepConsolidationState,
    type WorkingMemoryState,
    type BiologicalMechanismsResponse,
} from "./api";

// ============================================================================
// Data Loading Hooks
// ============================================================================

/**
 * Load memory graph data from backend
 */
export function useMemoryGraphData() {
    const setNodes = useSetAtom(nodesAtom);
    const setEdges = useSetAtom(edgesAtom);
    const setLoading = useSetAtom(loadingAtom);
    const setError = useSetAtom(errorAtom);
    const controls = useAtomValue(controlStateAtom);

    const loadData = useCallback(async (
        layout: string = "force-directed",
        limit: number = 500
    ) => {
        setLoading(true);
        setError(null);

        try {
            const { nodes, edges } = await wwApi.getGraph(layout, limit, true);

            setNodes(nodes);
            setEdges(edges);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load graph data");
        } finally {
            setLoading(false);
        }
    }, [setNodes, setEdges, setLoading, setError]);

    const refresh = useCallback(() => {
        loadData(controls.layoutAlgorithm);
    }, [loadData, controls.layoutAlgorithm]);

    return { loadData, refresh };
}

/**
 * Load biological mechanism states from backend
 */
export function useBiologicalMechanisms() {
    const [data, setData] = useState<BiologicalMechanismsResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const loadData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await wwApi.getAllBiologicalMechanisms();
            setData(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load biological mechanisms");
        } finally {
            setLoading(false);
        }
    }, []);

    return { data, loading, error, loadData };
}

/**
 * Load FSRS decay states
 */
export function useFSRSStates() {
    const [states, setStates] = useState<FSRSState[]>([]);
    const [loading, setLoading] = useState(false);

    const loadStates = useCallback(async (includeDecayCurve: boolean = true) => {
        setLoading(true);
        try {
            const response = await wwApi.getFSRSStates(50, includeDecayCurve);
            setStates(response);
        } finally {
            setLoading(false);
        }
    }, []);

    return { states, loading, loadStates };
}

/**
 * Load Hebbian synaptic weights
 */
export function useHebbianWeights() {
    const [weights, setWeights] = useState<HebbianWeight[]>([]);
    const [loading, setLoading] = useState(false);

    const loadWeights = useCallback(async (minWeight: number = 0.1) => {
        setLoading(true);
        try {
            const response = await wwApi.getHebbianWeights(100, minWeight);
            setWeights(response);
        } finally {
            setLoading(false);
        }
    }, []);

    return { weights, loading, loadWeights };
}

/**
 * Load activation spreading data
 */
export function useActivationSpreading() {
    const [spreading, setSpreading] = useState<ActivationSpread[]>([]);
    const [loading, setLoading] = useState(false);

    const loadSpreading = useCallback(async (sourceId?: string) => {
        setLoading(true);
        try {
            const response = await wwApi.getActivationSpreading(sourceId);
            setSpreading(response);
        } finally {
            setLoading(false);
        }
    }, []);

    return { spreading, loading, loadSpreading };
}

/**
 * Load sleep consolidation state
 */
export function useSleepConsolidation() {
    const [state, setState] = useState<SleepConsolidationState | null>(null);
    const [loading, setLoading] = useState(false);

    const loadState = useCallback(async () => {
        setLoading(true);
        try {
            const response = await wwApi.getSleepConsolidationState();
            setState(response);
        } finally {
            setLoading(false);
        }
    }, []);

    return { state, loading, loadState };
}

/**
 * Load working memory state
 */
export function useWorkingMemory() {
    const [state, setState] = useState<WorkingMemoryState | null>(null);
    const [loading, setLoading] = useState(false);

    const loadState = useCallback(async () => {
        setLoading(true);
        try {
            const response = await wwApi.getWorkingMemoryState();
            setState(response);
        } finally {
            setLoading(false);
        }
    }, []);

    return { state, loading, loadState };
}

// ============================================================================
// Visual Configuration Hooks
// ============================================================================

/**
 * Compute activation metrics for nodes based on recency and frequency
 */
export function useActivationMetrics(nodes: MemoryNode[]): Map<string, ActivationMetrics> {
    const timeline = useAtomValue(timelineStateAtom);

    return useMemo(() => {
        const metrics = new Map<string, ActivationMetrics>();
        const now = timeline.currentTime;

        nodes.forEach((node) => {
            const daysSinceAccess =
                (now - node.metadata.last_accessed) / (1000 * 60 * 60 * 24);
            const recency = Math.exp(-daysSinceAccess / 30); // 30-day decay

            const frequency = Math.min(node.metadata.access_count / 100, 1); // Normalize to 0-1

            const activation = 0.5 * recency + 0.5 * frequency;

            metrics.set(node.id, {
                nodeId: node.id,
                activation,
                recency,
                frequency,
            });
        });

        return metrics;
    }, [nodes, timeline.currentTime]);
}

/**
 * Compute visual configurations for nodes
 */
export function useNodeVisualConfigs(
    nodes: MemoryNode[],
    activationMetrics: Map<string, ActivationMetrics>
): Map<string, NodeVisualConfig> {
    const controls = useAtomValue(controlStateAtom);

    return useMemo(() => {
        const configs = new Map<string, NodeVisualConfig>();

        nodes.forEach((node) => {
            const metrics = activationMetrics.get(node.id);
            const activation = metrics?.activation || 0;

            // Guard against NaN values from undefined/invalid metadata
            const nodeScale = isFinite(controls.nodeScale) ? controls.nodeScale : 1.0;
            const importance = isFinite(node.metadata?.importance) ? node.metadata.importance : 0.5;

            const baseSize = DEFAULT_NODE_SIZE * nodeScale;
            const sizeMultiplier = MIN_NODE_SIZE + importance * (MAX_NODE_SIZE - MIN_NODE_SIZE);
            const size = isFinite(baseSize * sizeMultiplier) ? baseSize * sizeMultiplier : DEFAULT_NODE_SIZE;

            const color = MEMORY_TYPE_COLORS[node.type] || "#888888";
            const glowIntensity = activation > GLOW_THRESHOLD ? activation : 0;

            configs.set(node.id, {
                baseSize: size,
                sizeMultiplier: isFinite(sizeMultiplier) ? sizeMultiplier : 1.0,
                color,
                glowIntensity: isFinite(glowIntensity) ? glowIntensity : 0,
                opacity: 1.0,
                label: controls.showLabels ? node.content?.slice(0, 50) : undefined,
            });
        });

        return configs;
    }, [nodes, activationMetrics, controls.nodeScale, controls.showLabels]);
}

/**
 * Compute visual configurations for edges
 */
export function useEdgeVisualConfigs(
    edges: MemoryEdge[],
    activationMetrics: Map<string, ActivationMetrics>
): Map<string, EdgeVisualConfig> {
    const controls = useAtomValue(controlStateAtom);
    const timeline = useAtomValue(timelineStateAtom);

    return useMemo(() => {
        const configs = new Map<string, EdgeVisualConfig>();
        const now = timeline.currentTime;

        edges.forEach((edge) => {
            const sourceMetrics = activationMetrics.get(edge.source);
            const targetMetrics = activationMetrics.get(edge.target);

            const isActive =
                (sourceMetrics?.activation || 0) > GLOW_THRESHOLD ||
                (targetMetrics?.activation || 0) > GLOW_THRESHOLD;

            // Guard against NaN from undefined/invalid weight
            const weight = isFinite(edge.weight) ? edge.weight : 0.5;
            const rawWidth = DEFAULT_EDGE_WIDTH * (0.5 + weight * 0.5);
            const width = isFinite(rawWidth) && rawWidth > 0 ? rawWidth : DEFAULT_EDGE_WIDTH;
            const color = EDGE_TYPE_COLORS[edge.type] || "#888888";
            const edgeOpacity = isFinite(controls.edgeOpacity) ? controls.edgeOpacity : 0.6;

            // Check if edge was recently activated
            const lastActivated = edge.metadata?.last_activated || 0;
            const msSinceActivation = now - lastActivated;
            const animated = msSinceActivation < 5000; // Animate for 5 seconds after activation

            configs.set(edge.id, {
                width,
                color,
                opacity: edgeOpacity,
                animated: isActive || animated,
            });
        });

        return configs;
    }, [edges, activationMetrics, controls.edgeOpacity, timeline.currentTime]);
}

// ============================================================================
// Layout Hooks
// ============================================================================

/**
 * Compute 3D force-directed layout for nodes
 */
export function useForceLayout(
    nodes: MemoryNode[],
    edges: MemoryEdge[],
    layoutAlgorithm: string
): Map<string, Vector3> {
    return useMemo(() => {
        const positions = new Map<string, Vector3>();

        // Guard against empty nodes
        if (nodes.length === 0) {
            return positions;
        }

        if (layoutAlgorithm === "force-directed") {
            // Simple force-directed layout (this would be replaced with d3-force-3d or similar)
            nodes.forEach((node, index) => {
                const angle = (index / nodes.length) * Math.PI * 2;
                const radius = 20;
                positions.set(
                    node.id,
                    new Vector3(
                        Math.cos(angle) * radius,
                        Math.sin(angle) * radius,
                        (Math.random() - 0.5) * 10
                    )
                );
            });
        } else if (layoutAlgorithm === "hierarchical") {
            // Hierarchical layout based on edge relationships
            const levels = new Map<string, number>();
            const roots: string[] = [];

            // Find root nodes (no incoming edges)
            const hasIncoming = new Set(edges.map((e) => e.target));
            nodes.forEach((node) => {
                if (!hasIncoming.has(node.id)) {
                    roots.push(node.id);
                    levels.set(node.id, 0);
                }
            });

            // BFS to assign levels
            const queue = [...roots];
            while (queue.length > 0) {
                const nodeId = queue.shift()!;
                const level = levels.get(nodeId) || 0;

                edges
                    .filter((e) => e.source === nodeId)
                    .forEach((edge) => {
                        if (!levels.has(edge.target)) {
                            levels.set(edge.target, level + 1);
                            queue.push(edge.target);
                        }
                    });
            }

            // Position nodes by level
            const nodesByLevel = new Map<number, string[]>();
            levels.forEach((level, nodeId) => {
                if (!nodesByLevel.has(level)) {
                    nodesByLevel.set(level, []);
                }
                nodesByLevel.get(level)!.push(nodeId);
            });

            nodesByLevel.forEach((nodeIds, level) => {
                nodeIds.forEach((nodeId, index) => {
                    const x = (index - nodeIds.length / 2) * 10;
                    const y = level * -15;
                    const z = 0;
                    positions.set(nodeId, new Vector3(x, y, z));
                });
            });

            // Assign fallback positions for disconnected nodes not reached by BFS
            let orphanIndex = 0;
            nodes.forEach((node) => {
                if (!positions.has(node.id)) {
                    const angle = (orphanIndex / nodes.length) * Math.PI * 2;
                    const radius = 30;
                    positions.set(
                        node.id,
                        new Vector3(
                            Math.cos(angle) * radius,
                            -50, // Below the main hierarchy
                            Math.sin(angle) * radius
                        )
                    );
                    orphanIndex++;
                }
            });
        } else if (layoutAlgorithm === "circular") {
            // Circular layout
            const radius = Math.max(nodes.length * 2, 10);
            nodes.forEach((node, index) => {
                const angle = (index / nodes.length) * Math.PI * 2;
                positions.set(
                    node.id,
                    new Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, 0)
                );
            });
        } else {
            // Unknown layout algorithm - fallback to force-directed
            nodes.forEach((node, index) => {
                const angle = (index / nodes.length) * Math.PI * 2;
                const radius = 20;
                positions.set(
                    node.id,
                    new Vector3(
                        Math.cos(angle) * radius,
                        Math.sin(angle) * radius,
                        (Math.random() - 0.5) * 10
                    )
                );
            });
        }

        return positions;
    }, [nodes, edges, layoutAlgorithm]);
}

// ============================================================================
// Timeline Hooks
// ============================================================================

/**
 * Auto-advance timeline when playing
 */
export function useTimelinePlayback() {
    const [timeline, setTimeline] = useAtom(timelineStateAtom);
    const updateTimeline = useSetAtom(updateTimelineAtom);

    useEffect(() => {
        if (!timeline.isPlaying) {
            return;
        }

        const interval = setInterval(() => {
            const increment = 1000 * 60 * 60 * 24 * timeline.playbackSpeed; // Days per tick
            const newTime = timeline.currentTime + increment;

            if (newTime >= timeline.endTime) {
                updateTimeline({
                    currentTime: timeline.endTime,
                    isPlaying: false,
                });
            } else {
                updateTimeline({
                    currentTime: newTime,
                });
            }
        }, 100); // Update every 100ms

        return () => clearInterval(interval);
    }, [timeline, updateTimeline]);
}

// ============================================================================
// Search & Highlight Hooks
// ============================================================================

/**
 * Search nodes and update highlighted set
 */
export function useNodeSearch(
    nodes: MemoryNode[],
    query: string
): { results: MemoryNode[]; highlightIds: Set<string> } {
    return useMemo(() => {
        if (!query) {
            return { results: [], highlightIds: new Set() };
        }

        const lowerQuery = query.toLowerCase();
        const results = nodes.filter((node) => {
            const contentMatch = node.content.toLowerCase().includes(lowerQuery);
            const tagMatch = node.metadata.tags?.some((tag) =>
                tag.toLowerCase().includes(lowerQuery)
            );
            return contentMatch || tagMatch;
        });

        const highlightIds = new Set(results.map((n) => n.id));

        return { results, highlightIds };
    }, [nodes, query]);
}

// ============================================================================
// Camera Hooks
// ============================================================================

/**
 * Focus camera on specific node
 */
export function useFocusNode() {
    const nodes = useAtomValue(filteredNodesAtom);
    const positions = useForceLayout(nodes, [], "force-directed");

    return useCallback(
        (nodeId: string, cameraRef: any) => {
            const position = positions.get(nodeId);
            if (!position || !cameraRef.current) {
                return;
            }

            // Animate camera to node position
            const targetPosition = position.clone().add(new Vector3(0, 0, 10));
            cameraRef.current.position.lerp(targetPosition, 0.1);
            cameraRef.current.lookAt(position);
        },
        [positions]
    );
}

// ============================================================================
// Performance Hooks
// ============================================================================

/**
 * Throttle expensive computations
 */
export function useThrottle<T>(value: T, delay: number): T {
    const [throttledValue, setThrottledValue] = useState(value);
    const lastRan = useRef(Date.now());

    useEffect(() => {
        const handler = setTimeout(() => {
            if (Date.now() - lastRan.current >= delay) {
                setThrottledValue(value);
                lastRan.current = Date.now();
            }
        }, delay - (Date.now() - lastRan.current));

        return () => clearTimeout(handler);
    }, [value, delay]);

    return throttledValue;
}

/**
 * Debounce search queries
 */
export function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => clearTimeout(handler);
    }, [value, delay]);

    return debouncedValue;
}

// ============================================================================
// Interaction Hooks
// ============================================================================

/**
 * Handle node click with mode-specific behavior
 */
export function useNodeClickHandler(
    mode: "explore" | "inspect" | "prune" | "timeline",
    onSelect: (nodeId: string | null) => void,
    onToggleMultiSelect: (nodeId: string) => void
) {
    return useCallback(
        (nodeId: string) => {
            if (mode === "prune") {
                onToggleMultiSelect(nodeId);
            } else {
                onSelect(nodeId);
            }
        },
        [mode, onSelect, onToggleMultiSelect]
    );
}

// ============================================================================
// Export Hooks
// ============================================================================

/**
 * Export graph data to JSON
 */
export function useExportGraph() {
    const nodes = useAtomValue(nodesAtom);
    const edges = useAtomValue(edgesAtom);

    return useCallback(() => {
        const data = {
            nodes: nodes.map((n) => ({
                ...n,
                position: undefined, // Don't export computed positions
            })),
            edges,
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `memory-graph-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }, [nodes, edges]);
}
