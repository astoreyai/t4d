/**
 * Main 3D Memory Graph container using React Three Fiber
 */

import React, { Suspense, useRef, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useAtomValue, useSetAtom } from "jotai";
import {
    filteredNodesAtom,
    filteredEdgesAtom,
    controlStateAtom,
    selectionStateAtom,
    interactionModeAtom,
    selectNodeAtom,
    hoverNodeAtom,
    toggleMultiSelectAtom,
    resetCameraAtom,
    loadingAtom,
    nodesAtom,
} from "../memorygraph-state";
import {
    useActivationMetrics,
    useNodeVisualConfigs,
    useEdgeVisualConfigs,
    useForceLayout,
    useTimelinePlayback,
    useNodeClickHandler,
} from "../memorygraph-hooks";
import { NodeRenderer } from "./NodeRenderer";
import { EdgeRenderer } from "./EdgeRenderer";
import { MemoryGraph3DProps } from "../memorygraph-types";
import "./MemoryGraph3D.scss";

/**
 * Loading fallback component
 */
const LoadingFallback: React.FC = () => {
    return (
        <div className="memorygraph-loading">
            <div className="loading-spinner">
                <i className="fa fa-spinner fa-spin"></i>
            </div>
            <p>Loading memory graph...</p>
        </div>
    );
};

/**
 * Scene component containing the 3D graph
 */
const Scene: React.FC = () => {
    const nodes = useAtomValue(filteredNodesAtom);
    const edges = useAtomValue(filteredEdgesAtom);
    const controls = useAtomValue(controlStateAtom);
    const selection = useAtomValue(selectionStateAtom);
    const mode = useAtomValue(interactionModeAtom);

    const selectNode = useSetAtom(selectNodeAtom);
    const hoverNode = useSetAtom(hoverNodeAtom);
    const toggleMultiSelect = useSetAtom(toggleMultiSelectAtom);

    // Compute visual configurations
    const activationMetrics = useActivationMetrics(nodes);
    const nodeVisualConfigs = useNodeVisualConfigs(nodes, activationMetrics);
    const edgeVisualConfigs = useEdgeVisualConfigs(edges, activationMetrics);

    // Compute layout positions
    const positions = useForceLayout(nodes, edges, controls.layoutAlgorithm);

    // Update node positions (layout positions take precedence, but fall back to API positions)
    // Filter out nodes without valid positions to prevent THREE.js NaN errors
    const nodesWithPositions = nodes
        .map((node) => ({
            ...node,
            position: positions.get(node.id) || node.position,
        }))
        .filter((node) => {
            // Only include nodes with valid Vector3 positions
            if (!node.position) return false;
            const pos = node.position;
            return isFinite(pos.x) && isFinite(pos.y) && isFinite(pos.z);
        });

    // Handle node clicks based on interaction mode
    const handleNodeClick = useNodeClickHandler(
        mode.mode,
        selectNode,
        toggleMultiSelect
    );

    return (
        <>
            {/* Ambient lighting */}
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={0.8} />
            <pointLight position={[-10, -10, -10]} intensity={0.3} />

            {/* Grid helper - temporarily disabled to debug NaN errors
            <Grid
                args={[100, 100]}
                cellSize={5}
                cellThickness={0.5}
                cellColor="#6366f1"
                sectionSize={20}
                sectionThickness={1}
                sectionColor="#8b5cf6"
                fadeDistance={100}
                fadeStrength={1}
                followCamera={false}
            />
            */}

            {/* Render edges - only edges between valid nodes */}
            {controls.showEdges && nodesWithPositions.length > 0 && (
                <EdgeRenderer
                    edges={edges}
                    nodes={nodesWithPositions}
                    visualConfigs={edgeVisualConfigs}
                    selectedEdge={selection.selectedEdge}
                    hoveredEdge={selection.hoveredEdge}
                    showEdges={controls.showEdges}
                    edgeOpacity={controls.edgeOpacity}
                />
            )}

            {/* Render nodes - only if we have valid nodes */}
            {nodesWithPositions.length > 0 && (
                <NodeRenderer
                    nodes={nodesWithPositions}
                    visualConfigs={nodeVisualConfigs}
                    selectedNode={selection.selectedNode}
                    hoveredNode={selection.hoveredNode}
                    highlightedNodes={new Set()}
                    multiSelectNodes={selection.multiSelect}
                    showLabels={controls.showLabels}
                    onNodeClick={handleNodeClick}
                    onNodeHover={hoverNode}
                />
            )}
        </>
    );
};

/**
 * Main Memory Graph 3D component
 */
export const MemoryGraph3D: React.FC<MemoryGraph3DProps> = ({
    nodes,
    edges,
    onNodeSelect,
    onNodeEdit,
    onNodeDelete,
    onEdgeCreate,
    className,
}) => {
    const controls = useAtomValue(controlStateAtom);
    const loading = useAtomValue(loadingAtom);
    const allNodes = useAtomValue(nodesAtom);
    const controlsRef = useRef<any>(null);

    // Enable timeline playback
    useTimelinePlayback();

    // Handle camera reset
    const resetCamera = useSetAtom(resetCameraAtom);
    useEffect(() => {
        if (controlsRef.current) {
            controlsRef.current.reset();
        }
    }, [resetCamera]);

    // Show loading state while fetching data
    if (loading) {
        return (
            <div className={`memorygraph-3d ${className || ""}`}>
                <LoadingFallback />
            </div>
        );
    }

    // Show empty state if no data after loading
    if (allNodes.length === 0) {
        return (
            <div className={`memorygraph-3d ${className || ""}`}>
                <div className="memorygraph-empty">
                    <i className="fa fa-database"></i>
                    <p>No memory data available</p>
                    <p className="text-muted">Memory nodes will appear here when created</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`memorygraph-3d ${className || ""}`}>
            <Canvas
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: "low-power",
                    failIfMajorPerformanceCaveat: false,
                }}
                camera={{
                    position: [0, 0, 50],
                    fov: 60,
                    near: 0.1,
                    far: 1000,
                }}
                frameloop="demand"
                onCreated={(state) => {
                    // Disable automatic bounding sphere computation for raycasting
                    state.gl.localClippingEnabled = false;
                }}
            >
                <Suspense fallback={null}>
                    <OrbitControls
                        ref={controlsRef}
                        enableDamping={false}
                        rotateSpeed={0.5}
                        zoomSpeed={0.8}
                        panSpeed={0.8}
                        autoRotate={controls.autoRotate}
                        autoRotateSpeed={0.5}
                        minDistance={5}
                        maxDistance={200}
                        makeDefault
                    />

                    <Scene />
                </Suspense>
            </Canvas>

            <Suspense fallback={<LoadingFallback />}>
                {/* Additional overlays go here */}
            </Suspense>
        </div>
    );
};
