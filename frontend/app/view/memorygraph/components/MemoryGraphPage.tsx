/**
 * Memory Graph Page - Main container that loads data and orchestrates components
 */

import React, { useEffect, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { Provider } from "jotai";
import { MemoryGraph3D } from "./MemoryGraph3D";
import { ControlPanel } from "./ControlPanel";
import { InspectorPanel } from "./InspectorPanel";
import { TimelineSlider } from "./TimelineSlider";
import { PruneMode } from "./PruneMode";
import {
    loadingAtom,
    errorAtom,
    nodesAtom,
    edgesAtom,
    selectionStateAtom,
    interactionModeAtom,
    timelineStateAtom,
    selectNodeAtom,
    toggleMultiSelectAtom,
    updateTimelineAtom,
    toggleTimelinePlaybackAtom,
    setInteractionModeAtom,
} from "../memorygraph-state";
import {
    useMemoryGraphData,
    useBiologicalMechanisms,
} from "../memorygraph-hooks";
import "./MemoryGraphPage.scss";

/**
 * Error display component
 */
const ErrorDisplay: React.FC<{ message: string; onRetry: () => void }> = ({
    message,
    onRetry,
}) => (
    <div className="memorygraph-error">
        <div className="error-icon">
            <i className="fa fa-exclamation-triangle" />
        </div>
        <h3>Failed to load memory graph</h3>
        <p>{message}</p>
        <button onClick={onRetry} className="retry-button">
            <i className="fa fa-refresh" /> Retry
        </button>
    </div>
);

/**
 * Loading display component
 */
const LoadingDisplay: React.FC = () => (
    <div className="memorygraph-loading">
        <div className="loading-spinner">
            <i className="fa fa-spinner fa-spin" />
        </div>
        <p>Loading memory graph...</p>
    </div>
);

/**
 * Inner page component (must be inside Jotai Provider)
 */
const MemoryGraphPageInner: React.FC = () => {
    const loading = useAtomValue(loadingAtom);
    const error = useAtomValue(errorAtom);
    const nodes = useAtomValue(nodesAtom);
    const edges = useAtomValue(edgesAtom);
    const selection = useAtomValue(selectionStateAtom);
    const mode = useAtomValue(interactionModeAtom);
    const timeline = useAtomValue(timelineStateAtom);

    const selectNode = useSetAtom(selectNodeAtom);
    const toggleMultiSelect = useSetAtom(toggleMultiSelectAtom);
    const updateTimeline = useSetAtom(updateTimelineAtom);
    const togglePlayback = useSetAtom(toggleTimelinePlaybackAtom);
    const setMode = useSetAtom(setInteractionModeAtom);

    // Data loading
    const { loadData, refresh } = useMemoryGraphData();
    const { data: bioData, loadData: loadBioData } = useBiologicalMechanisms();

    // Load data on mount
    useEffect(() => {
        loadData();
        loadBioData();
    }, [loadData, loadBioData]);

    // Handle node selection
    const handleNodeSelect = (nodeId: string | null) => {
        selectNode(nodeId);
    };

    // Handle prune confirm
    const handlePruneConfirm = async (nodeIds: string[]) => {
        // TODO: Call API to prune nodes
        console.log("Pruning nodes:", nodeIds);
        setMode("explore");
        refresh();
    };

    // Handle prune cancel
    const handlePruneCancel = () => {
        setMode("explore");
    };

    // Handle timeline scrub
    const handleTimelineScrub = (time: number) => {
        updateTimeline({ currentTime: time });
    };

    // Handle playback toggle
    const handlePlaybackToggle = () => {
        togglePlayback();
    };

    // Render loading state
    if (loading && nodes.length === 0) {
        return <LoadingDisplay />;
    }

    // Render error state
    if (error && nodes.length === 0) {
        return <ErrorDisplay message={error} onRetry={refresh} />;
    }

    return (
        <div className="memorygraph-page">
            {/* Main 3D visualization */}
            <div className="memorygraph-main">
                <MemoryGraph3D
                    nodes={nodes}
                    edges={edges}
                    onNodeSelect={handleNodeSelect}
                    className={mode.mode === "prune" ? "prune-mode" : ""}
                />

                {/* Timeline slider */}
                <TimelineSlider
                    startTime={timeline.startTime}
                    endTime={timeline.endTime}
                    currentTime={timeline.currentTime}
                    isPlaying={timeline.isPlaying}
                    playbackSpeed={timeline.playbackSpeed}
                    onScrub={handleTimelineScrub}
                    onTogglePlayback={handlePlaybackToggle}
                    onSpeedChange={(speed) => updateTimeline({ playbackSpeed: speed })}
                />
            </div>

            {/* Right sidebar */}
            <div className="memorygraph-sidebar">
                {/* Control panel */}
                <ControlPanel onRefresh={refresh} />

                {/* Inspector panel (shown when node selected) */}
                {selection.selectedNode && (
                    <InspectorPanel
                        node={nodes.find((n) => n.id === selection.selectedNode) || null}
                        connectedNodes={[]}
                        connectedEdges={edges.filter(
                            (e) =>
                                e.source === selection.selectedNode ||
                                e.target === selection.selectedNode
                        )}
                        onClose={() => selectNode(null)}
                    />
                )}

                {/* Prune mode panel */}
                {mode.mode === "prune" && (
                    <PruneMode
                        selectedNodes={Array.from(selection.multiSelect)}
                        allNodes={nodes}
                        onConfirmPrune={handlePruneConfirm}
                        onCancel={handlePruneCancel}
                        onToggleNode={toggleMultiSelect}
                    />
                )}
            </div>

            {/* Loading overlay (for background refreshes) */}
            {loading && nodes.length > 0 && (
                <div className="loading-overlay">
                    <i className="fa fa-spinner fa-spin" />
                </div>
            )}

            {/* Error toast (for background errors) */}
            {error && nodes.length > 0 && (
                <div className="error-toast">
                    <i className="fa fa-exclamation-circle" />
                    <span>{error}</span>
                    <button onClick={refresh}>Retry</button>
                </div>
            )}
        </div>
    );
};

/**
 * Main Memory Graph Page component with Jotai Provider
 */
export const MemoryGraphPage: React.FC = () => {
    return (
        <Provider>
            <MemoryGraphPageInner />
        </Provider>
    );
};
