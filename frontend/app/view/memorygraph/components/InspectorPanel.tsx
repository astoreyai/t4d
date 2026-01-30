/**
 * Inspector panel showing selected node details and relationships
 */

import React from "react";
import { useAtomValue } from "jotai";
import {
    selectedNodeAtom,
    connectedNodesAtom,
    incomingEdgesAtom,
    outgoingEdgesAtom,
} from "../memorygraph-state";
import { InspectorPanelProps, MEMORY_TYPE_COLORS, EDGE_TYPE_COLORS } from "../memorygraph-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import "./InspectorPanel.scss";

/**
 * Metadata display component
 */
const MetadataSection: React.FC<{ metadata: any }> = ({ metadata }) => {
    const formatDate = (timestamp: number) => {
        return new Date(timestamp).toLocaleString();
    };

    const formatDuration = (timestamp: number) => {
        const now = Date.now();
        const diff = now - timestamp;
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));

        if (days > 0) {
            return `${days}d ${hours}h ago`;
        } else if (hours > 0) {
            return `${hours}h ago`;
        } else {
            return "Recently";
        }
    };

    return (
        <div className="metadata-section">
            <div className="metadata-item">
                <span className="metadata-label">Created:</span>
                <span className="metadata-value">{formatDate(metadata.created_at)}</span>
            </div>
            <div className="metadata-item">
                <span className="metadata-label">Last Accessed:</span>
                <span className="metadata-value">
                    {formatDuration(metadata.last_accessed)}
                </span>
            </div>
            <div className="metadata-item">
                <span className="metadata-label">Access Count:</span>
                <span className="metadata-value">{metadata.access_count}</span>
            </div>
            <div className="metadata-item">
                <span className="metadata-label">Importance:</span>
                <span className="metadata-value">
                    {(metadata.importance * 100).toFixed(0)}%
                </span>
            </div>
            {metadata.source && (
                <div className="metadata-item">
                    <span className="metadata-label">Source:</span>
                    <span className="metadata-value">{metadata.source}</span>
                </div>
            )}
            {metadata.tags && metadata.tags.length > 0 && (
                <div className="metadata-item">
                    <span className="metadata-label">Tags:</span>
                    <div className="tag-list">
                        {metadata.tags.map((tag: string, idx: number) => (
                            <Badge key={idx} variant="secondary">
                                {tag}
                            </Badge>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

/**
 * Connected nodes list
 */
const ConnectedNodesList: React.FC<{
    nodes: any[];
    edges: any[];
    direction: "incoming" | "outgoing";
    onNavigate: (nodeId: string) => void;
}> = ({ nodes, edges, direction, onNavigate }) => {
    // Create a map of edge types for each connected node
    const edgeTypeMap = new Map<string, string[]>();
    edges.forEach((edge) => {
        const nodeId = direction === "incoming" ? edge.source : edge.target;
        if (!edgeTypeMap.has(nodeId)) {
            edgeTypeMap.set(nodeId, []);
        }
        edgeTypeMap.get(nodeId)!.push(edge.type);
    });

    return (
        <div className="connected-nodes-list">
            {nodes.map((node) => {
                const edgeTypes = edgeTypeMap.get(node.id) || [];
                return (
                    <div
                        key={node.id}
                        className="connected-node-item"
                        onClick={() => onNavigate(node.id)}
                    >
                        <div className="node-indicator">
                            <div
                                className="node-color"
                                style={{ backgroundColor: MEMORY_TYPE_COLORS[node.type] }}
                            />
                            <i
                                className={`fa fa-arrow-${direction === "incoming" ? "right" : "left"}`}
                            ></i>
                        </div>
                        <div className="node-info">
                            <div className="node-content">{node.content.slice(0, 60)}...</div>
                            <div className="edge-types">
                                {edgeTypes.map((type, idx) => (
                                    <Badge
                                        key={idx}
                                        variant="outline"
                                        style={{
                                            borderColor: EDGE_TYPE_COLORS[type],
                                            color: EDGE_TYPE_COLORS[type],
                                        }}
                                    >
                                        {type.replace(/_/g, " ")}
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    </div>
                );
            })}
            {nodes.length === 0 && (
                <div className="empty-state">
                    <i className="fa fa-circle-info"></i>
                    <span>No {direction} connections</span>
                </div>
            )}
        </div>
    );
};

/**
 * Main inspector panel component
 */
export const InspectorPanel: React.FC<InspectorPanelProps> = ({
    node,
    connectedNodes,
    incomingEdges,
    outgoingEdges,
    onEdit,
    onDelete,
    onNavigate,
    onClose,
}) => {
    const selectedNode = useAtomValue(selectedNodeAtom);
    const connected = useAtomValue(connectedNodesAtom);
    const incoming = useAtomValue(incomingEdgesAtom);
    const outgoing = useAtomValue(outgoingEdgesAtom);

    const displayNode = node || selectedNode;
    const displayConnected = connectedNodes || connected;
    const displayIncoming = incomingEdges || incoming;
    const displayOutgoing = outgoingEdges || outgoing;

    if (!displayNode) {
        return (
            <div className="inspector-panel inspector-panel-empty">
                <Card>
                    <CardContent className="empty-state">
                        <i className="fa fa-hand-pointer"></i>
                        <p>Select a node to view details</p>
                    </CardContent>
                </Card>
            </div>
        );
    }

    const incomingNodes = displayConnected.filter((n) =>
        displayIncoming.some((e) => e.source === n.id)
    );
    const outgoingNodes = displayConnected.filter((n) =>
        displayOutgoing.some((e) => e.target === n.id)
    );

    return (
        <div className="inspector-panel">
            <Card>
                <CardHeader>
                    <div className="header-row">
                        <CardTitle>
                            <div
                                className="type-indicator"
                                style={{ backgroundColor: MEMORY_TYPE_COLORS[displayNode.type] }}
                            />
                            Memory Details
                        </CardTitle>
                        <Button variant="ghost" size="sm" onClick={onClose}>
                            <i className="fa fa-xmark"></i>
                        </Button>
                    </div>
                </CardHeader>
                <CardContent className="inspector-content">
                    <ScrollArea className="scroll-area">
                        {/* Memory Type Badge */}
                        <div className="type-badge-container">
                            <Badge
                                style={{
                                    backgroundColor: MEMORY_TYPE_COLORS[displayNode.type],
                                    color: "white",
                                }}
                            >
                                {displayNode.type.toUpperCase()}
                            </Badge>
                        </div>

                        {/* Content */}
                        <div className="content-section">
                            <h4>Content</h4>
                            <p className="content-text">{displayNode.content}</p>
                        </div>

                        <Separator />

                        {/* Metadata */}
                        <div className="metadata-container">
                            <h4>Metadata</h4>
                            <MetadataSection metadata={displayNode.metadata} />
                        </div>

                        <Separator />

                        {/* Incoming Connections */}
                        <div className="connections-section">
                            <h4>
                                Incoming Connections
                                <Badge variant="secondary">{incomingNodes.length}</Badge>
                            </h4>
                            <ConnectedNodesList
                                nodes={incomingNodes}
                                edges={displayIncoming}
                                direction="incoming"
                                onNavigate={onNavigate}
                            />
                        </div>

                        <Separator />

                        {/* Outgoing Connections */}
                        <div className="connections-section">
                            <h4>
                                Outgoing Connections
                                <Badge variant="secondary">{outgoingNodes.length}</Badge>
                            </h4>
                            <ConnectedNodesList
                                nodes={outgoingNodes}
                                edges={displayOutgoing}
                                direction="outgoing"
                                onNavigate={onNavigate}
                            />
                        </div>

                        <Separator />

                        {/* Action Buttons */}
                        <div className="action-buttons">
                            <Button
                                variant="outline"
                                className="full-width"
                                onClick={() => onEdit(displayNode.id)}
                            >
                                <i className="fa fa-pencil mr-2"></i>
                                Edit
                            </Button>
                            <Button
                                variant="destructive"
                                className="full-width"
                                onClick={() => onDelete(displayNode.id)}
                            >
                                <i className="fa fa-trash mr-2"></i>
                                Delete
                            </Button>
                        </div>
                    </ScrollArea>
                </CardContent>
            </Card>
        </div>
    );
};
