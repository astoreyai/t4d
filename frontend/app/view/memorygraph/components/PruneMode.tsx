/**
 * Prune mode for batch deletion of memory nodes
 */

import React from "react";
import { useAtom, useAtomValue } from "jotai";
import {
    selectionStateAtom,
    filteredNodesAtom,
    toggleMultiSelectAtom,
    clearMultiSelectAtom,
    setInteractionModeAtom,
} from "../memorygraph-state";
import { PruneModeProps } from "../memorygraph-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import "./PruneMode.scss";

/**
 * Prune mode component
 */
export const PruneMode: React.FC<PruneModeProps> = ({
    selectedNodes,
    nodes,
    onSelectionChange,
    onPrune,
    onCancel,
}) => {
    const [selection, setSelection] = useAtom(selectionStateAtom);
    const filteredNodes = useAtomValue(filteredNodesAtom);
    const toggleMultiSelect = useAtom(toggleMultiSelectAtom)[1];
    const clearMultiSelect = useAtom(clearMultiSelectAtom)[1];
    const setInteractionMode = useAtom(setInteractionModeAtom)[1];

    const [showConfirmDialog, setShowConfirmDialog] = React.useState(false);

    const displayNodes = nodes || filteredNodes;
    const displaySelection = selectedNodes || selection.multiSelect;

    const handleSelectAll = () => {
        const allIds = new Set(displayNodes.map((n) => n.id));
        setSelection({ ...selection, multiSelect: allIds });
        onSelectionChange?.(allIds);
    };

    const handleClearSelection = () => {
        clearMultiSelect();
        onSelectionChange?.(new Set());
    };

    const handleToggleNode = (nodeId: string) => {
        toggleMultiSelect(nodeId);
    };

    const handlePrune = () => {
        setShowConfirmDialog(true);
    };

    const handleConfirmPrune = () => {
        onPrune();
        setShowConfirmDialog(false);
        handleCancel();
    };

    const handleCancel = () => {
        clearMultiSelect();
        setInteractionMode("explore");
        onCancel?.();
    };

    // Calculate statistics
    const totalMemory = displayNodes.reduce((sum, node) => {
        // Estimate memory size (rough approximation)
        const contentSize = new Blob([node.content]).size;
        return sum + contentSize;
    }, 0);

    const selectedMemory = displayNodes
        .filter((n) => displaySelection.has(n.id))
        .reduce((sum, node) => {
            const contentSize = new Blob([node.content]).size;
            return sum + contentSize;
        }, 0);

    const formatBytes = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    return (
        <div className="prune-mode">
            <Card>
                <CardHeader>
                    <CardTitle>
                        <i className="fa fa-scissors mr-2"></i>
                        Prune Mode
                    </CardTitle>
                </CardHeader>
                <CardContent className="prune-content">
                    {/* Statistics */}
                    <div className="prune-stats">
                        <div className="stat-item">
                            <span className="stat-label">Total Nodes:</span>
                            <Badge variant="secondary">{displayNodes.length}</Badge>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Selected:</span>
                            <Badge variant="destructive">{displaySelection.size}</Badge>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Memory:</span>
                            <span className="stat-value">
                                {formatBytes(selectedMemory)} / {formatBytes(totalMemory)}
                            </span>
                        </div>
                    </div>

                    <Separator />

                    {/* Selection Controls */}
                    <div className="selection-controls">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleSelectAll}
                            disabled={displaySelection.size === displayNodes.length}
                        >
                            <i className="fa fa-check-double mr-2"></i>
                            Select All
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleClearSelection}
                            disabled={displaySelection.size === 0}
                        >
                            <i className="fa fa-xmark mr-2"></i>
                            Clear Selection
                        </Button>
                    </div>

                    <Separator />

                    {/* Selected Nodes List */}
                    <div className="selected-nodes-section">
                        <h4>Selected Nodes</h4>
                        <ScrollArea className="selected-nodes-list">
                            {displaySelection.size === 0 ? (
                                <div className="empty-state">
                                    <i className="fa fa-circle-info"></i>
                                    <p>Click nodes in the graph to select them for deletion</p>
                                </div>
                            ) : (
                                displayNodes
                                    .filter((n) => displaySelection.has(n.id))
                                    .map((node) => (
                                        <div key={node.id} className="selected-node-item">
                                            <Checkbox
                                                checked={true}
                                                onCheckedChange={() => handleToggleNode(node.id)}
                                            />
                                            <div className="node-preview">
                                                <Badge
                                                    variant="outline"
                                                    className="type-badge"
                                                >
                                                    {node.type}
                                                </Badge>
                                                <span className="node-content">
                                                    {node.content.slice(0, 80)}...
                                                </span>
                                            </div>
                                        </div>
                                    ))
                            )}
                        </ScrollArea>
                    </div>

                    <Separator />

                    {/* Action Buttons */}
                    <div className="prune-actions">
                        <Button
                            variant="destructive"
                            className="full-width"
                            onClick={handlePrune}
                            disabled={displaySelection.size === 0}
                        >
                            <i className="fa fa-trash mr-2"></i>
                            Delete {displaySelection.size} Node{displaySelection.size !== 1 ? "s" : ""}
                        </Button>
                        <Button
                            variant="outline"
                            className="full-width"
                            onClick={handleCancel}
                        >
                            <i className="fa fa-times mr-2"></i>
                            Cancel
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Confirmation Dialog */}
            <AlertDialog open={showConfirmDialog} onOpenChange={setShowConfirmDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm Deletion</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete {displaySelection.size} memory node
                            {displaySelection.size !== 1 ? "s" : ""}?
                            <br />
                            <br />
                            This action cannot be undone. All selected nodes and their relationships
                            will be permanently removed.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleConfirmPrune}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
};
