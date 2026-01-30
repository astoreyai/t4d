/**
 * Control panel for graph visualization settings
 */

import React from "react";
import { useAtom } from "jotai";
import {
    filterStateAtom,
    controlStateAtom,
    updateFiltersAtom,
    updateControlsAtom,
    resetFiltersAtom,
    resetCameraAtom,
} from "../memorygraph-state";
import { ControlPanelProps, MemoryType, EdgeType } from "../memorygraph-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import "./ControlPanel.scss";

/**
 * Memory type filter checkboxes
 */
const MemoryTypeFilters: React.FC = () => {
    const [filters, setFilters] = useAtom(filterStateAtom);

    const memoryTypes: MemoryType[] = ["episodic", "semantic", "procedural"];

    const toggleMemoryType = (type: MemoryType) => {
        const newTypes = new Set(filters.memoryTypes);
        if (newTypes.has(type)) {
            newTypes.delete(type);
        } else {
            newTypes.add(type);
        }
        setFilters({ ...filters, memoryTypes: newTypes });
    };

    return (
        <div className="filter-group">
            <Label className="filter-label">Memory Types</Label>
            <div className="checkbox-group">
                {memoryTypes.map((type) => (
                    <div key={type} className="checkbox-item">
                        <Checkbox
                            id={`memory-type-${type}`}
                            checked={filters.memoryTypes.has(type)}
                            onCheckedChange={() => toggleMemoryType(type)}
                        />
                        <Label htmlFor={`memory-type-${type}`} className="checkbox-label">
                            {type.charAt(0).toUpperCase() + type.slice(1)}
                        </Label>
                    </div>
                ))}
            </div>
        </div>
    );
};

/**
 * Edge type filter checkboxes
 */
const EdgeTypeFilters: React.FC = () => {
    const [filters, setFilters] = useAtom(filterStateAtom);

    const edgeTypes: EdgeType[] = [
        "CAUSED",
        "SIMILAR_TO",
        "PREREQUISITE",
        "CONTRADICTS",
        "REFERENCES",
        "DERIVED_FROM",
    ];

    const toggleEdgeType = (type: EdgeType) => {
        const newTypes = new Set(filters.edgeTypes);
        if (newTypes.has(type)) {
            newTypes.delete(type);
        } else {
            newTypes.add(type);
        }
        setFilters({ ...filters, edgeTypes: newTypes });
    };

    return (
        <div className="filter-group">
            <Label className="filter-label">Edge Types</Label>
            <div className="checkbox-group">
                {edgeTypes.map((type) => (
                    <div key={type} className="checkbox-item">
                        <Checkbox
                            id={`edge-type-${type}`}
                            checked={filters.edgeTypes.has(type)}
                            onCheckedChange={() => toggleEdgeType(type)}
                        />
                        <Label htmlFor={`edge-type-${type}`} className="checkbox-label">
                            {type.replace(/_/g, " ")}
                        </Label>
                    </div>
                ))}
            </div>
        </div>
    );
};

/**
 * Main control panel component
 */
export const ControlPanel: React.FC<ControlPanelProps> = ({
    filterState,
    controlState,
    onFilterChange,
    onControlChange,
    onResetCamera,
    onResetFilters,
}) => {
    const [filters, setFilters] = useAtom(filterStateAtom);
    const [controls, setControls] = useAtom(controlStateAtom);

    return (
        <div className="control-panel">
            <Card>
                <CardHeader>
                    <CardTitle>Controls</CardTitle>
                </CardHeader>
                <CardContent className="control-panel-content">
                    {/* Search */}
                    <div className="control-group">
                        <Label htmlFor="search-input">Search</Label>
                        <Input
                            id="search-input"
                            type="text"
                            placeholder="Search memories..."
                            value={filters.searchQuery}
                            onChange={(e) =>
                                setFilters({ ...filters, searchQuery: e.target.value })
                            }
                        />
                    </div>

                    <Separator />

                    {/* Memory Type Filters */}
                    <MemoryTypeFilters />

                    <Separator />

                    {/* Edge Type Filters */}
                    <EdgeTypeFilters />

                    <Separator />

                    {/* Threshold Sliders */}
                    <div className="control-group">
                        <Label>Activity Threshold: {(filters.activityThreshold * 100).toFixed(0)}%</Label>
                        <Slider
                            value={[filters.activityThreshold]}
                            min={0}
                            max={1}
                            step={0.01}
                            onValueChange={(values) =>
                                setFilters({ ...filters, activityThreshold: values[0] })
                            }
                        />
                    </div>

                    <div className="control-group">
                        <Label>Importance Threshold: {(filters.importanceThreshold * 100).toFixed(0)}%</Label>
                        <Slider
                            value={[filters.importanceThreshold]}
                            min={0}
                            max={1}
                            step={0.01}
                            onValueChange={(values) =>
                                setFilters({ ...filters, importanceThreshold: values[0] })
                            }
                        />
                    </div>

                    <Separator />

                    {/* Visual Controls */}
                    <div className="control-group">
                        <Label>Layout Algorithm</Label>
                        <Select
                            value={controls.layoutAlgorithm}
                            onValueChange={(value) =>
                                setControls({
                                    ...controls,
                                    layoutAlgorithm: value as any,
                                })
                            }
                        >
                            <SelectTrigger>
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="force-directed">Force Directed</SelectItem>
                                <SelectItem value="hierarchical">Hierarchical</SelectItem>
                                <SelectItem value="circular">Circular</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    <div className="control-group">
                        <Label>Node Scale: {controls.nodeScale.toFixed(2)}x</Label>
                        <Slider
                            value={[controls.nodeScale]}
                            min={0.5}
                            max={2.0}
                            step={0.1}
                            onValueChange={(values) =>
                                setControls({ ...controls, nodeScale: values[0] })
                            }
                        />
                    </div>

                    <div className="control-group">
                        <Label>Edge Opacity: {(controls.edgeOpacity * 100).toFixed(0)}%</Label>
                        <Slider
                            value={[controls.edgeOpacity]}
                            min={0}
                            max={1}
                            step={0.05}
                            onValueChange={(values) =>
                                setControls({ ...controls, edgeOpacity: values[0] })
                            }
                        />
                    </div>

                    <Separator />

                    {/* Toggle Switches */}
                    <div className="switch-group">
                        <div className="switch-item">
                            <Label htmlFor="show-labels">Show Labels</Label>
                            <Switch
                                id="show-labels"
                                checked={controls.showLabels}
                                onCheckedChange={(checked) =>
                                    setControls({ ...controls, showLabels: checked })
                                }
                            />
                        </div>

                        <div className="switch-item">
                            <Label htmlFor="show-edges">Show Edges</Label>
                            <Switch
                                id="show-edges"
                                checked={controls.showEdges}
                                onCheckedChange={(checked) =>
                                    setControls({ ...controls, showEdges: checked })
                                }
                            />
                        </div>

                        <div className="switch-item">
                            <Label htmlFor="auto-rotate">Auto Rotate</Label>
                            <Switch
                                id="auto-rotate"
                                checked={controls.autoRotate}
                                onCheckedChange={(checked) =>
                                    setControls({ ...controls, autoRotate: checked })
                                }
                            />
                        </div>
                    </div>

                    <Separator />

                    {/* Action Buttons */}
                    <div className="button-group">
                        <Button variant="outline" onClick={onResetCamera} className="full-width">
                            <i className="fa fa-camera mr-2"></i>
                            Reset Camera
                        </Button>
                        <Button variant="outline" onClick={onResetFilters} className="full-width">
                            <i className="fa fa-filter-circle-xmark mr-2"></i>
                            Reset Filters
                        </Button>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};
