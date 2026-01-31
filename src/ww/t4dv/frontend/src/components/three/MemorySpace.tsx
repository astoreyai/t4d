/**
 * V6.1–V6.5: Three.js 3D Memory Space
 *
 * Instanced memory nodes colored by κ, sized by importance,
 * with edge rendering, vis.js timeline sync, and 4D projection modes.
 */

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stats } from "@react-three/drei";
import { MemoryNodes } from "./MemoryNodes";
import { MemoryEdges } from "./MemoryEdges";
import { TimelineSync } from "./TimelineSync";
import { useState } from "react";

export type ProjectionMode = "slice" | "collapse" | "animate";

export interface MemoryNode {
  id: string;
  position: [number, number, number];
  kappa: number;
  importance: number;
  timestamp: number;
  label?: string;
}

export interface MemoryEdge {
  source: string;
  target: string;
  weight: number;
  edgeType: string;
}

interface Props {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
  timeRange?: [number, number];
  onTimeRangeChange?: (range: [number, number]) => void;
  projectionMode?: ProjectionMode;
  showTimeline?: boolean;
}

export function MemorySpace({
  nodes,
  edges,
  timeRange,
  onTimeRangeChange,
  projectionMode = "slice",
  showTimeline = true,
}: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Filter nodes by time range if provided
  const visibleNodes = timeRange
    ? nodes.filter(
        (n) => n.timestamp >= timeRange[0] && n.timestamp <= timeRange[1],
      )
    : nodes;

  // Build position lookup for edges
  const posMap = new Map(visibleNodes.map((n) => [n.id, n.position]));
  const visibleEdges = edges.filter(
    (e) => posMap.has(e.source) && posMap.has(e.target),
  );

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 relative" style={{ minHeight: 400 }}>
        <Canvas camera={{ position: [0, 0, 50], fov: 60 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 10]} intensity={0.8} />
          <MemoryNodes
            nodes={visibleNodes}
            selectedId={selectedId}
            hoveredId={hoveredId}
            onSelect={setSelectedId}
            onHover={setHoveredId}
            projectionMode={projectionMode}
          />
          <MemoryEdges
            edges={visibleEdges}
            posMap={posMap}
          />
          <OrbitControls
            enableDamping
            dampingFactor={0.1}
            minDistance={5}
            maxDistance={200}
          />
          <Stats />
        </Canvas>

        {/* Tooltip overlay */}
        {hoveredId && (
          <div className="absolute top-2 left-2 bg-gray-800 rounded px-2 py-1 text-xs pointer-events-none">
            {hoveredId} | κ={visibleNodes.find((n) => n.id === hoveredId)?.kappa.toFixed(2)}
          </div>
        )}
      </div>

      {showTimeline && (
        <TimelineSync
          nodes={nodes}
          timeRange={timeRange}
          onTimeRangeChange={onTimeRangeChange}
          selectedId={selectedId}
        />
      )}
    </div>
  );
}
