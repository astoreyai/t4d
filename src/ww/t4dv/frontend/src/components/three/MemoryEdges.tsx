/**
 * V6.3 Edge rendering
 *
 * Renders memory edges as line segments between node positions.
 * Color-coded by edge type, opacity by weight.
 */

import { useMemo } from "react";
import * as THREE from "three";
import type { MemoryEdge } from "./MemorySpace";

const EDGE_TYPE_COLORS: Record<string, string> = {
  temporal: "#3b82f6",     // blue
  causal: "#ef4444",       // red
  semantic: "#22c55e",     // green
  associative: "#a855f7",  // purple
  default: "#6b7280",      // gray
};

interface Props {
  edges: MemoryEdge[];
  posMap: Map<string, [number, number, number]>;
}

export function MemoryEdges({ edges, posMap }: Props) {
  // Group edges by type for batched rendering
  const grouped = useMemo(() => {
    const groups: Record<string, { positions: number[]; opacities: number[] }> = {};

    for (const edge of edges) {
      const srcPos = posMap.get(edge.source);
      const tgtPos = posMap.get(edge.target);
      if (!srcPos || !tgtPos) continue;

      const type = edge.edgeType in EDGE_TYPE_COLORS ? edge.edgeType : "default";
      if (!groups[type]) groups[type] = { positions: [], opacities: [] };

      groups[type].positions.push(...srcPos, ...tgtPos);
      groups[type].opacities.push(
        Math.max(0.1, Math.min(1.0, edge.weight)),
        Math.max(0.1, Math.min(1.0, edge.weight)),
      );
    }

    return groups;
  }, [edges, posMap]);

  return (
    <group>
      {Object.entries(grouped).map(([type, { positions }]) => {
        const posArray = new Float32Array(positions);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(posArray, 3),
        );

        const color = EDGE_TYPE_COLORS[type] || EDGE_TYPE_COLORS.default;

        return (
          <lineSegments key={type} geometry={geometry}>
            <lineBasicMaterial
              color={color}
              transparent
              opacity={0.4}
              linewidth={1}
            />
          </lineSegments>
        );
      })}
    </group>
  );
}
