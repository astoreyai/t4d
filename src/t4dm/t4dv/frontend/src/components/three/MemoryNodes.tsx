/**
 * V6.1 Instanced memory nodes + V6.2 κ-coloring + importance-sizing
 *
 * Uses Three.js InstancedMesh for rendering up to 100K nodes at 30+ FPS.
 * Color encodes κ (blue=0 → green=0.5 → gold=1.0).
 * Size encodes importance (0.3–1.5 radius).
 */

import { useRef, useMemo, useEffect } from "react";
import { useFrame, ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import type { MemoryNode, ProjectionMode } from "./MemorySpace";

// κ color gradient: blue(0) → cyan(0.25) → green(0.5) → yellow(0.75) → gold(1.0)
function kappaToColor(kappa: number): THREE.Color {
  if (kappa < 0.5) {
    return new THREE.Color().setHSL(0.55 - kappa * 0.5, 0.9, 0.5);
  }
  return new THREE.Color().setHSL(0.15 - (kappa - 0.5) * 0.15, 0.9, 0.5);
}

function importanceToScale(importance: number): number {
  return 0.3 + importance * 1.2;
}

interface Props {
  nodes: MemoryNode[];
  selectedId: string | null;
  hoveredId: string | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  projectionMode: ProjectionMode;
}

export function MemoryNodes({
  nodes,
  selectedId,
  hoveredId,
  onSelect,
  onHover,
  projectionMode,
}: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tmpObj = useMemo(() => new THREE.Object3D(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  // Animation time for "animate" projection mode
  const animTimeRef = useRef(0);

  useFrame((_, delta) => {
    if (projectionMode === "animate") {
      animTimeRef.current += delta;
    }
  });

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    nodes.forEach((node, i) => {
      let [x, y, z] = node.position;

      // V6.5: 4D projection modes
      if (projectionMode === "collapse") {
        // Collapse time into Y axis
        z = 0;
      } else if (projectionMode === "animate") {
        // Time-varying displacement
        const phase = (node.timestamp * 0.001 + animTimeRef.current) % (Math.PI * 2);
        z = Math.sin(phase) * 5;
      }
      // "slice" mode uses positions as-is

      const scale = importanceToScale(node.importance);
      tmpObj.position.set(x, y, z);
      tmpObj.scale.setScalar(scale);
      tmpObj.updateMatrix();
      mesh.setMatrixAt(i, tmpObj.matrix);

      // Color by κ
      const color = kappaToColor(node.kappa);
      if (node.id === selectedId) {
        color.set(0xffffff);
      } else if (node.id === hoveredId) {
        color.lerp(new THREE.Color(0xffffff), 0.4);
      }
      mesh.setColorAt(i, color);
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [nodes, selectedId, hoveredId, projectionMode]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    const idx = e.instanceId;
    if (idx !== undefined && idx < nodes.length) {
      onSelect(nodes[idx].id);
    }
  };

  const handlePointerOver = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const idx = e.instanceId;
    if (idx !== undefined && idx < nodes.length) {
      onHover(nodes[idx].id);
      document.body.style.cursor = "pointer";
    }
  };

  const handlePointerOut = () => {
    onHover(null);
    document.body.style.cursor = "auto";
  };

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, Math.max(nodes.length, 1)]}
      onClick={handleClick}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      <sphereGeometry args={[1, 16, 12]} />
      <meshStandardMaterial vertexColors toneMapped={false} />
    </instancedMesh>
  );
}
