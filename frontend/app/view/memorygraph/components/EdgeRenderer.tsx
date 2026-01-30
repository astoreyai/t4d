/**
 * Renders relationship edges between memory nodes
 */

import React, { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { EdgeRendererProps, MemoryNode } from "../memorygraph-types";

/**
 * Individual edge component
 */
const Edge: React.FC<{
    id: string;
    sourcePosition: THREE.Vector3;
    targetPosition: THREE.Vector3;
    width: number;
    color: string;
    opacity: number;
    animated: boolean;
    isSelected: boolean;
    isHovered: boolean;
    onClick?: (id: string) => void;
    onHover?: (id: string | null) => void;
}> = ({
    id,
    sourcePosition,
    targetPosition,
    width,
    color,
    opacity,
    animated,
    isSelected,
    isHovered,
    onClick,
    onHover,
}) => {
    const lineRef = useRef<THREE.Line>(null);

    // Create points for the line
    const points = useMemo(() => {
        return [sourcePosition, targetPosition];
    }, [sourcePosition, targetPosition]);

    // Animated dash offset for active edges
    useFrame((state) => {
        if (lineRef.current && animated) {
            const material = lineRef.current.material as THREE.LineDashedMaterial;
            if (material.dashSize !== undefined) {
                material.dashOffset = (state.clock.elapsedTime * 2) % (material.dashSize + material.gapSize);
            }
        }
    });

    // Edge appearance based on state
    const lineWidth = isHovered || isSelected ? width * 2 : width;
    const lineOpacity = isHovered || isSelected ? Math.min(opacity * 1.5, 1.0) : opacity;

    return (
        <Line
            ref={lineRef}
            points={points}
            color={color}
            lineWidth={lineWidth}
            opacity={lineOpacity}
            transparent
            dashed={animated}
            dashSize={0.5}
            gapSize={0.3}
            onClick={() => onClick?.(id)}
            onPointerOver={() => onHover?.(id)}
            onPointerOut={() => onHover?.(null)}
        />
    );
};

/**
 * Arrow head component for directed edges
 */
const ArrowHead: React.FC<{
    position: THREE.Vector3;
    direction: THREE.Vector3;
    color: string;
    size: number;
}> = ({ position, direction, color, size }) => {
    const rotation = useMemo(() => {
        // Guard against zero/invalid direction vectors that produce NaN when normalized
        const len = direction.length();
        if (!isFinite(len) || len < 0.001) {
            return new THREE.Euler(0, 0, 0);
        }
        const up = new THREE.Vector3(0, 1, 0);
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(up, direction.clone().normalize());
        return new THREE.Euler().setFromQuaternion(quaternion);
    }, [direction]);

    return (
        <mesh position={position} rotation={rotation}>
            <coneGeometry args={[size, size * 2, 8]} />
            <meshBasicMaterial color={color} />
        </mesh>
    );
};

/**
 * Edge renderer component - renders all visible edges
 */
export const EdgeRenderer: React.FC<EdgeRendererProps> = ({
    edges,
    nodes,
    visualConfigs,
    selectedEdge,
    hoveredEdge,
    showEdges,
    edgeOpacity,
    onEdgeClick,
    onEdgeHover,
}) => {
    // Create a map of node positions for quick lookup
    const nodePositions = useMemo(() => {
        const positions = new Map<string, THREE.Vector3>();
        nodes.forEach((node) => {
            if (node.position) {
                positions.set(node.id, node.position);
            }
        });
        return positions;
    }, [nodes]);

    if (!showEdges) {
        return null;
    }

    return (
        <group>
            {edges.map((edge) => {
                const sourcePos = nodePositions.get(edge.source);
                const targetPos = nodePositions.get(edge.target);

                // Guard against missing or invalid positions (NaN values cause THREE.js errors)
                if (!sourcePos || !targetPos ||
                    !isFinite(sourcePos.x) || !isFinite(sourcePos.y) || !isFinite(sourcePos.z) ||
                    !isFinite(targetPos.x) || !isFinite(targetPos.y) || !isFinite(targetPos.z)) {
                    return null;
                }

                const config = visualConfigs.get(edge.id);
                if (!config || !isFinite(config.width) || config.width <= 0) {
                    return null;
                }

                const isSelected = selectedEdge === edge.id;
                const isHovered = hoveredEdge === edge.id;

                // Calculate arrow position and direction
                const direction = new THREE.Vector3().subVectors(targetPos, sourcePos);
                const length = direction.length();

                // Guard against zero-length edges (normalize() on zero vector produces NaN)
                if (length < 0.001) {
                    return null;
                }
                direction.normalize();
                const arrowPos = targetPos.clone().sub(direction.clone().multiplyScalar(1.0));

                return (
                    <group key={edge.id}>
                        {/* Edge line */}
                        <Edge
                            id={edge.id}
                            sourcePosition={sourcePos}
                            targetPosition={targetPos}
                            width={config.width}
                            color={config.color}
                            opacity={config.opacity * edgeOpacity}
                            animated={config.animated}
                            isSelected={isSelected}
                            isHovered={isHovered}
                            onClick={onEdgeClick}
                            onHover={onEdgeHover}
                        />

                        {/* Arrow head */}
                        <ArrowHead
                            position={arrowPos}
                            direction={direction}
                            color={config.color}
                            size={config.width * 3}
                        />
                    </group>
                );
            })}
        </group>
    );
};
