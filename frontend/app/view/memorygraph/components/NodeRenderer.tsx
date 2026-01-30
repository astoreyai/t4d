/**
 * Renders memory nodes in 3D space with type-based coloring and activation effects
 */

import React, { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import * as THREE from "three";
import { NodeRendererProps } from "../memorygraph-types";

/**
 * Individual node component
 */
const Node: React.FC<{
    id: string;
    position: THREE.Vector3;
    size: number;
    color: string;
    glowIntensity: number;
    opacity: number;
    label?: string;
    isSelected: boolean;
    isHovered: boolean;
    isHighlighted: boolean;
    isMultiSelected: boolean;
    showLabel: boolean;
    onClick: (id: string) => void;
    onHover: (id: string | null) => void;
}> = ({
    id,
    position,
    size,
    color,
    glowIntensity,
    opacity,
    label,
    isSelected,
    isHovered,
    isHighlighted,
    isMultiSelected,
    showLabel,
    onClick,
    onHover,
}) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const glowRef = useRef<THREE.Mesh>(null);

    // Validate numeric values to prevent NaN errors in geometry/materials
    const safeSize = isFinite(size) && size > 0 ? size : 0.5;
    const safeOpacity = isFinite(opacity) && opacity >= 0 ? opacity : 1.0;
    const safeGlowIntensity = isFinite(glowIntensity) && glowIntensity >= 0 ? glowIntensity : 0;

    // Animate glow effect
    useFrame((state) => {
        if (glowRef.current && safeGlowIntensity > 0) {
            const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
            glowRef.current.scale.setScalar(scale);
        }
    });

    // Selection ring color
    const selectionColor = useMemo(() => {
        if (isSelected) return "#fbbf24"; // yellow
        if (isMultiSelected) return "#ef4444"; // red
        if (isHighlighted) return "#06b6d4"; // cyan
        return null;
    }, [isSelected, isMultiSelected, isHighlighted]);

    // Node scale when hovered
    const nodeScale = isHovered ? 1.2 : 1.0;

    // Validate position - create safe position if NaN values detected
    const safePosition = useMemo(() => {
        if (!position || !isFinite(position.x) || !isFinite(position.y) || !isFinite(position.z)) {
            return new THREE.Vector3(0, 0, 0);
        }
        return position;
    }, [position]);

    return (
        <group position={safePosition}>
            {/* Main node sphere */}
            <mesh
                ref={meshRef}
                scale={nodeScale}
                onClick={() => onClick(id)}
                onPointerOver={() => onHover(id)}
                onPointerOut={() => onHover(null)}
            >
                <sphereGeometry args={[safeSize, 32, 32]} />
                <meshStandardMaterial
                    color={color}
                    opacity={safeOpacity}
                    transparent
                    emissive={color}
                    emissiveIntensity={safeGlowIntensity * 0.5}
                />
            </mesh>

            {/* Glow effect for activated nodes */}
            {safeGlowIntensity > 0 && (
                <mesh ref={glowRef}>
                    <sphereGeometry args={[safeSize * 1.3, 32, 32]} />
                    <meshBasicMaterial
                        color={color}
                        opacity={safeGlowIntensity * 0.3}
                        transparent
                        side={THREE.BackSide}
                    />
                </mesh>
            )}

            {/* Selection ring */}
            {selectionColor && (
                <mesh rotation={[Math.PI / 2, 0, 0]}>
                    <ringGeometry args={[safeSize * 1.2, safeSize * 1.4, 32]} />
                    <meshBasicMaterial
                        color={selectionColor}
                        side={THREE.DoubleSide}
                        transparent
                        opacity={0.8}
                    />
                </mesh>
            )}

            {/* Label */}
            {showLabel && (isHovered || isSelected) && label && (
                <Text
                    position={[0, safeSize * 1.5, 0]}
                    fontSize={0.5}
                    color="white"
                    anchorX="center"
                    anchorY="middle"
                    outlineWidth={0.05}
                    outlineColor="#000000"
                >
                    {label}
                </Text>
            )}
        </group>
    );
};

/**
 * Node renderer component - renders all visible nodes
 */
export const NodeRenderer: React.FC<NodeRendererProps> = ({
    nodes,
    visualConfigs,
    selectedNode,
    hoveredNode,
    highlightedNodes,
    multiSelectNodes,
    showLabels,
    onNodeClick,
    onNodeHover,
}) => {
    return (
        <group>
            {nodes.map((node) => {
                // Guard against missing or invalid positions (NaN values cause THREE.js errors)
                if (!node.position ||
                    !isFinite(node.position.x) ||
                    !isFinite(node.position.y) ||
                    !isFinite(node.position.z)) {
                    return null;
                }

                const config = visualConfigs.get(node.id);
                if (!config || !isFinite(config.baseSize) || config.baseSize <= 0) {
                    return null;
                }

                const isSelected = selectedNode === node.id;
                const isHovered = hoveredNode === node.id;
                const isHighlighted = highlightedNodes.has(node.id);
                const isMultiSelected = multiSelectNodes.has(node.id);

                return (
                    <Node
                        key={node.id}
                        id={node.id}
                        position={node.position}
                        size={config.baseSize}
                        color={config.color}
                        glowIntensity={config.glowIntensity}
                        opacity={config.opacity}
                        label={config.label}
                        isSelected={isSelected}
                        isHovered={isHovered}
                        isHighlighted={isHighlighted}
                        isMultiSelected={isMultiSelected}
                        showLabel={showLabels}
                        onClick={onNodeClick}
                        onHover={onNodeHover}
                    />
                );
            })}
        </group>
    );
};
