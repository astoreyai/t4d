/**
 * Memory Graph Page - 3D Force-Directed Graph Visualization
 *
 * Uses React Three Fiber for 3D rendering and d3-force-3d for force simulation.
 * Displays episodic, semantic, and procedural memories as an interactive graph.
 */

import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Billboard, Line } from '@react-three/drei';
import * as THREE from 'three';
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
} from 'd3-force-3d';
import './MemoryGraphPage.scss';

// ============================================================================
// Types
// ============================================================================

type MemoryType = 'episodic' | 'semantic' | 'procedural';

interface GraphNode {
  id: string;
  type: MemoryType;
  content: string;
  activation: number;
  // Hinton-inspired learning dynamics
  retrievability: number;    // FSRS R-value (0-1): controls opacity
  eligibilityTrace: number;  // Trace strength (0-1): controls glow
  recentReconsolidation: boolean; // Pulse animation trigger
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
  type: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metrics: {
    node_count: number;
    edge_count: number;
    avg_activation: number;
    clusters: number;
  };
}

interface NodeProps {
  node: GraphNode;
  selected: boolean;
  onClick: () => void;
  onHover: (hovered: boolean) => void;
}

// ============================================================================
// Color Scheme
// ============================================================================

const COLORS = {
  episodic: '#64b5f6',     // Blue
  semantic: '#81c784',     // Green
  procedural: '#ffb74d',   // Orange
  edge: '#666666',
  edgeHighlight: '#ffffff',
  background: '#0a0a0f',
};

function getNodeColor(type: MemoryType): string {
  return COLORS[type] || COLORS.episodic;
}

// ============================================================================
// Node Component
// ============================================================================

const MemoryNode: React.FC<NodeProps> = ({ node, selected, onClick, onHover }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Size based on activation (0.3 - 1.0)
  const size = 0.3 + node.activation * 0.7;
  const color = getNodeColor(node.type);

  // Hinton-inspired dynamics:
  // Opacity from FSRS retrievability (forgotten memories fade)
  const baseOpacity = Math.max(0.15, node.retrievability);

  // Glow intensity from eligibility trace (recently accessed glows)
  const glowIntensity = node.eligibilityTrace * 0.8;

  // Reconsolidation pulse (learning just happened)
  const hasPulse = node.recentReconsolidation;

  useFrame((state) => {
    if (meshRef.current) {
      // Pulse animation for recent reconsolidation events
      if (hasPulse) {
        const pulse = 1 + Math.sin(state.clock.elapsedTime * 8) * 0.15;
        meshRef.current.scale.setScalar(pulse);
      } else if (hovered || selected) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
        meshRef.current.scale.setScalar(scale);
      } else {
        meshRef.current.scale.setScalar(1);
      }
    }

    // Animate glow based on eligibility trace
    if (glowRef.current && node.eligibilityTrace > 0.01) {
      const glowPulse = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
      glowRef.current.scale.setScalar(glowPulse);
    }
  });

  return (
    <group position={[node.x || 0, node.y || 0, node.z || 0]}>
      {/* Eligibility trace glow (outer halo) */}
      {node.eligibilityTrace > 0.01 && (
        <mesh ref={glowRef}>
          <sphereGeometry args={[size * 1.5, 16, 16]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={glowIntensity * 0.3}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Main node sphere */}
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        onPointerDown={(e) => {
          e.stopPropagation();
          if (meshRef.current) {
            meshRef.current.scale.setScalar(1.2);
          }
        }}
        onPointerUp={(e) => {
          e.stopPropagation();
          if (meshRef.current) {
            meshRef.current.scale.setScalar(1);
          }
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onHover(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setHovered(false);
          onHover(false);
          document.body.style.cursor = 'auto';
        }}
      >
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={
            hasPulse ? 0.7 :
            hovered || selected ? 0.5 :
            0.2 + glowIntensity * 0.3
          }
          transparent
          opacity={baseOpacity}
        />
      </mesh>

      {/* Reconsolidation pulse ring */}
      {hasPulse && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[size * 1.2, size * 1.4, 32]} />
          <meshBasicMaterial
            color="#22c55e"
            transparent
            opacity={0.6}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}

      {/* Label on hover */}
      {(hovered || selected) && (
        <Billboard follow lockX={false} lockY={false} lockZ={false}>
          <Text
            position={[0, size + 0.5, 0]}
            fontSize={0.4}
            color="white"
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.02}
            outlineColor="black"
          >
            {node.content.slice(0, 40)}
            {node.content.length > 40 ? '...' : ''}
          </Text>
        </Billboard>
      )}
    </group>
  );
};

// ============================================================================
// Edge Component
// ============================================================================

interface EdgeProps {
  source: GraphNode;
  target: GraphNode;
  weight: number;
  highlighted: boolean;
}

const MemoryEdge: React.FC<EdgeProps> = ({ source, target, weight, highlighted }) => {
  const points = useMemo(() => [
    new THREE.Vector3(source.x || 0, source.y || 0, source.z || 0),
    new THREE.Vector3(target.x || 0, target.y || 0, target.z || 0),
  ], [source.x, source.y, source.z, target.x, target.y, target.z]);

  return (
    <Line
      points={points}
      color={highlighted ? COLORS.edgeHighlight : COLORS.edge}
      lineWidth={weight * 2 + 0.5}
      transparent
      opacity={highlighted ? 0.8 : 0.3}
    />
  );
};

// ============================================================================
// Neuromodulator Ambient Background
// ============================================================================

interface NeuromodulatorState {
  acetylcholine_mode: 'encoding' | 'retrieval';
  dopamine_rpe: number;
  norepinephrine_gain: number;
}

interface AmbientBackgroundProps {
  neuromodulatorState: NeuromodulatorState | null;
}

const AmbientBackground: React.FC<AmbientBackgroundProps> = ({ neuromodulatorState }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [flashColor, setFlashColor] = useState<string | null>(null);

  // Default state if no data
  const state = neuromodulatorState || {
    acetylcholine_mode: 'retrieval' as const,
    dopamine_rpe: 0,
    norepinephrine_gain: 0.5,
  };

  // ACh mode determines base color: encoding=warm amber, retrieval=cool blue
  const baseColor = state.acetylcholine_mode === 'encoding' ? '#f59e0b' : '#3b82f6';

  // NE gain affects ambient light intensity
  const ambientIntensity = 0.3 + state.norepinephrine_gain * 0.3;

  // Dopamine RPE triggers color flash
  useEffect(() => {
    if (Math.abs(state.dopamine_rpe) > 0.1) {
      const color = state.dopamine_rpe > 0 ? '#22c55e' : '#ef4444';
      setFlashColor(color);
      const timer = setTimeout(() => setFlashColor(null), 500);
      return () => clearTimeout(timer);
    }
  }, [state.dopamine_rpe]);

  useFrame((frameState) => {
    if (meshRef.current) {
      // Subtle pulse for the ambient sphere
      const scale = 100 + Math.sin(frameState.clock.elapsedTime * 0.5) * 5;
      meshRef.current.scale.setScalar(scale);
    }
  });

  return (
    <>
      {/* Ambient light with NE-modulated intensity */}
      <ambientLight intensity={ambientIntensity} color={baseColor} />

      {/* Background sphere for ambient color wash */}
      <mesh ref={meshRef} position={[0, 0, 0]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial
          color={flashColor || baseColor}
          transparent
          opacity={flashColor ? 0.15 : 0.05}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>

      {/* Point lights for scene illumination */}
      <pointLight position={[10, 10, 10]} intensity={0.6} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />
    </>
  );
};

// ============================================================================
// Graph Scene
// ============================================================================

interface GraphSceneProps {
  data: GraphData;
  selectedNode: string | null;
  onNodeSelect: (id: string | null) => void;
  neuromodulatorState: NeuromodulatorState | null;
}

const GraphScene: React.FC<GraphSceneProps> = ({ data, selectedNode, onNodeSelect, neuromodulatorState }) => {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const simulationRef = useRef<any>(null);

  // Initialize force simulation
  useEffect(() => {
    if (data.nodes.length === 0) return;

    // Deep copy nodes for simulation
    const nodesCopy = data.nodes.map(n => ({ ...n, x: 0, y: 0, z: 0 }));

    // Initialize random positions
    nodesCopy.forEach(node => {
      node.x = (Math.random() - 0.5) * 50;
      node.y = (Math.random() - 0.5) * 50;
      node.z = (Math.random() - 0.5) * 50;
    });

    // Create node lookup
    const nodeMap = new Map(nodesCopy.map(n => [n.id, n]));

    // Create links with node references
    const links = data.edges
      .map(e => ({
        source: nodeMap.get(typeof e.source === 'string' ? e.source : e.source.id)!,
        target: nodeMap.get(typeof e.target === 'string' ? e.target : e.target.id)!,
        weight: e.weight,
      }))
      .filter(l => l.source && l.target) as Array<{ source: GraphNode; target: GraphNode; weight: number }>;

    // Create simulation
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const simulation = forceSimulation(nodesCopy as any, 3)
      .force('link', forceLink(links as any).id((d: any) => d.id).distance(10).strength((l: any) => l.weight * 0.5))
      .force('charge', forceManyBody().strength(-30))
      .force('center', forceCenter(0, 0, 0))
      .force('collide', forceCollide().radius(2))
      .alphaDecay(0.02)
      .on('tick', () => {
        setNodes([...nodesCopy]);
      });

    simulationRef.current = simulation;

    return () => {
      simulation.stop();
    };
  }, [data]);

  // Create node map for edge lookups
  const nodeMap = useMemo(() =>
    new Map(nodes.map(n => [n.id, n])),
    [nodes]
  );

  // Get connected node IDs for highlighting
  const connectedNodes = useMemo(() => {
    const target = selectedNode || hoveredNode;
    if (!target) return new Set<string>();

    const connected = new Set<string>();
    data.edges.forEach(e => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
      const targetId = typeof e.target === 'string' ? e.target : e.target.id;
      if (sourceId === target) connected.add(targetId);
      if (targetId === target) connected.add(sourceId);
    });
    return connected;
  }, [data.edges, selectedNode, hoveredNode]);

  return (
    <>
      {/* Neuromodulator-driven ambient background */}
      <AmbientBackground neuromodulatorState={neuromodulatorState} />

      {/* Edges */}
      {data.edges.map((edge, i) => {
        const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
        const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
        const sourceNode = nodeMap.get(sourceId);
        const targetNode = nodeMap.get(targetId);

        if (!sourceNode || !targetNode) return null;

        const highlighted =
          sourceId === selectedNode ||
          targetId === selectedNode ||
          sourceId === hoveredNode ||
          targetId === hoveredNode;

        return (
          <MemoryEdge
            key={`edge-${i}`}
            source={sourceNode}
            target={targetNode}
            weight={edge.weight}
            highlighted={highlighted}
          />
        );
      })}

      {/* Nodes */}
      {nodes.map(node => (
        <MemoryNode
          key={node.id}
          node={node}
          selected={node.id === selectedNode}
          onClick={() => onNodeSelect(node.id === selectedNode ? null : node.id)}
          onHover={(h) => setHoveredNode(h ? node.id : null)}
        />
      ))}

      {/* Camera controls - touch optimized */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={200}
        // Touch support
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        touches={{
          ONE: THREE.TOUCH.ROTATE,
          TWO: THREE.TOUCH.DOLLY_PAN
        }}
        // Improved touch sensitivity
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        panSpeed={0.8}
      />
    </>
  );
};

// ============================================================================
// Details Panel
// ============================================================================

interface DetailsPanelProps {
  node: GraphNode | null;
  edges: GraphEdge[];
  onClose: () => void;
}

const DetailsPanel: React.FC<DetailsPanelProps> = ({ node, edges, onClose }) => {
  if (!node) return null;

  const connections = edges.filter(e => {
    const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
    const targetId = typeof e.target === 'string' ? e.target : e.target.id;
    return sourceId === node.id || targetId === node.id;
  });

  return (
    <div className="details-panel">
      <div className="panel-header">
        <h3>{node.type.charAt(0).toUpperCase() + node.type.slice(1)} Memory</h3>
        <button onClick={onClose} className="close-btn">&times;</button>
      </div>
      <div className="panel-content">
        <div className="detail-row">
          <span className="label">Content:</span>
          <span className="value">{node.content}</span>
        </div>
        <div className="detail-row">
          <span className="label">Activation:</span>
          <span className="value">{(node.activation * 100).toFixed(1)}%</span>
        </div>
        <div className="detail-row">
          <span className="label">Connections:</span>
          <span className="value">{connections.length}</span>
        </div>
        <div className="detail-row">
          <span className="label">ID:</span>
          <span className="value mono">{node.id.slice(0, 8)}...</span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

// Helper interfaces for bio data
interface FSRSData {
  [memoryId: string]: { retrievability: number };
}

interface EligibilityTrace {
  memory_id: string;
  trace_value: number;
}

interface ReconsolidationEvent {
  memory_id: string;
  timestamp: string;
}

export const MemoryGraphPage: React.FC = () => {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'3d' | 'stats'>('3d');
  const [recentReconsolidations, setRecentReconsolidations] = useState<Set<string>>(new Set());
  const [neuromodulatorState, setNeuromodulatorState] = useState<NeuromodulatorState | null>(null);

  // Fetch graph data with Hinton-inspired bio enhancements
  const loadGraphData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch graph, FSRS, eligibility, reconsolidation, and neuromodulator data in parallel
      const [graphRes, fsrsRes, eligibilityRes, reconsolidationRes, neuromodRes] = await Promise.all([
        fetch('/api/v1/viz/graph?layout=force-directed&limit=500&include_edges=true'),
        fetch('/api/v1/viz/bio/fsrs').catch(() => null),
        fetch('/api/v1/viz/bio/eligibility/traces').catch(() => null),
        fetch('/api/v1/viz/surgery/reconsolidation-history?limit=20').catch(() => null),
        fetch('/api/v1/viz/bio/neuromodulators').catch(() => null),
      ]);

      if (!graphRes.ok) {
        throw new Error(`HTTP ${graphRes.status}: ${graphRes.statusText}`);
      }

      const json = await graphRes.json();

      // Parse FSRS retrievability data
      let fsrsData: FSRSData = {};
      if (fsrsRes?.ok) {
        try {
          const fsrsJson = await fsrsRes.json();
          // Handle both array and object formats
          if (Array.isArray(fsrsJson)) {
            fsrsJson.forEach((item: any) => {
              fsrsData[item.memory_id] = { retrievability: item.retrievability || 1 };
            });
          } else if (fsrsJson.memories) {
            fsrsJson.memories.forEach((item: any) => {
              fsrsData[item.memory_id] = { retrievability: item.retrievability || 1 };
            });
          }
        } catch { /* Use defaults */ }
      }

      // Parse eligibility trace data
      let eligibilityMap: Map<string, number> = new Map();
      if (eligibilityRes?.ok) {
        try {
          const eligibilityJson = await eligibilityRes.json();
          const traces: EligibilityTrace[] = eligibilityJson.traces || eligibilityJson || [];
          traces.forEach((trace) => {
            eligibilityMap.set(trace.memory_id, trace.trace_value);
          });
        } catch { /* Use defaults */ }
      }

      // Parse recent reconsolidation events (last 30 seconds)
      let recentRecons = new Set<string>();
      if (reconsolidationRes?.ok) {
        try {
          const reconJson = await reconsolidationRes.json();
          const events: ReconsolidationEvent[] = reconJson || [];
          const thirtySecondsAgo = Date.now() - 30000;
          events.forEach((event) => {
            const eventTime = new Date(event.timestamp).getTime();
            if (eventTime > thirtySecondsAgo) {
              recentRecons.add(event.memory_id);
            }
          });
          setRecentReconsolidations(recentRecons);
        } catch { /* Use defaults */ }
      }

      // Parse neuromodulator state for ambient background
      if (neuromodRes?.ok) {
        try {
          const neuroJson = await neuromodRes.json();
          setNeuromodulatorState({
            acetylcholine_mode: neuroJson.acetylcholine_mode === 'encoding' ? 'encoding' : 'retrieval',
            dopamine_rpe: neuroJson.dopamine_rpe || 0,
            norepinephrine_gain: neuroJson.norepinephrine_gain || 0.5,
          });
        } catch { /* Use defaults */ }
      }

      // Transform nodes with Hinton-inspired dynamics
      const now = Date.now() / 1000;
      const nodes: GraphNode[] = (json.nodes || []).map((node: any) => {
        // Compute activation from recency and frequency
        const lastAccessed = node.metadata?.last_accessed || now;
        const daysSinceAccess = (now - lastAccessed) / (60 * 60 * 24);
        const recency = Math.exp(-daysSinceAccess / 30);
        const frequency = Math.min((node.metadata?.access_count || 1) / 100, 1);
        const importance = node.metadata?.importance || 0.5;

        let activation = 0.4 * recency + 0.3 * frequency + 0.3 * importance;
        if (!isFinite(activation) || activation < 0) activation = 0.5;
        if (activation > 1) activation = 1;

        // Hinton dynamics
        const retrievability = fsrsData[node.id]?.retrievability ?? 0.8;
        const eligibilityTrace = eligibilityMap.get(node.id) ?? 0;
        const recentReconsolidation = recentRecons.has(node.id);

        return {
          id: node.id,
          type: node.type as MemoryType,
          content: node.content || '',
          activation,
          retrievability,
          eligibilityTrace,
          recentReconsolidation,
        };
      });

      // Compute average activation
      const avgActivation = nodes.length > 0
        ? nodes.reduce((sum, n) => sum + n.activation, 0) / nodes.length
        : 0;

      setData({
        nodes,
        edges: json.edges || [],
        metrics: {
          node_count: json.metrics?.total_nodes || nodes.length,
          edge_count: json.metrics?.total_edges || (json.edges || []).length,
          avg_activation: avgActivation,
          clusters: json.metrics?.clusters || 0,
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load graph data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGraphData();
  }, [loadGraphData]);

  // Find selected node
  const selectedNodeData = useMemo(() =>
    data?.nodes.find(n => n.id === selectedNode) || null,
    [data, selectedNode]
  );

  // Count by type
  const typeCounts = useMemo(() => {
    if (!data) return { episodic: 0, semantic: 0, procedural: 0 };
    return data.nodes.reduce((acc, n) => {
      acc[n.type] = (acc[n.type] || 0) + 1;
      return acc;
    }, { episodic: 0, semantic: 0, procedural: 0 } as Record<MemoryType, number>);
  }, [data]);

  return (
    <div className="memory-graph-page">
      {/* Header */}
      <div className="graph-header">
        <div className="header-left">
          <h1>Memory Graph</h1>
          <p className="subtitle">Interactive 3D knowledge visualization</p>
        </div>
        <div className="header-controls">
          <button
            className={`mode-btn ${viewMode === '3d' ? 'active' : ''}`}
            onClick={() => setViewMode('3d')}
          >
            3D View
          </button>
          <button
            className={`mode-btn ${viewMode === 'stats' ? 'active' : ''}`}
            onClick={() => setViewMode('stats')}
          >
            Statistics
          </button>
          <button className="refresh-btn" onClick={loadGraphData} disabled={loading}>
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="graph-content">
        {error ? (
          <div className="error-state">
            <p>Error loading graph: {error}</p>
            <button onClick={loadGraphData}>Retry</button>
          </div>
        ) : loading ? (
          <div className="loading-state">
            <div className="spinner" />
            <p>Loading memory graph...</p>
          </div>
        ) : data && data.nodes.length > 0 ? (
          viewMode === '3d' ? (
            <Canvas
              camera={{ position: [0, 0, 50], fov: 60 }}
              style={{ background: COLORS.background }}
            >
              <GraphScene
                data={data}
                selectedNode={selectedNode}
                onNodeSelect={setSelectedNode}
                neuromodulatorState={neuromodulatorState}
              />
            </Canvas>
          ) : (
            <div className="stats-view">
              <div className="stats-grid">
                <div className="stat-card">
                  <h4>Total Memories</h4>
                  <span className="big-number">{data.nodes.length}</span>
                </div>
                <div className="stat-card">
                  <h4>Connections</h4>
                  <span className="big-number">{data.edges.length}</span>
                </div>
                <div className="stat-card">
                  <h4>Avg. Activation</h4>
                  <span className="big-number">
                    {(data.metrics.avg_activation * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-card">
                  <h4>Clusters</h4>
                  <span className="big-number">{data.metrics.clusters || '-'}</span>
                </div>
              </div>
              <div className="type-breakdown">
                <h4>Memory Types</h4>
                <div className="type-bars">
                  <div className="type-bar">
                    <div className="bar-label">
                      <span className="dot episodic" />
                      Episodic
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill episodic"
                        style={{ width: `${(typeCounts.episodic / Math.max(data.nodes.length, 1)) * 100}%` }}
                      />
                    </div>
                    <span className="bar-count">{typeCounts.episodic}</span>
                  </div>
                  <div className="type-bar">
                    <div className="bar-label">
                      <span className="dot semantic" />
                      Semantic
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill semantic"
                        style={{ width: `${(typeCounts.semantic / Math.max(data.nodes.length, 1)) * 100}%` }}
                      />
                    </div>
                    <span className="bar-count">{typeCounts.semantic}</span>
                  </div>
                  <div className="type-bar">
                    <div className="bar-label">
                      <span className="dot procedural" />
                      Procedural
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill procedural"
                        style={{ width: `${(typeCounts.procedural / Math.max(data.nodes.length, 1)) * 100}%` }}
                      />
                    </div>
                    <span className="bar-count">{typeCounts.procedural}</span>
                  </div>
                </div>
              </div>
            </div>
          )
        ) : (
          <div className="empty-state">
            <p>No memories found. Create some memories to visualize them here.</p>
          </div>
        )}

        {/* Details panel */}
        {selectedNodeData && (
          <DetailsPanel
            node={selectedNodeData}
            edges={data?.edges || []}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>

      {/* Footer stats */}
      <div className="stats-bar">
        <div className="stat">
          <span className="stat-value">{data?.nodes.length || 0}</span>
          <span className="stat-label">Nodes</span>
        </div>
        <div className="stat">
          <span className="stat-value">{data?.edges.length || 0}</span>
          <span className="stat-label">Edges</span>
        </div>
        <div className="stat">
          <span className="stat-value">{data?.metrics.clusters || 0}</span>
          <span className="stat-label">Clusters</span>
        </div>
        <div className="stat">
          <span className="stat-value">
            {data ? (data.metrics.avg_activation * 100).toFixed(0) : 0}%
          </span>
          <span className="stat-label">Avg Active</span>
        </div>
      </div>
    </div>
  );
};

export default MemoryGraphPage;
