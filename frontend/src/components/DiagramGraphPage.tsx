/**
 * DiagramGraphPage - 3D Force-Directed Architecture Graph
 *
 * Renders all parsed Mermaid diagrams as an interactive 3D force graph.
 * Supports filtering by subgraph, search, and flow analysis overlay.
 */

import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Billboard, Line } from '@react-three/drei';
import * as THREE from 'three';
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
} from 'd3-force-3d';
import './DiagramGraphPage.scss';

// ============================================================================
// Types (mirrors Python schema)
// ============================================================================

interface DiagramNode {
  id: string;
  label: string;
  node_type: string;
  subgraph: string | null;
  source_file: string;
  diagram_type: string;
  style: Record<string, string>;
  metadata: Record<string, any>;
  // Simulation coords
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface DiagramEdge {
  source: string | DiagramNode;
  target: string | DiagramNode;
  label: string;
  edge_type: string;
  weight: number;
  source_file: string;
}

interface DiagramGraph {
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  subgraphs: any[];
  metadata: {
    node_count: number;
    edge_count: number;
    subgraph_count: number;
    diagram_count: number;
  };
}

interface FlowMetrics {
  betweenness: Record<string, number>;
  degrees: Record<string, { in: number; out: number; total: number }>;
  bottlenecks: Array<{
    id: string;
    label: string;
    betweenness: number;
    in_degree: number;
    out_degree: number;
    score: number;
  }>;
  coupling_matrix: Record<string, Record<string, number>>;
  summary: {
    total_nodes: number;
    total_edges: number;
    cycle_count: number;
    max_betweenness: number;
    avg_degree: number;
  };
}

// ============================================================================
// Color scale for subgraphs
// ============================================================================

const SUBGRAPH_COLORS = [
  '#64b5f6', '#81c784', '#ffb74d', '#ef5350', '#ab47bc',
  '#26c6da', '#ff7043', '#66bb6a', '#42a5f5', '#ec407a',
  '#7e57c2', '#26a69a', '#ffa726', '#78909c', '#8d6e63',
];

function getSubgraphColor(subgraph: string | null, subgraphList: string[]): string {
  if (!subgraph) return '#888888';
  const idx = subgraphList.indexOf(subgraph);
  return SUBGRAPH_COLORS[idx % SUBGRAPH_COLORS.length];
}

// ============================================================================
// DiagramNode3D Component
// ============================================================================

interface NodeProps {
  node: DiagramNode;
  color: string;
  size: number;
  selected: boolean;
  glowIntensity: number;
  onClick: () => void;
}

const DiagramNode3D: React.FC<NodeProps> = ({ node, color, size, selected, glowIntensity, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null!);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (!meshRef.current) return;
    meshRef.current.position.set(node.x || 0, node.y || 0, node.z || 0);
    const scale = selected ? 1.4 : hovered ? 1.2 : 1.0;
    meshRef.current.scale.setScalar(scale);
  });

  const emissiveColor = glowIntensity > 0.3
    ? new THREE.Color(1, 1 - glowIntensity, 0) // yellow -> red
    : new THREE.Color(color);

  return (
    <mesh
      ref={meshRef}
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={emissiveColor}
        emissiveIntensity={glowIntensity > 0 ? glowIntensity * 2 : hovered ? 0.3 : 0.1}
        transparent
        opacity={selected ? 1.0 : 0.85}
      />
      {(hovered || selected) && (
        <Billboard>
          <Text
            position={[0, size + 0.5, 0]}
            fontSize={0.4}
            color="white"
            anchorX="center"
            anchorY="bottom"
          >
            {node.label.slice(0, 40)}
          </Text>
        </Billboard>
      )}
    </mesh>
  );
};

// ============================================================================
// Edge3D Component
// ============================================================================

interface EdgeProps {
  edge: DiagramEdge;
  nodeMap: Map<string, DiagramNode>;
  highlight: boolean;
}

const Edge3D: React.FC<EdgeProps> = ({ edge, nodeMap, highlight }) => {
  const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
  const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
  const sourceNode = nodeMap.get(sourceId);
  const targetNode = nodeMap.get(targetId);

  if (!sourceNode || !targetNode) return null;

  const points = useMemo(() => [
    new THREE.Vector3(sourceNode.x || 0, sourceNode.y || 0, sourceNode.z || 0),
    new THREE.Vector3(targetNode.x || 0, targetNode.y || 0, targetNode.z || 0),
  ], [sourceNode.x, sourceNode.y, sourceNode.z, targetNode.x, targetNode.y, targetNode.z]);

  const lineWidth = edge.edge_type === 'thick' ? 2 : edge.edge_type === 'dotted' ? 0.5 : 1;
  const color = highlight ? '#ffffff' : edge.edge_type === 'dotted' ? '#555555' : '#444444';

  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      transparent
      opacity={highlight ? 0.8 : 0.3}
      dashed={edge.edge_type === 'dotted'}
      dashSize={0.3}
      gapSize={0.2}
    />
  );
};

// ============================================================================
// GraphScene Component
// ============================================================================

interface SceneProps {
  graph: DiagramGraph;
  metrics: FlowMetrics | null;
  selectedNode: DiagramNode | null;
  onSelectNode: (node: DiagramNode | null) => void;
  showFlowOverlay: boolean;
  subgraphList: string[];
  chargeStrength: number;
  linkDistance: number;
}

const GraphScene: React.FC<SceneProps> = ({
  graph, metrics, selectedNode, onSelectNode, showFlowOverlay, subgraphList,
  chargeStrength, linkDistance,
}) => {
  const simRef = useRef<any>(null);

  const nodeMap = useMemo(() => {
    const m = new Map<string, DiagramNode>();
    graph.nodes.forEach(n => m.set(n.id, n));
    return m;
  }, [graph.nodes]);

  const connectedIds = useMemo(() => {
    if (!selectedNode) return new Set<string>();
    const ids = new Set<string>();
    graph.edges.forEach(e => {
      const sid = typeof e.source === 'string' ? e.source : e.source.id;
      const tid = typeof e.target === 'string' ? e.target : e.target.id;
      if (sid === selectedNode.id) ids.add(tid);
      if (tid === selectedNode.id) ids.add(sid);
    });
    return ids;
  }, [selectedNode, graph.edges]);

  // Initialize force simulation
  useEffect(() => {
    const sim = forceSimulation(graph.nodes as any[], 3)
      .force('link', forceLink(graph.edges as any[])
        .id((d: any) => d.id)
        .distance(linkDistance)
        .strength((e: any) => e.weight * 0.1))
      .force('charge', forceManyBody().strength(chargeStrength))
      .force('center', forceCenter())
      .force('collide', forceCollide(1.5))
      .alpha(1)
      .alphaDecay(0.02);

    simRef.current = sim;
    return () => { sim.stop(); };
  }, [graph, chargeStrength, linkDistance]);

  // Tick simulation
  useFrame(() => {
    if (simRef.current) {
      simRef.current.tick();
    }
  });

  const maxBetweenness = metrics ? Math.max(...Object.values(metrics.betweenness), 0.001) : 1;

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />

      {graph.nodes.map(node => {
        const deg = metrics?.degrees[node.id]?.total || 1;
        const size = 0.3 + Math.min(deg * 0.08, 1.2);
        const bc = metrics?.betweenness[node.id] || 0;
        const glowIntensity = showFlowOverlay ? bc / maxBetweenness : 0;

        return (
          <DiagramNode3D
            key={node.id}
            node={node}
            color={getSubgraphColor(node.subgraph, subgraphList)}
            size={size}
            selected={selectedNode?.id === node.id}
            glowIntensity={glowIntensity}
            onClick={() => onSelectNode(selectedNode?.id === node.id ? null : node)}
          />
        );
      })}

      {graph.edges.map((edge, i) => {
        const sid = typeof edge.source === 'string' ? edge.source : edge.source.id;
        const tid = typeof edge.target === 'string' ? edge.target : edge.target.id;
        const highlight = selectedNode
          ? (sid === selectedNode.id || tid === selectedNode.id)
          : false;
        return (
          <Edge3D key={`e-${i}`} edge={edge} nodeMap={nodeMap} highlight={highlight} />
        );
      })}

      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
    </>
  );
};

// ============================================================================
// DetailsPanel Component
// ============================================================================

interface DetailsPanelProps {
  node: DiagramNode | null;
  metrics: FlowMetrics | null;
  connectedNodes: DiagramNode[];
}

const DetailsPanel: React.FC<DetailsPanelProps> = ({ node, metrics, connectedNodes }) => {
  if (!node) {
    return (
      <div className="details-panel empty">
        <p>Click a node to see details</p>
      </div>
    );
  }

  const deg = metrics?.degrees[node.id];
  const bc = metrics?.betweenness[node.id] || 0;

  return (
    <div className="details-panel">
      <h3>{node.label}</h3>
      <div className="detail-row">
        <span className="label">ID:</span>
        <span className="value">{node.id}</span>
      </div>
      <div className="detail-row">
        <span className="label">Type:</span>
        <span className="value">{node.node_type}</span>
      </div>
      <div className="detail-row">
        <span className="label">Source:</span>
        <span className="value">{node.source_file}</span>
      </div>
      <div className="detail-row">
        <span className="label">Subgraph:</span>
        <span className="value">{node.subgraph || 'none'}</span>
      </div>
      {deg && (
        <>
          <div className="detail-row">
            <span className="label">In-degree:</span>
            <span className="value">{deg.in}</span>
          </div>
          <div className="detail-row">
            <span className="label">Out-degree:</span>
            <span className="value">{deg.out}</span>
          </div>
        </>
      )}
      <div className="detail-row">
        <span className="label">Betweenness:</span>
        <span className="value">{bc.toFixed(6)}</span>
      </div>
      {node.metadata?.cross_diagram && (
        <div className="detail-row cross-diagram">
          <span className="label">Cross-diagram:</span>
          <span className="value">Appears in {node.metadata.diagram_count} diagrams</span>
        </div>
      )}
      {connectedNodes.length > 0 && (
        <div className="connected">
          <h4>Connected ({connectedNodes.length})</h4>
          <ul>
            {connectedNodes.slice(0, 15).map(n => (
              <li key={n.id}>{n.label}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// FlowAnalysisPanel Component
// ============================================================================

interface FlowPanelProps {
  metrics: FlowMetrics | null;
  showOverlay: boolean;
  onToggleOverlay: () => void;
}

const FlowAnalysisPanel: React.FC<FlowPanelProps> = ({ metrics, showOverlay, onToggleOverlay }) => {
  if (!metrics) return null;

  return (
    <div className="flow-panel">
      <div className="flow-header">
        <h3>Flow Analysis</h3>
        <button
          className={`overlay-toggle ${showOverlay ? 'active' : ''}`}
          onClick={onToggleOverlay}
        >
          {showOverlay ? 'Hide Overlay' : 'Show Overlay'}
        </button>
      </div>

      <div className="flow-stats">
        <div className="stat">
          <span className="stat-value">{metrics.summary.total_nodes}</span>
          <span className="stat-label">Nodes</span>
        </div>
        <div className="stat">
          <span className="stat-value">{metrics.summary.total_edges}</span>
          <span className="stat-label">Edges</span>
        </div>
        <div className="stat">
          <span className="stat-value">{metrics.summary.cycle_count}</span>
          <span className="stat-label">Cycles</span>
        </div>
        <div className="stat">
          <span className="stat-value">{metrics.summary.avg_degree.toFixed(1)}</span>
          <span className="stat-label">Avg Degree</span>
        </div>
      </div>

      <h4>Top Bottlenecks</h4>
      <table className="bottleneck-table">
        <thead>
          <tr>
            <th>Node</th>
            <th>BC</th>
            <th>In</th>
            <th>Out</th>
          </tr>
        </thead>
        <tbody>
          {metrics.bottlenecks.slice(0, 10).map(b => (
            <tr key={b.id}>
              <td title={b.label}>{b.label.slice(0, 20)}</td>
              <td>{b.betweenness.toFixed(4)}</td>
              <td>{b.in_degree}</td>
              <td>{b.out_degree}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ============================================================================
// Main Page Component
// ============================================================================

export const DiagramGraphPage: React.FC = () => {
  const [graph, setGraph] = useState<DiagramGraph | null>(null);
  const [metrics, setMetrics] = useState<FlowMetrics | null>(null);
  const [selectedNode, setSelectedNode] = useState<DiagramNode | null>(null);
  const [showFlowOverlay, setShowFlowOverlay] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [activeSubgraph, setActiveSubgraph] = useState<string | null>(null);

  // Force sliders
  const [chargeStrength, setChargeStrength] = useState(-30);
  const [linkDistance, setLinkDistance] = useState(15);

  // Fetch data
  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [graphRes, metricsRes] = await Promise.all([
        fetch('/api/v1/diagrams/graph'),
        fetch('/api/v1/diagrams/graph/metrics'),
      ]);
      if (!graphRes.ok) throw new Error(`Graph: ${graphRes.statusText}`);
      if (!metricsRes.ok) throw new Error(`Metrics: ${metricsRes.statusText}`);
      setGraph(await graphRes.json());
      setMetrics(await metricsRes.json());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Derived data
  const subgraphList = useMemo(() => {
    if (!graph) return [];
    const sgs = new Set<string>();
    graph.nodes.forEach(n => { if (n.subgraph) sgs.add(n.subgraph); });
    return Array.from(sgs).sort();
  }, [graph]);

  const filteredGraph = useMemo(() => {
    if (!graph) return null;
    let nodes = graph.nodes;

    if (activeSubgraph) {
      nodes = nodes.filter(n => n.subgraph && n.subgraph.includes(activeSubgraph));
    }
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      nodes = nodes.filter(n =>
        n.label.toLowerCase().includes(q) || n.id.toLowerCase().includes(q)
      );
    }

    const nodeIds = new Set(nodes.map(n => n.id));
    const edges = graph.edges.filter(e => {
      const sid = typeof e.source === 'string' ? e.source : e.source.id;
      const tid = typeof e.target === 'string' ? e.target : e.target.id;
      return nodeIds.has(sid) && nodeIds.has(tid);
    });

    return { ...graph, nodes, edges };
  }, [graph, activeSubgraph, searchTerm]);

  const connectedNodes = useMemo(() => {
    if (!selectedNode || !graph) return [];
    const ids = new Set<string>();
    graph.edges.forEach(e => {
      const sid = typeof e.source === 'string' ? e.source : e.source.id;
      const tid = typeof e.target === 'string' ? e.target : e.target.id;
      if (sid === selectedNode.id) ids.add(tid);
      if (tid === selectedNode.id) ids.add(sid);
    });
    return graph.nodes.filter(n => ids.has(n.id));
  }, [selectedNode, graph]);

  if (loading) {
    return <div className="diagram-graph-page loading"><p>Loading architecture graph...</p></div>;
  }
  if (error) {
    return <div className="diagram-graph-page error"><p>Error: {error}</p></div>;
  }
  if (!filteredGraph) {
    return <div className="diagram-graph-page empty"><p>No graph data</p></div>;
  }

  return (
    <div className="diagram-graph-page">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-section">
          <h3>Search</h3>
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="sidebar-section">
          <h3>Subgraphs</h3>
          <div className="subgraph-filters">
            <button
              className={`filter-btn ${!activeSubgraph ? 'active' : ''}`}
              onClick={() => setActiveSubgraph(null)}
            >
              All ({graph?.nodes.length})
            </button>
            {subgraphList.slice(0, 20).map((sg, i) => {
              // Extract short name from "filename::SubgraphId"
              const shortName = sg.includes('::') ? sg.split('::')[1] : sg;
              return (
                <button
                  key={sg}
                  className={`filter-btn ${activeSubgraph === sg ? 'active' : ''}`}
                  onClick={() => setActiveSubgraph(activeSubgraph === sg ? null : sg)}
                  style={{ borderLeft: `3px solid ${SUBGRAPH_COLORS[i % SUBGRAPH_COLORS.length]}` }}
                >
                  {shortName}
                </button>
              );
            })}
          </div>
        </div>

        <div className="sidebar-section">
          <h3>Forces</h3>
          <label>
            Charge: {chargeStrength}
            <input type="range" min={-100} max={-5} value={chargeStrength}
              onChange={e => setChargeStrength(Number(e.target.value))} />
          </label>
          <label>
            Distance: {linkDistance}
            <input type="range" min={5} max={50} value={linkDistance}
              onChange={e => setLinkDistance(Number(e.target.value))} />
          </label>
        </div>

        <FlowAnalysisPanel
          metrics={metrics}
          showOverlay={showFlowOverlay}
          onToggleOverlay={() => setShowFlowOverlay(!showFlowOverlay)}
        />

        <DetailsPanel
          node={selectedNode}
          metrics={metrics}
          connectedNodes={connectedNodes}
        />
      </aside>

      {/* 3D Canvas */}
      <div className="canvas-container">
        <Canvas
          camera={{ position: [0, 0, 50], fov: 60 }}
          style={{ background: '#0a0a0f' }}
          onClick={() => setSelectedNode(null)}
        >
          <GraphScene
            graph={filteredGraph}
            metrics={metrics}
            selectedNode={selectedNode}
            onSelectNode={setSelectedNode}
            showFlowOverlay={showFlowOverlay}
            subgraphList={subgraphList}
            chargeStrength={chargeStrength}
            linkDistance={linkDistance}
          />
        </Canvas>

        <div className="graph-stats">
          {filteredGraph.nodes.length} nodes | {filteredGraph.edges.length} edges
          {activeSubgraph && ` | Filtered: ${activeSubgraph.split('::').pop()}`}
        </div>
      </div>
    </div>
  );
};
