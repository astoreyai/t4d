/**
 * Sparse Encoding Visualization Panel
 * Visualizes k-WTA pattern separation inspired by hippocampal dentate gyrus
 */

import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
} from 'recharts';

interface EncodingResult {
  contentLength: number;
  sparseDim: number;
  activeCount: number;
  sparsity: number;
  targetSparsity: number;
  activeIndices?: number[];
}

interface EncoderStats {
  inputDim: number;
  hiddenDim: number;
  sparsity: number;
  expansionRatio: number;
}

const API_BASE = '/api/v1/viz';

// Generate heatmap grid from active indices
const generateHeatmapData = (indices: number[], gridSize: number = 64): number[][] => {
  const grid: number[][] = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0));
  indices.forEach(idx => {
    const row = Math.floor(idx / gridSize) % gridSize;
    const col = idx % gridSize;
    grid[row][col] = 1;
  });
  return grid;
};

export const SparseEncodingPanel: React.FC = () => {
  const [encoderStats, setEncoderStats] = useState<EncoderStats | null>(null);
  const [recentEncodings, setRecentEncodings] = useState<EncodingResult[]>([]);
  const [testContent, setTestContent] = useState('');
  const [currentEncoding, setCurrentEncoding] = useState<EncodingResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [encoding, setEncoding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/bio/sparse-encoder/stats`);
      if (res.ok) {
        const data = await res.json();
        setEncoderStats(data);
      }
      setLoading(false);
    } catch (err) {
      setError('Failed to load encoder stats');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const encodeContent = async () => {
    if (!testContent.trim()) return;

    setEncoding(true);
    try {
      const res = await fetch(`${API_BASE}/bio/encode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: testContent, returnIndices: true }),
      });

      if (res.ok) {
        const data = await res.json();
        setCurrentEncoding(data);
        setRecentEncodings(prev => [data, ...prev].slice(0, 10));
      }
    } catch (err) {
      setError('Failed to encode content');
    }
    setEncoding(false);
  };

  // Generate heatmap visualization
  const heatmapGrid = useMemo(() => {
    if (!currentEncoding?.activeIndices) return [];
    return generateHeatmapData(currentEncoding.activeIndices, 64);
  }, [currentEncoding]);

  // Sparsity distribution data
  const sparsityData = useMemo(() => {
    return recentEncodings.map((e, i) => ({
      name: `Enc ${i + 1}`,
      sparsity: e.sparsity * 100,
      activeCount: e.activeCount,
    }));
  }, [recentEncodings]);

  // Pie chart for active vs inactive
  const pieData = currentEncoding ? [
    { name: 'Active', value: currentEncoding.activeCount, fill: '#22c55e' },
    { name: 'Inactive', value: currentEncoding.sparseDim - currentEncoding.activeCount, fill: '#2a2a3a' },
  ] : [];

  if (loading) {
    return (
      <div className="panel loading">
        <div className="spinner" />
        <p>Loading sparse encoder...</p>
      </div>
    );
  }

  return (
    <div className="sparse-encoding-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>
          <span className="icon">🧠</span>
          Sparse Encoding (k-WTA)
        </h2>
        <div className="bio-tag">Hippocampal DG</div>
      </div>

      {/* Encoder Configuration */}
      <div className="config-section">
        <h3>Encoder Configuration</h3>
        <div className="config-grid">
          <div className="config-item">
            <span className="config-label">Input Dimension</span>
            <span className="config-value">{encoderStats?.inputDim ?? 1024}</span>
          </div>
          <div className="config-item">
            <span className="config-label">Hidden Dimension</span>
            <span className="config-value">{encoderStats?.hiddenDim ?? 8192}</span>
          </div>
          <div className="config-item">
            <span className="config-label">Target Sparsity</span>
            <span className="config-value">{((encoderStats?.sparsity ?? 0.02) * 100).toFixed(1)}%</span>
          </div>
          <div className="config-item">
            <span className="config-label">Expansion Ratio</span>
            <span className="config-value">{encoderStats?.expansionRatio ?? 8}x</span>
          </div>
        </div>
      </div>

      {/* Test Encoding */}
      <div className="test-section">
        <h3>Test Encoding</h3>
        <div className="input-row">
          <textarea
            value={testContent}
            onChange={(e) => setTestContent(e.target.value)}
            placeholder="Enter text to encode..."
            rows={3}
          />
          <button
            onClick={encodeContent}
            disabled={encoding || !testContent.trim()}
            className="btn-encode"
          >
            {encoding ? '⏳ Encoding...' : '🔄 Encode'}
          </button>
        </div>
      </div>

      {/* Current Encoding Results */}
      {currentEncoding && (
        <div className="encoding-results">
          <div className="results-grid">
            {/* Stats */}
            <div className="result-card">
              <h4>Encoding Statistics</h4>
              <div className="stat-list">
                <div className="stat-row">
                  <span>Content Length</span>
                  <span>{currentEncoding.contentLength} chars</span>
                </div>
                <div className="stat-row">
                  <span>Sparse Dimension</span>
                  <span>{currentEncoding.sparseDim}</span>
                </div>
                <div className="stat-row">
                  <span>Active Neurons</span>
                  <span className="highlight">{currentEncoding.activeCount}</span>
                </div>
                <div className="stat-row">
                  <span>Actual Sparsity</span>
                  <span className={
                    Math.abs(currentEncoding.sparsity - currentEncoding.targetSparsity) < 0.01
                      ? 'good' : 'warning'
                  }>
                    {(currentEncoding.sparsity * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="stat-row">
                  <span>Target Sparsity</span>
                  <span>{(currentEncoding.targetSparsity * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>

            {/* Pie Chart */}
            <div className="result-card">
              <h4>Active vs Inactive</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Activation Heatmap */}
          <div className="heatmap-section">
            <h4>Activation Pattern (64x64 grid)</h4>
            <div className="heatmap-container">
              <div className="heatmap-grid">
                {heatmapGrid.map((row, i) => (
                  <div key={i} className="heatmap-row">
                    {row.map((cell, j) => (
                      <div
                        key={j}
                        className={`heatmap-cell ${cell ? 'active' : ''}`}
                        title={`Index: ${i * 64 + j}`}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="heatmap-legend">
              <span className="legend-item inactive">Inactive</span>
              <span className="legend-item active">Active (~{currentEncoding.activeCount} neurons)</span>
            </div>
          </div>
        </div>
      )}

      {/* Sparsity History */}
      {recentEncodings.length > 0 && (
        <div className="history-section">
          <h3>Recent Encodings</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={sparsityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="name" stroke="#606070" />
              <YAxis domain={[0, 5]} stroke="#606070" label={{ value: 'Sparsity %', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{ background: '#1e1e2a', border: '1px solid #3a3a4a' }}
                formatter={(value: number) => [`${value.toFixed(2)}%`, 'Sparsity']}
              />
              <Bar dataKey="sparsity" fill="#6366f1" radius={[4, 4, 0, 0]}>
                {sparsityData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={Math.abs(entry.sparsity - 2) < 0.5 ? '#22c55e' : '#f97316'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Biological Context */}
      <div className="bio-context">
        <h4>Biological Inspiration</h4>
        <p>
          The sparse encoder implements <strong>k-Winner-Take-All (k-WTA)</strong> activation,
          inspired by the dentate gyrus (DG) of the hippocampus. Only the top ~2% of neurons
          remain active, creating orthogonal sparse representations that minimize interference
          between similar memories (pattern separation).
        </p>
        <div className="bio-stats">
          <div className="bio-stat">
            <span className="label">Biological DG sparsity:</span>
            <span className="value">1-5%</span>
          </div>
          <div className="bio-stat">
            <span className="label">Our target:</span>
            <span className="value">2%</span>
          </div>
          <div className="bio-stat">
            <span className="label">Expansion factor:</span>
            <span className="value">5-10x (ours: 8x)</span>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-toast">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}
    </div>
  );
};

export default SparseEncodingPanel;
