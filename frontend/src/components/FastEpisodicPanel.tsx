/**
 * Fast Episodic Store Visualization Panel
 * Visualizes one-shot learning with salience-based eviction
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface EpisodePreview {
  id: string;
  contentPreview: string;
  salience: number;
  timestamp: string;
  replayCount: number;
}

interface FESStats {
  count: number;
  capacity: number;
  capacityUsage: number;
  avgSalience: number;
  consolidationCandidates: number;
}

interface NeuromodState {
  dopamine: number;
  norepinephrine: number;
  acetylcholine: number;
}

const API_BASE = '/api/v1/viz';

// Color based on salience
const getSalienceColor = (salience: number): string => {
  if (salience > 0.8) return '#22c55e';
  if (salience > 0.5) return '#eab308';
  if (salience > 0.3) return '#f97316';
  return '#ef4444';
};

export const FastEpisodicPanel: React.FC = () => {
  const [stats, setStats] = useState<FESStats | null>(null);
  const [recentEpisodes, setRecentEpisodes] = useState<EpisodePreview[]>([]);
  const [capacityHistory, setCapacityHistory] = useState<any[]>([]);
  const [neuromod, setNeuromod] = useState<NeuromodState>({
    dopamine: 0.5,
    norepinephrine: 0.5,
    acetylcholine: 0.5,
  });
  const [testContent, setTestContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [writing, setWriting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [statsRes, episodesRes] = await Promise.all([
        fetch(`${API_BASE}/bio/fes/stats`),
        fetch(`${API_BASE}/bio/fes/recent?limit=20`),
      ]);

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);

        // Update capacity history
        setCapacityHistory(prev => {
          const newPoint = {
            time: new Date().toLocaleTimeString(),
            count: data.count,
            usage: data.capacityUsage * 100,
            candidates: data.consolidationCandidates,
          };
          return [...prev.slice(-30), newPoint];
        });
      }

      if (episodesRes.ok) {
        const data = await episodesRes.json();
        setRecentEpisodes(data.episodes || []);
      }

      setLoading(false);
    } catch (err) {
      setError('Failed to load FES data');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, [loadData]);

  const writeEpisode = async () => {
    if (!testContent.trim()) return;

    setWriting(true);
    try {
      const res = await fetch(`${API_BASE}/bio/fes/write`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: testContent,
          neuromodState: neuromod,
        }),
      });

      if (res.ok) {
        setTestContent('');
        loadData();
      }
    } catch (err) {
      setError('Failed to write episode');
    }
    setWriting(false);
  };

  const computeSalience = (): number => {
    // Salience = DA * NE * ACh (simplified)
    return neuromod.dopamine * neuromod.norepinephrine * neuromod.acetylcholine;
  };

  // Prepare salience distribution data
  const salienceDistribution = [
    { range: '0-0.2', count: recentEpisodes.filter(e => e.salience < 0.2).length },
    { range: '0.2-0.4', count: recentEpisodes.filter(e => e.salience >= 0.2 && e.salience < 0.4).length },
    { range: '0.4-0.6', count: recentEpisodes.filter(e => e.salience >= 0.4 && e.salience < 0.6).length },
    { range: '0.6-0.8', count: recentEpisodes.filter(e => e.salience >= 0.6 && e.salience < 0.8).length },
    { range: '0.8-1.0', count: recentEpisodes.filter(e => e.salience >= 0.8).length },
  ];

  if (loading) {
    return (
      <div className="panel loading">
        <div className="spinner" />
        <p>Loading Fast Episodic Store...</p>
      </div>
    );
  }

  return (
    <div className="fes-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>
          <span className="icon">⚡</span>
          Fast Episodic Store
        </h2>
        <div className="bio-tag">One-Shot Learning</div>
      </div>

      {/* Stats Overview */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-value">{stats?.count ?? 0}</div>
          <div className="stat-label">Episodes</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.capacity ?? 10000}</div>
          <div className="stat-label">Capacity</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{((stats?.capacityUsage ?? 0) * 100).toFixed(1)}%</div>
          <div className="stat-label">Usage</div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${(stats?.capacityUsage ?? 0) * 100}%`,
                backgroundColor: (stats?.capacityUsage ?? 0) > 0.8 ? '#ef4444' : '#22c55e',
              }}
            />
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.avgSalience?.toFixed(2) ?? '0.00'}</div>
          <div className="stat-label">Avg Salience</div>
        </div>
        <div className="stat-card highlight">
          <div className="stat-value">{stats?.consolidationCandidates ?? 0}</div>
          <div className="stat-label">Ready for Consolidation</div>
        </div>
      </div>

      {/* Write Episode */}
      <div className="write-section">
        <h3>Store New Episode</h3>
        <div className="write-form">
          <textarea
            value={testContent}
            onChange={(e) => setTestContent(e.target.value)}
            placeholder="Enter episode content..."
            rows={3}
          />

          <div className="neuromod-controls">
            <h4>Neuromodulator State</h4>
            <div className="sliders">
              <div className="slider-group">
                <label>
                  Dopamine (DA)
                  <span className="value">{neuromod.dopamine.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={neuromod.dopamine}
                  onChange={(e) => setNeuromod({ ...neuromod, dopamine: parseFloat(e.target.value) })}
                  className="slider dopamine"
                />
              </div>
              <div className="slider-group">
                <label>
                  Norepinephrine (NE)
                  <span className="value">{neuromod.norepinephrine.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={neuromod.norepinephrine}
                  onChange={(e) => setNeuromod({ ...neuromod, norepinephrine: parseFloat(e.target.value) })}
                  className="slider norepinephrine"
                />
              </div>
              <div className="slider-group">
                <label>
                  Acetylcholine (ACh)
                  <span className="value">{neuromod.acetylcholine.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={neuromod.acetylcholine}
                  onChange={(e) => setNeuromod({ ...neuromod, acetylcholine: parseFloat(e.target.value) })}
                  className="slider acetylcholine"
                />
              </div>
            </div>
            <div className="predicted-salience">
              Predicted Salience: <strong>{computeSalience().toFixed(3)}</strong>
            </div>
          </div>

          <button
            onClick={writeEpisode}
            disabled={writing || !testContent.trim()}
            className="btn-write"
          >
            {writing ? '⏳ Writing...' : '⚡ Store Episode (One-Shot)'}
          </button>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-grid">
        {/* Capacity Over Time */}
        <div className="chart-card">
          <h3>Store Capacity Over Time</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={capacityHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" stroke="#606070" />
              <YAxis stroke="#606070" />
              <Tooltip contentStyle={{ background: '#1e1e2a', border: '1px solid #3a3a4a' }} />
              <Legend />
              <Area
                type="monotone"
                dataKey="count"
                stroke="#6366f1"
                fill="#6366f1"
                fillOpacity={0.3}
                name="Episode Count"
              />
              <Area
                type="monotone"
                dataKey="candidates"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.3}
                name="Consolidation Candidates"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Salience Distribution */}
        <div className="chart-card">
          <h3>Salience Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={salienceDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="range" stroke="#606070" />
              <YAxis stroke="#606070" />
              <Tooltip contentStyle={{ background: '#1e1e2a', border: '1px solid #3a3a4a' }} />
              <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]}>
                {salienceDistribution.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={getSalienceColor(0.1 + index * 0.2)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Episodes */}
      <div className="episodes-section">
        <h3>Recent Episodes</h3>
        <div className="episodes-list">
          {recentEpisodes.length === 0 ? (
            <div className="empty-state">No episodes stored yet</div>
          ) : (
            recentEpisodes.slice(0, 10).map((ep) => (
              <div key={ep.id} className="episode-item">
                <div className="episode-header">
                  <span className="episode-id">{ep.id.substring(0, 8)}...</span>
                  <span
                    className="salience-badge"
                    style={{ backgroundColor: getSalienceColor(ep.salience) }}
                  >
                    S: {ep.salience.toFixed(2)}
                  </span>
                </div>
                <div className="episode-content">{ep.contentPreview}</div>
                <div className="episode-meta">
                  <span>Replays: {ep.replayCount}</span>
                  <span>{new Date(ep.timestamp).toLocaleString()}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Biological Context */}
      <div className="bio-context">
        <h4>Biological Inspiration</h4>
        <p>
          The Fast Episodic Store implements <strong>one-shot learning</strong> (100x faster
          than semantic memory), inspired by hippocampal rapid encoding. Episodes are stored
          immediately and prioritized by salience, which is computed from neuromodulator
          levels: <code>salience = DA × NE × ACh</code>
        </p>
        <div className="bio-features">
          <div className="feature">
            <span className="icon">📥</span>
            <span>Immediate storage (no training)</span>
          </div>
          <div className="feature">
            <span className="icon">🎯</span>
            <span>Salience-based eviction</span>
          </div>
          <div className="feature">
            <span className="icon">🔄</span>
            <span>Consolidation to semantic memory</span>
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

export default FastEpisodicPanel;
