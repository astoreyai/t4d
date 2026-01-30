/**
 * Eligibility Traces Visualization Panel
 * Visualizes temporal credit assignment with trace decay dynamics
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';

interface TraceEntry {
  memoryId: string;
  value: number;
  activations: number;
  lastUpdate: string;
}

interface EligibilityStats {
  count: number;
  meanTrace: number;
  maxTrace: number;
  totalUpdates: number;
  totalCreditsAssigned: number;
  traceType: 'standard' | 'layered';
}

interface CreditAssignment {
  memoryId: string;
  credit: number;
  timestamp: string;
}

const API_BASE = '/api/v1/viz';

// Color scale for trace strength
const getTraceColor = (value: number): string => {
  if (value > 0.7) return '#22c55e'; // Strong - green
  if (value > 0.4) return '#eab308'; // Medium - yellow
  if (value > 0.1) return '#f97316'; // Weak - orange
  return '#ef4444'; // Very weak - red
};

export const EligibilityTracesPanel: React.FC = () => {
  const [traces, setTraces] = useState<TraceEntry[]>([]);
  const [stats, setStats] = useState<EligibilityStats | null>(null);
  const [recentCredits, setRecentCredits] = useState<CreditAssignment[]>([]);
  const [decayHistory, setDecayHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [useLayered, setUseLayered] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [tracesRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/bio/eligibility/traces?layered=${useLayered}`),
        fetch(`${API_BASE}/bio/eligibility/stats?layered=${useLayered}`),
      ]);

      if (tracesRes.ok) {
        const data = await tracesRes.json();
        setTraces(data.traces || []);
      }
      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);

        // Add to decay history
        setDecayHistory(prev => {
          const newPoint = {
            time: new Date().toLocaleTimeString(),
            count: data.count || 0,
            meanTrace: data.meanTrace || 0,
            maxTrace: data.maxTrace || 0,
          };
          return [...prev.slice(-30), newPoint];
        });
      }

      setLoading(false);
    } catch (err) {
      setError('Failed to load eligibility trace data');
      setLoading(false);
    }
  }, [useLayered]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 2000);
    return () => clearInterval(interval);
  }, [loadData]);

  const triggerDecay = async () => {
    try {
      await fetch(`${API_BASE}/bio/eligibility/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dt: 1.0, useLayered }),
      });
      loadData();
    } catch (err) {
      setError('Failed to trigger decay step');
    }
  };

  const assignCredit = async (reward: number) => {
    try {
      const res = await fetch(`${API_BASE}/bio/eligibility/credit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reward, useLayered }),
      });
      if (res.ok) {
        const data = await res.json();
        const newCredits = Object.entries(data.topCredits || {}).map(([id, credit]) => ({
          memoryId: id,
          credit: credit as number,
          timestamp: new Date().toISOString(),
        }));
        setRecentCredits(prev => [...newCredits, ...prev].slice(0, 10));
      }
      loadData();
    } catch (err) {
      setError('Failed to assign credit');
    }
  };

  if (loading) {
    return (
      <div className="panel loading">
        <div className="spinner" />
        <p>Loading eligibility traces...</p>
      </div>
    );
  }

  // Prepare data for bar chart
  const topTraces = traces
    .sort((a, b) => b.value - a.value)
    .slice(0, 20)
    .map(t => ({
      name: t.memoryId.substring(0, 12) + '...',
      value: t.value,
      activations: t.activations,
      fullId: t.memoryId,
    }));

  return (
    <div className="eligibility-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>
          <span className="icon">⏱️</span>
          Eligibility Traces
        </h2>
        <div className="controls">
          <label className="toggle">
            <input
              type="checkbox"
              checked={useLayered}
              onChange={() => setUseLayered(!useLayered)}
            />
            <span>Layered (Fast/Slow)</span>
          </label>
          <button onClick={loadData} className="btn-refresh">
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-value">{stats?.count ?? 0}</div>
          <div className="stat-label">Active Traces</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.meanTrace?.toFixed(3) ?? '0.000'}</div>
          <div className="stat-label">Mean Trace</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.maxTrace?.toFixed(3) ?? '0.000'}</div>
          <div className="stat-label">Max Trace</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.totalUpdates ?? 0}</div>
          <div className="stat-label">Total Updates</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.totalCreditsAssigned?.toFixed(1) ?? '0.0'}</div>
          <div className="stat-label">Credits Assigned</div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="action-row">
        <button onClick={triggerDecay} className="btn-action">
          📉 Step Decay
        </button>
        <button onClick={() => assignCredit(1.0)} className="btn-action positive">
          ➕ Reward (+1)
        </button>
        <button onClick={() => assignCredit(-1.0)} className="btn-action negative">
          ➖ Punish (-1)
        </button>
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Top Traces Bar Chart */}
        <div className="chart-card">
          <h3>Top 20 Active Traces</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topTraces} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis type="number" domain={[0, 'auto']} stroke="#606070" />
              <YAxis dataKey="name" type="category" width={100} stroke="#606070" />
              <Tooltip
                contentStyle={{ background: '#1e1e2a', border: '1px solid #3a3a4a' }}
                formatter={(value: number) => [value.toFixed(4), 'Trace Value']}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {topTraces.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getTraceColor(entry.value)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Decay Over Time */}
        <div className="chart-card">
          <h3>Trace Dynamics Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={decayHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" stroke="#606070" />
              <YAxis yAxisId="left" stroke="#606070" />
              <YAxis yAxisId="right" orientation="right" stroke="#606070" />
              <Tooltip contentStyle={{ background: '#1e1e2a', border: '1px solid #3a3a4a' }} />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="count"
                stroke="#6366f1"
                name="Active Count"
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="meanTrace"
                stroke="#22c55e"
                name="Mean Trace"
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="maxTrace"
                stroke="#ef4444"
                name="Max Trace"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Credit Assignments */}
      <div className="credits-section">
        <h3>Recent Credit Assignments</h3>
        <div className="credits-list">
          {recentCredits.length === 0 ? (
            <div className="empty-state">No credit assignments yet</div>
          ) : (
            recentCredits.map((c, i) => (
              <div key={i} className={`credit-item ${c.credit >= 0 ? 'positive' : 'negative'}`}>
                <span className="memory-id">{c.memoryId.substring(0, 16)}...</span>
                <span className="credit-value">
                  {c.credit >= 0 ? '+' : ''}{c.credit.toFixed(4)}
                </span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Biological Context */}
      <div className="bio-context">
        <h4>Biological Inspiration</h4>
        <p>
          Eligibility traces implement <strong>TD(λ)-style temporal credit assignment</strong>,
          inspired by synaptic tagging and dopamine eligibility windows in the brain.
          Traces decay exponentially (τ = {useLayered ? '5s/60s' : '20s'}), allowing
          delayed rewards to reinforce earlier memories that led to the outcome.
        </p>
        <code>
          trace(t) = trace(t-1) × e^(-dt/τ) + activity<br/>
          credit = reward × trace
        </code>
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

export default EligibilityTracesPanel;
