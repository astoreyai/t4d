/**
 * Cognitive Activity Dashboard
 *
 * Visualizes AND controls neural activity traces, learning events, and memory operations.
 * NOT OpenTelemetry - this is cognitive tracing specific to World Weaver's
 * biologically-inspired architecture.
 *
 * Features:
 * - View learning traces (dopamine, reconsolidation, eligibility)
 * - Tune neuromodulator parameters in real-time
 * - Reset/adjust individual memory values
 * - Trigger learning operations (credit assignment, decay steps)
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from 'recharts';
import './TracingDashboard.scss';

// ============================================================================
// Types
// ============================================================================

interface NeuromodSnapshot {
  timestamp: string;
  dopamine_rpe: number;
  norepinephrine_gain: number;
  serotonin_mood: number;
  acetylcholine_mode: string;
  inhibition_sparsity: number;
}

interface LearningStats {
  dopamine: {
    total_signals: number;
    positive_surprises: number;
    negative_surprises: number;
    avg_rpe: number;
    avg_surprise: number;
    memories_tracked: number;
  };
  reconsolidation: {
    total_updates: number;
    positive_updates: number;
    negative_updates: number;
    avg_update_magnitude: number;
    memories_in_cooldown: number;
  };
  learning_health: {
    dopamine_active: boolean;
    reconsolidation_active: boolean;
    positive_surprise_ratio: number;
  };
}

interface ReconsolidationUpdate {
  memory_id: string;
  advantage: number;
  learning_rate: number;
  outcome_score: number;
  update_magnitude: number;
  timestamp: string;
}

interface MemoryValue {
  memory_id: string;
  expected_value: number;
  observation_count: number;
  uncertainty: number;
  confidence: number;
}

interface NeuromodConfig {
  dopamineBaseline: number;
  norepinephrineGain: number;
  serotoninDiscount: number;
  acetylcholineThreshold: number;
  gabaInhibition: number;
}

// ============================================================================
// Parameter Slider Component
// ============================================================================

interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
  description?: string;
}

const ParamSlider: React.FC<ParamSliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  unit = '',
  onChange,
  description,
}) => (
  <div className="param-slider">
    <div className="param-header">
      <label>{label}</label>
      <span className="param-value">{value.toFixed(step < 0.1 ? 2 : 1)}{unit}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
    />
    {description && <span className="param-desc">{description}</span>}
  </div>
);

// ============================================================================
// Component
// ============================================================================

export const TracingDashboard: React.FC = () => {
  const [learningStats, setLearningStats] = useState<LearningStats | null>(null);
  const [reconHistory, setReconHistory] = useState<ReconsolidationUpdate[]>([]);
  const [valueMap, setValueMap] = useState<MemoryValue[]>([]);
  const [neuromodHistory, setNeuromodHistory] = useState<NeuromodSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<ReconsolidationUpdate | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Control panel state
  const [showControls, setShowControls] = useState(true);
  const [neuromodConfig, setNeuromodConfig] = useState<NeuromodConfig>({
    dopamineBaseline: 0.5,
    norepinephrineGain: 1.0,
    serotoninDiscount: 0.5,
    acetylcholineThreshold: 0.5,
    gabaInhibition: 0.3,
  });
  const [configDirty, setConfigDirty] = useState(false);
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Memory surgery state
  const [selectedMemoryId, setSelectedMemoryId] = useState('');
  const [newExpectedValue, setNewExpectedValue] = useState(0.5);

  // Fetch all cognitive trace data
  const loadData = useCallback(async () => {
    try {
      setError(null);

      // Fetch in parallel
      const [statsRes, historyRes, valueRes, neuromodRes, configRes] = await Promise.all([
        fetch('/api/v1/viz/surgery/learning-stats'),
        fetch('/api/v1/viz/surgery/reconsolidation-history?limit=50'),
        fetch('/api/v1/viz/surgery/dopamine-value-map?min_observations=1'),
        fetch('/api/v1/viz/bio/neuromodulators'),
        fetch('/api/v1/config'),
      ]);

      // Parse responses
      if (statsRes.ok) {
        const stats = await statsRes.json();
        setLearningStats(stats);
      }

      if (historyRes.ok) {
        const history = await historyRes.json();
        setReconHistory(history);
      }

      if (valueRes.ok) {
        const values = await valueRes.json();
        setValueMap(values.slice(0, 20)); // Top 20
      }

      if (neuromodRes.ok) {
        const neuromod = await neuromodRes.json();
        // Add to history for timeline
        setNeuromodHistory(prev => {
          const newSnapshot: NeuromodSnapshot = {
            timestamp: neuromod.timestamp || new Date().toISOString(),
            dopamine_rpe: neuromod.dopamine_rpe,
            norepinephrine_gain: neuromod.norepinephrine_gain,
            serotonin_mood: neuromod.serotonin_mood,
            acetylcholine_mode: neuromod.acetylcholine_mode,
            inhibition_sparsity: neuromod.inhibition_sparsity,
          };
          const updated = [...prev, newSnapshot].slice(-30); // Keep last 30
          return updated;
        });
      }

      if (configRes.ok) {
        const config = await configRes.json();
        if (config.neuromod && !configDirty) {
          setNeuromodConfig({
            dopamineBaseline: config.neuromod.dopamineBaseline,
            norepinephrineGain: config.neuromod.norepinephrineGain,
            serotoninDiscount: config.neuromod.serotoninDiscount,
            acetylcholineThreshold: config.neuromod.acetylcholineThreshold,
            gabaInhibition: config.neuromod.gabaInhibition,
          });
        }
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [configDirty]);

  // Initial load
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  // Clear action message after delay
  useEffect(() => {
    if (actionMessage) {
      const timer = setTimeout(() => setActionMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [actionMessage]);

  // Format timestamp for display
  const formatTime = (ts: string) => {
    try {
      return new Date(ts).toLocaleTimeString();
    } catch {
      return ts;
    }
  };

  // Update neuromod config on backend
  const applyNeuromodConfig = async () => {
    try {
      const response = await fetch('/api/v1/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ neuromod: neuromodConfig }),
      });

      if (response.ok) {
        setConfigDirty(false);
        setActionMessage({ type: 'success', text: 'Neuromodulator config updated' });
      } else {
        throw new Error('Failed to update config');
      }
    } catch (err) {
      setActionMessage({ type: 'error', text: 'Failed to update config' });
    }
  };

  // Reset dopamine value for a memory
  const resetDopamineValue = async () => {
    if (!selectedMemoryId) {
      setActionMessage({ type: 'error', text: 'Enter a memory ID' });
      return;
    }

    try {
      const response = await fetch(
        `/api/v1/viz/surgery/reset-dopamine/${selectedMemoryId}?new_expected=${newExpectedValue}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const result = await response.json();
        setActionMessage({
          type: 'success',
          text: `Reset value: ${result.details?.old_expected?.toFixed(3)} → ${newExpectedValue.toFixed(3)}`,
        });
        loadData(); // Refresh
      } else {
        throw new Error('Failed to reset');
      }
    } catch (err) {
      setActionMessage({ type: 'error', text: 'Failed to reset dopamine value' });
    }
  };

  // Clear reconsolidation cooldown
  const clearCooldown = async (memoryId: string) => {
    try {
      const response = await fetch(
        `/api/v1/viz/surgery/clear-reconsolidation-cooldown/${memoryId}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const result = await response.json();
        setActionMessage({
          type: 'success',
          text: result.details?.was_in_cooldown ? 'Cooldown cleared' : 'Memory was not in cooldown',
        });
        loadData();
      }
    } catch (err) {
      setActionMessage({ type: 'error', text: 'Failed to clear cooldown' });
    }
  };

  // Trigger eligibility credit assignment
  const assignCredit = async (reward: number) => {
    try {
      const response = await fetch(
        `/api/v1/viz/bio/eligibility/credit?reward=${reward}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const result = await response.json();
        setActionMessage({
          type: 'success',
          text: `Credit assigned: ${result.totalAssigned.toFixed(3)} to ${Object.keys(result.topCredits).length} memories`,
        });
        loadData();
      }
    } catch (err) {
      setActionMessage({ type: 'error', text: 'Failed to assign credit' });
    }
  };

  // Trigger eligibility decay step
  const decayTraces = async () => {
    try {
      const response = await fetch('/api/v1/viz/bio/eligibility/step?dt=1.0', { method: 'POST' });

      if (response.ok) {
        const result = await response.json();
        setActionMessage({
          type: 'success',
          text: `Decay applied. ${result.remaining_traces} traces remaining`,
        });
        loadData();
      }
    } catch (err) {
      setActionMessage({ type: 'error', text: 'Failed to decay traces' });
    }
  };

  // Compute derived metrics
  const surpriseRatio = learningStats?.learning_health?.positive_surprise_ratio ?? 0;
  const totalSignals = learningStats?.dopamine?.total_signals ?? 0;
  const totalUpdates = learningStats?.reconsolidation?.total_updates ?? 0;

  if (loading) {
    return (
      <div className="tracing-dashboard loading">
        <i className="fa fa-spinner fa-spin" />
        <p>Loading cognitive traces...</p>
      </div>
    );
  }

  return (
    <div className="tracing-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h1>
          <i className="fa fa-brain" />
          Cognitive Activity Traces
        </h1>
        <div className="header-actions">
          <button
            className={`toggle-btn ${showControls ? 'active' : ''}`}
            onClick={() => setShowControls(!showControls)}
          >
            <i className="fa fa-sliders-h" />
            {showControls ? 'Hide' : 'Show'} Controls
          </button>
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button className="action-btn" onClick={loadData} disabled={loading}>
            <i className="fa fa-refresh" />
            Refresh
          </button>
        </div>
      </div>

      {/* Action message toast */}
      {actionMessage && (
        <div className={`action-toast ${actionMessage.type}`}>
          <i className={`fa fa-${actionMessage.type === 'success' ? 'check-circle' : 'exclamation-circle'}`} />
          {actionMessage.text}
        </div>
      )}

      {error && (
        <div className="error-banner">
          <i className="fa fa-exclamation-triangle" />
          {error}
          <button onClick={loadData}>Retry</button>
        </div>
      )}

      {/* Control Panel */}
      {showControls && (
        <div className="control-panel">
          <div className="control-section">
            <h3>
              <i className="fa fa-flask" />
              Neuromodulator Tuning
            </h3>
            <div className="param-grid">
              <ParamSlider
                label="Dopamine Baseline"
                value={neuromodConfig.dopamineBaseline}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => {
                  setNeuromodConfig(c => ({ ...c, dopamineBaseline: v }));
                  setConfigDirty(true);
                }}
                description="Base reward expectation level"
              />
              <ParamSlider
                label="Norepinephrine Gain"
                value={neuromodConfig.norepinephrineGain}
                min={0.1}
                max={5}
                step={0.1}
                onChange={(v) => {
                  setNeuromodConfig(c => ({ ...c, norepinephrineGain: v }));
                  setConfigDirty(true);
                }}
                description="Arousal/attention multiplier"
              />
              <ParamSlider
                label="Serotonin Discount"
                value={neuromodConfig.serotoninDiscount}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => {
                  setNeuromodConfig(c => ({ ...c, serotoninDiscount: v }));
                  setConfigDirty(true);
                }}
                description="Temporal discounting factor"
              />
              <ParamSlider
                label="ACh Threshold"
                value={neuromodConfig.acetylcholineThreshold}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => {
                  setNeuromodConfig(c => ({ ...c, acetylcholineThreshold: v }));
                  setConfigDirty(true);
                }}
                description="Encoding/retrieval mode switch"
              />
              <ParamSlider
                label="GABA Inhibition"
                value={neuromodConfig.gabaInhibition}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => {
                  setNeuromodConfig(c => ({ ...c, gabaInhibition: v }));
                  setConfigDirty(true);
                }}
                description="Pattern separation strength"
              />
            </div>
            <button
              className="apply-btn"
              onClick={applyNeuromodConfig}
              disabled={!configDirty}
            >
              <i className="fa fa-save" />
              Apply Changes
            </button>
          </div>

          <div className="control-section">
            <h3>
              <i className="fa fa-tools" />
              Memory Surgery
            </h3>
            <div className="surgery-controls">
              <div className="surgery-row">
                <input
                  type="text"
                  placeholder="Memory ID (UUID)"
                  value={selectedMemoryId}
                  onChange={(e) => setSelectedMemoryId(e.target.value)}
                  className="memory-id-input"
                />
                <div className="value-input">
                  <label>New Value:</label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={newExpectedValue}
                    onChange={(e) => setNewExpectedValue(parseFloat(e.target.value))}
                  />
                </div>
                <button onClick={resetDopamineValue} className="surgery-btn">
                  <i className="fa fa-undo" />
                  Reset DA Value
                </button>
                <button
                  onClick={() => clearCooldown(selectedMemoryId)}
                  className="surgery-btn"
                  disabled={!selectedMemoryId}
                >
                  <i className="fa fa-snowflake" />
                  Clear Cooldown
                </button>
              </div>
            </div>
          </div>

          <div className="control-section">
            <h3>
              <i className="fa fa-bolt" />
              Learning Actions
            </h3>
            <div className="action-buttons">
              <button onClick={() => assignCredit(1.0)} className="action-btn positive">
                <i className="fa fa-plus" />
                Assign Positive Credit (+1)
              </button>
              <button onClick={() => assignCredit(-1.0)} className="action-btn negative">
                <i className="fa fa-minus" />
                Assign Negative Credit (-1)
              </button>
              <button onClick={decayTraces} className="action-btn">
                <i className="fa fa-clock" />
                Decay Traces (1 step)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Learning Health Summary */}
      <div className="health-summary">
        <div className={`health-indicator ${learningStats?.learning_health?.dopamine_active ? 'active' : 'inactive'}`}>
          <span className="indicator-dot" />
          <span>Dopamine</span>
          <span className="count">{totalSignals} signals</span>
        </div>
        <div className={`health-indicator ${learningStats?.learning_health?.reconsolidation_active ? 'active' : 'inactive'}`}>
          <span className="indicator-dot" />
          <span>Reconsolidation</span>
          <span className="count">{totalUpdates} updates</span>
        </div>
        <div className="health-indicator">
          <span className="indicator-dot surprise" />
          <span>Surprise Ratio</span>
          <span className="count">{(surpriseRatio * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Neuromodulator Timeline */}
      {neuromodHistory.length > 1 && (
        <div className="metric-card full-width">
          <h3>
            <i className="fa fa-wave-square" />
            Neuromodulator State Timeline
          </h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={neuromodHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#606070"
                  tickFormatter={formatTime}
                />
                <YAxis stroke="#606070" domain={[-1, 2]} />
                <Tooltip
                  contentStyle={{ background: '#1e1e2a', border: '1px solid #2a2a3a' }}
                  labelFormatter={formatTime}
                />
                <Area
                  type="monotone"
                  dataKey="dopamine_rpe"
                  stroke="#f59e0b"
                  fill="#f59e0b"
                  fillOpacity={0.3}
                  name="DA (RPE)"
                />
                <Area
                  type="monotone"
                  dataKey="norepinephrine_gain"
                  stroke="#ef4444"
                  fill="#ef4444"
                  fillOpacity={0.2}
                  name="NE (Gain)"
                />
                <Area
                  type="monotone"
                  dataKey="serotonin_mood"
                  stroke="#22c55e"
                  fill="#22c55e"
                  fillOpacity={0.2}
                  name="5-HT (Mood)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="metrics-grid">
        {/* Dopamine Stats */}
        <div className="metric-card">
          <h3>
            <i className="fa fa-bolt" />
            Dopamine Learning
          </h3>
          <div className="stat-grid">
            <div className="stat-item">
              <span className="stat-value">{learningStats?.dopamine?.total_signals ?? 0}</span>
              <span className="stat-label">Total Signals</span>
            </div>
            <div className="stat-item positive">
              <span className="stat-value">{learningStats?.dopamine?.positive_surprises ?? 0}</span>
              <span className="stat-label">Positive</span>
            </div>
            <div className="stat-item negative">
              <span className="stat-value">{learningStats?.dopamine?.negative_surprises ?? 0}</span>
              <span className="stat-label">Negative</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">
                {(learningStats?.dopamine?.avg_rpe ?? 0).toFixed(3)}
              </span>
              <span className="stat-label">Avg RPE</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">
                {(learningStats?.dopamine?.avg_surprise ?? 0).toFixed(3)}
              </span>
              <span className="stat-label">Avg |δ|</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{learningStats?.dopamine?.memories_tracked ?? 0}</span>
              <span className="stat-label">Tracked</span>
            </div>
          </div>
        </div>

        {/* Reconsolidation Stats */}
        <div className="metric-card">
          <h3>
            <i className="fa fa-sync-alt" />
            Reconsolidation
          </h3>
          <div className="stat-grid">
            <div className="stat-item">
              <span className="stat-value">{learningStats?.reconsolidation?.total_updates ?? 0}</span>
              <span className="stat-label">Updates</span>
            </div>
            <div className="stat-item positive">
              <span className="stat-value">{learningStats?.reconsolidation?.positive_updates ?? 0}</span>
              <span className="stat-label">Positive</span>
            </div>
            <div className="stat-item negative">
              <span className="stat-value">{learningStats?.reconsolidation?.negative_updates ?? 0}</span>
              <span className="stat-label">Negative</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">
                {(learningStats?.reconsolidation?.avg_update_magnitude ?? 0).toFixed(4)}
              </span>
              <span className="stat-label">Avg Δ</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{learningStats?.reconsolidation?.memories_in_cooldown ?? 0}</span>
              <span className="stat-label">Cooldown</span>
            </div>
          </div>
        </div>
      </div>

      {/* Memory Value Distribution */}
      {valueMap.length > 0 && (
        <div className="metric-card full-width">
          <h3>
            <i className="fa fa-chart-bar" />
            Memory Expected Values (Top 20)
          </h3>
          <p className="card-subtitle">
            Click a bar to select that memory for surgery operations
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={valueMap} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                <XAxis type="number" stroke="#606070" domain={[0, 1]} />
                <YAxis
                  type="category"
                  dataKey="memory_id"
                  stroke="#606070"
                  width={80}
                  tickFormatter={(id: string) => id.slice(0, 8)}
                />
                <Tooltip
                  contentStyle={{ background: '#1e1e2a', border: '1px solid #2a2a3a' }}
                  formatter={(value: number, name: string) => [
                    name === 'expected_value' ? value.toFixed(3) : value,
                    name === 'expected_value' ? 'Value' : name
                  ]}
                />
                <Bar
                  dataKey="expected_value"
                  fill="#6366f1"
                  radius={[0, 4, 4, 0]}
                  onClick={(data) => setSelectedMemoryId(data.memory_id)}
                  cursor="pointer"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Reconsolidation History */}
      <div className="metric-card full-width">
        <h3>
          <i className="fa fa-history" />
          Recent Learning Events
        </h3>
        <p className="card-subtitle">
          Click a row to view details and perform operations on that memory
        </p>
        {reconHistory.length > 0 ? (
          <div className="trace-list">
            <div className="trace-header">
              <span>Memory</span>
              <span>Advantage</span>
              <span>LR</span>
              <span>Magnitude</span>
              <span>Time</span>
            </div>
            {reconHistory.map((update, i) => (
              <div
                key={`${update.memory_id}-${i}`}
                className={`trace-row ${selectedEvent === update ? 'selected' : ''} ${update.advantage > 0 ? 'positive' : update.advantage < 0 ? 'negative' : ''}`}
                onClick={() => {
                  setSelectedEvent(update === selectedEvent ? null : update);
                  setSelectedMemoryId(update.memory_id);
                }}
              >
                <span className="memory-id">{update.memory_id.slice(0, 8)}...</span>
                <span className={`advantage ${update.advantage > 0 ? 'pos' : 'neg'}`}>
                  {update.advantage > 0 ? '+' : ''}{update.advantage.toFixed(3)}
                </span>
                <span className="lr">{update.learning_rate.toFixed(4)}</span>
                <span className="magnitude">{update.update_magnitude.toFixed(4)}</span>
                <span className="time">{formatTime(update.timestamp)}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <i className="fa fa-info-circle" />
            <p>No reconsolidation events yet. Use the system to generate learning traces.</p>
          </div>
        )}
      </div>

      {/* Event Detail Panel */}
      {selectedEvent && (
        <div className="trace-detail">
          <div className="detail-header">
            <h4>Learning Event Details</h4>
            <button onClick={() => setSelectedEvent(null)}>
              <i className="fa fa-times" />
            </button>
          </div>
          <div className="detail-content">
            <div className="detail-row">
              <span className="label">Memory ID</span>
              <code>{selectedEvent.memory_id}</code>
            </div>
            <div className="detail-row">
              <span className="label">Advantage</span>
              <span className={selectedEvent.advantage > 0 ? 'positive' : 'negative'}>
                {selectedEvent.advantage > 0 ? '+' : ''}{selectedEvent.advantage.toFixed(4)}
              </span>
            </div>
            <div className="detail-row">
              <span className="label">Learning Rate</span>
              <span>{selectedEvent.learning_rate.toFixed(4)}</span>
            </div>
            <div className="detail-row">
              <span className="label">Outcome Score</span>
              <span>{selectedEvent.outcome_score.toFixed(4)}</span>
            </div>
            <div className="detail-row">
              <span className="label">Update Magnitude</span>
              <span>{selectedEvent.update_magnitude.toFixed(6)}</span>
            </div>
            <div className="detail-row">
              <span className="label">Timestamp</span>
              <span>{new Date(selectedEvent.timestamp).toLocaleString()}</span>
            </div>

            <div className="detail-actions">
              <h5>Quick Actions</h5>
              <button onClick={() => clearCooldown(selectedEvent.memory_id)}>
                <i className="fa fa-snowflake" />
                Clear Cooldown
              </button>
              <button onClick={() => {
                setNewExpectedValue(0.5);
                resetDopamineValue();
              }}>
                <i className="fa fa-undo" />
                Reset to 0.5
              </button>
            </div>

            <div className="detail-explanation">
              <h5>Interpretation</h5>
              <p>
                {selectedEvent.advantage > 0
                  ? `This memory performed better than expected (advantage = ${selectedEvent.advantage.toFixed(3)}). Its embedding was pulled toward the query context to reinforce this association.`
                  : selectedEvent.advantage < 0
                  ? `This memory performed worse than expected (advantage = ${selectedEvent.advantage.toFixed(3)}). Its embedding was pushed away from the query context to reduce future false matches.`
                  : `This memory performed as expected. Minimal embedding update.`
                }
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Footer explanation */}
      <div className="dashboard-footer">
        <div className="explanation">
          <h4>About Cognitive Traces</h4>
          <p>
            This dashboard provides <strong>full CRUD control</strong> over World Weaver's
            learning systems. Unlike infrastructure tracing (OpenTelemetry), cognitive traces track
            <em> neural learning dynamics</em>: dopamine reward prediction errors,
            memory reconsolidation, and eligibility traces. Use the controls above to
            tune neuromodulator parameters, reset memory values, and trigger learning operations.
          </p>
        </div>
      </div>
    </div>
  );
};

export default TracingDashboard;
