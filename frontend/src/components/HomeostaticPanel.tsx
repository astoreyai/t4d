/**
 * Homeostatic Panel
 *
 * Controls for homeostatic plasticity system:
 * - BCM-like sliding threshold
 * - Embedding norm regulation
 * - Decorrelation pressure
 */

import React, { useEffect, useState, useCallback } from 'react';
import './HomeostaticPanel.scss';

interface HomeostaticState {
  mean_norm: number;
  std_norm: number;
  mean_activation: number;
  sliding_threshold: number;
  needs_scaling: boolean;
  current_scaling_factor: number;
  observations: number;
  scaling_events: number;
  decorrelation_events: number;
  config: {
    target_norm: number;
    norm_tolerance: number;
    ema_alpha: number;
    decorrelation_strength: number;
    sliding_threshold_rate: number;
  };
}

interface HomeostaticConfig {
  target_norm: number;
  norm_tolerance: number;
  ema_alpha: number;
  decorrelation_strength: number;
  sliding_threshold_rate: number;
}

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
      <span className="param-value">{value.toFixed(step < 0.01 ? 3 : 2)}{unit}</span>
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

const defaultConfig: HomeostaticConfig = {
  target_norm: 1.0,
  norm_tolerance: 0.2,
  ema_alpha: 0.01,
  decorrelation_strength: 0.01,
  sliding_threshold_rate: 0.001,
};

export const HomeostaticPanel: React.FC = () => {
  const [state, setState] = useState<HomeostaticState | null>(null);
  const [config, setConfig] = useState<HomeostaticConfig>(defaultConfig);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const loadState = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/viz/bio/homeostatic');
      if (response.ok) {
        const data: HomeostaticState = await response.json();
        setState(data);
        // Sync config from server on load
        if (data.config) {
          setConfig({
            target_norm: data.config.target_norm,
            norm_tolerance: data.config.norm_tolerance,
            ema_alpha: data.config.ema_alpha,
            decorrelation_strength: data.config.decorrelation_strength,
            sliding_threshold_rate: data.config.sliding_threshold_rate,
          });
        }
        setError(null);
      } else {
        throw new Error('Failed to load homeostatic state');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadState();
  }, [loadState]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(loadState, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadState]);

  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => setMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  const applyConfig = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/v1/viz/bio/homeostatic', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        setDirty(false);
        setMessage({ type: 'success', text: 'Homeostatic config updated' });
        await loadState();
      } else {
        throw new Error('Failed to apply config');
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Update failed' });
    } finally {
      setSaving(false);
    }
  };

  const forceScaling = async () => {
    try {
      const response = await fetch('/api/v1/viz/bio/homeostatic/force-scaling', {
        method: 'POST',
      });

      if (response.ok) {
        const result = await response.json();
        setMessage({
          type: 'success',
          text: `Scaling applied: ${result.scaling_factor.toFixed(3)}x`,
        });
        await loadState();
      } else {
        throw new Error('Failed to force scaling');
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Scaling failed' });
    }
  };

  const updateConfig = (key: keyof HomeostaticConfig, value: number) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  if (loading) {
    return <div className="homeostatic-panel loading">Loading homeostatic state...</div>;
  }

  const normDeviation = state
    ? Math.abs(state.mean_norm - config.target_norm) / config.target_norm
    : 0;
  const isInTolerance = normDeviation <= config.norm_tolerance;

  return (
    <div className="homeostatic-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>Homeostatic Plasticity</h2>
        <div className="header-actions">
          <label className="auto-refresh">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button className="refresh-btn" onClick={loadState}>
            Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      {error && <div className="error-banner">{error}</div>}

      {/* Toast */}
      {message && (
        <div className={`toast-message ${message.type}`}>
          {message.text}
        </div>
      )}

      {/* Current State Overview */}
      <div className="state-overview">
        <h3>Current State</h3>
        <div className="state-grid">
          <div className="state-item">
            <span className="state-label">Mean Norm</span>
            <span className={`state-value ${isInTolerance ? 'good' : 'warning'}`}>
              {state?.mean_norm?.toFixed(3) ?? '-'}
            </span>
          </div>
          <div className="state-item">
            <span className="state-label">Std Norm</span>
            <span className="state-value">{state?.std_norm?.toFixed(3) ?? '-'}</span>
          </div>
          <div className="state-item">
            <span className="state-label">Sliding Threshold</span>
            <span className="state-value">{state?.sliding_threshold?.toFixed(4) ?? '-'}</span>
          </div>
          <div className="state-item">
            <span className="state-label">Scaling Factor</span>
            <span className="state-value">{state?.current_scaling_factor?.toFixed(3) ?? '1.000'}x</span>
          </div>
        </div>

        {/* Scaling Indicator */}
        <div className={`scaling-indicator ${state?.needs_scaling ? 'needs-scaling' : 'stable'}`}>
          {state?.needs_scaling ? (
            <>
              <span className="indicator-icon">!</span>
              <span>Scaling recommended (norm out of tolerance)</span>
            </>
          ) : (
            <>
              <span className="indicator-icon">✓</span>
              <span>System stable (norm within tolerance)</span>
            </>
          )}
        </div>

        {/* Stats */}
        <div className="stats-row">
          <span>Observations: {state?.observations ?? 0}</span>
          <span>Scaling Events: {state?.scaling_events ?? 0}</span>
          <span>Decorrelations: {state?.decorrelation_events ?? 0}</span>
        </div>
      </div>

      {/* BCM Threshold Visualization */}
      <div className="bcm-visualization">
        <h3>BCM Threshold (Sliding)</h3>
        <div className="bcm-chart">
          <div className="bcm-axis">
            <span>LTD</span>
            <div className="bcm-bar">
              <div
                className="threshold-marker"
                style={{
                  left: `${Math.min(100, (state?.sliding_threshold ?? 0.5) * 100)}%`,
                }}
              />
              <div className="bcm-regions">
                <div className="ltd-region" style={{ width: `${(state?.sliding_threshold ?? 0.5) * 100}%` }} />
                <div className="ltp-region" style={{ width: `${100 - (state?.sliding_threshold ?? 0.5) * 100}%` }} />
              </div>
            </div>
            <span>LTP</span>
          </div>
          <p className="bcm-note">
            Activity below threshold: Long-Term Depression (LTD)<br />
            Activity above threshold: Long-Term Potentiation (LTP)
          </p>
        </div>
      </div>

      {/* Configuration Controls */}
      <div className="config-section">
        <h3>Configuration</h3>
        <div className="param-grid">
          <ParamSlider
            label="Target Norm"
            value={config.target_norm}
            min={0.5}
            max={2.0}
            step={0.1}
            onChange={(v) => updateConfig('target_norm', v)}
            description="Target embedding vector norm"
          />
          <ParamSlider
            label="Norm Tolerance"
            value={config.norm_tolerance}
            min={0.05}
            max={0.5}
            step={0.05}
            onChange={(v) => updateConfig('norm_tolerance', v)}
            description="Acceptable deviation from target"
          />
          <ParamSlider
            label="EMA Alpha"
            value={config.ema_alpha}
            min={0.001}
            max={0.1}
            step={0.001}
            onChange={(v) => updateConfig('ema_alpha', v)}
            description="Smoothing factor for running stats"
          />
          <ParamSlider
            label="Decorrelation Strength"
            value={config.decorrelation_strength}
            min={0}
            max={0.1}
            step={0.005}
            onChange={(v) => updateConfig('decorrelation_strength', v)}
            description="Pressure to orthogonalize embeddings"
          />
          <ParamSlider
            label="Threshold Rate"
            value={config.sliding_threshold_rate}
            min={0.0001}
            max={0.01}
            step={0.0001}
            onChange={(v) => updateConfig('sliding_threshold_rate', v)}
            description="BCM threshold adaptation rate"
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="action-bar">
        <button
          className="force-scaling-btn"
          onClick={forceScaling}
          disabled={!state?.needs_scaling}
        >
          Force Scaling Now
        </button>
        <div className="spacer" />
        <button
          className="apply-btn"
          onClick={applyConfig}
          disabled={!dirty || saving}
        >
          {saving ? 'Applying...' : 'Apply Config'}
        </button>
        <button
          className="reset-btn"
          onClick={() => {
            setConfig(defaultConfig);
            setDirty(true);
          }}
        >
          Reset Defaults
        </button>
      </div>
    </div>
  );
};

export default HomeostaticPanel;
