/**
 * Neuromodulator Control Panel
 *
 * Comprehensive CRUD interface for all five neuromodulatory systems:
 * - Norepinephrine (NE): Arousal, novelty detection
 * - Acetylcholine (ACh): Encoding/retrieval mode switching
 * - Dopamine (DA): Reward prediction error
 * - Serotonin (5-HT): Long-term credit, patience
 * - GABA: Competitive inhibition, sparsity
 */

import React, { useEffect, useState, useCallback } from 'react';
import './NeuromodulatorControlPanel.scss';

// ============================================================================
// Types
// ============================================================================

interface NeuromodulatorState {
  dopamine_rpe: number;
  norepinephrine_gain: number;
  acetylcholine_mode: string;
  serotonin_mood: number;
  inhibition_sparsity: number;
  effective_learning_rate: number;
  exploration_exploitation: number;
  norepinephrine?: {
    current_gain: number;
    novelty_score: number;
    tonic_level: number;
    phasic_response: number;
  };
  acetylcholine?: {
    mode: string;
    encoding_level: number;
    retrieval_level: number;
  };
  serotonin?: {
    current_mood: number;
    total_outcomes: number;
    positive_rate: number;
  };
  inhibition?: {
    recent_sparsity: number;
    avg_sparsity: number;
    inhibition_events: number;
  };
}

interface NeuromodTuning {
  // NE
  ne_baseline_arousal: number;
  ne_min_gain: number;
  ne_max_gain: number;
  ne_novelty_decay: number;
  ne_phasic_decay: number;
  // ACh
  ach_baseline: number;
  ach_adaptation_rate: number;
  ach_encoding_threshold: number;
  ach_retrieval_threshold: number;
  // DA
  da_value_learning_rate: number;
  da_default_expected: number;
  da_surprise_threshold: number;
  // 5-HT
  serotonin_baseline_mood: number;
  serotonin_mood_adaptation_rate: number;
  serotonin_discount_rate: number;
  serotonin_eligibility_decay: number;
  // GABA
  inhibition_strength: number;
  sparsity_target: number;
  inhibition_temperature: number;
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
  disabled?: boolean;
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
  disabled = false,
}) => (
  <div className={`param-slider ${disabled ? 'disabled' : ''}`}>
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
      disabled={disabled}
    />
    {description && <span className="param-desc">{description}</span>}
  </div>
);

// ============================================================================
// Mode Button Component
// ============================================================================

interface ModeButtonProps {
  mode: string;
  currentMode: string;
  onClick: () => void;
  disabled?: boolean;
}

const ModeButton: React.FC<ModeButtonProps> = ({ mode, currentMode, onClick, disabled }) => (
  <button
    className={`mode-btn ${currentMode === mode ? 'active' : ''}`}
    onClick={onClick}
    disabled={disabled}
  >
    {mode.charAt(0).toUpperCase() + mode.slice(1)}
  </button>
);

// ============================================================================
// Main Component
// ============================================================================

const defaultTuning: NeuromodTuning = {
  ne_baseline_arousal: 0.5,
  ne_min_gain: 0.5,
  ne_max_gain: 2.0,
  ne_novelty_decay: 0.95,
  ne_phasic_decay: 0.7,
  ach_baseline: 0.5,
  ach_adaptation_rate: 0.2,
  ach_encoding_threshold: 0.7,
  ach_retrieval_threshold: 0.3,
  da_value_learning_rate: 0.1,
  da_default_expected: 0.5,
  da_surprise_threshold: 0.05,
  serotonin_baseline_mood: 0.5,
  serotonin_mood_adaptation_rate: 0.1,
  serotonin_discount_rate: 0.99,
  serotonin_eligibility_decay: 0.95,
  inhibition_strength: 0.5,
  sparsity_target: 0.2,
  inhibition_temperature: 1.0,
};

export const NeuromodulatorControlPanel: React.FC = () => {
  const [state, setState] = useState<NeuromodulatorState | null>(null);
  const [tuning, setTuning] = useState<NeuromodTuning>(defaultTuning);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [activeSystem, setActiveSystem] = useState<string>('ne');
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Load current state
  const loadState = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/viz/bio/neuromodulators');
      if (response.ok) {
        const data = await response.json();
        setState(data);
        setError(null);
      } else {
        throw new Error('Failed to load neuromodulator state');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load state');
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

  // Clear message after timeout
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => setMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  // Apply tuning changes
  const applyTuning = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/v1/viz/bio/neuromodulators', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tuning),
      });

      if (response.ok) {
        const result = await response.json();
        setDirty(false);
        setMessage({ type: 'success', text: `Updated: ${result.updated_systems.join(', ')}` });
        await loadState();
      } else {
        throw new Error('Failed to apply tuning');
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Update failed' });
    } finally {
      setSaving(false);
    }
  };

  // Reset all neuromodulators
  const resetAll = async () => {
    try {
      const response = await fetch('/api/v1/viz/bio/neuromodulators/reset', {
        method: 'POST',
      });

      if (response.ok) {
        setTuning(defaultTuning);
        setDirty(false);
        setMessage({ type: 'success', text: 'All systems reset to defaults' });
        await loadState();
      } else {
        throw new Error('Failed to reset');
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Reset failed' });
    }
  };

  // Force ACh mode
  const forceAchMode = async (mode: string) => {
    try {
      const response = await fetch('/api/v1/viz/bio/acetylcholine/switch-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });

      if (response.ok) {
        const result = await response.json();
        setMessage({
          type: 'success',
          text: `Mode: ${result.previous_mode} → ${result.new_mode}`,
        });
        await loadState();
      } else {
        throw new Error('Failed to switch mode');
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Mode switch failed' });
    }
  };

  // Update tuning value
  const updateTuning = (key: keyof NeuromodTuning, value: number) => {
    setTuning((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  if (loading) {
    return <div className="neuromod-panel loading">Loading neuromodulator state...</div>;
  }

  return (
    <div className="neuromod-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>Neuromodulator Control</h2>
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
            <i className="fa fa-refresh" /> Refresh
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && <div className="error-banner">{error}</div>}

      {/* Toast message */}
      {message && (
        <div className={`toast-message ${message.type}`}>
          <i className={`fa fa-${message.type === 'success' ? 'check-circle' : 'exclamation-circle'}`} />
          {message.text}
        </div>
      )}

      {/* Current State Overview */}
      <div className="state-overview">
        <div className="state-bar">
          <span className="state-label">DA (RPE)</span>
          <div className="bar-track">
            <div
              className="bar-fill da"
              style={{ width: `${Math.min(100, Math.abs(state?.dopamine_rpe || 0) * 100)}%` }}
            />
          </div>
          <span className="state-value">{(state?.dopamine_rpe || 0).toFixed(3)}</span>
        </div>
        <div className="state-bar">
          <span className="state-label">NE (Gain)</span>
          <div className="bar-track">
            <div
              className="bar-fill ne"
              style={{ width: `${Math.min(100, ((state?.norepinephrine_gain || 1) / 2) * 100)}%` }}
            />
          </div>
          <span className="state-value">{(state?.norepinephrine_gain || 1).toFixed(2)}x</span>
        </div>
        <div className="state-bar">
          <span className="state-label">ACh (Mode)</span>
          <div className="mode-display">
            <span className={`mode-tag ${state?.acetylcholine_mode || 'balanced'}`}>
              {state?.acetylcholine_mode || 'balanced'}
            </span>
          </div>
        </div>
        <div className="state-bar">
          <span className="state-label">5-HT (Mood)</span>
          <div className="bar-track">
            <div
              className="bar-fill serotonin"
              style={{ width: `${(state?.serotonin_mood || 0.5) * 100}%` }}
            />
          </div>
          <span className="state-value">{(state?.serotonin_mood || 0.5).toFixed(2)}</span>
        </div>
        <div className="state-bar">
          <span className="state-label">GABA (Sparsity)</span>
          <div className="bar-track">
            <div
              className="bar-fill gaba"
              style={{ width: `${(state?.inhibition_sparsity || 0) * 100}%` }}
            />
          </div>
          <span className="state-value">{(state?.inhibition_sparsity || 0).toFixed(2)}</span>
        </div>
      </div>

      {/* System Selector Tabs */}
      <div className="system-tabs">
        <button
          className={`tab ${activeSystem === 'ne' ? 'active' : ''}`}
          onClick={() => setActiveSystem('ne')}
        >
          NE (Arousal)
        </button>
        <button
          className={`tab ${activeSystem === 'ach' ? 'active' : ''}`}
          onClick={() => setActiveSystem('ach')}
        >
          ACh (Mode)
        </button>
        <button
          className={`tab ${activeSystem === 'da' ? 'active' : ''}`}
          onClick={() => setActiveSystem('da')}
        >
          DA (Reward)
        </button>
        <button
          className={`tab ${activeSystem === 'serotonin' ? 'active' : ''}`}
          onClick={() => setActiveSystem('serotonin')}
        >
          5-HT (Credit)
        </button>
        <button
          className={`tab ${activeSystem === 'gaba' ? 'active' : ''}`}
          onClick={() => setActiveSystem('gaba')}
        >
          GABA (Inhibition)
        </button>
      </div>

      {/* Tuning Controls */}
      <div className="tuning-section">
        {/* Norepinephrine */}
        {activeSystem === 'ne' && (
          <div className="system-controls">
            <h3>Norepinephrine (Arousal/Novelty)</h3>
            <p className="system-desc">
              Modulates arousal, novelty detection, and exploration-exploitation balance.
              Higher gain = more exploration, increased learning rate.
            </p>
            <div className="param-grid">
              <ParamSlider
                label="Baseline Arousal"
                value={tuning.ne_baseline_arousal}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => updateTuning('ne_baseline_arousal', v)}
                description="Tonic NE level when no novelty"
              />
              <ParamSlider
                label="Min Gain"
                value={tuning.ne_min_gain}
                min={0.1}
                max={1}
                step={0.1}
                unit="x"
                onChange={(v) => updateTuning('ne_min_gain', v)}
                description="Minimum arousal multiplier"
              />
              <ParamSlider
                label="Max Gain"
                value={tuning.ne_max_gain}
                min={1}
                max={5}
                step={0.5}
                unit="x"
                onChange={(v) => updateTuning('ne_max_gain', v)}
                description="Maximum arousal multiplier"
              />
              <ParamSlider
                label="Novelty Decay"
                value={tuning.ne_novelty_decay}
                min={0.8}
                max={0.99}
                step={0.01}
                onChange={(v) => updateTuning('ne_novelty_decay', v)}
                description="How fast novelty habituates"
              />
              <ParamSlider
                label="Phasic Decay"
                value={tuning.ne_phasic_decay}
                min={0.5}
                max={0.95}
                step={0.05}
                onChange={(v) => updateTuning('ne_phasic_decay', v)}
                description="Decay rate for phasic bursts"
              />
            </div>
          </div>
        )}

        {/* Acetylcholine */}
        {activeSystem === 'ach' && (
          <div className="system-controls">
            <h3>Acetylcholine (Encoding/Retrieval Mode)</h3>
            <p className="system-desc">
              Controls balance between encoding new information and retrieving stored patterns.
              High ACh = prioritize learning, Low ACh = prioritize recall.
            </p>
            <div className="mode-buttons">
              <span>Force Mode:</span>
              <ModeButton
                mode="encoding"
                currentMode={state?.acetylcholine_mode || 'balanced'}
                onClick={() => forceAchMode('encoding')}
              />
              <ModeButton
                mode="balanced"
                currentMode={state?.acetylcholine_mode || 'balanced'}
                onClick={() => forceAchMode('balanced')}
              />
              <ModeButton
                mode="retrieval"
                currentMode={state?.acetylcholine_mode || 'balanced'}
                onClick={() => forceAchMode('retrieval')}
              />
            </div>
            <div className="param-grid">
              <ParamSlider
                label="Baseline ACh"
                value={tuning.ach_baseline}
                min={0.1}
                max={0.9}
                step={0.05}
                onChange={(v) => updateTuning('ach_baseline', v)}
                description="Default ACh level"
              />
              <ParamSlider
                label="Adaptation Rate"
                value={tuning.ach_adaptation_rate}
                min={0.01}
                max={1}
                step={0.05}
                onChange={(v) => updateTuning('ach_adaptation_rate', v)}
                description="How fast mode adapts"
              />
              <ParamSlider
                label="Encoding Threshold"
                value={tuning.ach_encoding_threshold}
                min={0.5}
                max={0.9}
                step={0.05}
                onChange={(v) => updateTuning('ach_encoding_threshold', v)}
                description="ACh level for encoding mode"
              />
              <ParamSlider
                label="Retrieval Threshold"
                value={tuning.ach_retrieval_threshold}
                min={0.1}
                max={0.5}
                step={0.05}
                onChange={(v) => updateTuning('ach_retrieval_threshold', v)}
                description="ACh level for retrieval mode"
              />
            </div>
          </div>
        )}

        {/* Dopamine */}
        {activeSystem === 'da' && (
          <div className="system-controls">
            <h3>Dopamine (Reward Prediction Error)</h3>
            <p className="system-desc">
              Signals surprise: positive when outcome better than expected, negative when worse.
              Learning is proportional to prediction error, not raw reward.
            </p>
            <div className="param-grid">
              <ParamSlider
                label="Value Learning Rate"
                value={tuning.da_value_learning_rate}
                min={0.01}
                max={0.5}
                step={0.01}
                onChange={(v) => updateTuning('da_value_learning_rate', v)}
                description="Alpha for updating expectations"
              />
              <ParamSlider
                label="Default Expected"
                value={tuning.da_default_expected}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => updateTuning('da_default_expected', v)}
                description="Initial expectation for new memories"
              />
              <ParamSlider
                label="Surprise Threshold"
                value={tuning.da_surprise_threshold}
                min={0.01}
                max={0.2}
                step={0.01}
                onChange={(v) => updateTuning('da_surprise_threshold', v)}
                description="Min |RPE| to count as surprising"
              />
            </div>
          </div>
        )}

        {/* Serotonin */}
        {activeSystem === 'serotonin' && (
          <div className="system-controls">
            <h3>Serotonin (Long-term Credit Assignment)</h3>
            <p className="system-desc">
              Supports patience and long-term value estimation.
              High mood = more patience, less temporal discounting.
            </p>
            <div className="param-grid">
              <ParamSlider
                label="Baseline Mood"
                value={tuning.serotonin_baseline_mood}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => updateTuning('serotonin_baseline_mood', v)}
                description="Default mood level"
              />
              <ParamSlider
                label="Mood Adaptation"
                value={tuning.serotonin_mood_adaptation_rate}
                min={0.01}
                max={0.5}
                step={0.01}
                onChange={(v) => updateTuning('serotonin_mood_adaptation_rate', v)}
                description="How fast mood responds to outcomes"
              />
              <ParamSlider
                label="Discount Rate"
                value={tuning.serotonin_discount_rate}
                min={0.9}
                max={1}
                step={0.01}
                onChange={(v) => updateTuning('serotonin_discount_rate', v)}
                description="Gamma for temporal discounting"
              />
              <ParamSlider
                label="Eligibility Decay"
                value={tuning.serotonin_eligibility_decay}
                min={0.8}
                max={0.99}
                step={0.01}
                onChange={(v) => updateTuning('serotonin_eligibility_decay', v)}
                description="Per-hour trace decay rate"
              />
            </div>
          </div>
        )}

        {/* GABA */}
        {activeSystem === 'gaba' && (
          <div className="system-controls">
            <h3>GABA (Competitive Inhibition)</h3>
            <p className="system-desc">
              Implements winner-take-all dynamics for sparse representations.
              Higher inhibition = sparser, more distinct outputs.
            </p>
            <div className="param-grid">
              <ParamSlider
                label="Inhibition Strength"
                value={tuning.inhibition_strength}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => updateTuning('inhibition_strength', v)}
                description="How strongly winners suppress losers"
              />
              <ParamSlider
                label="Sparsity Target"
                value={tuning.sparsity_target}
                min={0.05}
                max={0.5}
                step={0.05}
                onChange={(v) => updateTuning('sparsity_target', v)}
                description="Target fraction of surviving items"
              />
              <ParamSlider
                label="Temperature"
                value={tuning.inhibition_temperature}
                min={0.1}
                max={5}
                step={0.1}
                onChange={(v) => updateTuning('inhibition_temperature', v)}
                description="Softmax temperature for competition"
              />
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="action-bar">
        <button
          className="apply-btn"
          onClick={applyTuning}
          disabled={!dirty || saving}
        >
          {saving ? 'Applying...' : 'Apply Changes'}
        </button>
        <button className="reset-btn" onClick={resetAll}>
          Reset All Systems
        </button>
      </div>
    </div>
  );
};

export default NeuromodulatorControlPanel;
