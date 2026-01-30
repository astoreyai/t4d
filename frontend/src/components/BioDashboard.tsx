/**
 * Biological Systems Dashboard
 * Visualizes neuromodulation, Hebbian learning, FSRS decay, and pattern separation
 */

import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  AreaChart,
  Area,
} from 'recharts';
import { EligibilityTracesPanel } from './EligibilityTracesPanel';
import { SparseEncodingPanel } from './SparseEncodingPanel';
import { FastEpisodicPanel } from './FastEpisodicPanel';
import { NeuromodulatorControlPanel } from './NeuromodulatorControlPanel';
import { HomeostaticPanel } from './HomeostaticPanel';
import './BioDashboard.scss';
import './BioinspiredPanels.scss';

type BioSubTab = 'overview' | 'eligibility' | 'sparse' | 'episodic' | 'neuromod' | 'homeostatic';

// Backend response types (raw)
interface NeuromodulatorOrchestraResponse {
  dopamine_rpe: number;
  norepinephrine_gain: number;
  acetylcholine_mode: string;
  serotonin_mood: number;
  inhibition_sparsity: number;
  effective_learning_rate: number;
  exploration_exploitation: number;
  norepinephrine?: { gain: number; novelty: number; tonic_level: number; phasic_level: number };
  acetylcholine?: { mode: string; encoding_weight: number; retrieval_weight: number };
  serotonin?: { mood: number; positive_rate: number; traces_active: number };
  inhibition?: { sparsity: number; k_winners: number; lateral_inhibition: number };
}

interface FSRSStateResponse {
  memory_id: string;
  stability: number;
  difficulty: number;
  retrievability: number;
  last_review: number;
  next_review: number;
  review_count: number;
}

interface HebbianWeightResponse {
  source_id: string;
  target_id: string;
  weight: number;
  co_activation_count: number;
  last_potentiation?: number;
  last_depression?: number;
  eligibility_trace: number;
}

interface PatternSeparationResponse {
  input_similarity: number;
  output_similarity: number;
  separation_ratio: number;
  sparsity: number;
  orthogonalization_strength: number;
}

// Transformed types for UI display
interface NeuromodulatorState {
  dopamine: number;
  norepinephrine: number;
  serotonin: number;
  acetylcholine: number;
  gaba: number;
}

interface FSRSMetrics {
  avgStability: number;
  avgRetrievability: number;
  decayRate: number;
  reviewsToday: number;
}

interface HebbianMetrics {
  avgWeight: number;
  totalConnections: number;
  strengthenedToday: number;
  learningRate: number;
}

interface PatternSeparationMetrics {
  orthogonality: number;
  sparsity: number;
  dgActivity: number;
  ca3Activity: number;
}

// Transform functions to convert backend responses to UI types
function transformNeuromodulators(data: NeuromodulatorOrchestraResponse): NeuromodulatorState {
  return {
    dopamine: Math.abs(data.dopamine_rpe),  // Use absolute RPE as activity level
    norepinephrine: Math.min(1, data.norepinephrine_gain / 2),  // Normalize gain (typically 0.5-2.0)
    serotonin: data.serotonin_mood,
    acetylcholine: data.acetylcholine_mode === 'encoding' ? 0.8 :
                   data.acetylcholine_mode === 'retrieval' ? 0.2 : 0.5,
    gaba: data.inhibition_sparsity,
  };
}

function transformFSRS(data: FSRSStateResponse[]): FSRSMetrics {
  if (!data || data.length === 0) {
    return { avgStability: 0, avgRetrievability: 0, decayRate: 0.5, reviewsToday: 0 };
  }
  const now = Date.now() / 1000;
  const today = now - (now % 86400);
  return {
    avgStability: data.reduce((sum, s) => sum + s.stability, 0) / data.length,
    avgRetrievability: data.reduce((sum, s) => sum + s.retrievability, 0) / data.length,
    decayRate: 0.5,  // Standard FSRS decay exponent
    reviewsToday: data.filter(s => s.last_review >= today).length,
  };
}

function transformHebbian(data: HebbianWeightResponse[]): HebbianMetrics {
  if (!data || data.length === 0) {
    return { avgWeight: 0, totalConnections: 0, strengthenedToday: 0, learningRate: 0.01 };
  }
  const now = Date.now() / 1000;
  const today = now - (now % 86400);
  return {
    avgWeight: data.reduce((sum, h) => sum + h.weight, 0) / data.length,
    totalConnections: data.length,
    strengthenedToday: data.filter(h => h.last_potentiation && h.last_potentiation >= today).length,
    learningRate: 0.01,  // Default learning rate
  };
}

function transformPatternSeparation(data: PatternSeparationResponse): PatternSeparationMetrics {
  return {
    orthogonality: data.orthogonalization_strength,
    sparsity: data.sparsity,
    dgActivity: data.separation_ratio > 1 ? Math.min(1, data.separation_ratio / 5) : 0.2,
    ca3Activity: 1 - data.output_similarity,  // Lower output similarity = higher CA3 activity
  };
}

const API_BASE = '/api/v1/viz';

export const BioDashboard: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<BioSubTab>('overview');
  const [neuroState, setNeuroState] = useState<NeuromodulatorState | null>(null);
  const [fsrs, setFSRS] = useState<FSRSMetrics | null>(null);
  const [hebbian, setHebbian] = useState<HebbianMetrics | null>(null);
  const [patternSep, setPatternSep] = useState<PatternSeparationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Simulated time-series data for charts
  const [activationHistory, setActivationHistory] = useState<any[]>([]);

  useEffect(() => {
    loadBioData();
    const interval = setInterval(loadBioData, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const loadBioData = async () => {
    try {
      const [neuroRes, fsrsRes, hebbianRes, patternRes] = await Promise.all([
        fetch(`${API_BASE}/bio/neuromodulators`),
        fetch(`${API_BASE}/bio/fsrs`),
        fetch(`${API_BASE}/bio/hebbian`),
        fetch(`${API_BASE}/bio/pattern-separation`),
      ]);

      let newNeuroState: NeuromodulatorState | null = null;

      if (neuroRes.ok) {
        const data: NeuromodulatorOrchestraResponse = await neuroRes.json();
        newNeuroState = transformNeuromodulators(data);
        setNeuroState(newNeuroState);
      }
      if (fsrsRes.ok) {
        const data: FSRSStateResponse[] = await fsrsRes.json();
        setFSRS(transformFSRS(data));
      }
      if (hebbianRes.ok) {
        const data: HebbianWeightResponse[] = await hebbianRes.json();
        setHebbian(transformHebbian(data));
      }
      if (patternRes.ok) {
        const data: PatternSeparationResponse = await patternRes.json();
        setPatternSep(transformPatternSeparation(data));
      }

      // Add to activation history using the new state
      if (newNeuroState) {
        setActivationHistory((prev) => {
          const newPoint = {
            time: new Date().toLocaleTimeString(),
            dopamine: newNeuroState?.dopamine ?? 0.5,
            norepinephrine: newNeuroState?.norepinephrine ?? 0.5,
            serotonin: newNeuroState?.serotonin ?? 0.5,
          };
          return [...prev.slice(-20), newPoint];
        });
      }

      setLoading(false);
    } catch (err) {
      setError('Failed to load biological metrics');
      setLoading(false);
    }
  };

  const radarData = neuroState
    ? [
        {
          subject: 'Dopamine',
          value: neuroState.dopamine * 100,
          fullMark: 100,
        },
        {
          subject: 'Norepinephrine',
          value: neuroState.norepinephrine * 100,
          fullMark: 100,
        },
        {
          subject: 'Serotonin',
          value: neuroState.serotonin * 100,
          fullMark: 100,
        },
        {
          subject: 'Acetylcholine',
          value: neuroState.acetylcholine * 100,
          fullMark: 100,
        },
        {
          subject: 'GABA',
          value: neuroState.gaba * 100,
          fullMark: 100,
        },
      ]
    : [];

  if (loading) {
    return (
      <div className="bio-dashboard loading">
        <i className="fa fa-spinner fa-spin" />
        <p>Loading biological metrics...</p>
      </div>
    );
  }

  // Render sub-panel based on active tab
  if (activeSubTab === 'eligibility') {
    return (
      <div className="bio-dashboard">
        <div className="sub-nav">
          <button onClick={() => setActiveSubTab('overview')}>← Overview</button>
          <span className="sub-title">Eligibility Traces</span>
        </div>
        <EligibilityTracesPanel />
      </div>
    );
  }

  if (activeSubTab === 'sparse') {
    return (
      <div className="bio-dashboard">
        <div className="sub-nav">
          <button onClick={() => setActiveSubTab('overview')}>← Overview</button>
          <span className="sub-title">Sparse Encoding</span>
        </div>
        <SparseEncodingPanel />
      </div>
    );
  }

  if (activeSubTab === 'episodic') {
    return (
      <div className="bio-dashboard">
        <div className="sub-nav">
          <button onClick={() => setActiveSubTab('overview')}>← Overview</button>
          <span className="sub-title">Fast Episodic Store</span>
        </div>
        <FastEpisodicPanel />
      </div>
    );
  }

  if (activeSubTab === 'neuromod') {
    return (
      <div className="bio-dashboard">
        <div className="sub-nav">
          <button onClick={() => setActiveSubTab('overview')}>← Overview</button>
          <span className="sub-title">Neuromodulator Control</span>
        </div>
        <NeuromodulatorControlPanel />
      </div>
    );
  }

  if (activeSubTab === 'homeostatic') {
    return (
      <div className="bio-dashboard">
        <div className="sub-nav">
          <button onClick={() => setActiveSubTab('overview')}>← Overview</button>
          <span className="sub-title">Homeostatic Plasticity</span>
        </div>
        <HomeostaticPanel />
      </div>
    );
  }

  return (
    <div className="bio-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h1>
          <i className="fa fa-brain" />
          Biological Systems Monitor
        </h1>
        <button onClick={loadBioData} className="refresh-btn">
          <i className="fa fa-refresh" />
          Refresh
        </button>
      </div>

      {/* Sub-panel Navigation */}
      <div className="bio-panels-nav">
        <button onClick={() => setActiveSubTab('neuromod')} className="panel-link">
          <span className="icon">🎛️</span>
          <span>Neuromod Control</span>
          <span className="arrow">→</span>
        </button>
        <button onClick={() => setActiveSubTab('homeostatic')} className="panel-link">
          <span className="icon">⚖️</span>
          <span>Homeostatic</span>
          <span className="arrow">→</span>
        </button>
        <button onClick={() => setActiveSubTab('eligibility')} className="panel-link">
          <span className="icon">⏱️</span>
          <span>Eligibility Traces</span>
          <span className="arrow">→</span>
        </button>
        <button onClick={() => setActiveSubTab('sparse')} className="panel-link">
          <span className="icon">🧠</span>
          <span>Sparse Encoding (k-WTA)</span>
          <span className="arrow">→</span>
        </button>
        <button onClick={() => setActiveSubTab('episodic')} className="panel-link">
          <span className="icon">⚡</span>
          <span>Fast Episodic Store</span>
          <span className="arrow">→</span>
        </button>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        {/* Neuromodulator Radar */}
        <div className="metric-card neuromod">
          <h3>
            <i className="fa fa-flask" />
            Neuromodulator Orchestra
          </h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#2a2a3a" />
                <PolarAngleAxis dataKey="subject" stroke="#a0a0b0" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#606070" />
                <Radar
                  name="Level"
                  dataKey="value"
                  stroke="#6366f1"
                  fill="#6366f1"
                  fillOpacity={0.4}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="neuro-levels">
            <div className="level dopamine">
              <span>DA</span>
              <div className="bar">
                <div style={{ width: `${(neuroState?.dopamine ?? 0) * 100}%` }} />
              </div>
              <span>{((neuroState?.dopamine ?? 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="level norepinephrine">
              <span>NE</span>
              <div className="bar">
                <div style={{ width: `${(neuroState?.norepinephrine ?? 0) * 100}%` }} />
              </div>
              <span>{((neuroState?.norepinephrine ?? 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="level serotonin">
              <span>5-HT</span>
              <div className="bar">
                <div style={{ width: `${(neuroState?.serotonin ?? 0) * 100}%` }} />
              </div>
              <span>{((neuroState?.serotonin ?? 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="level acetylcholine">
              <span>ACh</span>
              <div className="bar">
                <div style={{ width: `${(neuroState?.acetylcholine ?? 0) * 100}%` }} />
              </div>
              <span>{((neuroState?.acetylcholine ?? 0) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* FSRS Decay */}
        <div className="metric-card fsrs">
          <h3>
            <i className="fa fa-clock" />
            FSRS Memory Decay
          </h3>
          <div className="stat-grid">
            <div className="stat">
              <span className="value">{fsrs?.avgStability?.toFixed(1) ?? '-'}</span>
              <span className="label">Avg Stability</span>
            </div>
            <div className="stat">
              <span className="value">{((fsrs?.avgRetrievability ?? 0) * 100).toFixed(0)}%</span>
              <span className="label">Avg Retrievability</span>
            </div>
            <div className="stat">
              <span className="value">{fsrs?.decayRate?.toFixed(3) ?? '-'}</span>
              <span className="label">Decay Rate</span>
            </div>
            <div className="stat">
              <span className="value">{fsrs?.reviewsToday ?? 0}</span>
              <span className="label">Reviews Today</span>
            </div>
          </div>
          <div className="decay-formula">
            <code>R(t) = (1 + t/9S)^(-0.5)</code>
          </div>
        </div>

        {/* Hebbian Learning */}
        <div className="metric-card hebbian">
          <h3>
            <i className="fa fa-link" />
            Hebbian Learning
          </h3>
          <div className="stat-grid">
            <div className="stat">
              <span className="value">{hebbian?.avgWeight?.toFixed(3) ?? '-'}</span>
              <span className="label">Avg Weight</span>
            </div>
            <div className="stat">
              <span className="value">{hebbian?.totalConnections ?? 0}</span>
              <span className="label">Connections</span>
            </div>
            <div className="stat">
              <span className="value">{hebbian?.strengthenedToday ?? 0}</span>
              <span className="label">Strengthened</span>
            </div>
            <div className="stat">
              <span className="value">{hebbian?.learningRate?.toFixed(2) ?? '-'}</span>
              <span className="label">Learning Rate</span>
            </div>
          </div>
          <div className="learning-formula">
            <code>dw = eta * pre * post * DA</code>
          </div>
        </div>

        {/* Pattern Separation */}
        <div className="metric-card pattern-sep">
          <h3>
            <i className="fa fa-random" />
            Pattern Separation (DG/CA3)
          </h3>
          <div className="stat-grid">
            <div className="stat">
              <span className="value">{((patternSep?.orthogonality ?? 0) * 100).toFixed(0)}%</span>
              <span className="label">Orthogonality</span>
            </div>
            <div className="stat">
              <span className="value">{((patternSep?.sparsity ?? 0.04) * 100).toFixed(1)}%</span>
              <span className="label">Sparsity</span>
            </div>
            <div className="stat">
              <span className="value">{((patternSep?.dgActivity ?? 0) * 100).toFixed(0)}%</span>
              <span className="label">DG Activity</span>
            </div>
            <div className="stat">
              <span className="value">{((patternSep?.ca3Activity ?? 0) * 100).toFixed(0)}%</span>
              <span className="label">CA3 Activity</span>
            </div>
          </div>
          <div className="bio-note">
            Target: ~4% sparsity (biological DG)
          </div>
        </div>
      </div>

      {/* Time-series Chart */}
      <div className="metric-card full-width">
        <h3>
          <i className="fa fa-chart-line" />
          Neuromodulator Activity Over Time
        </h3>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={activationHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" stroke="#606070" />
              <YAxis domain={[0, 1]} stroke="#606070" />
              <Tooltip
                contentStyle={{ background: '#1e1e2a', border: '1px solid #2a2a3a' }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="dopamine"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.2}
              />
              <Area
                type="monotone"
                dataKey="norepinephrine"
                stroke="#f97316"
                fill="#f97316"
                fillOpacity={0.2}
              />
              <Area
                type="monotone"
                dataKey="serotonin"
                stroke="#eab308"
                fill="#eab308"
                fillOpacity={0.2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {error && (
        <div className="error-toast">
          <i className="fa fa-exclamation-circle" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default BioDashboard;
