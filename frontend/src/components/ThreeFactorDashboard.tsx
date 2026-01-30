/**
 * Three-Factor Learning Dashboard
 *
 * Visualizes the core learning equation from Hinton's perspective:
 * effective_lr = eligibility × neuromod_gate × dopamine_surprise
 *
 * Shows when learning is happening and WHY it's blocked when it's not.
 */

import React, { useEffect, useState, useCallback } from 'react';
import './ThreeFactorDashboard.scss';

// ============================================================================
// Types
// ============================================================================

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

interface NeuromodulatorState {
  dopamine_rpe: number;
  norepinephrine_gain: number;
  serotonin_mood: number;
  acetylcholine_mode: string;
  inhibition_sparsity: number;
  effective_learning_rate: number;
  exploration_exploitation: number;
}

interface EligibilityTrace {
  memory_id: string;
  trace_value: number;
  age_seconds: number;
}

interface ThreeFactorState {
  eligibility: number;      // 0-1: How "hot" are the traces
  neuromodGate: number;     // 0-1: Is the system in learning mode
  dopamineSurprise: number; // -1 to 1: Prediction error
  effectiveLR: number;      // Product of the three
  learningBlocked: boolean;
  blockReason: string | null;
}

interface RecentEvent {
  id: string;
  memoryId: string;
  advantage: number;
  magnitude: number;
  timestamp: string;
}

// ============================================================================
// Gauge Component
// ============================================================================

interface GaugeProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  color: string;
  icon: string;
  subtitle?: string;
  showSign?: boolean;
}

const Gauge: React.FC<GaugeProps> = ({
  label,
  value,
  min = 0,
  max = 1,
  color,
  icon,
  subtitle,
  showSign = false,
}) => {
  // Normalize to 0-1 range for display
  const range = max - min;
  const normalized = Math.max(0, Math.min(1, (value - min) / range));
  const percentage = normalized * 100;

  // Arc calculations (semicircle)
  const radius = 45;
  const circumference = Math.PI * radius;
  const offset = circumference * (1 - normalized);

  const displayValue = showSign && value > 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
  const isLow = normalized < 0.1;

  return (
    <div className={`gauge ${isLow ? 'low' : ''}`}>
      <div className="gauge-header">
        <i className={`fa ${icon}`} style={{ color }} />
        <span className="gauge-label">{label}</span>
      </div>

      <div className="gauge-visual">
        <svg viewBox="0 0 100 60" className="gauge-arc">
          {/* Background arc */}
          <path
            d="M 5 55 A 45 45 0 0 1 95 55"
            fill="none"
            stroke="var(--color-bg-tertiary)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Value arc */}
          <path
            d="M 5 55 A 45 45 0 0 1 95 55"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          />
        </svg>
        <div className="gauge-value" style={{ color }}>
          {displayValue}
        </div>
      </div>

      {subtitle && <div className="gauge-subtitle">{subtitle}</div>}

      {isLow && (
        <div className="gauge-warning">
          <i className="fa fa-exclamation-triangle" />
          <span>Blocking learning</span>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Learning Equation Display
// ============================================================================

interface EquationProps {
  eligibility: number;
  neuromod: number;
  dopamine: number;
  result: number;
}

const LearningEquation: React.FC<EquationProps> = ({
  eligibility,
  neuromod,
  dopamine,
  result
}) => {
  const isBlocked = result < 0.001;

  return (
    <div className={`learning-equation ${isBlocked ? 'blocked' : 'active'}`}>
      <div className="equation-header">
        <i className={`fa fa-${isBlocked ? 'times-circle' : 'check-circle'}`} />
        <span>{isBlocked ? 'Learning Blocked' : 'Learning Active'}</span>
      </div>

      <div className="equation-body">
        <span className={`term ${eligibility < 0.1 ? 'blocking' : ''}`}>
          {eligibility.toFixed(2)}
        </span>
        <span className="operator">×</span>
        <span className={`term ${neuromod < 0.1 ? 'blocking' : ''}`}>
          {neuromod.toFixed(2)}
        </span>
        <span className="operator">×</span>
        <span className={`term ${Math.abs(dopamine) < 0.01 ? 'blocking' : ''}`}>
          {dopamine.toFixed(2)}
        </span>
        <span className="operator">=</span>
        <span className="result">{result.toFixed(4)}</span>
      </div>

      <div className="equation-labels">
        <span>eligibility</span>
        <span></span>
        <span>neuromod</span>
        <span></span>
        <span>δ (RPE)</span>
        <span></span>
        <span>effective LR</span>
      </div>
    </div>
  );
};

// ============================================================================
// Recent Events List
// ============================================================================

interface EventsListProps {
  events: RecentEvent[];
  onEventClick?: (memoryId: string) => void;
}

const RecentEventsList: React.FC<EventsListProps> = ({ events, onEventClick }) => {
  const formatTime = (ts: string) => {
    try {
      return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  };

  return (
    <div className="recent-events">
      <h4>
        <i className="fa fa-bolt" />
        Recent Learning Events
      </h4>

      {events.length === 0 ? (
        <div className="no-events">
          <i className="fa fa-hourglass-half" />
          <span>Waiting for learning events...</span>
        </div>
      ) : (
        <div className="events-list">
          {events.slice(0, 8).map((event) => (
            <div
              key={event.id}
              className={`event-row ${event.advantage > 0 ? 'positive' : 'negative'}`}
              onClick={() => onEventClick?.(event.memoryId)}
            >
              <span className="event-icon">
                {event.advantage > 0 ? '↑' : '↓'}
              </span>
              <span className="event-memory">{event.memoryId.slice(0, 8)}</span>
              <span className={`event-advantage ${event.advantage > 0 ? 'pos' : 'neg'}`}>
                {event.advantage > 0 ? '+' : ''}{event.advantage.toFixed(3)}
              </span>
              <span className="event-time">{formatTime(event.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ThreeFactorDashboard: React.FC = () => {
  const [threeFactorState, setThreeFactorState] = useState<ThreeFactorState>({
    eligibility: 0,
    neuromodGate: 0.5,
    dopamineSurprise: 0,
    effectiveLR: 0,
    learningBlocked: true,
    blockReason: 'Loading...',
  });

  const [recentEvents, setRecentEvents] = useState<RecentEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch and compute three-factor state
  const loadData = useCallback(async () => {
    try {
      const [statsRes, neuroRes, tracesRes, historyRes] = await Promise.all([
        fetch('/api/v1/viz/surgery/learning-stats'),
        fetch('/api/v1/viz/bio/neuromodulators'),
        fetch('/api/v1/viz/bio/eligibility/traces'),
        fetch('/api/v1/viz/surgery/reconsolidation-history?limit=10'),
      ]);

      let eligibility = 0;
      let neuromodGate = 0.5;
      let dopamineSurprise = 0;

      // Parse eligibility traces
      if (tracesRes.ok) {
        const tracesData = await tracesRes.json();
        const traces: EligibilityTrace[] = tracesData.traces || [];
        if (traces.length > 0) {
          // Average trace strength
          eligibility = traces.reduce((sum, t) => sum + t.trace_value, 0) / traces.length;
        }
      }

      // Parse neuromodulator state
      if (neuroRes.ok) {
        const neuroData: NeuromodulatorState = await neuroRes.json();
        // Neuromod gate is a combination of ACh mode and effective LR
        const achGate = neuroData.acetylcholine_mode === 'encoding' ? 0.8 : 0.4;
        neuromodGate = Math.min(1, (achGate + neuroData.effective_learning_rate) / 2);
        dopamineSurprise = neuroData.dopamine_rpe;
      }

      // Parse learning stats for additional context
      if (statsRes.ok) {
        const stats: LearningStats = await statsRes.json();
        // Use avg_rpe if available
        if (stats.dopamine?.avg_rpe !== undefined) {
          dopamineSurprise = stats.dopamine.avg_rpe;
        }
      }

      // Parse recent events
      if (historyRes.ok) {
        const history = await historyRes.json();
        const events: RecentEvent[] = (history || []).map((h: any, i: number) => ({
          id: `${h.memory_id}-${i}`,
          memoryId: h.memory_id,
          advantage: h.advantage,
          magnitude: h.update_magnitude,
          timestamp: h.timestamp,
        }));
        setRecentEvents(events);
      }

      // Compute effective learning rate
      const effectiveLR = Math.abs(eligibility * neuromodGate * dopamineSurprise);

      // Determine if learning is blocked and why
      let learningBlocked = effectiveLR < 0.001;
      let blockReason: string | null = null;

      if (learningBlocked) {
        if (eligibility < 0.1) {
          blockReason = 'No active eligibility traces - memories need recent access';
        } else if (neuromodGate < 0.1) {
          blockReason = 'Neuromodulator gate closed - system not in learning mode';
        } else if (Math.abs(dopamineSurprise) < 0.01) {
          blockReason = 'No prediction error - outcomes match expectations';
        }
      }

      setThreeFactorState({
        eligibility,
        neuromodGate,
        dopamineSurprise,
        effectiveLR,
        learningBlocked,
        blockReason,
      });

      setLoading(false);
    } catch (err) {
      console.error('Failed to load three-factor data:', err);
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(loadData, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  if (loading) {
    return (
      <div className="three-factor-dashboard loading">
        <i className="fa fa-spinner fa-spin" />
        <p>Analyzing learning dynamics...</p>
      </div>
    );
  }

  return (
    <div className="three-factor-dashboard">
      <div className="dashboard-header">
        <h2>
          <i className="fa fa-graduation-cap" />
          Three-Factor Learning Signal
        </h2>
        <div className="header-controls">
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Live
          </label>
          <button onClick={loadData} className="refresh-btn">
            <i className="fa fa-sync" />
          </button>
        </div>
      </div>

      <div className="dashboard-subtitle">
        Hinton's insight: Learning requires three factors to align
      </div>

      {/* The Three Gauges */}
      <div className="gauges-row">
        <Gauge
          label="Eligibility"
          value={threeFactorState.eligibility}
          color="#22c55e"
          icon="fa-clock"
          subtitle="Trace strength"
        />
        <Gauge
          label="Neuromod Gate"
          value={threeFactorState.neuromodGate}
          color="#6366f1"
          icon="fa-flask"
          subtitle="Learning mode"
        />
        <Gauge
          label="DA Surprise (δ)"
          value={threeFactorState.dopamineSurprise}
          min={-1}
          max={1}
          color="#f59e0b"
          icon="fa-bolt"
          subtitle="Prediction error"
          showSign
        />
      </div>

      {/* The Equation */}
      <LearningEquation
        eligibility={threeFactorState.eligibility}
        neuromod={threeFactorState.neuromodGate}
        dopamine={threeFactorState.dopamineSurprise}
        result={threeFactorState.effectiveLR}
      />

      {/* Block Reason */}
      {threeFactorState.blockReason && (
        <div className="block-reason">
          <i className="fa fa-info-circle" />
          {threeFactorState.blockReason}
        </div>
      )}

      {/* Recent Events */}
      <RecentEventsList events={recentEvents} />

      {/* Educational Footer */}
      <div className="dashboard-footer">
        <h5>How It Works</h5>
        <ul>
          <li><strong>Eligibility traces</strong> mark recently-active memories for potential update</li>
          <li><strong>Neuromodulator gate</strong> determines if the system is ready to learn</li>
          <li><strong>Dopamine surprise</strong> signals whether outcomes exceeded (or missed) expectations</li>
          <li>All three must be non-zero for learning to occur</li>
        </ul>
      </div>
    </div>
  );
};

export default ThreeFactorDashboard;
