/**
 * Reconsolidation Timeline
 *
 * Live event stream showing memory reconsolidation as it happens.
 * Hinton-inspired visualization of learning dynamics in real-time.
 */

import React, { useEffect, useState, useCallback } from 'react';
import './ReconsolidationTimeline.scss';

// ============================================================================
// Types
// ============================================================================

interface ReconsolidationEvent {
  id: string;
  memory_id: string;
  advantage: number;
  update_magnitude: number;
  learning_rate: number;
  timestamp: string;
  memory_type?: 'episodic' | 'semantic' | 'procedural';
}

interface TimelineProps {
  onMemorySelect?: (memoryId: string) => void;
  maxEvents?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

// ============================================================================
// Event Row Component
// ============================================================================

interface EventRowProps {
  event: ReconsolidationEvent;
  onClick?: () => void;
}

const EventRow: React.FC<EventRowProps> = ({ event, onClick }) => {
  const isPositive = event.advantage > 0;
  const magnitude = Math.abs(event.advantage);

  // Format timestamp
  const formatTime = (ts: string) => {
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  // Relative time
  const getRelativeTime = (ts: string) => {
    try {
      const now = Date.now();
      const eventTime = new Date(ts).getTime();
      const seconds = Math.floor((now - eventTime) / 1000);

      if (seconds < 5) return 'just now';
      if (seconds < 60) return `${seconds}s ago`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
      return formatTime(ts);
    } catch {
      return ts;
    }
  };

  // Magnitude indicator (1-5 bars)
  const getMagnitudeBars = (mag: number) => {
    const normalizedMag = Math.min(5, Math.ceil(mag * 10));
    return Array(5)
      .fill(0)
      .map((_, i) => (
        <span
          key={i}
          className={`mag-bar ${i < normalizedMag ? 'active' : ''} ${isPositive ? 'pos' : 'neg'}`}
        />
      ));
  };

  return (
    <div
      className={`event-row ${isPositive ? 'positive' : 'negative'}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick?.()}
    >
      {/* Direction indicator */}
      <div className={`event-direction ${isPositive ? 'up' : 'down'}`}>
        <i className={`fa fa-arrow-${isPositive ? 'up' : 'down'}`} />
      </div>

      {/* Memory info */}
      <div className="event-info">
        <span className="memory-id">{event.memory_id.slice(0, 8)}</span>
        {event.memory_type && (
          <span className={`memory-type ${event.memory_type}`}>
            {event.memory_type.slice(0, 3)}
          </span>
        )}
      </div>

      {/* Advantage value */}
      <div className={`event-advantage ${isPositive ? 'pos' : 'neg'}`}>
        {isPositive ? '+' : ''}{event.advantage.toFixed(4)}
      </div>

      {/* Magnitude bars */}
      <div className="event-magnitude">
        {getMagnitudeBars(magnitude)}
      </div>

      {/* Learning rate */}
      <div className="event-lr">
        <span className="lr-label">LR:</span>
        <span className="lr-value">{event.learning_rate.toFixed(4)}</span>
      </div>

      {/* Timestamp */}
      <div className="event-time" title={formatTime(event.timestamp)}>
        {getRelativeTime(event.timestamp)}
      </div>
    </div>
  );
};

// ============================================================================
// Stats Summary
// ============================================================================

interface StatsSummaryProps {
  events: ReconsolidationEvent[];
}

const StatsSummary: React.FC<StatsSummaryProps> = ({ events }) => {
  if (events.length === 0) return null;

  const positiveCount = events.filter((e) => e.advantage > 0).length;
  const negativeCount = events.filter((e) => e.advantage < 0).length;
  const avgMagnitude =
    events.reduce((sum, e) => sum + Math.abs(e.advantage), 0) / events.length;
  const avgLR =
    events.reduce((sum, e) => sum + e.learning_rate, 0) / events.length;

  return (
    <div className="stats-summary">
      <div className="stat">
        <span className="stat-value positive">{positiveCount}</span>
        <span className="stat-label">
          <i className="fa fa-arrow-up" /> Reinforced
        </span>
      </div>
      <div className="stat">
        <span className="stat-value negative">{negativeCount}</span>
        <span className="stat-label">
          <i className="fa fa-arrow-down" /> Weakened
        </span>
      </div>
      <div className="stat">
        <span className="stat-value">{avgMagnitude.toFixed(4)}</span>
        <span className="stat-label">Avg Magnitude</span>
      </div>
      <div className="stat">
        <span className="stat-value">{avgLR.toFixed(4)}</span>
        <span className="stat-label">Avg LR</span>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ReconsolidationTimeline: React.FC<TimelineProps> = ({
  onMemorySelect,
  maxEvents = 20,
  autoRefresh = true,
  refreshInterval = 3000,
}) => {
  const [events, setEvents] = useState<ReconsolidationEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [paused, setPaused] = useState(false);

  // Fetch reconsolidation events
  const loadEvents = useCallback(async () => {
    try {
      const response = await fetch(
        `/api/v1/viz/surgery/reconsolidation-history?limit=${maxEvents}`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Transform and deduplicate events
      const newEvents: ReconsolidationEvent[] = (data || []).map(
        (e: any, i: number) => ({
          id: `${e.memory_id}-${e.timestamp}-${i}`,
          memory_id: e.memory_id,
          advantage: e.advantage || 0,
          update_magnitude: e.update_magnitude || Math.abs(e.advantage || 0),
          learning_rate: e.learning_rate || e.effective_lr || 0.001,
          timestamp: e.timestamp,
          memory_type: e.memory_type,
        })
      );

      setEvents(newEvents);
      setLoading(false);
    } catch (err) {
      console.error('Failed to load reconsolidation events:', err);
      setLoading(false);
    }
  }, [maxEvents]);

  // Initial load
  useEffect(() => {
    loadEvents();
  }, [loadEvents]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh || paused) return;

    const interval = setInterval(loadEvents, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, paused, refreshInterval, loadEvents]);

  if (loading) {
    return (
      <div className="reconsolidation-timeline loading">
        <i className="fa fa-spinner fa-spin" />
        <span>Loading learning events...</span>
      </div>
    );
  }

  return (
    <div className="reconsolidation-timeline">
      {/* Header */}
      <div className="timeline-header">
        <h3>
          <i className="fa fa-bolt" />
          Live Reconsolidation
        </h3>
        <div className="header-controls">
          <button
            className={`pause-btn ${paused ? 'paused' : ''}`}
            onClick={() => setPaused(!paused)}
            title={paused ? 'Resume' : 'Pause'}
          >
            <i className={`fa fa-${paused ? 'play' : 'pause'}`} />
          </button>
          <button className="refresh-btn" onClick={loadEvents} title="Refresh">
            <i className="fa fa-sync" />
          </button>
        </div>
      </div>

      {/* Stats summary */}
      <StatsSummary events={events} />

      {/* Events list */}
      <div className="events-container">
        {events.length === 0 ? (
          <div className="no-events">
            <i className="fa fa-hourglass-half" />
            <span>Waiting for learning events...</span>
            <p className="hint">
              Learning occurs when memories are retrieved and prediction errors are detected.
            </p>
          </div>
        ) : (
          <div className="events-list">
            {events.map((event) => (
              <EventRow
                key={event.id}
                event={event}
                onClick={() => onMemorySelect?.(event.memory_id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer legend */}
      <div className="timeline-footer">
        <div className="legend">
          <span className="legend-item">
            <i className="fa fa-arrow-up positive" /> Reinforced
          </span>
          <span className="legend-item">
            <i className="fa fa-arrow-down negative" /> Weakened
          </span>
          <span className="legend-item">
            <span className="mag-indicator">|||</span> Magnitude
          </span>
        </div>
        <div className="status">
          {paused ? (
            <span className="status-paused">
              <i className="fa fa-pause" /> Paused
            </span>
          ) : (
            <span className="status-live">
              <span className="pulse-dot" /> Live
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ReconsolidationTimeline;
