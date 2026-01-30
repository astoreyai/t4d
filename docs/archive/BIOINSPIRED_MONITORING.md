# Bioinspired Systems Monitoring & Feedback

## Overview

This document defines the monitoring, feedback, and observability systems for World Weaver's bioinspired neural memory components. Real-time visibility into biological dynamics enables debugging, tuning, and validation.

---

## 1. Metrics Architecture

### 1.1 Metrics Collection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Bioinspired Components                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │Dendritic │ │  Sparse  │ │ Attractor│ │  Fast Episodic   │   │
│  │ Neuron   │ │ Encoder  │ │ Network  │ │     Store        │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       │            │            │                 │              │
│       └────────────┴────────────┴─────────────────┘              │
│                            │                                      │
│                    ┌───────▼───────┐                             │
│                    │MetricsEmitter │                             │
│                    └───────┬───────┘                             │
└────────────────────────────┼────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌──────▼──────┐ ┌─────▼─────┐
        │ Prometheus│ │   InfluxDB  │ │  Console  │
        │  (system) │ │(time-series)│ │  (debug)  │
        └─────┬─────┘ └──────┬──────┘ └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Grafana /     │
                    │   WW Dashboard  │
                    └─────────────────┘
```

### 1.2 Core Metrics

| Metric | Type | Description | Biological Target |
|--------|------|-------------|-------------------|
| `bio_sparsity_ratio` | Gauge | Current activation sparsity | 0.01-0.05 |
| `bio_learning_rate_effective` | Gauge | Modulated learning rate | Variable |
| `bio_learning_rate_ratio` | Gauge | Fast/slow learning ratio | ~100x |
| `bio_attractor_energy` | Gauge | Current attractor energy | Decreasing |
| `bio_attractor_settling_steps` | Histogram | Steps to convergence | <100 |
| `bio_eligibility_trace_magnitude` | Gauge | Active trace strength | 0-1 |
| `bio_fast_episodic_usage` | Gauge | Store capacity utilization | <10,000 |
| `bio_consolidation_candidates` | Counter | Patterns ready to consolidate | Increasing |
| `bio_dendritic_coupling` | Gauge | Context influence strength | Config-dependent |
| `bio_pattern_orthogonality` | Gauge | Average pattern decorrelation | >0.9 |

---

## 2. Real-Time Dashboard Components

### 2.1 BiologicalMetricsPanel (React Component)

```typescript
// frontend/src/components/BiologicalMetricsPanel.tsx

interface BiologicalMetrics {
  sparsity: number;
  learningRateEffective: number;
  learningRateRatio: number;
  attractorEnergy: number;
  eligibilityMagnitude: number;
  episodicUsage: number;
  consolidationCount: number;
}

interface ValidationStatus {
  sparsityValid: boolean;
  learningRatioValid: boolean;
  capacityValid: boolean;
}

const BiologicalMetricsPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<BiologicalMetrics | null>(null);
  const [validation, setValidation] = useState<ValidationStatus | null>(null);
  const [history, setHistory] = useState<BiologicalMetrics[]>([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765/bio-metrics');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMetrics(data.metrics);
      setValidation(data.validation);
      setHistory(prev => [...prev.slice(-100), data.metrics]);
    };
    return () => ws.close();
  }, []);

  return (
    <div className="bio-metrics-panel">
      <div className="metrics-header">
        <h2><i className="fa fa-dna" /> Biological Metrics</h2>
        <ValidationBadge status={validation} />
      </div>

      <div className="metrics-grid">
        <MetricCard
          label="Sparsity"
          value={metrics?.sparsity}
          target={0.02}
          tolerance={0.01}
          unit="%"
          icon="fa-th-large"
        />
        <MetricCard
          label="Learning Rate"
          value={metrics?.learningRateEffective}
          format="scientific"
          icon="fa-graduation-cap"
        />
        <MetricCard
          label="Fast/Slow Ratio"
          value={metrics?.learningRateRatio}
          target={100}
          tolerance={50}
          unit="x"
          icon="fa-tachometer-alt"
        />
        <MetricCard
          label="Attractor Energy"
          value={metrics?.attractorEnergy}
          trend="decreasing"
          icon="fa-magnet"
        />
        <MetricCard
          label="Eligibility Trace"
          value={metrics?.eligibilityMagnitude}
          range={[0, 1]}
          icon="fa-chart-line"
        />
        <MetricCard
          label="Episodic Store"
          value={metrics?.episodicUsage}
          max={10000}
          icon="fa-database"
        />
      </div>

      <div className="metrics-charts">
        <SparsityChart data={history} />
        <LearningDynamicsChart data={history} />
        <EnergyLandscapeChart data={history} />
      </div>
    </div>
  );
};
```

### 2.2 SparsityVisualization Component

```typescript
// frontend/src/components/SparsityVisualization.tsx

interface SparsityProps {
  encoding: number[];  // Sparse vector
  targetSparsity: number;
}

const SparsityVisualization: React.FC<SparsityProps> = ({
  encoding,
  targetSparsity
}) => {
  const actualSparsity = encoding.filter(x => x !== 0).length / encoding.length;
  const isValid = Math.abs(actualSparsity - targetSparsity) < 0.01;

  return (
    <div className="sparsity-viz">
      <div className="sparsity-header">
        <span>Encoding Pattern ({encoding.length} dims)</span>
        <span className={isValid ? 'valid' : 'invalid'}>
          {(actualSparsity * 100).toFixed(1)}% active
        </span>
      </div>

      <div className="sparsity-grid">
        {encoding.map((value, i) => (
          <div
            key={i}
            className={`cell ${value !== 0 ? 'active' : ''}`}
            style={{
              opacity: value !== 0 ? Math.abs(value) : 0.1,
              backgroundColor: value > 0 ? '#00d4aa' : value < 0 ? '#ff6b6b' : '#2a2f35'
            }}
            title={`dim ${i}: ${value.toFixed(4)}`}
          />
        ))}
      </div>

      <div className="sparsity-stats">
        <span>Target: {(targetSparsity * 100).toFixed(1)}%</span>
        <span>Actual: {(actualSparsity * 100).toFixed(1)}%</span>
        <span className={isValid ? 'valid' : 'warning'}>
          {isValid ? 'Within tolerance' : 'Outside target range'}
        </span>
      </div>
    </div>
  );
};
```

### 2.3 NeuromodulatorGauge Component

```typescript
// frontend/src/components/NeuromodulatorGauge.tsx

interface NeuromodulatorState {
  dopamine: number;      // 0-1
  norepinephrine: number;
  acetylcholine: number;
  serotonin: number;
  gaba: number;
}

const NeuromodulatorGauge: React.FC<{ state: NeuromodulatorState }> = ({ state }) => {
  const modulators = [
    { key: 'dopamine', label: 'DA', color: '#ff6b6b', role: 'Reward/Learning' },
    { key: 'norepinephrine', label: 'NE', color: '#ffd93d', role: 'Attention/Arousal' },
    { key: 'acetylcholine', label: 'ACh', color: '#6bcb77', role: 'Memory/Encoding' },
    { key: 'serotonin', label: '5-HT', color: '#4d96ff', role: 'Mood/Stability' },
    { key: 'gaba', label: 'GABA', color: '#9c88ff', role: 'Inhibition' },
  ];

  return (
    <div className="neuromod-gauge">
      <h3>Neuromodulator State</h3>
      <div className="gauge-container">
        {modulators.map(mod => (
          <div key={mod.key} className="modulator">
            <div className="mod-header">
              <span className="mod-label">{mod.label}</span>
              <span className="mod-value">
                {(state[mod.key as keyof NeuromodulatorState] * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mod-bar">
              <div
                className="mod-fill"
                style={{
                  width: `${state[mod.key as keyof NeuromodulatorState] * 100}%`,
                  backgroundColor: mod.color
                }}
              />
            </div>
            <span className="mod-role">{mod.role}</span>
          </div>
        ))}
      </div>

      <div className="effective-lr">
        <span>Effective Learning Rate:</span>
        <code>
          η_eff = η_base × {state.dopamine.toFixed(2)} × {state.norepinephrine.toFixed(2)} × {state.acetylcholine.toFixed(2)}
        </code>
      </div>
    </div>
  );
};
```

### 2.4 AttractorDynamicsPanel Component

```typescript
// frontend/src/components/AttractorDynamicsPanel.tsx

interface AttractorState {
  currentEnergy: number;
  energyHistory: number[];
  settlingStep: number;
  totalSteps: number;
  convergenceThreshold: number;
  patternCount: number;
  capacity: number;
}

const AttractorDynamicsPanel: React.FC<{ state: AttractorState }> = ({ state }) => {
  const convergenceProgress = (state.settlingStep / state.totalSteps) * 100;
  const capacityUsage = (state.patternCount / state.capacity) * 100;

  return (
    <div className="attractor-panel">
      <h3><i className="fa fa-magnet" /> Attractor Dynamics</h3>

      <div className="energy-landscape">
        <EnergyChart
          data={state.energyHistory}
          threshold={state.convergenceThreshold}
        />
        <div className="energy-stats">
          <span>Current: {state.currentEnergy.toFixed(4)}</span>
          <span>Δ: {(state.energyHistory.slice(-2).reduce((a, b) => a - b, 0)).toFixed(6)}</span>
        </div>
      </div>

      <div className="settling-progress">
        <label>Settling Progress</label>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${convergenceProgress}%` }}
          />
        </div>
        <span>{state.settlingStep} / {state.totalSteps} steps</span>
      </div>

      <div className="capacity-meter">
        <label>Pattern Capacity</label>
        <div className="capacity-bar">
          <div
            className={`capacity-fill ${capacityUsage > 80 ? 'warning' : ''}`}
            style={{ width: `${capacityUsage}%` }}
          />
        </div>
        <span>{state.patternCount} / {state.capacity} patterns ({capacityUsage.toFixed(0)}%)</span>
      </div>
    </div>
  );
};
```

---

## 3. Alerting System

### 3.1 Alert Rules

```yaml
# monitoring/alerts/bioinspired.yml

groups:
  - name: bioinspired_alerts
    rules:
      # Sparsity out of range
      - alert: SparsityOutOfRange
        expr: bio_sparsity_ratio < 0.01 or bio_sparsity_ratio > 0.10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Sparsity outside biological range"
          description: "Current sparsity {{ $value }} is outside 1-10% range"

      # Learning rate ratio degraded
      - alert: LearningRatioLow
        expr: bio_learning_rate_ratio < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Fast/slow learning ratio too low"
          description: "Ratio {{ $value }}x is below 50x threshold (target: ~100x)"

      # Attractor not converging
      - alert: AttractorNotConverging
        expr: increase(bio_attractor_energy[5m]) > 0
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Attractor energy increasing"
          description: "Energy should decrease during settling, currently rising"

      # Episodic store near capacity
      - alert: EpisodicStoreNearCapacity
        expr: bio_fast_episodic_usage / bio_fast_episodic_capacity > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Fast episodic store near capacity"
          description: "Store at {{ $value | humanizePercentage }} capacity"

      # Eligibility trace vanished
      - alert: EligibilityTraceVanished
        expr: bio_eligibility_trace_magnitude < 0.001
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "No active eligibility traces"
          description: "All traces have decayed - no recent activity"

      # Pattern orthogonality degraded
      - alert: PatternOrthogonalityLow
        expr: bio_pattern_orthogonality < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Pattern orthogonality degraded"
          description: "Average pattern correlation {{ $value }} (target: >0.9)"
```

### 3.2 Alert Handler

```python
# src/monitoring/alerts.py

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: Severity
    message: str
    metric_value: float
    threshold: float
    timestamp: float

class BioinspiredAlertHandler:
    """Handles alerts from bioinspired monitoring."""

    def __init__(self):
        self.handlers: Dict[Severity, List[Callable]] = {
            Severity.INFO: [],
            Severity.WARNING: [],
            Severity.CRITICAL: [],
        }
        self.active_alerts: Dict[str, Alert] = {}

    def register_handler(self, severity: Severity, handler: Callable):
        """Register handler for severity level."""
        self.handlers[severity].append(handler)

    def emit(self, alert: Alert):
        """Emit alert to registered handlers."""
        self.active_alerts[alert.name] = alert

        for handler in self.handlers[alert.severity]:
            handler(alert)

    def resolve(self, alert_name: str):
        """Mark alert as resolved."""
        if alert_name in self.active_alerts:
            del self.active_alerts[alert_name]

    def check_sparsity(self, value: float, target: float = 0.02, tolerance: float = 0.01):
        """Check sparsity against biological target."""
        if abs(value - target) > tolerance * 3:
            self.emit(Alert(
                name="SparsityOutOfRange",
                severity=Severity.WARNING,
                message=f"Sparsity {value:.2%} outside range [{target-tolerance:.2%}, {target+tolerance:.2%}]",
                metric_value=value,
                threshold=target,
                timestamp=time.time()
            ))
        else:
            self.resolve("SparsityOutOfRange")

    def check_learning_ratio(self, fast_lr: float, slow_lr: float, target_ratio: float = 100):
        """Check fast/slow learning rate separation."""
        ratio = fast_lr / slow_lr if slow_lr > 0 else 0

        if ratio < target_ratio / 2:
            self.emit(Alert(
                name="LearningRatioLow",
                severity=Severity.WARNING,
                message=f"Learning ratio {ratio:.1f}x below target {target_ratio}x",
                metric_value=ratio,
                threshold=target_ratio,
                timestamp=time.time()
            ))
        else:
            self.resolve("LearningRatioLow")
```

---

## 4. Logging Strategy

### 4.1 Structured Logging

```python
# src/monitoring/logging.py

import structlog
from typing import Any, Dict

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class BioinspiredLogger:
    """Structured logging for bioinspired components."""

    def __init__(self, component: str):
        self.log = structlog.get_logger(component)

    def encoding_event(
        self,
        input_dim: int,
        output_dim: int,
        sparsity: float,
        latency_ms: float,
        metadata: Dict[str, Any] = None
    ):
        """Log sparse encoding event."""
        self.log.info(
            "sparse_encoding",
            input_dim=input_dim,
            output_dim=output_dim,
            sparsity=sparsity,
            sparsity_valid=0.01 < sparsity < 0.05,
            latency_ms=latency_ms,
            **(metadata or {})
        )

    def learning_event(
        self,
        effective_lr: float,
        base_lr: float,
        modulation_factor: float,
        weight_delta_norm: float
    ):
        """Log modulated learning event."""
        self.log.info(
            "modulated_learning",
            effective_lr=effective_lr,
            base_lr=base_lr,
            modulation_factor=modulation_factor,
            weight_delta_norm=weight_delta_norm,
            lr_ratio=modulation_factor
        )

    def attractor_settling(
        self,
        initial_energy: float,
        final_energy: float,
        steps: int,
        converged: bool
    ):
        """Log attractor network settling."""
        self.log.info(
            "attractor_settling",
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_delta=initial_energy - final_energy,
            steps=steps,
            converged=converged
        )

    def consolidation_event(
        self,
        pattern_id: str,
        source: str,
        destination: str,
        success: bool
    ):
        """Log memory consolidation."""
        self.log.info(
            "consolidation",
            pattern_id=pattern_id,
            source=source,
            destination=destination,
            success=success
        )
```

### 4.2 Log Aggregation Query Examples

```sql
-- Kibana/Elasticsearch queries for bioinspired monitoring

-- Sparsity distribution over time
SELECT
  date_histogram(timestamp, '1m') as time,
  avg(sparsity) as avg_sparsity,
  percentile(sparsity, 95) as p95_sparsity,
  count(*) as encoding_count
FROM bioinspired_logs
WHERE event = 'sparse_encoding'
GROUP BY time

-- Learning rate modulation patterns
SELECT
  modulation_factor,
  count(*) as count,
  avg(weight_delta_norm) as avg_delta
FROM bioinspired_logs
WHERE event = 'modulated_learning'
GROUP BY modulation_factor
ORDER BY modulation_factor

-- Attractor convergence rate
SELECT
  converged,
  avg(steps) as avg_steps,
  avg(initial_energy - final_energy) as avg_energy_drop
FROM bioinspired_logs
WHERE event = 'attractor_settling'
GROUP BY converged
```

---

## 5. WebSocket API for Real-Time Updates

### 5.1 Backend Handler

```python
# src/api/websocket.py

from fastapi import WebSocket
import asyncio
import json

class BioinspiredMetricsSocket:
    """WebSocket handler for real-time bioinspired metrics."""

    def __init__(self, metrics_collector):
        self.collector = metrics_collector
        self.clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await websocket.accept()
        self.clients.append(websocket)

        try:
            while True:
                metrics = self.collector.get_current_metrics()
                validation = self.validate_metrics(metrics)

                await websocket.send_json({
                    "metrics": metrics.to_dict(),
                    "validation": validation.to_dict(),
                    "timestamp": time.time()
                })

                await asyncio.sleep(0.1)  # 10 Hz update rate

        except Exception:
            self.clients.remove(websocket)

    def validate_metrics(self, metrics) -> ValidationStatus:
        """Validate metrics against biological targets."""
        return ValidationStatus(
            sparsity_valid=0.01 < metrics.sparsity < 0.05,
            learning_ratio_valid=metrics.learning_ratio > 50,
            capacity_valid=metrics.episodic_usage < metrics.episodic_capacity * 0.95,
            attractor_stable=metrics.energy_delta <= 0
        )


# Router setup
@router.websocket("/bio-metrics")
async def bio_metrics_endpoint(websocket: WebSocket):
    handler = BioinspiredMetricsSocket(metrics_collector)
    await handler.connect(websocket)
```

---

## 6. Grafana Dashboard JSON

```json
{
  "title": "World Weaver - Bioinspired Metrics",
  "panels": [
    {
      "title": "Sparsity Over Time",
      "type": "timeseries",
      "targets": [
        {
          "expr": "bio_sparsity_ratio",
          "legendFormat": "Actual Sparsity"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "steps": [
              {"color": "red", "value": 0},
              {"color": "yellow", "value": 0.01},
              {"color": "green", "value": 0.015},
              {"color": "yellow", "value": 0.04},
              {"color": "red", "value": 0.06}
            ]
          }
        }
      }
    },
    {
      "title": "Learning Rate Dynamics",
      "type": "timeseries",
      "targets": [
        {
          "expr": "bio_learning_rate_effective",
          "legendFormat": "Effective LR"
        },
        {
          "expr": "bio_learning_rate_ratio",
          "legendFormat": "Fast/Slow Ratio"
        }
      ]
    },
    {
      "title": "Attractor Energy Landscape",
      "type": "timeseries",
      "targets": [
        {
          "expr": "bio_attractor_energy",
          "legendFormat": "Network Energy"
        }
      ]
    },
    {
      "title": "Fast Episodic Store Usage",
      "type": "gauge",
      "targets": [
        {
          "expr": "bio_fast_episodic_usage / bio_fast_episodic_capacity",
          "legendFormat": "Usage %"
        }
      ],
      "options": {
        "showThresholdLabels": true,
        "showThresholdMarkers": true
      }
    },
    {
      "title": "Neuromodulator Levels",
      "type": "bargauge",
      "targets": [
        {"expr": "bio_neuromod_da", "legendFormat": "Dopamine"},
        {"expr": "bio_neuromod_ne", "legendFormat": "Norepinephrine"},
        {"expr": "bio_neuromod_ach", "legendFormat": "Acetylcholine"},
        {"expr": "bio_neuromod_5ht", "legendFormat": "Serotonin"},
        {"expr": "bio_neuromod_gaba", "legendFormat": "GABA"}
      ]
    }
  ]
}
```

---

## 7. Integration with BioDashboard

The BioDashboard component in the WW frontend should be extended with these monitoring panels:

```typescript
// Updates to BioDashboard.tsx

import { BiologicalMetricsPanel } from './BiologicalMetricsPanel';
import { SparsityVisualization } from './SparsityVisualization';
import { NeuromodulatorGauge } from './NeuromodulatorGauge';
import { AttractorDynamicsPanel } from './AttractorDynamicsPanel';

const BioDashboard: React.FC = () => {
  // ... existing code ...

  return (
    <div className="bio-dashboard">
      <div className="dashboard-grid">
        {/* Existing panels */}
        <SystemOverviewCard />
        <NeuromodulatorPanel />

        {/* New bioinspired monitoring panels */}
        <BiologicalMetricsPanel />
        <SparsityVisualization encoding={currentEncoding} targetSparsity={0.02} />
        <NeuromodulatorGauge state={neuromodState} />
        <AttractorDynamicsPanel state={attractorState} />
      </div>

      {/* Alert banner */}
      {activeAlerts.length > 0 && (
        <AlertBanner alerts={activeAlerts} />
      )}
    </div>
  );
};
```

---

## Summary

This monitoring system provides:

1. **Real-time metrics**: 10 Hz updates on all bioinspired components
2. **Biological validation**: Continuous checking against neuroscience targets
3. **Visual dashboards**: Grafana + custom React components
4. **Alerting**: Prometheus-style rules with severity levels
5. **Structured logging**: JSON logs for aggregation and analysis
6. **WebSocket API**: Live streaming to frontend dashboards

Target: Sub-100ms latency on all monitoring operations, <1% overhead on core system.
