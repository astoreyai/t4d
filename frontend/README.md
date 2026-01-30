# World Weaver Frontend

**Version**: 0.1.0 | **Last Updated**: 2025-12-09

React TypeScript dashboard for visualizing and monitoring the World Weaver memory system.

---

## Quick Start

```bash
# Install dependencies
npm install

# Development server (proxies to API at localhost:8765)
npm run dev

# Production build
npm run build
```

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.0 | Build tool |
| Recharts | 2.10.3 | 2D charts |
| Three.js | 0.160.0 | 3D graphics |
| React Three Fiber | 8.15.0 | React Three.js bindings |
| d3-force-3d | 3.0.5 | Force-directed layouts |
| Jotai | 2.6.0 | State management |
| SCSS | - | Styling |

## Components

### Production Ready

| Component | Lines | Purpose |
|-----------|-------|---------|
| **BioDashboard** | 430 | Neuromodulator, FSRS, Hebbian, Pattern Separation metrics |
| **ConfigPanel** | 1,256 | System configuration and parameter tuning |
| **TracingDashboard** | 278 | OpenTelemetry traces and latency metrics |
| **FastEpisodicPanel** | 394 | Episode encoding and consolidation |
| **EligibilityTracesPanel** | 325 | Learning traces and credit assignment |
| **SparseEncodingPanel** | 331 | Sparse encoding visualization |
| **DocumentationPanel** | 381 | System documentation and help |
| **MemoryGraphPage** | 580 | 3D force-directed knowledge graph with React Three Fiber |

## Architecture

```
src/
├── App.tsx              # Main app with tab navigation
├── main.tsx             # Entry point
├── components/
│   ├── BioDashboard.tsx
│   ├── ConfigPanel.tsx
│   ├── TracingDashboard.tsx
│   ├── FastEpisodicPanel.tsx
│   ├── EligibilityTracesPanel.tsx
│   ├── SparseEncodingPanel.tsx
│   ├── DocumentationPanel.tsx
│   └── MemoryGraphPage.tsx  # 3D force-directed graph
├── types/
│   └── d3-force-3d.d.ts     # Type declarations
└── styles/
    └── variables.scss
```

## API Integration

### Development Proxy

Vite proxies API and WebSocket connections:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:8765',
    '/ws': {
      target: 'ws://localhost:8765',
      ws: true
    }
  }
}
```

### API Endpoints Used

| Endpoint | Component | Purpose |
|----------|-----------|---------|
| `GET /api/v1/viz/bio/all` | BioDashboard | All biological mechanisms |
| `GET /api/v1/viz/bio/neuromodulators` | BioDashboard | Neuromodulator state |
| `GET /api/v1/viz/bio/fsrs` | BioDashboard | FSRS retrievability |
| `GET /api/v1/viz/bio/hebbian` | BioDashboard | Hebbian weights |
| `GET /api/v1/viz/graph` | MemoryGraphPage | Graph nodes/edges |
| `GET /api/v1/config` | ConfigPanel | System configuration |
| `WS /ws/health` | All | Real-time health metrics |
| `WS /ws/learning` | EligibilityTracesPanel | Learning updates |

## WebSocket Channels

| Channel | Events | Usage |
|---------|--------|-------|
| `/ws/events` | All system events | General monitoring |
| `/ws/memory` | ADDED, PROMOTED, REMOVED | Memory operations |
| `/ws/learning` | GATE_UPDATED, TRACE_CREATED | Learning updates |
| `/ws/health` | UPDATE, WARNING, ERROR | Health metrics |

## Building for Production

```bash
# Build optimized bundle
npm run build

# Output in dist/
ls -la dist/
```

The built files can be served by any static file server or integrated with the FastAPI backend.

## TODO

- [x] Implement MemoryGraphPage with React Three Fiber
- [ ] Add unified metrics dashboard
- [ ] WebSocket event consumption in components
- [ ] Dark mode toggle
- [ ] Mobile responsive layout

## Development Notes

### Adding a New Component

1. Create component in `src/components/`
2. Add tab in `App.tsx`
3. Create corresponding SCSS file
4. Add API endpoint calls as needed

### State Management

Using Jotai for lightweight atomic state:

```typescript
import { atom, useAtom } from 'jotai';

const metricsAtom = atom<MetricsState>({ ... });

function MyComponent() {
  const [metrics, setMetrics] = useAtom(metricsAtom);
}
```

## Related Documentation

- [API Walkthrough](../docs/API_WALKTHROUGH.md)
- [Visualization Walkthrough](../docs/VISUALIZATION_WALKTHROUGH.md)
- [System Walkthrough](../docs/SYSTEM_WALKTHROUGH.md)
