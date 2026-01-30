# Frontend
**Path**: `/mnt/projects/t4d/t4dm/frontend/`

## What
React + Three.js 3D memory visualization dashboard for T4DM. Renders memory graphs, neuromodulator controls, bio-inspired panels, and tracing dashboards.

## How
- **Vite** + **TypeScript** + **React 18**
- **@react-three/fiber** + **drei** for WebGL 3D rendering
- **D3** force-3d for graph layout, **Recharts** for charts
- **Jotai** for state management, **SCSS** for styling
- Run: `npm run dev` (dev), `npm run build` (production)

## Why
Provides interactive visual exploration of memory structures, neuromodulator dynamics, eligibility traces, reconsolidation timelines, and sparse encoding patterns.

## Key Files
| File | Purpose |
|------|---------|
| `src/App.tsx` | Root application component |
| `src/main.tsx` | Entry point |
| `src/components/MemoryGraphPage.tsx` | 3D memory graph visualization |
| `src/components/BioDashboard.tsx` | Biological metrics dashboard |
| `src/components/NeuromodulatorControlPanel.tsx` | Neuromodulator dynamics controls |
| `src/components/TracingDashboard.tsx` | OpenTelemetry tracing view |
| `src/components/ReconsolidationTimeline.tsx` | Memory reconsolidation timeline |
| `src/components/ThreeFactorDashboard.tsx` | Three-factor learning display |
| `src/components/HomeostaticPanel.tsx` | Homeostatic plasticity controls |
| `src/components/EligibilityTracesPanel.tsx` | Eligibility trace visualization |
| `src/components/SparseEncodingPanel.tsx` | Sparse encoding patterns |
| `HINTON_UI_PLAN.md` | UI design plan for Hinton-inspired features |
| `vite.config.ts` | Vite build configuration |

## Data Flow
```
T4DM API (FastAPI) → WebSocket/REST → React frontend → Three.js (3D render)
                                                      → Recharts (2D charts)
```

## Integration Points
- **T4DM API**: Consumes REST endpoints from `src/t4dm/api/`
- **T4DV**: Will eventually be replaced by/merged with T4DV visualization component
