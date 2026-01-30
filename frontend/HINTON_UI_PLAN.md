# Hinton-Inspired UI Enhancement Plan

## Goal
Transform the UI from "static knowledge graph viewer" to "living learning system visualizer"

## Phase 1: Three-Factor Learning Dashboard (Priority: HIGH)
**Component:** `ThreeFactorDashboard.tsx`

Shows the core learning equation in real-time:
```
effective_lr = eligibility × neuromod_gate × dopamine_surprise
```

### Implementation
1. Create dashboard component with three gauge visualizations
2. Connect to `/api/v1/viz/surgery/learning-stats` endpoint
3. Show when learning is blocked (any factor near zero)
4. Add history timeline of learning events

### API Endpoints Used
- GET `/api/v1/viz/surgery/learning-stats`
- GET `/api/v1/viz/bio/neuromodulators`
- GET `/api/v1/viz/bio/eligibility/traces`

---

## Phase 2: Enhanced Memory Nodes (Priority: HIGH)
**Modify:** `MemoryGraphPage.tsx`

### Node Enhancements
1. **Opacity** → FSRS retrievability (R value)
   - R=1.0: fully opaque
   - R=0.3: semi-transparent (at risk)
   - R<0.1: nearly invisible (forgotten)

2. **Glow intensity** → Eligibility trace strength
   - Just accessed: bright glow
   - Decaying: fading glow
   - No trace: no glow

3. **Pulse animation** → Recent reconsolidation event

### API Endpoints Needed
- GET `/api/v1/viz/bio/fsrs` (batch retrievability)
- GET `/api/v1/viz/bio/eligibility/traces`
- WebSocket for real-time updates (future)

---

## Phase 3: Neuromodulator Ambience (Priority: MEDIUM)
**Modify:** Canvas background in `MemoryGraphPage.tsx`

### Ambient Effects
1. Background gradient based on neuromodulator state:
   - High ACh (encoding): warm amber (#f59e0b @ 0.1 opacity)
   - Low ACh (retrieval): cool blue (#3b82f6 @ 0.1 opacity)
   - High NE (arousal): increased contrast
   - Low NE: softer rendering

2. Dopamine RPE pulse:
   - Positive surprise: brief green flash
   - Negative surprise: brief red flash

### Implementation
- Use Three.js fog/ambient light to shift scene mood
- Poll neuromodulator state every 2-5 seconds
- Smooth transitions between states

---

## Phase 4: Reconsolidation Event Stream (Priority: MEDIUM)
**Component:** `ReconsolidationTimeline.tsx`

### Features
1. Live event stream showing learning as it happens
2. Each event shows:
   - Memory ID (truncated)
   - Advantage (+/-) with color coding
   - Learning rate applied
   - Timestamp
3. Click event to highlight memory in graph

### API Endpoints
- GET `/api/v1/viz/surgery/reconsolidation-history?limit=20`
- Future: WebSocket for real-time push

---

## Phase 5: Memory Surgery Improvements (Priority: LOW)
**Modify:** `MemoryListPanel.tsx` surgery modal

### Enhancements
1. Show memory health metrics before surgery
2. Show consequences (edges that will be severed)
3. Add "soft delete" option (set R→0)
4. Show reconsolidation alternative

---

## Implementation Order

1. **ThreeFactorDashboard** - New standalone component
2. **Enhanced nodes** - Modify MemoryGraphPage
3. **Ambient background** - Add to MemoryGraphPage
4. **Reconsolidation timeline** - New component, integrate with graph
5. **Surgery improvements** - Enhance existing modal

## Files to Create
- `src/components/ThreeFactorDashboard.tsx`
- `src/components/ThreeFactorDashboard.scss`
- `src/components/ReconsolidationTimeline.tsx`
- `src/components/ReconsolidationTimeline.scss`

## Files to Modify
- `src/components/MemoryGraphPage.tsx` (node opacity, glow, ambient)
- `src/components/MemoryGraphPage.scss` (new styles)
- `src/App.tsx` (add ThreeFactorDashboard to layout)
