# Docs
**Path**: `/mnt/projects/t4d/t4dm/docs/`

## What
MkDocs-based documentation covering architecture, neuroscience foundations, API reference, deployment guides, and mathematical foundations for T4DM.

## How
- **MkDocs** with `mkdocs.yml` at project root
- Organized into concepts, guides, reference, operations, science, and plans
- Includes architecture diagrams and brain region mapping

## Why
Documents the bio-inspired memory system's design rationale, maps software components to neuroscience concepts (hippocampus, neocortex, neuromodulators), and provides operational guides for deployment.

## Key Files
| File/Directory | Purpose |
|----------------|---------|
| `ARCHITECTURE.md` | System architecture overview |
| `MEMORY_ARCHITECTURE.md` | Tripartite memory design |
| `BRAIN_REGION_MAPPING.md` | Software-to-neuroscience mapping |
| `LEARNING_ARCHITECTURE.md` | Hebbian, STDP, Forward-Forward |
| `MATHEMATICAL_FOUNDATIONS.md` | Core equations and proofs |
| `NEUROTRANSMITTER_ARCHITECTURE.md` | Neuromodulator dynamics |
| `LIMITATIONS.md` | Honest capability disclosure and validation status |
| `COMPARISON.md` | SimpleBaseline vs Full T4DM guide |
| `BENCHMARK_RESULTS.md` | Benchmark test results (51 tests) |
| `VALIDATION_REPORT.md` | Biological validation evidence |
| `concepts/` | Architecture, bio-inspired, capsules, NCA, Forward-Forward |
| `guides/` | Hooks, performance, testing guides, ablation study |
| `reference/` | API docs (REST, SDK, CLI, capsule, FF, NCA, glymphatic) |
| `operations/` | Operational runbooks |
| `runbooks/` | Debugging guides (memory, storage, spiking, performance) |
| `science/` | Research papers and references |
| `plans/` | Roadmaps and planning docs |

## Data Flow
```
Markdown sources (docs/) → MkDocs (mkdocs.yml) → Static site
```

## Integration Points
- **MkDocs**: `mkdocs serve` for local preview, `mkdocs build` for static site
- **GitHub Pages**: Deployable documentation site
- **API docs**: Reference docs mirror FastAPI endpoint structure
