# Examples
**Path**: `/mnt/projects/t4d/t4dm/examples/`

## What
Runnable Python demos showcasing T4DM features: hooks system, interface usage, Neural Cellular Automata, and 3D visualization.

## How
- Standalone Python scripts importable from `src/t4dm/`
- Each demo is self-contained with inline documentation

## Why
Provides working code examples for developers integrating with T4DM, demonstrating key subsystems without needing full infrastructure.

## Key Files
| File | Purpose |
|------|---------|
| `hooks_examples.py` | Lifecycle hooks for memory operations |
| `interface_demo.py` | Core memory interface usage patterns |
| `nca_demo.py` | Neural Cellular Automata demonstration |
| `visualization_demo.py` | 3D memory visualization setup |

## Data Flow
```
Example script → src/t4dm/ (imports) → Console output / visualization
```

## Integration Points
- **Docs**: Referenced from guides and README
- **Testing**: Examples serve as smoke tests for API stability
