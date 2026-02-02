# T4DM Architecture Diagrams -- Summary

**Path**: `/mnt/projects/t4d/t4dm/docs/diagrams/`
**Updated**: 2026-02-02
**Status**: Active

---

## D2 Diagrams (9 files, all rendered to SVG)

| File | Description | Status |
|------|-------------|--------|
| `t4dm_full_system.d2` | Full system architecture: Qwen + spiking cortical stack + T4DX | Current |
| `t4dm_data_flow.d2` | End-to-end data flow through the platform | Current |
| `t4dm_memory_lifecycle.d2` | Memory lifecycle from encoding to consolidation | Current |
| `t4dm_spiking_block.d2` | 6-stage spiking cortical block detail | Current |
| `t4dm_snn_vaswani.d2` | Vaswani-style SNN architecture layout | Current |
| `t4dm_training_pipeline.d2` | Phase 1 surrogate gradient + QLoRA training | Current |
| `t4dx_storage_engine.d2` | T4DX embedded LSM engine internals | Current |
| `t4dv_data_flow.d2` | T4DV visualization data flow | Current |
| `t4dv_observation_bus.d2` | T4DV observation bus architecture | Current |

All D2 diagrams have corresponding `.svg` files rendered via `d2` CLI.

---

## Mermaid Architecture Diagrams

### System-Level (`.mmd` format)

| File | Description | Status |
|------|-------------|--------|
| `01_system_architecture.mmd` | Top-level system layers and components | Current |
| `03_data_flow.mmd` | Data flow through memory pipeline | Current |
| `05_storage_architecture.mmd` | T4DX embedded storage engine architecture | Current |
| `06_memory_systems.mmd` | Tripartite memory systems and interactions | Current |
| `08_consolidation_pipeline.mmd` | Sleep-phase consolidation pipeline | Current |
| `10_observability.mmd` | Observability and metrics infrastructure | Current |
| `11_memory_subsystem.mmd` | Memory subsystem detail | Current |
| `12_learning_subsystem.mmd` | Learning subsystem (Hebbian, STDP, FF) | Current |
| `13_neuromodulation_subsystem.mmd` | Neuromodulator dynamics (DA, NE, ACh, 5-HT) | Current |
| `14_storage_subsystem.mmd` | T4DX storage subsystem detail | Current |
| `15_api_subsystem.mmd` | FastAPI REST endpoints and SDK | Current |

### Class Diagrams (`.mmd` format)

| File | Description | Status |
|------|-------------|--------|
| `02_bioinspired_components.mmd` | Bio-inspired component relationships | Current |
| `07_class_bioinspired.mmd` | Bio-inspired class hierarchy | Current |
| `21_class_memory.mmd` | Memory store class relationships | Current |
| `22_class_learning.mmd` | Learning system classes | Current |
| `23_class_storage.mmd` | T4DX storage classes | Current |
| `24_class_neuromod.mmd` | Neuromodulator class hierarchy | Current |
| `25_class_consolidation.mmd` | Consolidation class hierarchy | Current |

### State Diagrams (`.mmd` format)

| File | Description | Status |
|------|-------------|--------|
| `31_state_circuit_breaker.mmd` | Circuit breaker FSM (CLOSED/OPEN/HALF_OPEN) | Current |
| `32_state_consolidation.mmd` | Consolidation state machine | Current |
| `33_state_neuromod.mmd` | Neuromodulator state transitions | Current |
| `34_state_memory_gate.mmd` | Memory gate decision states | Current |

### Sequence Diagrams (`.mmd` format)

| File | Description | Status |
|------|-------------|--------|
| `41_seq_store_memory.mmd` | Store memory sequence | Current |
| `42_seq_retrieve_memory.mmd` | Retrieve memory sequence | Current |
| `43_seq_consolidation.mmd` | Consolidation sequence | Current |
| `44_seq_sleep_replay.mmd` | Sleep replay sequence | Current |

### Biological Diagrams (`.mermaid` format)

| File | Description | Status |
|------|-------------|--------|
| `50_vae_generator.mermaid` | VAE generative replay model | Current |
| `51_glymphatic_system.mermaid` | Glymphatic waste clearance | Current |
| `52_spatial_cells.mermaid` | Place/grid cell spatial encoding | Current |
| `53_theta_gamma_integration.mermaid` | Theta-gamma phase coupling | Current |
| `54_transmission_delays.mermaid` | Axonal transmission delay model | Current |
| `55_astrocyte_tripartite.mermaid` | Astrocyte tripartite synapse | Current |
| `56_striatal_msn_d1d2.mermaid` | Striatal D1/D2 MSN pathways | Current |
| `57_connectome_regions.mermaid` | Brain region connectome mapping | Current |
| `58_causal_discovery.mermaid` | Causal discovery in memory traces | Current |
| `59_self_supervised_credit.mermaid` | Self-supervised credit assignment | Current |

### Neuroscience Detail (`.mermaid` format)

| File | Description | Status |
|------|-------------|--------|
| `adenosine_homeostasis.mermaid` | Adenosine sleep pressure model | Current |
| `capsule_routing.mermaid` | Capsule routing-by-agreement | Current |
| `consolidation_stages.mermaid` | NREM/REM consolidation stages | Current |
| `credit_assignment_flow.mermaid` | Credit assignment data flow | Current |
| `eligibility_trace.mermaid` | Eligibility trace dynamics | Current |
| `ff_nca_coupling.mermaid` | Forward-Forward + NCA coupling | Current |
| `hippocampal_circuit.mermaid` | Hippocampal trisynaptic circuit | Current |
| `hpc_trisynaptic.mermaid` | HPC trisynaptic pathway detail | Current |
| `memory_lifecycle.mermaid` | Memory item lifecycle | Current |
| `nca_module_map.mermaid` | NCA module dependency map | Current |
| `neuromod_orchestra.mermaid` | Neuromodulator orchestra integration | Current |
| `neuromodulator_pathways.mermaid` | NT pathway diagrams | Current |
| `sleep_cycle.mermaid` | Full sleep cycle model | Current |
| `sleep_subsystems.mermaid` | Sleep subsystem interactions | Current |
| `spindle_ripple_coupling.mermaid` | Sleep spindle-ripple coupling | Current |
| `swr_replay.mermaid` | Sharp-wave ripple replay | Current |
| `three_factor_rule.mermaid` | Three-factor learning rule | Current |
| `vta_circuit.mermaid` | VTA dopamine circuit | Current |

### New Architecture Diagrams (`.mermaid` format)

| File | Description | Status |
|------|-------------|--------|
| `kappa_gradient_consolidation.mermaid` | Kappa progression through LSM compaction phases | **New** |
| `tau_temporal_gate.mermaid` | Tau(t) temporal write gate with gate variants | **New** |
| `sdk_client_architecture.mermaid` | 3-tier SDK: Client -> AgentClient -> WWAgent + MCP | **New** |
| `plugin_integration_map.mermaid` | All integration surfaces: protocols, API, MCP, hooks, SDK | **New** |

### SNN Architecture (`.mermaid` format)

| File | Description | Status |
|------|-------------|--------|
| `t4dm_snn_transformer_architecture.mermaid` | SNN transformer overview | Current |

---

## Markdown Documentation Diagrams

| File | Description | Status |
|------|-------------|--------|
| `system_architecture.md` | High-level system overview with embedded Mermaid | Current |
| `memory_subsystems.md` | Tripartite memory interactions | Current |
| `neural_pathways.md` | Neuromodulator orchestra and plasticity | Current |
| `storage_resilience.md` | Circuit breakers and fault tolerance | Current |
| `embedding_pipeline.md` | Embedding generation and caching | Current |
| `consolidation_flow.md` | Sleep consolidation with SWR | Current |
| `learning_signals.md` | Learning signal documentation | Current |
| `memory_lifecycle.md` | Memory lifecycle documentation | Current |
| `neuromodulator_pathways.md` | Neuromodulator pathway documentation | Current |
| `storage_patterns.md` | Storage pattern documentation | Current |
| `system_network_map.md` | System network topology | Current |
| `module_dependencies.md` | Module dependency documentation | Current |

---

## Other Diagram Files

| File | Description |
|------|-------------|
| `checkpoint_lifecycle.mmd` | Checkpoint save/load lifecycle |
| `persistence_layer.mmd` | WAL + checkpoint persistence |
| `recovery_flow.mmd` | Crash recovery flow |
| `shutdown_flow.mmd` | Graceful shutdown sequence |
| `wal_flow.mmd` | Write-ahead log flow |
| `architecture_3d.html` | Interactive 3D architecture visualization |
| `interactive_network.html` | Interactive network graph |
| `generate_diagrams.py` | Script to render diagrams |

---

## Audit and Planning Documents

| File | Purpose |
|------|---------|
| `ATOMIC_HARDENING_PLAN.md` | Plan for hardening diagram accuracy |
| `ATOMIC_SYNC_PLAN.md` | Plan for syncing diagrams with code |
| `CODE_DIAGRAM_DIFF.md` | Differences between code and diagrams |
| `CODE_VS_DIAGRAM_AUDIT.md` | Audit of code vs diagram accuracy |
| `COMPUTATIONAL_BIOLOGY_ANALYSIS.md` | Bio-plausibility analysis |
| `DIAGRAM_RECONCILIATION_PLAN.md` | Reconciliation plan |
| `NEURAL_PATHWAY_AUDIT.md` | Neural pathway accuracy audit |
| `SECURITY_CORRECTNESS_AUDIT.md` | Security audit of diagrams |
| `WIRING_AUDIT.md` | Wiring correctness audit |

---

## Rendering

### D2 Diagrams
```bash
d2 diagram.d2 diagram.svg
```

### Mermaid Diagrams
```bash
# CLI
npx @mermaid-js/mermaid-cli mmdc -i diagram.mermaid -o diagram.svg

# Or use generate_diagrams.py
python docs/diagrams/generate_diagrams.py
```

### Supported Viewers
- GitHub / GitLab (auto-render `.mermaid` in Markdown)
- VS Code (Mermaid extension)
- Obsidian (built-in Mermaid support)
- D2 Playground (https://play.d2lang.com/)

---

## Color Scheme

Consistent across all new diagrams:

| Color | Hex | Usage |
|-------|-----|-------|
| Light Blue | #e1f5ff | Input / API / Client layer |
| Light Yellow | #fff4e1 | Working memory / Active processing |
| Light Green | #e8f5e9 | Storage / Persistent memory |
| Light Purple | #f3e5f5 | Neural / Computation |
| Light Orange | #ffe0b2 | Consolidation / Transitions |
| Light Teal | #e0f2f1 | Semantic / Final state |
| Light Red | #ffebee | Error / Discard / Pruning |
| Light Gray | #f5f5f5 | Legend / Planned / Inactive |

---

## Statistics

- **D2 Diagrams**: 9 (all with SVG)
- **Mermaid Diagrams**: 60+ files
- **Markdown Docs**: 12 files
- **HTML Interactive**: 2 files
- **Total Diagram Files**: ~90
