"""
T4DM REST API - Phase 5 Visualization Module Routes.

Exposes the 22 visualization modules via REST endpoints:
- κ gradient distribution and flow
- T4DX storage metrics (LSM compaction, write amplification)
- Spiking dynamics (spike rasters, membrane potentials)
- Qwen metrics (hidden states, LoRA weights)
- Neuromodulator layer injection
- Oscillator phase injection (theta/gamma/delta)
- Consolidation replay sequences
- Energy landscape visualization
"""

import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from t4dm.api.deps import MemoryServices
from t4dm.visualization import (
    KappaGradientVisualizer,
    KappaSnapshot,
    KAPPA_BANDS,
    KAPPA_COLORS,
    T4DXMetricsVisualizer,
    T4DXSnapshot,
    CompactionEvent,
    CompactionType,
    SpikingDynamicsVisualizer,
    SpikingSnapshot,
    QwenMetricsVisualizer,
    QwenSnapshot,
    OscillatorInjectionVisualizer,
    OscillatorSnapshot,
    EnergyLandscapeVisualizer,
    ConsolidationVisualizer,
    NeuromodLayerVisualizer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/viz", tags=["visualization-modules"])

# Global visualizer instances (lazily initialized)
_kappa_viz: KappaGradientVisualizer | None = None
_t4dx_viz: T4DXMetricsVisualizer | None = None
_spiking_viz: SpikingDynamicsVisualizer | None = None
_qwen_viz: QwenMetricsVisualizer | None = None
_oscillator_viz: OscillatorInjectionVisualizer | None = None
_energy_viz: EnergyLandscapeVisualizer | None = None
_consolidation_viz: ConsolidationVisualizer | None = None
_neuromod_layer_viz: NeuromodLayerVisualizer | None = None


def get_kappa_viz() -> KappaGradientVisualizer:
    """Get or create κ gradient visualizer."""
    global _kappa_viz
    if _kappa_viz is None:
        _kappa_viz = KappaGradientVisualizer()
    return _kappa_viz


def get_t4dx_viz() -> T4DXMetricsVisualizer:
    """Get or create T4DX metrics visualizer."""
    global _t4dx_viz
    if _t4dx_viz is None:
        _t4dx_viz = T4DXMetricsVisualizer()
    return _t4dx_viz


def get_spiking_viz() -> SpikingDynamicsVisualizer:
    """Get or create spiking dynamics visualizer."""
    global _spiking_viz
    if _spiking_viz is None:
        _spiking_viz = SpikingDynamicsVisualizer()
    return _spiking_viz


def get_qwen_viz() -> QwenMetricsVisualizer:
    """Get or create Qwen metrics visualizer."""
    global _qwen_viz
    if _qwen_viz is None:
        _qwen_viz = QwenMetricsVisualizer()
    return _qwen_viz


def get_oscillator_viz() -> OscillatorInjectionVisualizer:
    """Get or create oscillator injection visualizer."""
    global _oscillator_viz
    if _oscillator_viz is None:
        _oscillator_viz = OscillatorInjectionVisualizer()
    return _oscillator_viz


def get_energy_viz() -> EnergyLandscapeVisualizer:
    """Get or create energy landscape visualizer."""
    global _energy_viz
    if _energy_viz is None:
        _energy_viz = EnergyLandscapeVisualizer()
    return _energy_viz


def get_consolidation_viz() -> ConsolidationVisualizer:
    """Get or create consolidation visualizer."""
    global _consolidation_viz
    if _consolidation_viz is None:
        _consolidation_viz = ConsolidationVisualizer()
    return _consolidation_viz


def get_neuromod_layer_viz() -> NeuromodLayerVisualizer:
    """Get or create neuromodulator layer visualizer."""
    global _neuromod_layer_viz
    if _neuromod_layer_viz is None:
        _neuromod_layer_viz = NeuromodLayerVisualizer()
    return _neuromod_layer_viz


# =============================================================================
# Response Models
# =============================================================================


class KappaBandInfo(BaseModel):
    """Information about a kappa band."""

    name: str
    lower: float
    upper: float
    color: str


class KappaDistributionResponse(BaseModel):
    """Response for kappa gradient distribution."""

    timestamp: float
    total_items: int
    band_counts: dict[str, int]
    kappa_mean: float
    kappa_std: float
    bands: list[KappaBandInfo]


class KappaFlowResponse(BaseModel):
    """Response for kappa flow over time."""

    timestamps: list[float]
    band_series: dict[str, list[int]]
    bands: list[KappaBandInfo]


class T4DXStorageResponse(BaseModel):
    """Response for T4DX storage metrics."""

    timestamp: float
    memtable_size: int
    segment_count: int
    total_items: int
    total_edges: int
    wal_size: int
    tombstone_count: int


class T4DXCompactionResponse(BaseModel):
    """Response for T4DX compaction events."""

    events: list[dict[str, Any]]
    write_amplification: dict[str, Any]


class SpikingDynamicsResponse(BaseModel):
    """Response for spiking dynamics data."""

    snapshots: list[dict[str, Any]]
    meta: dict[str, Any]


class QwenMetricsResponse(BaseModel):
    """Response for Qwen adapter metrics."""

    snapshots: list[dict[str, Any]]
    meta: dict[str, Any]


class OscillatorPhaseResponse(BaseModel):
    """Response for oscillator phase data."""

    theta_phase: float
    gamma_phase: float
    delta_phase: float
    bias_mean: float
    bias_max: float
    timestamp: float


class OscillatorHistoryResponse(BaseModel):
    """Response for oscillator history."""

    snapshots: list[dict[str, Any]]
    oscillator_names: list[str]


class EnergyLandscapeResponse(BaseModel):
    """Response for energy landscape data."""

    surface: dict[str, Any] | None = None
    gradient_field: dict[str, Any] | None = None
    trajectory: list[list[float]]
    attractors: dict[str, tuple[float, float, float]]
    metrics: dict[str, float]
    basin_occupancy: dict[str, float]


class ConsolidationReplayResponse(BaseModel):
    """Response for consolidation replay data."""

    total_sequences: int
    nrem_count: int
    rem_count: int
    recent_sequences: list[dict[str, Any]]


class NeuromodLayerResponse(BaseModel):
    """Response for neuromodulator layer injection."""

    layers: dict[str, dict[str, float]]
    timestamp: float


# =============================================================================
# Kappa Gradient Endpoints (A5.1)
# =============================================================================


@router.get("/kappa/distribution", response_model=KappaDistributionResponse)
async def get_kappa_distribution(services: MemoryServices):
    """
    Get κ (kappa) gradient distribution.

    Returns the distribution of memory items across consolidation bands:
    - episodic (0.0-0.15): Just encoded
    - replayed (0.15-0.4): NREM strengthened
    - transitional (0.4-0.85): Being abstracted
    - semantic (0.85-1.0): Fully consolidated
    """
    try:
        viz = get_kappa_viz()

        # Collect current kappa values from T4DX
        episodic = services.get("episodic")
        kappa_values = []
        level_counts: dict[int, int] = {}

        if episodic is not None and hasattr(episodic, "_storage"):
            storage = episodic._storage
            # Scan all items for kappa values
            for item in storage.scan():
                if hasattr(item, "kappa"):
                    kappa_values.append(item.kappa)

        # Record snapshot
        snapshot = KappaSnapshot(
            kappa_values=kappa_values,
            item_count=len(kappa_values),
            timestamp=time.time(),
            level_counts=level_counts,
        )
        viz.record_snapshot(snapshot)

        # Compute band counts
        band_counts = {name: 0 for name in KAPPA_BANDS}
        for v in kappa_values:
            for band_name, (lo, hi) in KAPPA_BANDS.items():
                if lo <= v < hi or (band_name == "semantic" and v == 1.0):
                    band_counts[band_name] += 1
                    break

        bands = [
            KappaBandInfo(
                name=name,
                lower=bounds[0],
                upper=bounds[1],
                color=KAPPA_COLORS[name],
            )
            for name, bounds in KAPPA_BANDS.items()
        ]

        return KappaDistributionResponse(
            timestamp=snapshot.timestamp,
            total_items=len(kappa_values),
            band_counts=band_counts,
            kappa_mean=float(np.mean(kappa_values)) if kappa_values else 0.0,
            kappa_std=float(np.std(kappa_values)) if kappa_values else 0.0,
            bands=bands,
        )

    except Exception as e:
        logger.error(f"Failed to get kappa distribution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get kappa distribution: {e}",
        )


@router.get("/kappa/flow", response_model=KappaFlowResponse)
async def get_kappa_flow():
    """
    Get κ flow over time.

    Returns time series of item counts per consolidation band,
    showing how memories flow through the consolidation pipeline.
    """
    try:
        viz = get_kappa_viz()
        data = viz.export_data()

        timestamps = [s["timestamp"] for s in data["snapshots"]]
        band_series = {name: [] for name in KAPPA_BANDS}

        for s in data["snapshots"]:
            for name in KAPPA_BANDS:
                band_series[name].append(s["band_counts"].get(name, 0))

        bands = [
            KappaBandInfo(
                name=name,
                lower=bounds[0],
                upper=bounds[1],
                color=KAPPA_COLORS[name],
            )
            for name, bounds in KAPPA_BANDS.items()
        ]

        return KappaFlowResponse(
            timestamps=timestamps,
            band_series=band_series,
            bands=bands,
        )

    except Exception as e:
        logger.error(f"Failed to get kappa flow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get kappa flow: {e}",
        )


@router.get("/kappa/export")
async def export_kappa_data():
    """Export full kappa visualization data for external analysis."""
    try:
        viz = get_kappa_viz()
        return viz.export_data()
    except Exception as e:
        logger.error(f"Failed to export kappa data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export kappa data: {e}",
        )


# =============================================================================
# T4DX Metrics Endpoints (A5.2)
# =============================================================================


@router.get("/t4dx/storage", response_model=T4DXStorageResponse)
async def get_t4dx_storage_metrics(services: MemoryServices):
    """
    Get T4DX storage engine metrics.

    Returns current memtable size, segment count, total items,
    edges, WAL size, and tombstone count.
    """
    try:
        viz = get_t4dx_viz()

        # Get metrics from storage
        episodic = services.get("episodic")
        memtable_size = 0
        segment_count = 0
        total_items = 0
        total_edges = 0
        wal_size = 0
        tombstone_count = 0

        if episodic is not None and hasattr(episodic, "_storage"):
            storage = episodic._storage
            if hasattr(storage, "stats"):
                stats = storage.stats()
                memtable_size = stats.get("memtable_size", 0)
                segment_count = stats.get("segment_count", 0)
                total_items = stats.get("total_items", 0)
                total_edges = stats.get("total_edges", 0)
                wal_size = stats.get("wal_size", 0)
                tombstone_count = stats.get("tombstone_count", 0)

        now = time.time()
        snapshot = T4DXSnapshot(
            memtable_size=memtable_size,
            segment_count=segment_count,
            total_items=total_items,
            total_edges=total_edges,
            wal_size=wal_size,
            last_flush_time=now,
            last_compact_time=now,
            tombstone_count=tombstone_count,
            timestamp=now,
        )
        viz.record_snapshot(snapshot)

        return T4DXStorageResponse(
            timestamp=now,
            memtable_size=memtable_size,
            segment_count=segment_count,
            total_items=total_items,
            total_edges=total_edges,
            wal_size=wal_size,
            tombstone_count=tombstone_count,
        )

    except Exception as e:
        logger.error(f"Failed to get T4DX storage metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get T4DX storage metrics: {e}",
        )


@router.get("/t4dx/compaction", response_model=T4DXCompactionResponse)
async def get_t4dx_compaction_events():
    """
    Get T4DX compaction events and write amplification.

    Returns history of FLUSH, NREM, REM, and PRUNE compaction events
    along with write amplification ratio.
    """
    try:
        viz = get_t4dx_viz()
        data = viz.export_data()

        return T4DXCompactionResponse(
            events=data["compaction_events"],
            write_amplification=data["write_amplification"],
        )

    except Exception as e:
        logger.error(f"Failed to get T4DX compaction events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get T4DX compaction events: {e}",
        )


@router.get("/t4dx/export")
async def export_t4dx_data():
    """Export full T4DX metrics for external analysis."""
    try:
        viz = get_t4dx_viz()
        return viz.export_data()
    except Exception as e:
        logger.error(f"Failed to export T4DX data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export T4DX data: {e}",
        )


# =============================================================================
# Spiking Dynamics Endpoints (A5.3)
# =============================================================================


@router.get("/spiking/dynamics", response_model=SpikingDynamicsResponse)
async def get_spiking_dynamics():
    """
    Get spiking cortical block dynamics.

    Returns spike raster data, membrane potentials, thalamic gate state,
    and apical modulation (prediction error).
    """
    try:
        viz = get_spiking_viz()
        return SpikingDynamicsResponse(**viz.export_data())
    except Exception as e:
        logger.error(f"Failed to get spiking dynamics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get spiking dynamics: {e}",
        )


@router.post("/spiking/record")
async def record_spiking_snapshot(
    membrane_potentials: list[float],
    spike_mask: list[bool],
    thalamic_gate: list[float],
    apical_error: list[float],
    block_index: int = 0,
):
    """
    Record a spiking block snapshot.

    Used by the spiking cortical stack to report state for visualization.
    """
    try:
        viz = get_spiking_viz()
        snapshot = SpikingSnapshot(
            membrane_potentials=np.array(membrane_potentials),
            spike_mask=np.array(spike_mask),
            thalamic_gate=np.array(thalamic_gate),
            apical_error=np.array(apical_error),
            block_index=block_index,
            timestamp=time.time(),
        )
        viz.record_snapshot(snapshot)
        return {"success": True, "snapshot_count": len(viz._snapshots)}
    except Exception as e:
        logger.error(f"Failed to record spiking snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record spiking snapshot: {e}",
        )


@router.get("/spiking/export")
async def export_spiking_data():
    """Export full spiking dynamics data for external analysis."""
    try:
        viz = get_spiking_viz()
        return viz.export_data()
    except Exception as e:
        logger.error(f"Failed to export spiking data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export spiking data: {e}",
        )


# =============================================================================
# Qwen Metrics Endpoints (A5.4)
# =============================================================================


@router.get("/qwen/metrics", response_model=QwenMetricsResponse)
async def get_qwen_metrics():
    """
    Get Qwen adapter metrics.

    Returns hidden state norms, projection norms, LoRA weight norms,
    and residual blend alpha over time.
    """
    try:
        viz = get_qwen_viz()
        return QwenMetricsResponse(**viz.export_data())
    except Exception as e:
        logger.error(f"Failed to get Qwen metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Qwen metrics: {e}",
        )


@router.post("/qwen/record")
async def record_qwen_snapshot(
    hidden_state_norm: float,
    projection_norm: float,
    lora_weight_norms: dict[str, float] | None = None,
    residual_blend_alpha: float = 0.5,
    block_index: int = 0,
):
    """
    Record a Qwen adapter snapshot.

    Used by the Qwen adapter to report state for visualization.
    """
    try:
        viz = get_qwen_viz()
        snapshot = QwenSnapshot(
            hidden_state_norm=hidden_state_norm,
            projection_norm=projection_norm,
            lora_weight_norms=lora_weight_norms or {},
            residual_blend_alpha=residual_blend_alpha,
            block_index=block_index,
            timestamp=time.time(),
        )
        viz.record_snapshot(snapshot)
        return {"success": True, "snapshot_count": len(viz._snapshots)}
    except Exception as e:
        logger.error(f"Failed to record Qwen snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record Qwen snapshot: {e}",
        )


@router.get("/qwen/export")
async def export_qwen_data():
    """Export full Qwen metrics for external analysis."""
    try:
        viz = get_qwen_viz()
        return viz.export_data()
    except Exception as e:
        logger.error(f"Failed to export Qwen data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export Qwen data: {e}",
        )


# =============================================================================
# Neuromodulator Layer Endpoints (A5.5)
# =============================================================================


@router.get("/neuromod/layers", response_model=NeuromodLayerResponse)
async def get_neuromod_layer_injection(services: MemoryServices):
    """
    Get neuromodulator injection per spiking block layer.

    Returns DA, NE, ACh, 5-HT levels for each cortical block,
    showing how neuromodulators are distributed across the stack.
    """
    try:
        viz = get_neuromod_layer_viz()

        # Get neuromodulator state from services
        episodic = services.get("episodic")
        layers: dict[str, dict[str, float]] = {}

        if episodic is not None and hasattr(episodic, "orchestra"):
            orchestra = episodic.orchestra
            # Get current levels
            da_level = orchestra.dopamine.get_current_level() if hasattr(orchestra, "dopamine") else 0.5
            ne_level = orchestra.norepinephrine.tonic_level if hasattr(orchestra, "norepinephrine") else 0.5
            ach_level = orchestra.acetylcholine.get_ach_level() if hasattr(orchestra, "acetylcholine") else 0.5
            serotonin_level = orchestra.serotonin.current_mood if hasattr(orchestra, "serotonin") else 0.5

            # Distribute across 6 blocks with layer-specific modulation
            for i in range(6):
                layer_name = f"block_{i}"
                # Earlier blocks get more encoding, later blocks more retrieval
                encoding_bias = 1.0 - (i / 5) * 0.3
                retrieval_bias = (i / 5) * 0.3

                layers[layer_name] = {
                    "DA": da_level * encoding_bias,
                    "NE": ne_level,
                    "ACh": ach_level * (1.0 if i < 3 else 0.8),  # More ACh in early layers
                    "5-HT": serotonin_level * retrieval_bias if i > 2 else serotonin_level * 0.5,
                }

        return NeuromodLayerResponse(
            layers=layers,
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Failed to get neuromod layer injection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get neuromod layer injection: {e}",
        )


# =============================================================================
# Oscillator Phase Endpoints (A5.6)
# =============================================================================


@router.get("/oscillator/phase", response_model=OscillatorPhaseResponse)
async def get_oscillator_phase():
    """
    Get current oscillator phases.

    Returns theta (4-8 Hz), gamma (30-100 Hz), and delta (0.5-4 Hz)
    phase angles and the resulting bias injection magnitude.
    """
    try:
        viz = get_oscillator_viz()

        if viz.snapshot_count > 0:
            s = viz._snapshots[-1]
            return OscillatorPhaseResponse(
                theta_phase=s.theta_phase,
                gamma_phase=s.gamma_phase,
                delta_phase=s.delta_phase,
                bias_mean=float(np.mean(np.abs(s.bias_values))),
                bias_max=float(np.max(np.abs(s.bias_values))),
                timestamp=s.timestamp,
            )
        else:
            # Return default values if no data
            return OscillatorPhaseResponse(
                theta_phase=0.0,
                gamma_phase=0.0,
                delta_phase=0.0,
                bias_mean=0.0,
                bias_max=0.0,
                timestamp=time.time(),
            )

    except Exception as e:
        logger.error(f"Failed to get oscillator phase: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get oscillator phase: {e}",
        )


@router.get("/oscillator/history", response_model=OscillatorHistoryResponse)
async def get_oscillator_history():
    """
    Get oscillator phase history.

    Returns time series of theta/gamma/delta phases and bias magnitudes
    for phase-amplitude coupling analysis.
    """
    try:
        viz = get_oscillator_viz()
        data = viz.export_data()
        return OscillatorHistoryResponse(
            snapshots=data["snapshots"],
            oscillator_names=data["oscillator_names"],
        )
    except Exception as e:
        logger.error(f"Failed to get oscillator history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get oscillator history: {e}",
        )


@router.post("/oscillator/record")
async def record_oscillator_snapshot(
    theta_phase: float,
    gamma_phase: float,
    delta_phase: float,
    bias_values: list[float],
):
    """
    Record an oscillator snapshot.

    Used by the OscillatorBias module to report state for visualization.
    """
    try:
        viz = get_oscillator_viz()
        snapshot = OscillatorSnapshot(
            theta_phase=theta_phase,
            gamma_phase=gamma_phase,
            delta_phase=delta_phase,
            bias_values=np.array(bias_values),
            timestamp=time.time(),
        )
        viz.record_snapshot(snapshot)
        return {"success": True, "snapshot_count": viz.snapshot_count}
    except Exception as e:
        logger.error(f"Failed to record oscillator snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record oscillator snapshot: {e}",
        )


# =============================================================================
# Consolidation Replay Endpoints (A5.7)
# =============================================================================


@router.get("/consolidation/replay", response_model=ConsolidationReplayResponse)
async def get_consolidation_replay():
    """
    Get consolidation replay sequences.

    Returns NREM and REM replay sequences with priority distributions,
    showing which memories are being replayed during consolidation.
    """
    try:
        viz = get_consolidation_viz()

        nrem_count = sum(1 for s in viz._sequences if s.phase == "nrem")
        rem_count = sum(1 for s in viz._sequences if s.phase == "rem")

        # Get recent sequences
        recent = viz._sequences[-10:] if viz._sequences else []
        recent_data = [
            {
                "sequence_id": s.sequence_id,
                "memory_count": len(s.memory_ids),
                "phase": s.phase,
                "avg_priority": float(np.mean(s.priority_scores)) if s.priority_scores else 0.0,
            }
            for s in recent
        ]

        return ConsolidationReplayResponse(
            total_sequences=len(viz._sequences),
            nrem_count=nrem_count,
            rem_count=rem_count,
            recent_sequences=recent_data,
        )

    except Exception as e:
        logger.error(f"Failed to get consolidation replay: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get consolidation replay: {e}",
        )


@router.post("/consolidation/record")
async def record_replay_sequence(
    memory_ids: list[str],
    priority_scores: list[float],
    phase: str = "nrem",
):
    """
    Record a replay sequence.

    Used by the consolidation system to report replay events for visualization.
    """
    try:
        viz = get_consolidation_viz()
        viz.record_replay_sequence(
            memory_ids=memory_ids,
            priority_scores=priority_scores,
            phase=phase,
        )
        return {"success": True, "sequence_count": len(viz._sequences)}
    except Exception as e:
        logger.error(f"Failed to record replay sequence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record replay sequence: {e}",
        )


# =============================================================================
# Energy Landscape Endpoints (A5.8)
# =============================================================================


@router.get("/energy/landscape", response_model=EnergyLandscapeResponse)
async def get_energy_landscape(
    include_surface: bool = Query(False, description="Include full energy surface (expensive)"),
    include_gradient: bool = Query(False, description="Include gradient field"),
):
    """
    Get energy landscape visualization data.

    Returns 2D PCA projection of NT state trajectory, attractor positions,
    basin occupancy, and stability metrics.

    Optionally includes full energy surface and gradient field for contour plots
    (computationally expensive).
    """
    try:
        viz = get_energy_viz()
        data = viz.export_data()

        response_data = {
            "trajectory": data["trajectory"],
            "attractors": data["attractors"],
            "metrics": data["metrics"],
            "basin_occupancy": data["basin_occupancy"],
        }

        if include_surface:
            response_data["surface"] = data["surface"]

        if include_gradient:
            response_data["gradient_field"] = data["gradient_field"]

        return EnergyLandscapeResponse(**response_data)

    except Exception as e:
        logger.error(f"Failed to get energy landscape: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get energy landscape: {e}",
        )


@router.post("/energy/record")
async def record_energy_state(
    nt_state: list[float],
    cognitive_state: str | None = None,
):
    """
    Record an NT state for energy landscape tracking.

    Used to update the trajectory and compute energy metrics.
    """
    try:
        viz = get_energy_viz()
        snapshot = viz.record_state(
            nt_state=np.array(nt_state),
            cognitive_state=cognitive_state,
        )
        return {
            "success": True,
            "total_energy": snapshot.total_energy,
            "nearest_attractor": snapshot.nearest_attractor,
        }
    except Exception as e:
        logger.error(f"Failed to record energy state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record energy state: {e}",
        )


@router.get("/energy/export")
async def export_energy_data():
    """Export full energy landscape data for external analysis."""
    try:
        viz = get_energy_viz()
        return viz.export_data()
    except Exception as e:
        logger.error(f"Failed to export energy data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export energy data: {e}",
        )


# =============================================================================
# Aggregated Export Endpoint (A5.9, A5.10)
# =============================================================================


@router.get("/all/export")
async def export_all_viz_data(services: MemoryServices):
    """
    Export all visualization module data in a single response.

    This is the primary endpoint for real-time streaming aggregation.
    Returns data from all 8 visualization modules.
    """
    try:
        return {
            "timestamp": time.time(),
            "modules": {
                "kappa": get_kappa_viz().export_data(),
                "t4dx": get_t4dx_viz().export_data(),
                "spiking": get_spiking_viz().export_data(),
                "qwen": get_qwen_viz().export_data(),
                "oscillator": get_oscillator_viz().export_data(),
                "energy": get_energy_viz().export_data(),
                "consolidation": {
                    "total_sequences": len(get_consolidation_viz()._sequences),
                    "nrem_count": sum(1 for s in get_consolidation_viz()._sequences if s.phase == "nrem"),
                    "rem_count": sum(1 for s in get_consolidation_viz()._sequences if s.phase == "rem"),
                },
            },
        }
    except Exception as e:
        logger.error(f"Failed to export all viz data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export all viz data: {e}",
        )


@router.get("/realtime/metrics")
async def get_realtime_metrics(services: MemoryServices):
    """
    Get aggregated real-time metrics from all visualization modules.

    Lightweight endpoint for high-frequency polling (< 100ms target).
    Returns summary statistics rather than full data.
    """
    try:
        kappa_viz = get_kappa_viz()
        t4dx_viz = get_t4dx_viz()
        spiking_viz = get_spiking_viz()
        oscillator_viz = get_oscillator_viz()

        # Aggregate lightweight metrics
        metrics = {
            "timestamp": time.time(),
            "kappa": {
                "snapshot_count": len(kappa_viz._snapshots),
                "latest_mean": (
                    kappa_viz._snapshots[-1].kappa_values
                    and float(np.mean(kappa_viz._snapshots[-1].kappa_values))
                    if kappa_viz._snapshots
                    else 0.0
                ),
            },
            "t4dx": {
                "snapshot_count": len(t4dx_viz._snapshots),
                "latest_items": (
                    t4dx_viz._snapshots[-1].total_items if t4dx_viz._snapshots else 0
                ),
                "compaction_events": len(t4dx_viz._compaction_events),
            },
            "spiking": {
                "snapshot_count": len(spiking_viz._snapshots),
                "latest_spike_rate": (
                    spiking_viz._snapshots[-1].spike_mask.mean()
                    if spiking_viz._snapshots
                    else 0.0
                ),
            },
            "oscillator": {
                "snapshot_count": oscillator_viz.snapshot_count,
                "theta_phase": (
                    oscillator_viz._snapshots[-1].theta_phase
                    if oscillator_viz._snapshots
                    else 0.0
                ),
            },
        }

        return metrics

    except Exception as e:
        logger.error(f"Failed to get realtime metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get realtime metrics: {e}",
        )
