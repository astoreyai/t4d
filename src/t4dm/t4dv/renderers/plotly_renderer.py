"""Plotly renderer adapters."""

from __future__ import annotations

from typing import Any


class _PlotlyRenderer:
    def __init__(self, view_id: str):
        self.view_id = view_id


class PlotlySpikeRasterRenderer(_PlotlyRenderer):
    def __init__(self):
        super().__init__("spiking.raster")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        import plotly.graph_objects as go
        events = data.get("events", [])
        times = [e.get("timestamp_offset", i) for i, e in enumerate(events)]
        blocks = [e.get("block_index", 0) for e in events]
        rates = [e.get("firing_rate", 0.0) for e in events]
        fig = go.Figure(data=go.Scatter(
            x=times, y=blocks, mode="markers",
            marker=dict(size=3, color=rates, colorscale="Hot", showscale=True),
        ))
        fig.update_layout(title="Spike Raster", xaxis_title="Time", yaxis_title="Block")
        return fig


class PlotlyKappaSankeyRenderer(_PlotlyRenderer):
    """Sankey diagram showing kappa band transitions."""

    def __init__(self):
        super().__init__("kappa.sankey")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        import plotly.graph_objects as go
        transitions = data.get("transitions", [])
        labels = ["Episodic (0-0.2)", "Replayed (0.2-0.4)", "Transitional (0.4-0.6)",
                  "Semantic (0.6-0.85)", "Stable (0.85-1.0)"]

        source, target, value = [], [], []
        for t in transitions:
            source.append(t.get("from_band", 0))
            target.append(t.get("to_band", 1))
            value.append(t.get("count", 1))

        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels),
            link=dict(source=source, target=target, value=value),
        )])
        fig.update_layout(title="Kappa Consolidation Flow")
        return fig


class PlotlyNeuromodRenderer(_PlotlyRenderer):
    def __init__(self):
        super().__init__("neuromod.dashboard")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        import plotly.graph_objects as go
        fig = go.Figure()
        for nt_name, values in data.get("series", {}).items():
            fig.add_trace(go.Scatter(y=values, name=nt_name, mode="lines"))
        fig.update_layout(title="Neuromodulator Dashboard", yaxis_title="Level")
        return fig


class PlotlyStorageOpsRenderer(_PlotlyRenderer):
    def __init__(self):
        super().__init__("storage.ops")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        import plotly.graph_objects as go
        ops = data.get("op_counts", {})
        fig = go.Figure(data=go.Bar(x=list(ops.values()), y=list(ops.keys()), orientation="h"))
        fig.update_layout(title="Storage Operations")
        return fig


def register_plotly_renderers(registry: Any) -> None:
    for cls in (
        PlotlySpikeRasterRenderer,
        PlotlyKappaSankeyRenderer,
        PlotlyNeuromodRenderer,
        PlotlyStorageOpsRenderer,
    ):
        registry.register(cls())
