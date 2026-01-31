"""CLI entry point: t4dm viz <view_id> [--backend mpl|plotly] [--last 60s] [--save path.png]."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ww.t4dv.bus import get_bus
from ww.t4dv.renderers.protocol import get_registry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="t4dm viz", description="Render a visualization view.")
    parser.add_argument("view_id", help="View ID to render (e.g. spiking.raster)")
    parser.add_argument("--backend", choices=["mpl", "plotly"], default="mpl")
    parser.add_argument("--last", default="60s", help="Time window (e.g. 60s, 10s)")
    parser.add_argument("--save", default=None, help="Save to file path")
    parser.add_argument("--list", action="store_true", dest="list_views", help="List available views")

    args = parser.parse_args(argv)
    registry = get_registry()

    # Lazy-register renderers
    if args.backend == "mpl":
        from ww.t4dv.renderers.mpl import register_mpl_renderers
        register_mpl_renderers(registry)
    else:
        from ww.t4dv.renderers.plotly_renderer import register_plotly_renderers
        register_plotly_renderers(registry)

    if args.list_views:
        for v in registry.list_views():
            print(v)
        return

    bus = get_bus()
    topic = args.view_id.split(".")[0]
    events = bus.snapshot(topic)

    # Build data from events
    data: dict[str, Any] = {"events": [e.model_dump() for e in events]}

    fig = registry.render(args.view_id, data)

    if args.save:
        if hasattr(fig, "savefig"):
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
        elif hasattr(fig, "write_image"):
            fig.write_image(args.save)
        print(f"Saved to {args.save}")
    else:
        if hasattr(fig, "show"):
            fig.show()


if __name__ == "__main__":
    main()
