"""CLI entry point for diagram parsing and export.

Usage:
    python -m ww.diagrams --input-dir docs/diagrams/ --format all --output-dir exports/
    python -m ww.diagrams --input-dir docs/diagrams/ --format json --output exports/graph.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ww.diagrams.export import export_all, export_graph
from ww.diagrams.graph_merger import merge_from_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse Mermaid diagrams and export unified graph")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .mmd/.mermaid files")
    parser.add_argument("--format", choices=["json", "gexf", "graphml", "all"], default="all")
    parser.add_argument("--output", type=Path, help="Output file path (for single format)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (for --format all)")
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    graph = merge_from_directory(args.input_dir)
    print(f"Parsed: {graph.metadata.node_count} nodes, {graph.metadata.edge_count} edges, "
          f"{graph.metadata.subgraph_count} subgraphs from {graph.metadata.diagram_count} diagrams")

    if args.format == "all":
        out_dir = args.output_dir or Path("exports")
        paths = export_all(graph, out_dir)
        for fmt, p in paths.items():
            print(f"  {fmt}: {p}")
    else:
        if not args.output:
            print("Error: --output required for single format", file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_graph(graph, args.output, args.format)
        print(f"  {args.format}: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
