"""Export UnifiedGraph to GraphML, GEXF, and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, register_namespace, tostring

register_namespace("viz", "http://gexf.net/1.3/viz")

from ww.diagrams.schema import UnifiedGraph


def _indent_xml(elem: Element, level: int = 0) -> None:
    """Add indentation to XML tree for readability."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def to_graphml(graph: UnifiedGraph) -> str:
    """Export graph to GraphML XML format."""
    root = Element("graphml")
    root.set("xmlns", "http://graphml.graphstruct.org/xmlns")

    # Attribute keys
    for kid, kfor, kname, ktype in [
        ("d0", "node", "label", "string"),
        ("d1", "node", "node_type", "string"),
        ("d2", "node", "subgraph", "string"),
        ("d3", "node", "fill_color", "string"),
        ("d4", "node", "source_file", "string"),
        ("d5", "edge", "label", "string"),
        ("d6", "edge", "edge_type", "string"),
        ("d7", "edge", "weight", "double"),
    ]:
        key = SubElement(root, "key")
        key.set("id", kid)
        key.set("for", kfor)
        key.set("attr.name", kname)
        key.set("attr.type", ktype)

    g = SubElement(root, "graph")
    g.set("id", "G")
    g.set("edgedefault", "directed")

    for node in graph.nodes:
        n = SubElement(g, "node")
        n.set("id", node.id)
        _add_data(n, "d0", node.label)
        _add_data(n, "d1", node.node_type.value)
        _add_data(n, "d2", node.subgraph or "")
        _add_data(n, "d3", node.style.get("fill", ""))
        _add_data(n, "d4", node.source_file)

    for i, edge in enumerate(graph.edges):
        e = SubElement(g, "edge")
        e.set("id", f"e{i}")
        e.set("source", edge.source)
        e.set("target", edge.target)
        _add_data(e, "d5", edge.label)
        _add_data(e, "d6", edge.edge_type.value)
        _add_data(e, "d7", str(edge.weight))

    _indent_xml(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="unicode")


def _add_data(parent: Element, key: str, value: str) -> None:
    d = SubElement(parent, "data")
    d.set("key", key)
    d.text = value


def to_gexf(graph: UnifiedGraph) -> str:
    """Export graph to GEXF 1.3 format for Gephi."""
    root = Element("gexf")
    root.set("xmlns", "http://gexf.net/1.3")
    root.set("version", "1.3")

    # Meta
    meta = SubElement(root, "meta")
    creator = SubElement(meta, "creator")
    creator.text = "T4DM Diagram Pipeline"
    desc = SubElement(meta, "description")
    desc.text = f"Unified architecture graph ({graph.metadata.node_count} nodes, {graph.metadata.edge_count} edges)"

    g = SubElement(root, "graph")
    g.set("defaultedgetype", "directed")
    g.set("mode", "static")

    # Node attributes
    attrs_nodes = SubElement(g, "attributes")
    attrs_nodes.set("class", "node")
    for aid, title, atype in [
        ("0", "label", "string"),
        ("1", "node_type", "string"),
        ("2", "modularity_class", "string"),
        ("3", "source_file", "string"),
    ]:
        a = SubElement(attrs_nodes, "attribute")
        a.set("id", aid)
        a.set("title", title)
        a.set("type", atype)

    # Edge attributes
    attrs_edges = SubElement(g, "attributes")
    attrs_edges.set("class", "edge")
    for aid, title, atype in [
        ("0", "label", "string"),
        ("1", "edge_type", "string"),
    ]:
        a = SubElement(attrs_edges, "attribute")
        a.set("id", aid)
        a.set("title", title)
        a.set("type", atype)

    # Nodes
    nodes_el = SubElement(g, "nodes")
    for node in graph.nodes:
        n = SubElement(nodes_el, "node")
        n.set("id", node.id)
        n.set("label", node.label)
        avs = SubElement(n, "attvalues")
        _add_attvalue(avs, "0", node.label)
        _add_attvalue(avs, "1", node.node_type.value)
        _add_attvalue(avs, "2", node.subgraph or "ungrouped")
        _add_attvalue(avs, "3", node.source_file)

        # Color from style
        if "fill" in node.style:
            color = node.style["fill"].lstrip("#")
            if len(color) == 6:
                viz = SubElement(n, "{http://gexf.net/1.3/viz}color")
                viz.set("r", str(int(color[0:2], 16)))
                viz.set("g", str(int(color[2:4], 16)))
                viz.set("b", str(int(color[4:6], 16)))

    # Edges
    edges_el = SubElement(g, "edges")
    for i, edge in enumerate(graph.edges):
        e = SubElement(edges_el, "edge")
        e.set("id", str(i))
        e.set("source", edge.source)
        e.set("target", edge.target)
        e.set("weight", str(edge.weight))
        avs = SubElement(e, "attvalues")
        _add_attvalue(avs, "0", edge.label)
        _add_attvalue(avs, "1", edge.edge_type.value)

    _indent_xml(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="unicode")


def _add_attvalue(parent: Element, for_id: str, value: str) -> None:
    av = SubElement(parent, "attvalue")
    av.set("for", for_id)
    av.set("value", value)


def to_json(graph: UnifiedGraph) -> str:
    """Export graph to JSON format."""
    return graph.model_dump_json(indent=2)


def export_graph(graph: UnifiedGraph, output: Path, fmt: str) -> None:
    """Export graph to specified format."""
    if fmt == "json":
        output.write_text(to_json(graph), encoding="utf-8")
    elif fmt == "graphml":
        output.write_text(to_graphml(graph), encoding="utf-8")
    elif fmt == "gexf":
        output.write_text(to_gexf(graph), encoding="utf-8")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def export_all(graph: UnifiedGraph, output_dir: Path) -> dict[str, Path]:
    """Export graph to all formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fmt, filename in [
        ("json", "t4dm_unified_graph.json"),
        ("gexf", "t4dm_architecture.gexf"),
        ("graphml", "t4dm_architecture.graphml"),
    ]:
        p = output_dir / filename
        export_graph(graph, p, fmt)
        paths[fmt] = p
    return paths
