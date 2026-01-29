"""Parse Mermaid diagram files into UnifiedGraph structures."""

from __future__ import annotations

import html
import re
from pathlib import Path

from ww.diagrams.schema import (
    DiagramType,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    Subgraph,
    UnifiedGraph,
)

# Arrow patterns for splitting edge lines
_ARROW_PATTERNS = [
    (r"<-\.->", EdgeType.DOTTED, 0.5, True),   # bidirectional dotted
    (r"-\.->", EdgeType.DOTTED, 0.5, False),    # dotted
    (r"==>", EdgeType.THICK, 2.0, False),       # thick
    (r"-->", EdgeType.SOLID, 1.0, False),        # solid
]

# Compiled arrow splitter: matches any arrow with optional |label|
_ARROW_SPLIT_RE = re.compile(
    r"(<-\.->|==>|-\.->|--\s+[^-][^>]*?\s*-->|-->)"
)

# Node ref: extract ID and optional shape+label from a node token
_NODE_REF_RE = re.compile(
    r"^(?P<id>[A-Za-z_][\w]*)"
    r"(?:"
    r"\[\"(?P<label_dq>[^\"]*)\"\]"
    r"|\(\[\"?(?P<label_stadium>[^\)\"]*?)\"?\]\)"
    r"|\[\(\"?(?P<label_cyl>[^\)\"]*?)\"?\)\]"
    r"|\[/\"?(?P<label_trap>[^/\"]*?)\"?/\]"
    r"|\{\"?(?P<label_dec>[^\}\"]*?)\"?\}"
    r"|\[\[\"?(?P<label_sub>[^\]\"]*?)\"?\]\]"
    r"|\[\"?(?P<label_box>[^\]\"]*?)\"?\]"
    r")?$"
)

_SUBGRAPH_RE = re.compile(
    r"^\s*subgraph\s+(?P<id>\S+?)(?:\[\"(?P<label>[^\"]*)\"\])?\s*$"
)

_STYLE_RE = re.compile(
    r"^\s*style\s+(?P<id>\S+)\s+(?P<attrs>.+)$"
)

_CLASSDEF_RE = re.compile(
    r"^\s*classDef\s+(?P<name>\S+)\s+(?P<attrs>.+)$"
)

# State diagram patterns
_STATE_BLOCK_RE = re.compile(r"^\s*state\s+(?P<id>\S+)\s*\{\s*$")
_STATE_TRANS_RE = re.compile(
    r"^\s*(?P<src>\[?\*?\]?[\w]+|\[\*\])"
    r"\s*-->\s*"
    r"(?P<tgt>\[?\*?\]?[\w]+|\[\*\])"
    r"(?:\s*:\s*(?P<label>.+))?\s*$"
)

# Sequence diagram patterns
_PARTICIPANT_RE = re.compile(
    r"^\s*participant\s+(?P<id>\S+)\s+as\s+(?P<label>.+)$"
)
_SEQ_MSG_RE = re.compile(
    r"^\s*(?P<src>\S+?)\s*(?P<arrow>->>|-->>|-\))\+?\s*(?P<tgt>\S+?)\s*:\s*(?P<label>.+)$"
)
_SEQ_RETURN_RE = re.compile(
    r"^\s*(?P<src>\S+?)\s*(?P<arrow>-->>|-\))-?\s*(?P<tgt>\S+?)\s*:\s*(?P<label>.+)$"
)


def _clean_label(text: str) -> str:
    """Clean HTML entities and line breaks from label text."""
    text = text.replace("<br/>", " ").replace("<br>", " ")
    text = html.unescape(text)
    return text.strip()


def _parse_style_attrs(attrs: str) -> dict[str, str]:
    """Parse 'fill:#color,stroke:#color' into dict."""
    result: dict[str, str] = {}
    for part in attrs.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def detect_diagram_type(content: str) -> DiagramType:
    """Detect the diagram type from file content."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("%%{"):
            continue
        if line.startswith(("flowchart", "graph ")):
            return DiagramType.FLOWCHART
        if line.startswith("stateDiagram"):
            return DiagramType.STATE
        if line.startswith("sequenceDiagram"):
            return DiagramType.SEQUENCE
        if line.startswith("classDiagram"):
            return DiagramType.CLASS
    return DiagramType.UNKNOWN


def _parse_node_token(token: str) -> tuple[str, str, NodeType]:
    """Parse a node token like 'A[Label]' or bare 'A'. Returns (id, label, type)."""
    token = token.strip()
    m = _NODE_REF_RE.match(token)
    if not m:
        return token, token, NodeType.PROCESS

    node_id = m.group("id")
    for gname, gtype in [
        ("label_dq", NodeType.PROCESS),
        ("label_dec", NodeType.DECISION),
        ("label_stadium", NodeType.STORE),
        ("label_cyl", NodeType.CYLINDER),
        ("label_trap", NodeType.TRAPEZOID),
        ("label_sub", NodeType.SUBROUTINE),
        ("label_box", NodeType.PROCESS),
    ]:
        val = m.group(gname)
        if val is not None:
            return node_id, _clean_label(val), gtype

    return node_id, node_id, NodeType.PROCESS


def _extract_arrow_info(arrow: str) -> tuple[EdgeType, float, bool, str]:
    """Determine edge type, weight, bidirectional flag, and inline label from arrow string."""
    arrow = arrow.strip()
    if "<-.->" in arrow:
        return EdgeType.DOTTED, 0.5, True, ""
    if ".->" in arrow:
        return EdgeType.DOTTED, 0.5, False, ""
    if "==>" in arrow:
        return EdgeType.THICK, 2.0, False, ""
    # Check for inline label: -- label -->
    m = re.match(r"--\s+(.+?)\s*-->", arrow)
    if m:
        return EdgeType.SOLID, 1.0, False, _clean_label(m.group(1))
    return EdgeType.SOLID, 1.0, False, ""


def parse_flowchart(content: str, source_file: str = "") -> UnifiedGraph:
    """Parse a flowchart/graph TD/TB/LR diagram."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    subgraphs: list[Subgraph] = []
    styles: dict[str, dict[str, str]] = {}
    classdefs: dict[str, dict[str, str]] = {}

    subgraph_stack: list[str] = []

    def ensure_node(node_id: str, label: str = "", ntype: NodeType = NodeType.PROCESS) -> None:
        if node_id not in nodes:
            nodes[node_id] = GraphNode(
                id=node_id,
                label=label or node_id,
                node_type=ntype,
                source_file=source_file,
                diagram_type=DiagramType.FLOWCHART,
                subgraph=subgraph_stack[-1] if subgraph_stack else None,
            )
        else:
            if label and nodes[node_id].label == node_id:
                nodes[node_id].label = label
            if ntype != NodeType.PROCESS:
                nodes[node_id].node_type = ntype
        if subgraph_stack and not nodes[node_id].subgraph:
            nodes[node_id].subgraph = subgraph_stack[-1]

    for raw_line in content.splitlines():
        line = raw_line.strip()

        # Skip comments, empty, directives, direction
        if not line or line.startswith("%%") or line.startswith("direction"):
            continue
        if line.startswith(("flowchart", "graph ")):
            continue

        # classDef
        m = _CLASSDEF_RE.match(line)
        if m:
            classdefs[m.group("name")] = _parse_style_attrs(m.group("attrs"))
            continue

        # style
        m = _STYLE_RE.match(line)
        if m:
            styles[m.group("id")] = _parse_style_attrs(m.group("attrs"))
            continue

        # subgraph
        m = _SUBGRAPH_RE.match(line)
        if m:
            sg_id = m.group("id")
            sg_label = m.group("label") or sg_id
            sg = Subgraph(
                id=sg_id,
                label=_clean_label(sg_label),
                parent=subgraph_stack[-1] if subgraph_stack else None,
            )
            subgraphs.append(sg)
            subgraph_stack.append(sg_id)
            continue

        if line == "end":
            if subgraph_stack:
                subgraph_stack.pop()
            continue

        # Try to split on arrow operators
        parts = _ARROW_SPLIT_RE.split(line)
        if len(parts) >= 3:
            # parts = [left_token, arrow, right_token, arrow, right_token, ...]
            i = 0
            while i + 2 < len(parts):
                left_raw = parts[i].strip()
                arrow_raw = parts[i + 1].strip()
                right_raw = parts[i + 2].strip()

                # Extract |label| from right side
                edge_label = ""
                label_match = re.match(r'^\|"?([^|"]*?)"?\|\s*(.*)', right_raw)
                if label_match:
                    edge_label = _clean_label(label_match.group(1))
                    right_raw = label_match.group(2).strip()

                if not left_raw or not right_raw:
                    i += 2
                    continue

                src_id, src_label, src_type = _parse_node_token(left_raw)
                tgt_id, tgt_label, tgt_type = _parse_node_token(right_raw)
                etype, weight, bidi, inline_label = _extract_arrow_info(arrow_raw)
                if not edge_label:
                    edge_label = inline_label

                ensure_node(src_id, src_label, src_type)
                ensure_node(tgt_id, tgt_label, tgt_type)

                edges.append(GraphEdge(
                    source=src_id, target=tgt_id, label=edge_label,
                    edge_type=etype, weight=weight, source_file=source_file,
                ))
                if bidi:
                    edges.append(GraphEdge(
                        source=tgt_id, target=src_id, label=edge_label,
                        edge_type=etype, weight=weight, source_file=source_file,
                    ))

                # For chained edges, next iteration uses right as left
                parts[i + 2] = right_raw
                i += 2
            continue

        # Standalone node definition (no arrow)
        nid, nlabel, ntype = _parse_node_token(line)
        if nid and nid != line:  # Had a shape bracket
            ensure_node(nid, nlabel, ntype)
            continue

        # Bare ID (could be a node ref in a subgraph)
        if re.match(r'^[A-Za-z_]\w*$', line):
            ensure_node(line)

    # Apply styles
    for node_id, sdict in styles.items():
        if node_id in nodes:
            nodes[node_id].style.update(sdict)

    # Assign subgraph children
    sg_map = {sg.id: sg for sg in subgraphs}
    for node in nodes.values():
        if node.subgraph and node.subgraph in sg_map:
            sg_map[node.subgraph].children.append(node.id)

    graph = UnifiedGraph(
        nodes=list(nodes.values()),
        edges=edges,
        subgraphs=subgraphs,
    )
    graph.update_metadata()
    return graph


def parse_state_diagram(content: str, source_file: str = "") -> UnifiedGraph:
    """Parse a stateDiagram-v2 file."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    subgraphs: list[Subgraph] = []
    state_stack: list[str] = []

    def ensure_node(node_id: str) -> None:
        display_id = node_id
        ntype = NodeType.STATE
        if node_id == "[*]":
            display_id = f"__star_{len(nodes)}__"
            ntype = NodeType.START_END
        if display_id not in nodes:
            nodes[display_id] = GraphNode(
                id=display_id,
                label=node_id,
                node_type=ntype,
                source_file=source_file,
                diagram_type=DiagramType.STATE,
                subgraph=state_stack[-1] if state_stack else None,
            )
        return display_id

    # Track [*] counters per context for unique IDs
    star_counter = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%%") or line.startswith("stateDiagram") or line.startswith("note "):
            continue
        if line.startswith("end note"):
            continue

        # state block
        m = _STATE_BLOCK_RE.match(line)
        if m:
            sid = m.group("id")
            sg = Subgraph(
                id=sid,
                label=sid,
                parent=state_stack[-1] if state_stack else None,
            )
            subgraphs.append(sg)
            state_stack.append(sid)
            # Also create a node for the state itself
            if sid not in nodes:
                nodes[sid] = GraphNode(
                    id=sid, label=sid, node_type=NodeType.STATE,
                    source_file=source_file, diagram_type=DiagramType.STATE,
                    subgraph=state_stack[-2] if len(state_stack) > 1 else None,
                )
            continue

        if line == "end":
            if state_stack:
                state_stack.pop()
            continue
        if line == "}":
            if state_stack:
                state_stack.pop()
            continue

        # Transition
        m = _STATE_TRANS_RE.match(line)
        if m:
            src_raw = m.group("src")
            tgt_raw = m.group("tgt")
            label = m.group("label") or ""

            # Handle [*]
            if src_raw == "[*]":
                star_counter += 1
                src_id = f"__star_{star_counter}__"
                if src_id not in nodes:
                    nodes[src_id] = GraphNode(
                        id=src_id, label="[*]", node_type=NodeType.START_END,
                        source_file=source_file, diagram_type=DiagramType.STATE,
                        subgraph=state_stack[-1] if state_stack else None,
                    )
            else:
                src_id = src_raw
                if src_id not in nodes:
                    nodes[src_id] = GraphNode(
                        id=src_id, label=src_id, node_type=NodeType.STATE,
                        source_file=source_file, diagram_type=DiagramType.STATE,
                        subgraph=state_stack[-1] if state_stack else None,
                    )

            if tgt_raw == "[*]":
                star_counter += 1
                tgt_id = f"__star_{star_counter}__"
                if tgt_id not in nodes:
                    nodes[tgt_id] = GraphNode(
                        id=tgt_id, label="[*]", node_type=NodeType.START_END,
                        source_file=source_file, diagram_type=DiagramType.STATE,
                        subgraph=state_stack[-1] if state_stack else None,
                    )
            else:
                tgt_id = tgt_raw
                if tgt_id not in nodes:
                    nodes[tgt_id] = GraphNode(
                        id=tgt_id, label=tgt_id, node_type=NodeType.STATE,
                        source_file=source_file, diagram_type=DiagramType.STATE,
                        subgraph=state_stack[-1] if state_stack else None,
                    )

            edges.append(GraphEdge(
                source=src_id, target=tgt_id, label=label.strip(),
                source_file=source_file,
            ))
            continue

    # Assign children
    sg_map = {sg.id: sg for sg in subgraphs}
    for node in nodes.values():
        if node.subgraph and node.subgraph in sg_map:
            sg_map[node.subgraph].children.append(node.id)

    graph = UnifiedGraph(nodes=list(nodes.values()), edges=edges, subgraphs=subgraphs)
    graph.update_metadata()
    return graph


def parse_sequence_diagram(content: str, source_file: str = "") -> UnifiedGraph:
    """Parse a sequenceDiagram file."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%%") or line.startswith("sequenceDiagram"):
            continue
        if line.startswith(("autonumber", "Note ", "alt ", "else ", "end", "loop ", "rect ", "par ", "and ")):
            continue

        # Participant
        m = _PARTICIPANT_RE.match(line)
        if m:
            pid = m.group("id")
            plabel = m.group("label").strip()
            nodes[pid] = GraphNode(
                id=pid, label=plabel, node_type=NodeType.PARTICIPANT,
                source_file=source_file, diagram_type=DiagramType.SEQUENCE,
            )
            continue

        # Message (covers ->> -->> and return --)
        m = _SEQ_MSG_RE.match(line)
        if not m:
            m = _SEQ_RETURN_RE.match(line)
        if m:
            src = m.group("src").rstrip("+").rstrip("-")
            tgt = m.group("tgt").rstrip("+").rstrip("-")
            arrow = m.group("arrow")
            label = m.group("label").strip()

            for nid in (src, tgt):
                if nid not in nodes:
                    nodes[nid] = GraphNode(
                        id=nid, label=nid, node_type=NodeType.PARTICIPANT,
                        source_file=source_file, diagram_type=DiagramType.SEQUENCE,
                    )

            etype = EdgeType.DOTTED if "-->" in arrow else EdgeType.SOLID
            edges.append(GraphEdge(
                source=src, target=tgt, label=label,
                edge_type=etype, weight=0.5 if etype == EdgeType.DOTTED else 1.0,
                source_file=source_file,
            ))
            continue

    graph = UnifiedGraph(nodes=list(nodes.values()), edges=edges)
    graph.update_metadata()
    return graph


def parse_file(path: Path) -> UnifiedGraph:
    """Parse a single mermaid file, auto-detecting diagram type."""
    content = path.read_text(encoding="utf-8")
    source_file = path.name
    dtype = detect_diagram_type(content)

    if dtype == DiagramType.FLOWCHART:
        return parse_flowchart(content, source_file)
    elif dtype == DiagramType.STATE:
        return parse_state_diagram(content, source_file)
    elif dtype == DiagramType.SEQUENCE:
        return parse_sequence_diagram(content, source_file)
    else:
        # Best-effort: try flowchart
        return parse_flowchart(content, source_file)


def parse_directory(input_dir: Path) -> dict[str, UnifiedGraph]:
    """Parse all .mmd and .mermaid files in a directory."""
    results: dict[str, UnifiedGraph] = {}
    for ext in ("*.mmd", "*.mermaid"):
        for path in sorted(input_dir.glob(ext)):
            results[path.name] = parse_file(path)
    return results
