"""Parse mermaid source into a plain graph so diagram rules can score *semantics*, not text.

Two correct mermaid transcriptions of the same figure rarely share source text: node ids are
arbitrary, shapes and orientation are stylistic, edge text may sit in ``|pipes|`` or between
the dashes, and a subgraph is optional. Everything the rules care about is a set of labelled
nodes and (optionally labelled, optionally directed) edges. This module turns the mermaid
dialects LlamaParse emits into that ``Graph``:

* ``flowchart`` / ``graph`` — full statement grammar (shapes, chains, ``&`` fan-out, inline and
  pipe edge text, all line styles, ``subgraph … end``, style/class/click lines ignored).
* ``stateDiagram``, ``sequenceDiagram``, ``mindmap``, ``classDiagram``, ``erDiagram`` — nodes and
  relations only, enough for type detection and node/edge alignment.

The parser is deliberately lenient (it is scoring an LLM's output, not validating it) but it
reports ``errors`` for statements it could not read, and a block whose body yields no node at
all is treated as unparseable by the rules.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

# Fenced code blocks whose info string is ``mermaid`` (case-insensitive, optional attributes).
MERMAID_FENCE_RE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})[ \t]*mermaid\b[^\n]*\n(.*?)^\1\2[ \t]*$", re.M | re.S | re.I)

FLOW_HEADER_RE = re.compile(r"^\s*(flowchart|graph)\b\s*(TB|TD|BT|LR|RL)?\s*;?\s*$", re.I)
_TYPE_HEADERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(flowchart|graph)\b", re.I), "flowchart"),
    (re.compile(r"^\s*stateDiagram(-v2)?\b", re.I), "state"),
    (re.compile(r"^\s*sequenceDiagram\b", re.I), "sequence"),
    (re.compile(r"^\s*mindmap\b", re.I), "mindmap"),
    (re.compile(r"^\s*classDiagram(-v2)?\b", re.I), "class"),
    (re.compile(r"^\s*erDiagram\b", re.I), "er"),
    (re.compile(r"^\s*gantt\b", re.I), "gantt"),
    (re.compile(r"^\s*pie\b", re.I), "pie"),
    (re.compile(r"^\s*timeline\b", re.I), "timeline"),
    (re.compile(r"^\s*journey\b", re.I), "journey"),
    (re.compile(r"^\s*gitGraph\b", re.I), "gitgraph"),
    (re.compile(r"^\s*quadrantChart\b", re.I), "quadrant"),
    (re.compile(r"^\s*xychart(-beta)?\b", re.I), "xychart"),
    (re.compile(r"^\s*block(-beta)?\b", re.I), "block"),
    (re.compile(r"^\s*sankey(-beta)?\b", re.I), "sankey"),
    (re.compile(r"^\s*C4(Context|Container|Component|Dynamic|Deployment)\b"), "c4"),
    (re.compile(r"^\s*requirementDiagram\b", re.I), "requirement"),
    (re.compile(r"^\s*architecture(-beta)?\b", re.I), "architecture"),
    (re.compile(r"^\s*packet(-beta)?\b", re.I), "packet"),
    (re.compile(r"^\s*kanban\b", re.I), "kanban"),
]


@dataclass
class Node:
    id: str
    label: str
    group: str | None = None


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    directed: bool = True
    # True for ``<-->`` style links: the relation exists in both directions.
    bidirectional: bool = False


@dataclass
class Graph:
    type: str
    direction: str | None = None
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)  # group id -> title
    errors: list[str] = field(default_factory=list)
    source: str = ""

    def node(self, node_id: str, label: str | None = None, group: str | None = None) -> Node:
        """Return the node, creating it; the first explicit label wins over the bare id."""
        n = self.nodes.get(node_id)
        if n is None:
            # A bare id stands in for its label; ``Buying_Box`` reads as "Buying Box".
            n = Node(id=node_id, label=clean_label(label) if label else node_id.replace("_", " "), group=group)
            self.nodes[node_id] = n
        else:
            if label and n.label == n.id:
                n.label = clean_label(label)
            if group and n.group is None:
                n.group = group
        return n

    @property
    def is_empty(self) -> bool:
        return not self.nodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "direction": self.direction,
            "nodes": [
                {"id": n.id, "label": n.label, **({"group": n.group} if n.group else {})} for n in self.nodes.values()
            ],
            "edges": [
                {
                    "from": e.source,
                    "to": e.target,
                    **({"label": e.label} if e.label else {}),
                    **({"directed": False} if not e.directed else {}),
                    **({"bidirectional": True} if e.bidirectional else {}),
                }
                for e in self.edges
            ],
            "groups": dict(self.groups),
            "errors": list(self.errors),
        }


_TAG_RE = re.compile(r"<[^>]+>")
_MD_RE = re.compile(r"(\*\*|__|\*|_|`)")


def clean_label(raw: str) -> str:
    """Strip quoting, HTML breaks/tags, markdown emphasis and entity codes from a node/edge label."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    # ``"`text`"`` markdown-string labels
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        s = s[1:-1]
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = _TAG_RE.sub(" ", s)
    s = s.replace("\\n", " ")
    s = re.sub(r"#(\w+);", lambda m: html.unescape(f"&{m.group(1)};"), s)  # mermaid entity codes ``#quot;``
    s = html.unescape(s)
    s = _MD_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_mermaid_blocks(markdown: str) -> list[dict[str, Any]]:
    """Return ``{"source", "offset"}`` for every fenced mermaid block, in document order."""
    return [{"source": m.group(3).strip("\n"), "offset": m.start()} for m in MERMAID_FENCE_RE.finditer(markdown)]


def detect_type(source: str) -> str | None:
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%") or stripped.startswith("---"):
            continue
        for pattern, name in _TYPE_HEADERS:
            if pattern.match(stripped):
                return name
        return None
    return None


def _strip_frontmatter_and_comments(source: str) -> list[str]:
    lines = source.splitlines()
    # YAML front matter ``--- ... ---`` at the very top (title/config)
    if lines and lines[0].strip() == "---":
        try:
            end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
            lines = lines[end + 1 :]
        except StopIteration:
            pass
    out = []
    for line in lines:
        if line.strip().startswith("%%"):
            continue
        # trailing ``%% comment``
        idx = line.find("%%")
        if idx > 0:
            line = line[:idx]
        out.append(line.rstrip())
    return out


def parse_mermaid(source: str) -> Graph:
    """Parse one mermaid block. Unknown diagram kinds return an empty graph of that type."""
    kind = detect_type(source)
    lines = _strip_frontmatter_and_comments(source)
    if kind == "flowchart":
        graph = _parse_flowchart(lines)
    elif kind == "state":
        graph = _parse_state(lines)
    elif kind == "sequence":
        graph = _parse_sequence(lines)
    elif kind == "mindmap":
        graph = _parse_mindmap(lines)
    elif kind == "class":
        graph = _parse_class(lines)
    elif kind == "er":
        graph = _parse_er(lines)
    else:
        graph = Graph(type=kind or "unknown")
        if kind is None:
            graph.errors.append("no recognised diagram header")
    graph.source = source
    return graph


# ---------------------------------------------------------------------------
# flowchart / graph
# ---------------------------------------------------------------------------

# Shape openers, longest first, with the closer each expects. ``@{`` is the v11 generic-shape syntax.
_SHAPES: list[tuple[str, str]] = [
    ("(((", ")))"),
    ("((", "))"),
    ("([", "])"),
    ("[[", "]]"),
    ("[(", ")]"),
    ("[/", "/]"),  # parallelogram / trapezoid: closer may also be ``\]``
    ("[\\", "\\]"),
    ("{{", "}}"),
    ("@{", "}"),
    ("[", "]"),
    ("(", ")"),
    ("{", "}"),
    (">", "]"),
]
# Ids may contain ``-`` and ``.`` but must stop before an edge start (``--``, ``==``, ``-.``, ``~~``) so
# ``a-->b`` reads as node ``a``, edge, node ``b``.
_NODE_ID_RE = re.compile(r"(?:(?!--|==|-\.|~~)[A-Za-z0-9_\-.:À-ÿ])+")
_EDGE_TEXT_RE = re.compile(
    r"\s*(?P<lhead><)?(?P<open>-{2,}|={2,}|-\.)\s*(?P<text>[^\n]+?)\s*(?P<close>-{2,}|={2,}|\.-+|\.-*\.?-)(?P<rhead>[>xo])?"
)
_EDGE_PLAIN_RE = re.compile(r"\s*(?P<lhead><)?(?P<body>={2,}[>xo]?|-{2,}[>xo]?|-\.+-[>xo]?|~{3,})")
_PIPE_TEXT_RE = re.compile(r"\s*\|(?P<text>[^|]*)\|")
_SUBGRAPH_RE = re.compile(r"^subgraph\s+(?P<rest>.+)$", re.I)
_IGNORED_FLOW_PREFIX = ("classDef", "class ", "style ", "linkStyle", "click ", "direction ", "accTitle", "accDescr")


def _find_closer(text: str, start: int, closer: str, alt_closer: str | None = None) -> int:
    """Index of ``closer`` after ``start``, skipping over quoted strings."""
    i = start
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if text.startswith(closer, i):
            return i
        if alt_closer and text.startswith(alt_closer, i):
            return i
        i += 1
    return -1


def _parse_flow_node(text: str, pos: int, graph: Graph, group: str | None) -> tuple[str | None, int]:
    """Parse ``id`` or ``id[label]`` at ``pos``; return (node id, new pos)."""
    m = _NODE_ID_RE.match(text, pos)
    if not m:
        return None, pos
    node_id = m.group(0)
    pos = m.end()
    label: str | None = None
    for opener, closer in _SHAPES:
        if text.startswith(opener, pos):
            alt = None
            if opener in ("[/", "[\\"):
                alt = "\\]" if closer == "/]" else "/]"
            end = _find_closer(text, pos + len(opener), closer, alt)
            if end < 0:
                graph.errors.append(f"unclosed shape for node {node_id!r}")
                return None, len(text)
            inner = text[pos + len(opener) : end]
            if opener == "@{":
                lm = re.search(r"label\s*:\s*(\"[^\"]*\"|'[^']*'|[^,}]+)", inner)
                label = lm.group(1) if lm else node_id
            else:
                label = inner
            closer_len = len(closer) if not alt or text.startswith(closer, end) else len(alt)
            pos = end + closer_len
            break
    # ``:::className`` suffix
    if text.startswith(":::", pos):
        cm = re.match(r":::\s*[\w-]+", text[pos:])
        pos += cm.end() if cm else 3
    graph.node(node_id, label, group)
    return node_id, pos


def _parse_flow_node_group(text: str, pos: int, graph: Graph, group: str | None) -> tuple[list[str], int]:
    """``A & B & C`` — a fan-out group of nodes."""
    ids: list[str] = []
    while True:
        while pos < len(text) and text[pos] in " \t":
            pos += 1
        node_id, pos = _parse_flow_node(text, pos, graph, group)
        if node_id is None:
            break
        ids.append(node_id)
        am = re.match(r"\s*&\s*", text[pos:])
        if not am:
            break
        pos += am.end()
    return ids, pos


def _parse_flow_edge(text: str, pos: int) -> tuple[dict[str, Any] | None, int]:
    m = _EDGE_TEXT_RE.match(text, pos)
    edge: dict[str, Any] | None = None
    if m and not m.group("text").lstrip().startswith(("-", "=", ">", "|", ".")):
        edge = {
            "label": m.group("text"),
            "directed": bool(m.group("rhead")) or bool(m.group("lhead")),
            "bidirectional": bool(m.group("lhead") and m.group("rhead")),
            "invisible": False,
        }
        pos = m.end()
    else:
        m = _EDGE_PLAIN_RE.match(text, pos)
        if not m:
            return None, pos
        body = m.group("body")
        rhead = body[-1] in ">xo"
        edge = {
            "label": "",
            "directed": rhead or bool(m.group("lhead")),
            "bidirectional": bool(m.group("lhead")) and rhead,
            "invisible": body.startswith("~"),
        }
        pos = m.end()
    pm = _PIPE_TEXT_RE.match(text, pos)
    if pm:
        edge["label"] = pm.group("text")
        pos = pm.end()
    return edge, pos


def _split_statements(line: str) -> list[str]:
    """Split on ``;`` outside quotes/brackets."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in line:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(0, depth - 1)
        if ch == ";" and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return [s.strip() for s in out if s.strip()]


def _parse_flowchart(lines: list[str]) -> Graph:
    graph = Graph(type="flowchart")
    group_stack: list[str] = []
    header_seen = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not header_seen:
            hm = FLOW_HEADER_RE.match(line)
            if hm:
                graph.direction = (hm.group(2) or "TB").upper()
                header_seen = True
                continue
            # header with trailing statement on the same line (``graph LR; A-->B``)
            hm2 = re.match(r"^(flowchart|graph)\s+(TB|TD|BT|LR|RL)\s*;?\s*(.*)$", line, re.I)
            if hm2:
                graph.direction = hm2.group(2).upper()
                header_seen = True
                line = hm2.group(3).strip()
                if not line:
                    continue
        for stmt in _split_statements(line):
            if stmt.startswith(_IGNORED_FLOW_PREFIX):
                continue
            sm = _SUBGRAPH_RE.match(stmt)
            if sm:
                rest = sm.group("rest").strip()
                gm = re.match(r"^(?P<id>[\w\-.]+)\s*\[(?P<title>.*)\]\s*$", rest)
                if gm:
                    gid, title = gm.group("id"), clean_label(gm.group("title"))
                else:
                    title = clean_label(rest)
                    gid = title
                graph.groups[gid] = title
                group_stack.append(gid)
                continue
            if re.match(r"^end\b", stmt, re.I):
                if group_stack:
                    group_stack.pop()
                continue
            group = group_stack[-1] if group_stack else None
            _parse_flow_statement(stmt, graph, group)
    return graph


def _parse_flow_statement(stmt: str, graph: Graph, group: str | None) -> None:
    pos = 0
    left, pos = _parse_flow_node_group(stmt, pos, graph, group)
    if not left:
        graph.errors.append(f"unreadable statement: {stmt[:60]!r}")
        return
    while pos < len(stmt):
        edge, npos = _parse_flow_edge(stmt, pos)
        if edge is None:
            rest = stmt[pos:].strip()
            if rest:
                graph.errors.append(f"trailing text in statement: {rest[:40]!r}")
            break
        right, pos = _parse_flow_node_group(stmt, npos, graph, group)
        if not right:
            graph.errors.append(f"edge without target: {stmt[:60]!r}")
            break
        if not edge["invisible"]:
            for s in left:
                for t in right:
                    graph.edges.append(
                        Edge(
                            source=s,
                            target=t,
                            label=clean_label(edge["label"]),
                            directed=edge["directed"],
                            bidirectional=edge["bidirectional"],
                        )
                    )
        left = right


# ---------------------------------------------------------------------------
# stateDiagram
# ---------------------------------------------------------------------------

_STATE_EDGE_RE = re.compile(r"^(?P<a>\[\*\]|[\w.\-]+)\s*-->\s*(?P<b>\[\*\]|[\w.\-]+)\s*(?::\s*(?P<label>.*))?$")
_STATE_ALIAS_RE = re.compile(r"^state\s+\"(?P<label>[^\"]+)\"\s+as\s+(?P<id>[\w.\-]+)")
_STATE_DESC_RE = re.compile(r"^(?P<id>[\w.\-]+)\s*:\s*(?P<label>.+)$")
_STATE_OPEN_RE = re.compile(r"^state\s+(?:\"(?P<label>[^\"]+)\"\s+as\s+)?(?P<id>[\w.\-]+)\s*\{")


def _parse_state(lines: list[str]) -> Graph:
    graph = Graph(type="state")
    stack: list[str] = []
    start_count = 0
    for raw in lines:
        line = raw.strip()
        if (
            not line
            or re.match(r"^stateDiagram", line, re.I)
            or line.startswith(("direction", "note", "classDef", "class "))
        ):
            continue
        om = _STATE_OPEN_RE.match(line)
        if om:
            graph.node(om.group("id"), om.group("label"))
            graph.groups[om.group("id")] = clean_label(om.group("label") or om.group("id"))
            stack.append(om.group("id"))
            continue
        if line == "}":
            if stack:
                stack.pop()
            continue
        if line in ("--", "---"):
            continue
        am = _STATE_ALIAS_RE.match(line)
        if am:
            graph.node(am.group("id"), am.group("label"), stack[-1] if stack else None)
            continue
        em = _STATE_EDGE_RE.match(line)
        if em:
            ids = []
            for key in ("a", "b"):
                v = em.group(key)
                if v == "[*]":
                    start_count += 1
                    v = f"[*]{start_count}"
                    graph.node(v, "start/end", stack[-1] if stack else None)
                else:
                    graph.node(v, None, stack[-1] if stack else None)
                ids.append(v)
            graph.edges.append(Edge(ids[0], ids[1], clean_label(em.group("label") or "")))
            continue
        dm = _STATE_DESC_RE.match(line)
        if dm and not line.startswith("state "):
            n = graph.node(dm.group("id"), None, stack[-1] if stack else None)
            n.label = clean_label(dm.group("label"))
            continue
        if re.match(r"^state\s+[\w.\-]+\s*(<<\w+>>)?$", line):
            graph.node(line.split()[1])
            continue
        graph.errors.append(f"unreadable statement: {line[:60]!r}")
    return graph


# ---------------------------------------------------------------------------
# sequenceDiagram
# ---------------------------------------------------------------------------

_SEQ_PARTICIPANT_RE = re.compile(
    r"^(participant|actor)\s+(?P<id>[^\s]+(?:\s+[^\s]+)*?)(?:\s+as\s+(?P<alias>.+))?$", re.I
)
_SEQ_MSG_RE = re.compile(
    r"^(?P<a>[^-\s][^-]*?)\s*(?P<arrow>-{1,2}(?:>>|>|x|\)|\))?)\s*(?P<b>[^:]+?)\s*(?::\s*(?P<label>.*))?$"
)


def _parse_sequence(lines: list[str]) -> Graph:
    graph = Graph(type="sequence")
    for raw in lines:
        line = raw.strip()
        if not line or re.match(r"^sequenceDiagram", line, re.I):
            continue
        if re.match(
            r"^(activate|deactivate|loop|alt|else|opt|par|and|critical|option|break|rect|end|note|autonumber|box|title|links?)\b",
            line,
            re.I,
        ):
            continue
        pm = _SEQ_PARTICIPANT_RE.match(line)
        if pm:
            graph.node(pm.group("id").strip(), (pm.group("alias") or pm.group("id")).strip())
            continue
        mm = _SEQ_MSG_RE.match(line)
        if mm and "-" in mm.group("arrow"):
            a, b = mm.group("a").strip(), mm.group("b").strip()
            for pid in (a, b):
                graph.node(pid)
            graph.edges.append(Edge(a, b, clean_label(mm.group("label") or "")))
            continue
        graph.errors.append(f"unreadable statement: {line[:60]!r}")
    return graph


# ---------------------------------------------------------------------------
# mindmap
# ---------------------------------------------------------------------------

_MIND_NODE_RE = re.compile(
    r"^(?P<id>[\w\-.]+)?(?:\(\((?P<circle>.*)\)\)|\)(?P<bang>.*)\(|\((?P<round>.*)\)|\[(?P<square>.*)\]|\{\{(?P<hex>.*)\}\})?(?P<plain>.*)$"
)


def _parse_mindmap(lines: list[str]) -> Graph:
    graph = Graph(type="mindmap")
    stack: list[tuple[int, str]] = []
    counter = 0
    for raw in lines:
        if not raw.strip() or re.match(r"^\s*mindmap\b", raw, re.I):
            continue
        stripped = raw.strip()
        if stripped.startswith(("::icon", ":::")):
            continue
        indent = len(raw) - len(raw.lstrip())
        m = _MIND_NODE_RE.match(stripped)
        label = None
        if m:
            label = next(
                (m.group(k) for k in ("circle", "bang", "round", "square", "hex") if m.group(k) is not None), None
            )
            if label is None:
                label = (m.group("id") or "") + (m.group("plain") or "")
        counter += 1
        node_id = f"m{counter}"
        graph.node(node_id, label or stripped)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            graph.edges.append(Edge(stack[-1][1], node_id))
        stack.append((indent, node_id))
    return graph


# ---------------------------------------------------------------------------
# classDiagram / erDiagram (relations only)
# ---------------------------------------------------------------------------

_CLASS_REL_RE = re.compile(
    r"^(?P<a>[\w.\-~]+)\s*(?P<rel>(?:<\|--|<\|\.\.|\*--|o--|-->|\.\.>|--\|>|\.\.\|>|--\*|--o|--|\.\.|<--|<\.\.))\s*(?P<b>[\w.\-~]+)\s*(?::\s*(?P<label>.*))?$"
)
_ER_REL_RE = re.compile(r"^(?P<a>[\w\-]+)\s+(?P<rel>[|}{o]+[-.]+[|}{o]+)\s+(?P<b>[\w\-]+)\s*:\s*(?P<label>.*)$")


def _parse_class(lines: list[str]) -> Graph:
    graph = Graph(type="class")
    depth = 0
    for raw in lines:
        line = raw.strip()
        if not line or re.match(r"^classDiagram", line, re.I):
            continue
        if depth:
            depth += line.count("{") - line.count("}")
            continue
        cm = re.match(r"^class\s+(?P<id>[\w.\-~]+)(?:\[\"(?P<label>[^\"]+)\"\])?\s*(\{)?", line)
        if cm:
            graph.node(cm.group("id"), cm.group("label"))
            if cm.group(3):
                depth = 1 + line.count("{") - 1 - line.count("}")
            continue
        rm = _CLASS_REL_RE.match(line)
        if rm:
            a, b, rel = rm.group("a"), rm.group("b"), rm.group("rel")
            graph.node(a)
            graph.node(b)
            # ``A <|-- B`` reads "B inherits A": arrow head on the left points at the source concept.
            if rel.startswith("<"):
                graph.edges.append(Edge(b, a, clean_label(rm.group("label") or "")))
            else:
                graph.edges.append(
                    Edge(a, b, clean_label(rm.group("label") or ""), directed=rel.endswith((">", "*", "o")))
                )
            continue
        if ":" in line and not line.startswith("note"):
            continue  # member line ``Class : +method()``
        if not line.startswith(("note", "direction", "style", "classDef", "cssClass", "link", "click", "callback")):
            graph.errors.append(f"unreadable statement: {line[:60]!r}")
    return graph


def _parse_er(lines: list[str]) -> Graph:
    graph = Graph(type="er")
    depth = 0
    for raw in lines:
        line = raw.strip()
        if not line or re.match(r"^erDiagram", line, re.I):
            continue
        if depth:
            depth += line.count("{") - line.count("}")
            continue
        rm = _ER_REL_RE.match(line)
        if rm:
            graph.node(rm.group("a"))
            graph.node(rm.group("b"))
            graph.edges.append(Edge(rm.group("a"), rm.group("b"), clean_label(rm.group("label")), directed=False))
            continue
        em = re.match(r"^(?P<id>[\w\-]+)\s*(\{)?", line)
        if em:
            graph.node(em.group("id"))
            if em.group(2):
                depth = 1
            continue
        graph.errors.append(f"unreadable statement: {line[:60]!r}")
    return graph


def graph_from_dict(data: dict[str, Any]) -> Graph:
    """Build a ``Graph`` from the ground-truth JSON shape (``nodes``/``edges``/``groups``)."""
    graph = Graph(type=str(data.get("type") or "flowchart"))
    for n in data.get("nodes") or []:
        graph.node(str(n["id"]), str(n.get("label") or n["id"]), n.get("group"))
    for e in data.get("edges") or []:
        graph.edges.append(
            Edge(
                source=str(e["from"]),
                target=str(e["to"]),
                label=clean_label(str(e.get("label") or "")),
                directed=bool(e.get("directed", True)),
                bidirectional=bool(e.get("bidirectional", False)),
            )
        )
    for g in data.get("groups") or []:
        if isinstance(g, dict):
            graph.groups[str(g.get("id") or g.get("label"))] = str(g.get("label") or g.get("id"))
            for member in g.get("members") or []:
                if member in graph.nodes:
                    graph.nodes[member].group = str(g.get("id") or g.get("label"))
    return graph
