"""Rulebook client (sealed decision tables; the synthesa-decide engine is not distributed):

Every rule-based decision

Python measures facts from the PDF (and reads the proposer's labels); the sealed
synthesa-decide oracles (research/parsebench/oracle/rules/*.rules.yaml, proven complete
by completeprobe) decide. Oracles are verified offline at start-up; a failed
verification refuses to run. Decisions are memoized by fact vector — exact, since
an oracle is a pure function of its facts — and each distinct (oracle, facts) cell
used is recorded with its result_digest for the run receipt.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

ORACLES = {
    "text_layer_validity": ("o0_text_layer_validity", "authority"),
    "line_boundary": ("o1_line_boundary", "boundary"),
    "element_role": ("o2_element_role", "role"),
    "picture_reconcile": ("o3_picture_reconcile", "action"),
    "markup_inject": ("o4_markup_inject", "action"),
    "chart_title": ("o5_chart_title", "action"),
    "decoration": ("o6_decoration", "style"),
    "bold_evidence": ("o7_bold_evidence", "verdict"),
    "heading_join": ("o8_heading_join", "action"),
    "content_source": ("o9_content_source", "source"),
    "pixel_floor": ("o10_pixel_floor", "action"),
    "emphasis_residual": ("o11_emphasis_residual", "verdict"),
    "table_source": ("o12_table_source", "action"),
    "table_requery": ("o13_table_requery_gate", "verdict"),
    "layout_source": ("o14_layout_source", "source"),
}

_LOCK = threading.Lock()
_BOOK: TableRulebook | None = None


class TableRulebook:
    """Same interface as Rulebook, backed by the sealed decision tables (no engine)."""

    def __init__(self) -> None:
        from .tables import table

        self._t = {name: table(d) for name, (d, _) in ORACLES.items()}
        self._memo: dict[tuple[str, str], str] = {}
        self.cells: dict[tuple[str, str], str] = {}

    def decide(self, oracle: str, /, **facts: Any) -> str:
        key = (oracle, json.dumps(facts, sort_keys=True))
        v = self._memo.get(key)
        if v is not None:
            return v
        r = self._t[oracle].evaluate(facts)
        if r.get("status") != "DECIDED":
            raise RuntimeError(f"rulebook {oracle} did not decide {facts}: {r.get('status')} {r.get('errors')}")
        v = r["decision"][self._t[oracle].field]
        self._memo[key] = v
        self.cells[key] = r["result_digest"]
        return v


def rules() -> TableRulebook | None:
    """The rulebook when SX_ORACLE_RULES=1, else None (legacy in-code rules)."""
    global _BOOK
    if os.environ.get("SX_ORACLE_RULES", "1") != "1":  # default: sealed rulebook (shadow-equivalent on 2,078 docs)
        return None
    with _LOCK:
        if _BOOK is None:
            _BOOK = TableRulebook()
    return _BOOK
