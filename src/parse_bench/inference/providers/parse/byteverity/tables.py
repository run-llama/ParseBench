"""Sealed decision tables: the product's decisions without the synthesa-decide engine.

Each runtime oracle is finite, so its complete truth table (every cell of the declared input domain, decision +
engine result digest) is enumerated once by the engine and shipped here. Before serving any decision a table is
checked for INTEGRITY (its SHA-256 recomputes and equals the pinned digest in PINS.json) and COMPLETENESS (cell
count == product of the domain sizes, every key a valid point of the domain). Lookups are exact: a fact vector
outside the domain is refused, never guessed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import threading
from typing import Any

from .config import PACKAGE_DIR

TABLES_DIR = os.environ.get("BYTEVERITY_TABLES", str(PACKAGE_DIR / "rulebook" / "tables"))
_LOCK = threading.Lock()
_CACHE: dict[str, DecisionTable] = {}


class DecisionTable:
    def __init__(self, name: str):
        path = os.path.join(TABLES_DIR, f"{name}.table.json")
        t = json.load(open(path))
        pins = json.load(open(os.path.join(TABLES_DIR, "PINS.json")))["tables"]
        claimed = t.pop("table_sha256")
        body = json.dumps(t, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(body).hexdigest() != claimed or pins.get(name) != claimed:
            raise RuntimeError(f"decision table {name}: integrity check failed")
        dom = [i["values"] for i in t["inputs"]]
        if t["n_cells"] != math.prod(len(v) for v in dom) or len(t["cells"]) != t["n_cells"]:
            raise RuntimeError(f"decision table {name}: incomplete")
        for combo in itertools.product(*dom):
            key = json.dumps({i["name"]: v for i, v in zip(t["inputs"], combo, strict=False)}, sort_keys=True)
            if key not in t["cells"]:
                raise RuntimeError(f"decision table {name}: cell {key} missing")
        self.name, self.field, self.oracle_digest = name, t["decision_field"], t["oracle_digest"]
        self.cells, self.table_sha256 = t["cells"], claimed

    def evaluate(self, facts: dict[str, Any]) -> dict[str, Any]:
        cell = self.cells.get(json.dumps(facts, sort_keys=True))
        if cell is None:
            return {"status": "INVALID_INPUT", "errors": [f"facts outside the declared domain: {facts}"]}
        return {
            "status": "DECIDED",
            "decision": {self.field: cell[0]},
            "result_digest": cell[1],
            "oracle_digest": self.oracle_digest,
        }


def table(name: str) -> DecisionTable:
    with _LOCK:
        if name not in _CACHE:
            _CACHE[name] = DecisionTable(name)
        return _CACHE[name]


def mode() -> str:
    return "tables"
