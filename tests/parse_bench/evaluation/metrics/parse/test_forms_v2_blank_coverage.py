"""forms_v2 coverage cap on blank form tables: blank cells left out of a found table do not cap the score."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

COLS = ["Name", "Address", "City"]


def _ref(*extra: dict[str, Any]) -> Any:
    table = {"type": "table", "label": "Directors", "columns": COLS, "rows": [[None, None, None] for _ in range(6)]}
    return reference_from_tree([{"page_number": 1, "items": [*extra, table]}])


def _text(label: str, value: str | None = None) -> dict[str, Any]:
    field: dict[str, Any] = {"type": "field", "field": "text", "label": label}
    field.update({"value": value} if value is not None else {"isEmpty": True})
    return field


def test_blank_table_written_as_lines_is_not_capped():
    ref = _ref(_text("Phone number"))
    md = "Phone number:\n\n" + "\n\n".join("\n".join(f"{c}: <empty>" for c in COLS) for _ in range(6))
    res = score_forms(ref, md)
    assert res.coverage == 1.0
    assert res.score >= 0.95


def test_blank_table_written_header_only_is_not_capped():
    ref = _ref(_text("Phone number"))
    md = "Phone number:\n\n<table><tr><th>Name</th><th>Address</th><th>City</th></tr></table>"
    res = score_forms(ref, md)
    assert res.coverage == 1.0
    assert res.score >= 0.8  # the dropped blank cells still cost minor events


def test_table_captions_alone_do_not_hide_lost_values():
    ref = _ref(*(_text(f"Field {i}", f"value {i}") for i in range(4)))
    md = "<table><tr><th>Name</th><th>Address</th><th>City</th></tr></table>"
    assert score_forms(ref, md).score == 0.0
    assert score_forms(ref, "").score == 0.0
