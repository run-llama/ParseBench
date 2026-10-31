"""forms_v2: a blank field read as text printed on the form is a minor event, an invented value stays critical."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _blank(label: str, pid: str) -> dict[str, Any]:
    return {"type": "field", "field": "text", "id": pid, "label": label, "isEmpty": True}


REF = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                _blank("Tax-exempt interest", "2a"),
                _blank("Taxable interest", "2b"),
                _blank("Qualified dividends (see instructions)", "3b"),
            ],
        }
    ]
)


def _kinds(md: str) -> dict[str, list[str]]:
    kinds: dict[str, list[str]] = {}
    for e in score_forms(REF, md).events:
        kinds.setdefault(e.label, []).append(f"{e.cls}:{e.kind}")
    return kinds


def test_blank_field_read_as_printed_text_is_minor() -> None:
    md = "- **Tax-exempt interest**: 2a b\n- **Taxable interest**:\n- **Qualified dividends (see instructions)**: see\n"
    kinds = _kinds(md)
    critical = [k for ks in kinds.values() for k in ks if k.startswith("critical")]
    assert critical == []
    assert "minor:printed text in a blank field" in kinds["Tax-exempt interest"]
    assert "minor:printed text in a blank field" in kinds["Qualified dividends (see instructions)"]


def test_invented_value_in_a_blank_field_stays_critical() -> None:
    md = (
        "- **Tax-exempt interest**: Marie b\n"
        "- **Taxable interest**:\n"
        "- **Qualified dividends (see instructions)**: 1200\n"
    )
    kinds = _kinds(md)
    assert "critical:invented value" in kinds["Tax-exempt interest"]
    # a bare number may be an invented amount, even where the form prints numbers
    assert "critical:invented value" in kinds["Qualified dividends (see instructions)"]
