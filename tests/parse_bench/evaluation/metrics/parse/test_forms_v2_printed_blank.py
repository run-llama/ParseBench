"""forms_v2: a blank field read as text printed on the form is a minor event, an invented value stays critical."""

from __future__ import annotations

from typing import Any

import pytest

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


ID_REF = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": label, "isEmpty": True}
                for label in ("ID No", "Branch/Transit #", "Area")
            ],
        }
    ]
)


def _id_kinds(md: str) -> dict[str, list[str]]:
    kinds: dict[str, list[str]] = {}
    for e in score_forms(ID_REF, md).events:
        kinds.setdefault(e.label, []).append(f"{e.cls}:{e.kind}")
    return kinds


def test_printed_text_after_a_blank_marker_is_minor():
    """``[___] /OR``, ``[___] (5 digits)``, ``[___] SQ FT``: the marker is the blank field, the rest is printed."""
    md = "**ID No:** [___] /OR\n**Branch/Transit #:** [___] (5 digits)\n**Area:** [___] SQ FT\n"
    assert _id_kinds(md) == {
        "ID No": ["minor:printed text in a blank field"],
        "Branch/Transit #": ["minor:printed text in a blank field"],
        "Area": ["minor:printed text in a blank field"],
    }


def test_number_after_a_blank_marker_stays_invented():
    md = "**ID No:** [___] 8501015800083\n**Branch/Transit #:** [___] 7\n**Area:** [___] (sq) 40\n"
    assert _id_kinds(md) == {
        "ID No": ["critical:invented value"],
        "Branch/Transit #": ["critical:invented value"],
        "Area": ["critical:invented value"],
    }


@pytest.mark.parametrize(
    "line",
    [
        "**ID No:** [___] (12345)",  # a number in brackets is a value, not a worded hint
        "**ID No:** [___] \uff17",  # a full-width digit
        "**ID No:** OR [___]",  # the marker after the text: the text is no reading past a blank
        "**ID No:** [_] OR",  # one underscore is no blank marker
    ],
)
def test_near_miss_blank_markers_stay_invented(line: str) -> None:
    assert _id_kinds(line + "\n**Branch/Transit #:** [___]\n**Area:** [___]\n") == {
        "ID No": ["critical:invented value"]
    }


def _one_blank(md_value: str) -> list[str]:
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "Name", "isEmpty": True}]}]
    )
    return [f"{e.cls}:{e.kind}" for e in score_forms(ref, f"**Name:** {md_value}\n").events if e.charged]


@pytest.mark.parametrize(
    "written",
    [
        "[___] SQ FT",
        "[___] CT",
        "[___] and/or",
        "[___] plus costs",
        "[___] per year",
        "[___] Deputy",
        "[___] OR",
        "[___] (5 digits)",
        "[___] Taxpayer signature",
    ],
)
def test_form_text_after_a_blank_marker_is_minor(written: str) -> None:
    assert _one_blank(written) == ["minor:printed text in a blank field"]


@pytest.mark.parametrize(
    "written",
    [
        "[___] John Smith",  # a name
        "________ John Smith",  # a fill line is a blank marker too
        "........ Engineer",  # a free word after a dot leader
        "[___] five hundred",  # a spelled-out amount
        "[___] john@example.com",  # an email
        "[___] www.example.org",  # a web address
    ],
)
def test_free_value_after_a_blank_marker_stays_invented(written: str) -> None:
    assert _one_blank(written) == ["critical:invented value"]
