"""forms_v2: grids that write a reference row over several output rows (captions over values, stacked or
rowspan headers)."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms


def _page(*items: dict[str, Any]) -> Any:
    return reference_from_tree([{"page_number": 1, "items": list(items)}])


def _text(label: str, value: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"type": "field", "field": "text", "label": label, "isEmpty": True}
    return {"type": "field", "field": "text", "label": label, "value": value}


def _charged(ref: Any, md: str) -> list[tuple[str, str]]:
    return [(e.kind, e.cls) for e in score_forms(ref, md).events if e.charged]


DEPENDENTS = _page(
    {
        "type": "table",
        "label": "Dependents",
        "columns": ["", "Dependent's SSN", "Date of birth", "Dependent's first name", "Dependent's last name"],
        "rows": [
            ["8.", "579092115", "1976-03-11", "Elena", "Okafor"],
            ["9.", "095334023", "2009-01-12", "Caleb", "Sorensen"],
            ["10.", "478114284", "1980-10-28", None, None],
        ],
    }
)


def _dependents_md(rows: list[tuple[str, str, str, str, str]]) -> str:
    """One HTML table; per dependent a caption row, its value row, a second caption row and its value row."""
    body = ""
    for n, ssn, dob, first, last in rows:
        body += f"<tr><td>{n} Dependent's SSN</td><td>Date of birth</td></tr><tr><td>{ssn}</td><td>{dob}</td></tr>"
        body += "<tr><td>Dependent's first name</td><td>Dependent's last name</td></tr>"
        if first or last:
            body += f"<tr><td>{first}</td><td>{last}</td></tr>"
    return f"<table>{body}</table>"


def test_reference_renders_to_one():
    assert score_forms(DEPENDENTS, render_reference(DEPENDENTS)).score == 1.0


def test_caption_rows_alternating_with_value_rows_read_the_row_below():
    md = _dependents_md(
        [
            ("8.", "579092115", "1976-03-11", "Elena", "Okafor"),
            ("9.", "095334023", "2009-01-12", "Caleb", "Sorensen"),
            ("10.", "478114284", "1980-10-28", "", ""),
        ]
    )
    assert _charged(DEPENDENTS, md) == []


def test_swapped_names_in_a_caption_value_block_stay_critical():
    md = _dependents_md(
        [
            ("8.", "579092115", "1976-03-11", "Okafor", "Elena"),
            ("9.", "095334023", "2009-01-12", "Caleb", "Sorensen"),
            ("10.", "478114284", "1980-10-28", "", ""),
        ]
    )
    assert [c for _, c in _charged(DEPENDENTS, md)] == ["critical", "critical"]


W2 = _page(
    {
        "type": "table",
        "label": "W-2s",
        "columns": ["", "Box b - EIN", "Box 1 - Wages", "Box 15 - Employer's ID", "Box 16 - State wages"],
        "rows": [
            ["1.", "219499854", "367,683", "Silverline Roofing", "45,274.84"],
            ["2.", "784593058", "655,814", None, "197,372"],
        ],
    }
)


def test_stacked_header_reads_each_line_of_a_row():
    md = (
        "<table><tr><th>No.</th><th>Box b - EIN</th><th>Box 1 - Wages</th></tr>"
        "<tr><th></th><th>Box 15 - Employer's ID</th><th>Box 16 - State wages</th></tr>"
        "<tr><td>1.</td><td>219499854</td><td>367,683</td></tr>"
        "<tr><td></td><td>Silverline Roofing</td><td>45,274.84</td></tr>"
        "<tr><td>2.</td><td>784593058</td><td>655,814</td></tr><tr><td></td><td></td><td>197,372</td></tr></table>"
    )
    assert _charged(W2, md) == []


PARENT = _page(_text("Relationship to you", "Spouse"), _text("Family name", "Tanaka"), _text("Any other names", "Noor"))


def _parent_md(values: tuple[str, str, str]) -> str:
    """A two-row header: rowspan captions beside a column whose caption is split over both rows (Sex / M/F/X)."""
    cells = "".join(f"<td>{v}</td>" for v in (*values, "N"))
    return (
        '<table><tr><th rowspan="2">Relationship to you</th><th rowspan="2">Family name</th>'
        '<th rowspan="2">Any other names</th><th>Sex</th></tr><tr><th>M/F/X</th></tr>'
        f"<tr>{cells}</tr></table>"
    )


def test_rowspan_caption_reads_the_cell_under_its_span():
    assert _charged(PARENT, _parent_md(("Spouse", "Tanaka", "Noor"))) == []


def test_wrong_value_under_a_rowspan_caption_stays_critical():
    assert _charged(PARENT, _parent_md(("Spouse", "Tanaka", "Nora Lee"))) == [("wrong value", "critical")]
