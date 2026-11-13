"""forms_v2: a run of captions the reference repeats block after block (``LOC # .. ZIP`` per premises), printed
by the output in a different order inside each block."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

LABELS = ["LOC #", "BLD #", "STREET", "CITY:", "STATE:", "ZIP:", "# FULL TIME EMPL:", "ANNUAL REVENUES:"]
FILLED = ["1", "1", "958 Katherine Crescent", "Paulbury", "MD", "21908", "60", "1,385,307.03"]
PRINTED = ["LOC #", "STREET", "# FULL TIME EMPL:", "ANNUAL REVENUES:", "BLD #", "CITY:", "STATE:", "ZIP:"]


def _block(values: list[str | None]) -> list[dict]:
    return [
        {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})}
        for k, v in zip(LABELS, values, strict=True)
    ]


PREMISES = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "section", "label": "PREMISES INFORMATION", "items": _block(FILLED) + _block([None] * 8) * 2}
            ],
        }
    ]
)


def _md(*blocks: dict[str, str]) -> str:
    lines = [f"**{k.rstrip(':')}:** {b.get(k, '[___]')}" for b in blocks for k in PRINTED]
    return "## PREMISES INFORMATION\n\n" + "\n\n".join(lines) + "\n"


def _charged(md: str) -> list[tuple[str, str, str]]:
    return [(e.label, e.expected, e.got) for e in score_forms(PREMISES, md).events if e.charged]


def test_each_block_reads_its_own_captions_when_printed_in_another_order():
    assert _charged(_md(dict(zip(LABELS, FILLED, strict=True)), {}, {})) == []


def test_a_value_printed_in_the_wrong_block_is_still_charged():
    first = dict(zip(LABELS, FILLED, strict=True))
    first.pop("CITY:")
    assert ("CITY:", "Paulbury", "") in _charged(_md(first, {"CITY:": "Paulbury"}, {}))


def test_a_block_that_repeats_a_caption_pair_inside_is_still_one_block():
    labels = ["LOC #", "PHONE", "LOC #", "PHONE", *LABELS[1:]]
    filled = ["1", "555-0101", "1", "555-0102", *FILLED[1:]]
    printed = ["LOC #", "PHONE", "LOC #", "PHONE", *PRINTED[1:]]

    def block(values: list[str | None]) -> list[dict]:
        return [
            {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})}
            for k, v in zip(labels, values, strict=True)
        ]

    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [{"type": "section", "label": "PREMISES", "items": block(filled) + block([None] * 11) * 2}],
            }
        ]
    )
    values = [dict(zip(labels[:2], filled[:2], strict=True)) | dict(zip(labels[4:], filled[4:], strict=True)), {}, {}]
    lines: list[str] = []
    for b in values:
        seen: dict[str, int] = {}
        for k in printed:
            seen[k] = seen.get(k, 0) + 1
            v = filled[labels.index(k) + 2 * (seen[k] - 1)] if b and k in ("LOC #", "PHONE") else b.get(k, "[___]")
            lines.append(f"**{k.rstrip(':')}:** {v}")
    md = "## PREMISES\n\n" + "\n\n".join(lines) + "\n"
    assert [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged] == []
