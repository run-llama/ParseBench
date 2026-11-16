"""Row descriptions containing column names are not repeated column headers."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms


def ref():
    def field(v):
        return {"items": [{"type": "field", "field": "text", "value": v}]}

    return reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "table",
                        "columns": ["", "Capital allowances", "Balancing charges"],
                        "rows": [
                            ["Capital allowances at 18% on equipment", field("3,251"), field("908")],
                            ["Capital allowances at 6% on equipment", field("977"), field("")],
                        ],
                    }
                ],
            }
        ]
    )


def test_canonical_table_with_overlapping_row_and_column_captions():
    r = ref()
    assert score_forms(r, render_reference(r)).score == 1


@pytest.mark.parametrize("old,new", [("3,251", "9,999"), ("977", "908"), ("3,251 | 908", "908 | 3,251")])
def test_wrong_and_swapped_cells_still_fail(old, new):
    r = ref()
    result = score_forms(r, render_reference(r).replace(old, new))
    assert result.score < 1
    assert any(e.charged and e.cls == "critical" for e in result.events)
