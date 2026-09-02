from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.rules_heading import HeadingStructureRule


def _rule(headings: list[tuple[str, int]]) -> HeadingStructureRule:
    rule = create_test_rule(
        {
            "type": "heading_structure",
            "headings": [{"text": text, "level": level} for text, level in headings],
        }
    )
    assert isinstance(rule, HeadingStructureRule)
    return rule


def test_exact_page_heading_structure_passes() -> None:
    rule = _rule([("Overview", 2), ("Financial results", 3)])

    passed, _, score = rule.run("## Overview\n\n### Financial **results**\n")

    assert passed
    assert score == 1.0
    assert rule.result_details["mean_level_distance"] == 0.0


def test_missed_and_extra_headings_are_penalized() -> None:
    rule = _rule([("Overview", 2), ("Financial results", 3)])

    passed, _, score = rule.run("## Overview\n\n### Unrelated section\n")

    assert not passed
    assert score == pytest.approx(0.5)
    assert rule.result_details["missed_count"] == 1
    assert rule.result_details["extra_count"] == 1


def test_absolute_level_distance_is_scored() -> None:
    rule = _rule([("Overview", 2), ("Financial results", 3)])

    _, _, score = rule.run("## Overview\n\n#### Financial results\n")

    assert score == pytest.approx(0.75)
    assert rule.result_details["mean_level_distance"] == 0.5


def test_empty_annotation_rejects_any_output_heading() -> None:
    assert _rule([]).run("ordinary prose")[2] == 1.0
    assert _rule([]).run("**bold lead-in**\n\nordinary prose")[2] == 1.0
    assert _rule([]).run("# Invented heading")[2] == 0.0


def test_heading_like_content_in_tables_and_code_is_ignored() -> None:
    markdown = """```markdown
# code sample
```

| # cell heading |
| --- |

<table><tr><td><h2>cell heading</h2></td></tr></table>
"""

    assert _rule([]).run(markdown)[2] == 1.0


def test_reordered_headings_cannot_both_match() -> None:
    rule = _rule([("First section", 2), ("Second section", 2)])

    _, _, score = rule.run("## Second section\n\n## First section\n")

    assert score == pytest.approx(0.5)
    assert rule.result_details["matched_count"] == 1
