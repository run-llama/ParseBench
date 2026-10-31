"""Rule and prediction builders shared by the layout_grounding_pass_rate test files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parse_bench.evaluation.metrics.attribution.core import PredBlock
from parse_bench.evaluation.metrics.parse.layout_grounding import (
    compute_layout_grounding_metrics,
)
from parse_bench.schemas.evaluation import MetricValue
from parse_bench.test_cases.schema import LayoutTestRule

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "layout_grounding_presets.json").read_text())
PW, PH = 612.0, 792.0


def _rule(rid: str, box: list[float], cls: str = "Text", **kw: Any) -> LayoutTestRule:
    return LayoutTestRule(id=rid, page=1, bbox=box, canonical_class=cls, **kw)


def _word(rid: str, parent: str, box: list[float], text: str) -> LayoutTestRule:
    return LayoutTestRule(
        id=rid,
        page=1,
        bbox=box,
        canonical_class="Text",
        granularity="word",
        parent_test_id=parent,
        content={"type": "text", "text": text},
    )


def _line(rid: str, parent: str, box: list[float], ro: int) -> LayoutTestRule:
    return LayoutTestRule(
        id=rid, page=1, bbox=box, canonical_class="Text", granularity="line", parent_test_id=parent, ro_index=ro
    )


def _block(box: list[float], text: str, order: int) -> PredBlock:
    x, y, w, h = box
    return PredBlock(
        bbox_xyxy=[x, y, x + w, y + h],
        block_type="text",
        label="text",
        text=text,
        normalized_text=text,
        tokens=[],
        order_index=order,
    )


def _single_score(rules: list[LayoutTestRule], blocks: list[PredBlock]) -> MetricValue:
    (metric,) = compute_layout_grounding_metrics(rules, lambda page: blocks)
    return metric


def _math_doc(content: str, words: list[str]) -> list[LayoutTestRule]:
    box = [0.1, 0.1, 0.6, 0.05]
    rules = [_rule("R", box, content={"type": "text", "text": content})]
    for n, text in enumerate(words):
        rules.append(_word(f"W{n}", "R", [0.1 + n * 0.1, 0.11, 0.08, 0.03], text))
    return rules
