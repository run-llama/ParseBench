"""Portable Metric Lab policies and rescoring of saved, content-free event counts.

Scores and penalties are fractions (0–1). Policy labels are frozen by format version;
new scorer kinds retain their recorded severity unless a later format exposes them.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Severity = Literal["critical", "major", "minor", "ignore"]


class SeverityValues:
    CRITICAL: Severity = "critical"
    MAJOR: Severity = "major"
    MINOR: Severity = "minor"
    IGNORE: Severity = "ignore"


LEVELS = (SeverityValues.CRITICAL, SeverityValues.MAJOR, SeverityValues.MINOR)
POLICY_INPUTS = "policy_inputs"
METRIC_KEYS = {
    "text_next": "text",
    "formatting_v2": "fmt",
    "table_next": "tables",
    "chart_next": "charts",
    "forms_v2": "forms",
}
PAPER = json.loads(Path(__file__).with_name("paper_policy.json").read_text(encoding="utf-8"))
SYSTEMATIC = re.compile(r" \(systematic, \d+x\)$")
# These are the v1 labels for kinds whose recorded severity changes their meaning.
TABLE_LABELS = {
    "cell errors in row": {
        "critical": "wrong or misplaced number in a row",
        "major": "wrong text, lost unit or partly different value in a row",
    },
    "systematic column error": {
        "critical": "systematic column error: wrong numbers (3+ in one column)",
        "major": "systematic column error: wrong text or units (3+ in one column)",
        "minor": "systematic column error: small differences (3+ in one column)",
    },
    "contradicting copy": {
        "critical": "contradicting copy of a table: number",
        "major": "contradicting copy of a table: text",
    },
    "label reworded": {
        "major": "label reworded (different wording)",
        "minor": "label reworded (close wording)",
        "ignore": "label reworded (repeat, not charged)",
    },
    "row-label header differs": {"major": "row-label header differs", "minor": "row-label header differs (mild)"},
    "table transposed": {"critical": "table transposed, structure lost", "major": "table transposed"},
    "table missing": {
        "critical": "table missing",
        "ignore": "table missing (all rows of one table; its rows are charged)",
    },
}
CELL_EVENTS = {
    "wrong value",
    "value off",
    "point missing",
    "label typo",
    "repeated point",
    "range instead of value",
    "category label truncated",
    "category label reworded",
}
OMISSION_EVENTS = {"point missing", "category missing", "series missing"}


class MetricPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format: Literal["parsebench-metric-policy"]
    version: Literal[1]
    weights: dict[str, float]
    dimension_weights: dict[str, float]
    scaling: dict[str, float]
    error_classes: dict[str, dict[str, Severity]]

    @classmethod
    def load(cls, path: str | Path) -> MetricPolicy:
        policy = cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        for field in ("weights", "dimension_weights", "scaling"):
            values = getattr(policy, field)
            if values.keys() != PAPER[field].keys():
                raise ValueError(f"{field}: expected keys {sorted(PAPER[field])}")
            if any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError(f"{field}: values must be finite and nonnegative")
        for key in ("minorCap", "orderCap"):
            if policy.scaling[key] > 1:
                raise ValueError(f"{key} must be in 0–1 units")
        if any(v > 1 for v in policy.weights.values()):
            raise ValueError("weights must be in 0–1 units")
        if not sum(policy.dimension_weights.values()):
            raise ValueError("at least one dimension weight must be positive")
        if policy.error_classes.keys() != PAPER["error_classes"].keys():
            raise ValueError("error_classes must name text, fmt, tables, charts and forms")
        for metric, entries in policy.error_classes.items():
            if entries.keys() != PAPER["error_classes"][metric].keys():
                raise ValueError(f"{metric}: error class keys do not match policy format v1")
        for label in ("moved block (reading order)", "cut sentence (reading order)"):
            if policy.error_classes["text"][label] not in (SeverityValues.MAJOR, SeverityValues.IGNORE):
                raise ValueError("reading-order events can only be major or ignore")
        return policy


def event_label(metric: str, kind: str, severity: str) -> str:
    if metric == "tables":
        return TABLE_LABELS.get(kind, {}).get(severity, kind)
    if metric == "charts" and kind == "legend lost":
        return "legend lost (several series)" if severity == SeverityValues.CRITICAL else "legend lost (one series)"
    if metric == "forms":
        if kind.startswith("form fields missing"):
            return "form fields missing (under half found; coverage cap applies)"
        if kind == "contradicting copy":
            return "contradicting copy of a field"
    return kind


def atom_inputs(events: list[Any], metric: str) -> list[dict[str, Any]]:
    """Keep charged counts and systematic repeats separately, without document text."""
    systematic = {e.group for e in events if SYSTEMATIC.search(e.kind)}
    counts: Counter[tuple[str, str, bool]] = Counter()
    for e in events:
        severity = e.cls if e.cls in LEVELS else SeverityValues.IGNORE
        # Coverage diagnostics are independently reclassifiable in Metric Lab.
        # Other uncharged events can be redundant flips within one choice group.
        if metric == "forms" and e.kind.startswith("form fields missing"):
            severity = SeverityValues.IGNORE
        grouped = not e.charged and e.group in systematic
        if not e.charged and not grouped and severity != SeverityValues.IGNORE:
            continue  # e.g. a second flip in the same exclusive-choice group
        kind = SYSTEMATIC.sub("", e.kind)
        label = f"all {severity} errors" if metric == "fmt" else event_label(metric, kind, severity)
        counts[label, severity, grouped] += 1
    return [{"label": k, "severity": s, "grouped": g, "count": n} for (k, s, g), n in counts.items()]


def table_inputs(items: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for kind, _, _, severity in items:
        if severity == "detail":
            continue
        severity = severity if severity in LEVELS else SeverityValues.IGNORE
        counts[event_label("tables", kind, severity), severity] += 2 if kind == "systematic column error" else 1
    return [{"label": k, "severity": s, "count": n} for (k, s), n in counts.items()]


def score_metric(metric: str, metadata: dict[str, Any], policy: MetricPolicy) -> float:
    """Recompute a page cap from sufficient statistics, keeping its parsing/coverage score."""
    key = METRIC_KEYS[metric]
    if "proportional" not in metadata:
        raise ValueError(f"{metric}: missing parsing score; rerun evaluation with --next --next_only")
    parsing = metadata["proportional"]
    if key in ("fmt", "forms"):
        parsing = min(parsing, metadata["coverage"] / 0.5)
    w, p = policy.weights, policy.scaling
    if key == "text":
        events = [{"label": f"all {s} errors", "severity": s, "count": metadata["events"][s]} for s in LEVELS]
        order = sum(
            metadata["events"][field] * mult
            for field, mult, label in (
                ("moved", 1, "moved block (reading order)"),
                ("broken", 0.5, "cut sentence (reading order)"),
            )
            if policy.error_classes[key][label] != SeverityValues.IGNORE
        )
        size, constant = metadata["n_ref_tokens"], p["textL0"]
    else:
        if POLICY_INPUTS not in metadata:
            raise ValueError(
                f"{metric}: saved report lacks exact policy counts; rerun evaluation with --next --next_only"
            )
        events = metadata[POLICY_INPUTS]
        if key == "charts":
            return float(min(parsing, _chart_cap(events, policy)))
        size = metadata["reference_cells"] if key == "tables" else metadata["atoms"]
        constant = p["tableK"] if key == "tables" else p["atomsK"]
    totals: Counter[str] = Counter()
    for event in events:
        severity = policy.error_classes[key].get(event["label"], event["severity"])
        if event.get("grouped") and severity != SeverityValues.CRITICAL:
            continue
        totals[severity] += event["count"]
    scale = min(1, constant / max(1, size))
    charged = {s: w[s] * (min(totals[s], 1) + max(0, totals[s] - 1) * scale) for s in LEVELS}
    if key != "tables":
        charged[SeverityValues.MINOR] = w[SeverityValues.MINOR] * totals[SeverityValues.MINOR]
    penalty = (
        charged[SeverityValues.CRITICAL]
        + charged[SeverityValues.MAJOR]
        + min(p["minorCap"], charged[SeverityValues.MINOR])
    )
    if key == "text":
        count = totals[SeverityValues.MAJOR] + order
        with_order = w[SeverityValues.MAJOR] * (min(count, 1) + max(0, count - 1) * scale)
        penalty += min(p["orderCap"], with_order - charged[SeverityValues.MAJOR])
    return float(min(parsing, max(0, min(1, 1 - penalty))))


def _chart_cap(groups: list[dict[str, Any]], policy: MetricPolicy) -> float:
    # Retain each chart's omission limit and page-level first-event charge. The Lab's
    # legacy snapshot estimates these groups; fresh CLI reports preserve them exactly.
    costs: dict[str, float] = defaultdict(float)
    structural: Counter[str] = Counter()
    first_cell: dict[str, float] = {}
    credit = 0.0
    for group in groups:
        unit = min(1, policy.scaling["chartK"] / max(1, group["size"]))
        omitted = 0.0
        for event in group["events"]:
            kind, original, count = event
            severity = policy.error_classes["charts"].get(event_label("charts", kind, original), original)
            if severity == SeverityValues.IGNORE:
                continue
            cell = group["mode"] == "tables" and kind in CELL_EVENTS
            cost = policy.weights[severity] * count * (unit if cell else 1)
            costs[severity] += cost
            if cell:
                first_cell.setdefault(severity, unit)
            else:
                structural[severity] += count
            if group["mode"] == "tables" and kind in OMISSION_EVENTS and severity != SeverityValues.MINOR:
                omitted += cost
        if group.get("cap_omissions", False):
            credit += max(0, omitted - policy.weights[SeverityValues.CRITICAL])
    for severity, unit in first_cell.items():
        if not structural[severity]:
            costs[severity] += policy.weights[severity] * (1 - unit)
    return max(
        0,
        min(
            1,
            1
            - costs[SeverityValues.CRITICAL]
            - costs[SeverityValues.MAJOR]
            + credit
            - min(policy.scaling["minorCap"], costs[SeverityValues.MINOR]),
        ),
    )
