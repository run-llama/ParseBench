"""Combined score of the extended scorers (``evaluation run --next``).

The score is the unweighted mean of six dimensions. Each dimension is a weighted mean of slots; a slot is
one metric averaged over the documents of one subset (a dataset evaluated on its own). ``NEXT_DIMENSIONS``
holds every weight. A dimension is reported missing, with the subsets it lacks, until all of its slots are
scored, and the combined score is reported only when all six dimensions are present.

Subsets are named in ``NEXT_SUBSETS``. A run names the subset of its documents in one of three ways:

- ``--next=SUBSET``: every document of the run belongs to SUBSET.
- ``--next=SUBSET=RESULTS_DIR[::TEST_CASES_DIR],...``: each listed results directory is evaluated as its
  subset; a test cases directory that is not given is read from the results' ``_metadata.json``.
- ``--next``: a document belongs to the subset its group maps to in ``GROUP_SUBSETS`` (the groups of the
  downloadable dataset); documents of other groups get the extended metrics but join no slot.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parse_bench.schemas.evaluation import EvaluationResult


@dataclass(frozen=True)
class NextSlot:
    """One weighted part of a dimension: ``metric`` averaged over the documents of ``subset``."""

    subset: str
    metric: str
    weight: float


NEXT_DIMENSIONS: dict[str, tuple[NextSlot, ...]] = {
    "text": (NextSlot("text", "text_next", 1.0),),
    "formatting": (
        NextSlot("text", "formatting_v2", 3 / 4),
        NextSlot("headings", "formatting_v2", 1 / 8),
        # strikeout and underline share one eighth: 1/8 * mean(strikeout, underline)
        NextSlot("strikeout", "formatting_v2", 1 / 16),
        NextSlot("underline", "formatting_v2", 1 / 16),
    ),
    "tables": (
        NextSlot("tables", "table_next", 1 / 2),
        NextSlot("complex_table_headers", "table_next", 1 / 8),
        NextSlot("financial_tables", "table_next", 1 / 8),
        NextSlot("insurance_filing_tables", "table_next", 1 / 8),
        NextSlot("dense_tables", "table_next", 1 / 8),
    ),
    "charts": (NextSlot("charts", "chart_next", 1.0),),
    "layout": (NextSlot("layout", "layout_grounding_pass_rate", 1.0),),
    "forms": (
        NextSlot("forms", "forms_v2", 7 / 8),
        NextSlot("fillable_forms", "forms_v2", 1 / 8),
    ),
}

NEXT_SUBSETS: tuple[str, ...] = tuple(dict.fromkeys(s.subset for slots in NEXT_DIMENSIONS.values() for s in slots))

# Groups of the downloadable dataset and the subset their documents score in (``--next`` without a subset).
GROUP_SUBSETS: dict[str, str] = {
    "text": "text",
    "text_content": "text",
    "text_formatting": "text",
    "table": "tables",
    "chart": "charts",
    "layout": "layout",
}

SUBSET_DIR_SEPARATOR = "::"
QUESTION_MARK = "#q"  # per-question results carry ``<document test id>#q<n>``


@dataclass(frozen=True)
class SubsetSource:
    """Where one subset's documents come from: a results directory and, optionally, its test cases."""

    subset: str
    output_dir: Path
    test_cases_dir: Path | None = None


def parse_next_option(value: Any, base_dir: Path) -> tuple[str | None, list[SubsetSource]]:
    """Read the ``--next`` value.

    Returns ``(subset, [])`` for ``--next`` (subset None: by group) or ``--next=SUBSET``, and
    ``(None, sources)`` for a list of ``SUBSET=RESULTS_DIR[::TEST_CASES_DIR]`` entries (a comma-separated
    string, a list of such strings, or a mapping of subset to directory). Relative directories resolve
    against ``base_dir``. Raises ``ValueError`` on an unknown subset or a malformed entry.
    """
    if value is True:
        return None, []
    entries: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        entries = [(str(k), v) for k, v in value.items()]
    else:
        items = value.split(",") if isinstance(value, str) else [str(v) for v in value]
        for item in (i.strip() for i in items):
            if not item:
                continue
            if "=" not in item:
                if len(items) != 1:
                    raise ValueError(f"--next entry {item!r} is not SUBSET=RESULTS_DIR")
                _check_subset(item)
                return item, []
            name, _, target = item.partition("=")
            entries.append((name.strip(), target.strip()))
    if not entries:
        raise ValueError("--next got an empty subset list")
    sources = []
    for name, target in entries:
        _check_subset(name)
        if isinstance(target, Mapping):
            out, cases = target.get("output_dir"), target.get("test_cases_dir")
        else:
            out, _, cases = str(target).partition(SUBSET_DIR_SEPARATOR)
        if not out:
            raise ValueError(f"--next subset {name!r} has no results directory")
        sources.append(
            SubsetSource(
                subset=name,
                output_dir=_resolve(base_dir, str(out)),
                test_cases_dir=_resolve(base_dir, str(cases)) if cases else None,
            )
        )
    if len({s.subset for s in sources}) != len(sources):
        raise ValueError("--next names a subset more than once")
    return None, sources


def _check_subset(name: str) -> None:
    if name not in NEXT_SUBSETS:
        raise ValueError(f"unknown --next subset {name!r}; expected one of: {', '.join(NEXT_SUBSETS)}")


def _resolve(base_dir: Path, path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else base_dir / p


def group_subset(test_id: str) -> str | None:
    """Subset of a document of the downloadable dataset, from the group prefix of its test id."""
    group = test_id.split("/", 1)[0] if "/" in test_id else ""
    return GROUP_SUBSETS.get(group)


def split_by_subset(
    results: Iterable[EvaluationResult], subset_of: Callable[[str], str | None]
) -> dict[str, list[EvaluationResult]]:
    """Group document results by the subset of their test id.

    Results of no subset are dropped, and so are per-question results (``<test id>#q<n>``): a slot averages
    documents, and a question's failure says nothing about the document's extended metrics.
    """
    by_subset: dict[str, list[EvaluationResult]] = {}
    for result in results:
        if QUESTION_MARK in result.test_id:
            continue
        subset = subset_of(result.test_id)
        if subset is not None:
            by_subset.setdefault(subset, []).append(result)
    return by_subset


def combine_next_score(
    subset_metrics: Mapping[str, Mapping[str, float]], subset_documents: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Combine per-subset averages into dimension scores and the overall score.

    ``subset_metrics`` maps a subset to its aggregate metrics (``avg_<metric>`` keys, as in an evaluation
    summary). A slot whose subset or metric is absent is missing; so is every dimension holding a missing
    slot, and the overall score unless all dimensions are present.
    """
    dimensions: dict[str, Any] = {}
    for name, slots in NEXT_DIMENSIONS.items():
        rows = []
        missing: list[str] = []
        total = weight = 0.0
        for slot in slots:
            value = subset_metrics.get(slot.subset, {}).get(f"avg_{slot.metric}")
            score = float(value) if value is not None and math.isfinite(value) else None
            rows.append(
                {
                    "subset": slot.subset,
                    "metric": slot.metric,
                    "weight": slot.weight,
                    "score": score,
                    "documents": (subset_documents or {}).get(slot.subset),
                }
            )
            if score is None:
                missing.append(slot.subset)
            else:
                total += slot.weight * score
                weight += slot.weight
        complete = not missing and weight > 0
        dimensions[name] = {
            "score": total / weight if complete else None,
            "missing_subsets": list(dict.fromkeys(missing)),
            "slots": rows,
        }
    present = [d["score"] for d in dimensions.values() if d["score"] is not None]
    missing_dimensions = [name for name, d in dimensions.items() if d["score"] is None]
    return {
        "next_score": sum(present) / len(present) if not missing_dimensions else None,
        "missing_dimensions": missing_dimensions,
        "dimensions": dimensions,
    }


def next_score_aggregates(combined: Mapping[str, Any]) -> dict[str, float]:
    """``next_score`` and ``next_<dimension>`` entries for an aggregate metrics dict (present values only)."""
    out = {f"next_{name}": d["score"] for name, d in combined["dimensions"].items() if d["score"] is not None}
    if combined["next_score"] is not None:
        out["next_score"] = combined["next_score"]
    return out


def format_next_score(combined: Mapping[str, Any]) -> str:
    """Console lines for a combined score."""
    lines = ["Next score:"]
    overall = combined["next_score"]
    lines.append(f"  next_score: {overall:.4f}" if overall is not None else "  next_score: missing")
    for name, d in combined["dimensions"].items():
        if d["score"] is not None:
            lines.append(f"  {name}: {d['score']:.4f}")
        else:
            lines.append(f"  {name}: missing (subsets not evaluated: {', '.join(d['missing_subsets'])})")
    return "\n".join(lines)
