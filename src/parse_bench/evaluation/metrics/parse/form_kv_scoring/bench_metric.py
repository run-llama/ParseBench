"""Bench bridge: the `form_kv_list_*` (enriched `list.md`) and `form_kv_md_*` (page markdown)
families, gated on `metadata["forms"]` like `form_*`. Coverage is a GT property: prediction pages
are always exhaustive and never read a coverage comment. GT is the FormNode tree under
`metadata.forms.gt` (rendered and parsed) or key-value markdown under `metadata.forms.gt_pages_md`
(`[{page_number, markdown, coverage?}]`). Each arm is emitted only when its source exists; a
broken GT yields a loud `..._composite = 0` row; a doc with only spot pages emits `spot_*` rows
and no composite. Cost and latency ride on the `form_*` rows of the same doc.

The page markdown goes through `kv_markdown_adapter` first (its rewrites are counted on the row as
`pred_md_adapted`); the enriched list is parsed as written. A dataset whose GT scores only the
ordinary markdown of a plain parse run says so with `metadata.forms.scope == "markdown"`: the list
arm is then not emitted even when an enriched result is present.
"""

from __future__ import annotations

import copy
from typing import Any

from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_grammar import parse_kv_markdown
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_markdown_adapter import (
    DIAG_KEYS as ADAPTER_DIAG_KEYS,
)
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_markdown_adapter import (
    adapt_page_markdown,
    adapter_summary,
)
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_records import (
    COVERAGE_EXHAUSTIVE,
    COVERAGE_SPOT,
    KvPage,
)
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_render import pages_from_tree
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_submetrics import score_pages
from parse_bench.evaluation.metrics.parse.formnode_scoring.formnode_compat import normalize_formnode_types
from parse_bench.schemas.evaluation import MetricValue

ARMS = ("list", "md")
SCOPE_MARKDOWN = "markdown"


def scores_markdown_only(forms_meta: dict[str, Any]) -> bool:
    """True for a forms GT that scores the ordinary page markdown only (no enriched forms output expected)."""
    return forms_meta.get("scope") == SCOPE_MARKDOWN


# finalize() key -> emitted suffix; None values are not emitted
SCORE_METRICS: dict[str, str] = {
    "composite": "composite",
    "core_kv_f1": "core_kv_f1",
    "core_kv_recall": "core_kv_recall",
    "core_kv_precision": "core_kv_precision",
    "value_f1": "value_f1",
    "value_strict_f1": "value_f1_strict",
    "label_f1": "label_f1",
    "label_strict_f1": "label_f1_strict",
    "id_f1": "id_f1",
    "id_lenient_f1": "id_f1_lenient",
    "checkbox_f1": "checkbox_f1",
    "signature_f1": "signature_f1",
    "structure": "structure",
    "spot_composite": "spot_composite",
    "spot_core_kv_recall": "spot_core_kv_recall",
    "spot_declared_precision": "spot_declared_precision",
}
_DIAG_KEYS = ("records", "ids", "unparsed", "bold_colon_unparsed", "continuation_lines", "tables", "md_adapted_lines")
_DIAG_NAMES = {"md_adapted_lines": "md_adapted"}


SOURCE_FORM_FIELD_RULES = "form_field_rules"


def _rule_attr(rule: Any, name: str) -> Any:
    return rule.get(name) if isinstance(rule, dict) else getattr(rule, name, None)


def _esc_md(text: Any) -> str:
    return str(text).strip().replace("*", "\\*")


def _rule_value(rule: Any) -> str:
    """`[x]`/`[ ]` for a box, `[signature]` or `[blank]` for a signature line, the text or `[blank]`."""
    value, kind = _rule_attr(rule, "value"), _rule_attr(rule, "value_type") or "text"
    if kind == "checkbox":
        return "[x]" if value is True else "[ ]"
    if kind == "signature":
        return "[signature]" if value is True else "[blank]"
    text = "" if value is None else str(value).strip()
    return _esc_md(text) if text else "[blank]"


def _rule_labels(rule: Any) -> list[str]:
    """A rule's label path: `"City"`, or `["3. Gender", "M"]` for an option under its caption."""
    label = _rule_attr(rule, "label")
    parts = label if isinstance(label, list) else [label]
    return [str(p).strip() for p in parts if p is not None and str(p).strip()]


def _key_line(label: str, value: str, indent: str = "") -> str:
    """The guide's field line: the colon inside the bold when the label prints one."""
    key = _esc_md(label)
    return f"{indent}- **{key}** {value}" if key.endswith(":") else f"{indent}- **{key}**: {value}"


def _rules_page_markdown(rules: list[Any]) -> str:
    """One page's rules as guide list lines. A rule whose label is `[caption, option]` is an option
    nested under its caption: `- **caption**` then `  - [ ] option` (box first) or
    `  - **option**: value`; options sharing a caption share one caption line."""
    lines: list[str] = []
    groups: dict[str, int] = {}  # caption -> index of its last line
    for rule in rules:
        labels = _rule_labels(rule)
        value = _rule_value(rule)
        if len(labels) == 1:
            lines.append(_key_line(labels[0], value))
            continue
        caption, option = labels[0], " ".join(labels[1:])
        box = _rule_attr(rule, "value_type") == "checkbox"
        child = f"  - {value} {_esc_md(option)}" if box else _key_line(option, value, "  ")
        if caption in groups:
            at = groups[caption] + 1
            lines.insert(at, child)
            groups = {c: (i + 1 if i >= at else i) for c, i in groups.items()}
            groups[caption] = at
        else:
            lines.append(f"- **{_esc_md(caption)}**")
            lines.append(child)
            groups[caption] = len(lines) - 1
    return "\n".join(lines)


def forms_meta_from_rules(test_rules: list[Any] | None) -> dict[str, Any] | None:
    """A markdown-scoped forms GT built from a dataset's `form_field` rules, or None when it has none.

    The rules list the fields an annotator could identify, not every field on the page, so each page
    is a spot page: the recall side is scored (`form_kv_md_spot_*`), a transcribed field the rules
    do not list costs nothing."""
    by_page: dict[int, list[Any]] = {}
    for rule in test_rules or []:
        if _rule_attr(rule, "type") != "form_field" or not _rule_labels(rule):
            continue
        by_page.setdefault(int(_rule_attr(rule, "page") or 1), []).append(rule)
    if not by_page:
        return None
    return {
        "scope": SCOPE_MARKDOWN,
        "source": SOURCE_FORM_FIELD_RULES,
        "gt_pages_md": [
            {"page_number": page, "markdown": _rules_page_markdown(rules), "coverage": COVERAGE_SPOT}
            for page, rules in sorted(by_page.items())
        ],
    }


def metric_name(arm: str, suffix: str) -> str:
    return f"form_kv_{arm}_{suffix}"


def _zero_row(arm: str, error: str) -> MetricValue:
    return MetricValue(metric_name=metric_name(arm, "composite"), value=0.0, metadata={"error": error}, details=[error])


def gt_pages_from_meta(forms_meta: dict[str, Any]) -> tuple[list[KvPage] | None, str | None]:
    gt = forms_meta.get("gt")
    if gt:
        tree = normalize_formnode_types(copy.deepcopy(gt))
        pages = tree.get("pages") if isinstance(tree, dict) else None
        if not pages:
            return None, "metadata.forms.gt has no pages — broken dataset export"
        return pages_from_tree(pages), None
    md_pages = forms_meta.get("gt_pages_md")
    if isinstance(md_pages, list) and md_pages:
        out = []
        try:
            for i, pg in enumerate(md_pages):
                if not isinstance(pg, dict):
                    continue
                out.append(
                    parse_kv_markdown(pg.get("markdown") or "", pg.get("page_number", i + 1), pg.get("coverage"))
                )
        except ValueError as exc:
            return None, f"metadata.forms.gt_pages_md: {exc}"
        if out:
            return out, None
    return None, "metadata.forms carries neither a FormNode gt tree nor gt_pages_md"


def list_arm_pages(raw_output: Any) -> tuple[list[KvPage] | None, int]:
    """Pages from the enriched result's `list.md`; a failed page scores as empty."""
    forms = raw_output.get("forms") if isinstance(raw_output, dict) else None
    if not isinstance(forms, dict) or not forms.get("pages"):
        return None, 0
    pages: list[KvPage] = []
    failed = 0
    for pg in forms.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        page_number = pg.get("page_number", len(pages) + 1)
        if pg.get("success") is False:
            failed += 1
            pages.append(KvPage(page_number=page_number, records=[], diagnostics={"parse_error": True}))
            continue
        md = "\n".join(
            form["list"]["md"]
            for form in pg.get("forms") or []
            if isinstance(form, dict) and isinstance(form.get("list"), dict) and form["list"].get("md")
        )
        pages.append(parse_kv_markdown(md, page_number, COVERAGE_EXHAUSTIVE))
    return (pages or None), failed


def _parse_page_markdown(markdown: str, page_number: int) -> KvPage:
    """The page markdown through the dialect adapter, its rewrite counts kept on the page."""
    adapted, adapter_diag = adapt_page_markdown(markdown)
    page = parse_kv_markdown(adapted, page_number, COVERAGE_EXHAUSTIVE)
    page.diagnostics.update(adapter_diag)
    return page


def md_arm_pages(output: Any, gt_page_count: int) -> tuple[list[KvPage] | None, str | None]:
    """Pages from the page markdown; a provider that keeps only document-level markdown
    scores it as page 1 against a one-page GT and is skipped, with the reason, otherwise."""
    pages = getattr(output, "pages", None)
    if pages:
        out = []
        for i, pg in enumerate(pages):
            index = getattr(pg, "page_index", i)
            out.append(_parse_page_markdown(getattr(pg, "markdown", "") or "", int(index) + 1))
        return out, None
    doc_md = getattr(output, "markdown", None)
    if isinstance(doc_md, str) and doc_md.strip():
        if gt_page_count == 1:
            return [_parse_page_markdown(doc_md, 1)], "document-level markdown scored as page 1"
        return None, f"document-level markdown only; it cannot be paired with a {gt_page_count}-page GT"
    return None, None


def _arm_diagnostics(gt_pages: list[KvPage], pred_pages: list[KvPage], r: dict[str, Any]) -> dict[str, Any]:
    diag: dict[str, Any] = {
        f"pred_{_DIAG_NAMES.get(k, k)}": sum(p.diagnostics.get(k, 0) for p in pred_pages) for k in _DIAG_KEYS
    }
    diag["gt_records"] = sum(len(p.records) for p in gt_pages)
    diag["matched_gt"] = r["counts"]["all"]["matched_gt"]
    diag["rescued_selects"] = r["counts"]["all"]["rescued_selects"]
    diag["baked_ids"] = r["counts"]["all"]["baked_ids"]
    diag["exhaustive_pages"] = r["exhaustive_pages"]
    diag["spot_pages"] = r["spot_pages"]
    diag["gt_roundtrip_loss"] = sum(abs(p.diagnostics.get("roundtrip_loss", 0)) for p in gt_pages)
    return diag


def _arm_rows(arm: str, r: dict[str, Any], diag: dict[str, Any], details: list[str]) -> list[MetricValue]:
    rows: list[MetricValue] = []
    composite = r.get("composite")
    if composite is None and r.get("spot_composite") is None:
        composite = 0.0
        details = [*details, "composite unscorable: no axis had a denominator"]
    if composite is not None:
        rows.append(
            MetricValue(
                metric_name=metric_name(arm, "composite"), value=float(composite), metadata=diag, details=details
            )
        )
    for key, suffix in SCORE_METRICS.items():
        if key == "composite":
            continue
        value = r.get(key)
        if value is not None:
            row_meta = diag if (key == "spot_composite" and r.get("composite") is None) else {}
            rows.append(MetricValue(metric_name=metric_name(arm, suffix), value=float(value), metadata=row_meta))
    return rows


def compute_form_kv_metrics(inference_result: Any, forms_meta: dict[str, Any]) -> list[MetricValue]:
    gt_pages, err = gt_pages_from_meta(forms_meta)
    if scores_markdown_only(forms_meta):
        list_pages, failed_pages = None, 0
    else:
        list_pages, failed_pages = list_arm_pages(getattr(inference_result, "raw_output", None))
    md_pages, md_note = md_arm_pages(getattr(inference_result, "output", None), len(gt_pages or []))
    arms = [(arm, pages) for arm, pages in (("list", list_pages), ("md", md_pages)) if pages is not None]

    if gt_pages is None:
        return [_zero_row(arm, err or "no ground truth") for arm, _ in arms] or [
            _zero_row("md", err or "no ground truth")
        ]
    if not arms:
        why = md_note or "no page markdown and no forms result in the inference result"
        return [_zero_row("md", f"nothing to score: {why}")]

    metrics: list[MetricValue] = []
    for arm, pages in arms:
        r = score_pages(gt_pages, pages)
        diag = _arm_diagnostics(gt_pages, pages, r)
        diag["doc_id"] = forms_meta.get("doc_id")
        details: list[str] = []
        if arm == "list" and failed_pages:
            diag["failed_pages"] = failed_pages
            details.append(f"{failed_pages}/{len(pages)} form pages failed in the product output")
        if arm == "md" and md_note:
            details.append(md_note)
        if arm == "md":
            adapted = adapter_summary({k: sum(p.diagnostics.get(k, 0) for p in pages) for k in ADAPTER_DIAG_KEYS})
            if adapted:
                details.append(adapted)
        if diag["pred_unparsed"]:
            unparsed, bold = diag["pred_unparsed"], diag["pred_bold_colon_unparsed"]
            details.append(f"{unparsed} non-list lines unparsed ({bold} bold-colon paragraphs)")
        if diag["gt_roundtrip_loss"]:
            details.append(
                f"GT render round trip changed {diag['gt_roundtrip_loss']} record(s): a shape the list cannot express"
            )
        metrics.extend(_arm_rows(arm, r, diag, details))
    return metrics
