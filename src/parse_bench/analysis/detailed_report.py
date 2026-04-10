"""Detailed HTML report generation for evaluation results.

Generates a self-contained, interactive HTML evaluation report with:
- Summary cards with key metrics
- Aggregate metrics panel with color-coded score bars
- Collapsible aggregate stats (latency, cost, tokens)
- Interactive examples table with metric selector, filters, sort, search, pagination
- Detail panel with per-example metrics, rule results, PDF viewer, and stats

This module provides the fancy interactive report (_evaluation_report_detailed.html).
It should be run as a separate step after evaluation to explore results in detail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import bleach
import markdown2

from parse_bench.analysis.metric_definitions import (
    TOOLTIP_CSS,
    TOOLTIP_JS,
    display_name,
    display_name_dict,
    tooltip_dict,
)
from parse_bench.schemas.evaluation import EvaluationSummary


def _render_markdown_to_html(md_text: str) -> str:
    """Render markdown to sanitised HTML, preserving HTML tables with colspan/rowspan."""
    if not md_text:
        return ""

    # Extract HTML tables before markdown2 processing (it can mangle colspan/rowspan)
    html_table_pattern = r"<table[^>]*>.*?</table>"
    processed_md = md_text
    table_placeholders: dict[str, str] = {}
    matches = list(re.finditer(html_table_pattern, md_text, re.DOTALL | re.IGNORECASE))
    for i, match in enumerate(reversed(matches)):
        placeholder = f"<!--HTMLTABLE_{len(matches) - 1 - i}-->"
        table_placeholders[placeholder] = match.group(0)
        s, e = match.span()
        processed_md = processed_md[:s] + placeholder + processed_md[e:]

    rendered = markdown2.markdown(processed_md, extras=["tables", "fenced-code-blocks", "break-on-newline"])

    # Restore original HTML tables
    for placeholder, table_html in table_placeholders.items():
        rendered = rendered.replace(placeholder, table_html)

    allowed_tags = bleach.sanitizer.ALLOWED_TAGS | {
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "caption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "br",
        "hr",
        "pre",
        "code",
        "img",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "div",
        "span",
        "sup",
        "sub",
    }
    allowed_attrs = {
        **bleach.sanitizer.ALLOWED_ATTRIBUTES,
        "th": ["colspan", "rowspan", "scope"],
        "td": ["colspan", "rowspan"],
        "img": ["src", "alt", "width", "height"],
        "code": ["class"],
        "pre": ["class"],
    }
    return str(bleach.clean(rendered, tags=allowed_tags, attributes=allowed_attrs))


def _build_data_blob(
    summary: EvaluationSummary,
    output_dir: Path | None = None,
    test_cases_dir: Path | None = None,
    pdf_base_url: str = "",
) -> dict[str, Any]:
    """Build the JSON data blob that powers the client-side rendering."""

    # --- load predicted/expected output from files ---
    predicted_map: dict[str, str] = {}
    expected_map: dict[str, str] = {}
    job_id_map: dict[str, str] = {}
    parse_job_logs_url_map: dict[str, str] = {}
    parse_job_logs_local_path_map: dict[str, str] = {}
    parse_job_logs_html_path_map: dict[str, str] = {}

    if output_dir and output_dir.exists():
        for result_file in output_dir.rglob("*.result.json"):
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                test_id = result_file.stem.replace(".result", "")
                output = data.get("output") or {}
                raw_output = data.get("raw_output") or {}
                # Parse output: markdown field
                if isinstance(output, dict) and output.get("markdown"):
                    predicted_map[test_id] = output["markdown"]
                # Extract output: extracted_data field
                elif isinstance(output, dict) and output.get("extracted_data"):
                    predicted_map[test_id] = json.dumps(output["extracted_data"], indent=2, ensure_ascii=False)
                # Job ID from output (e.g. LlamaParse)
                if isinstance(output, dict) and output.get("job_id"):
                    job_id_map[test_id] = output["job_id"]
                if isinstance(raw_output, dict):
                    job_logs_url = raw_output.get("job_logs_url")
                    if not isinstance(job_logs_url, str) or not job_logs_url:
                        job_logs = raw_output.get("job_logs")
                        if isinstance(job_logs, dict):
                            nested_url = job_logs.get("url")
                            if isinstance(nested_url, str) and nested_url:
                                job_logs_url = nested_url
                    if isinstance(job_logs_url, str) and job_logs_url:
                        parse_job_logs_url_map[test_id] = job_logs_url

                    job_logs_local = raw_output.get("job_logs_local_path")
                    if isinstance(job_logs_local, str) and job_logs_local:
                        parse_job_logs_local_path_map[test_id] = job_logs_local

                    job_logs_html = raw_output.get("job_logs_html_local_path")
                    if isinstance(job_logs_html, str) and job_logs_html:
                        parse_job_logs_html_path_map[test_id] = job_logs_html
            except Exception:
                pass

    if test_cases_dir and test_cases_dir.exists():
        for test_file in test_cases_dir.rglob("*.test.json"):
            try:
                data = json.loads(test_file.read_text(encoding="utf-8"))
                test_id = test_file.stem.replace(".test", "")
                if data.get("expected_markdown"):
                    expected_map[test_id] = data["expected_markdown"]
                elif data.get("expected_output"):
                    expected_map[test_id] = json.dumps(data["expected_output"], indent=2, ensure_ascii=False)
            except Exception:
                pass

    # --- aggregate metrics (group avg/min/max) ---
    # Per-doc table count metrics are bookkeeping, not quality scores --
    # exclude them from the detailed report's aggregate metric panel.
    _hidden_table_count_metrics = {
        "tables_expected",
        "tables_actual",
        "tables_paired",
        "tables_unmatched_expected",
        "tables_unmatched_pred",
        "tables_unparseable_pred",
    }
    metric_groups: dict[str, dict[str, float]] = {}
    for key, value in summary.aggregate_metrics.items():
        for prefix in ("avg_", "min_", "max_"):
            if key.startswith(prefix):
                base = key[len(prefix) :]
                if base in _hidden_table_count_metrics:
                    break
                metric_groups.setdefault(base, {})[prefix.rstrip("_")] = value
                break

    agg_metrics_unsorted = [
        {
            "name": name,
            "displayName": display_name(name),
            "avg": vals.get("avg", 0.0),
            "min": vals.get("min", 0.0),
            "max": vals.get("max", 0.0),
        }
        for name, vals in metric_groups.items()
    ]
    agg_metrics = sorted(
        agg_metrics_unsorted,
        key=lambda m: cast(float, m["avg"]),
        reverse=True,
    )

    # --- aggregate stats ---
    agg_stats = []
    for stat_name, agg in sorted(summary.aggregate_stats.items()):
        agg_stats.append(
            {
                "name": stat_name,
                "displayName": stat_name.replace("_", " ").title(),
                "unit": agg.get("unit", ""),
                "avg": agg.get("avg", 0),
                "min": agg.get("min", 0),
                "max": agg.get("max", 0),
                "p50": agg.get("p50", 0),
                "p95": agg.get("p95", 0),
                "p99": agg.get("p99", 0),
                "total": agg.get("total", 0),
                "count": agg.get("count", 0),
            }
        )

    # --- metric names lookup ---
    metric_names_map: dict[str, str] = {}
    for base_name in metric_groups:
        metric_names_map[base_name] = display_name(base_name)

    # --- collect all tags ---
    all_tags: set[str] = set()
    for result in summary.per_example_results:
        all_tags.update(result.tags)

    # --- per-example data ---
    examples = []
    for result in summary.per_example_results:
        metrics_dict: dict[str, float] = {}
        rule_details: dict[str, dict[str, int]] = {}
        rule_results_map: dict[str, list[dict[str, Any]]] = {}
        metric_details_map: dict[str, list[str]] = {}

        for mv in result.metrics:
            if mv.metric_name in _hidden_table_count_metrics:
                continue
            metrics_dict[mv.metric_name] = mv.value
            # Add to metric_names_map if not already there
            if mv.metric_name not in metric_names_map:
                metric_names_map[mv.metric_name] = display_name(mv.metric_name)

            # Collect human-readable detail strings
            if mv.details:
                metric_details_map[mv.metric_name] = mv.details

            # Extract rule details from metadata
            if "rule_results" in mv.metadata:
                passed = sum(1 for r in mv.metadata["rule_results"] if r.get("passed"))
                total = len(mv.metadata["rule_results"])
                rule_details[mv.metric_name] = {"passed": passed, "total": total}
                rule_results_map[mv.metric_name] = [
                    {
                        "type": r.get("type", ""),
                        "passed": r.get("passed", False),
                        "id": r.get("id", ""),
                        "message": r.get("message", ""),
                    }
                    for r in mv.metadata["rule_results"]
                ]

        stats_dict: dict[str, float] = {}
        for s in result.stats:
            stats_dict[s.name] = s.value

        examples.append(
            {
                "id": result.test_id,
                "success": result.success,
                "error": result.error,
                "tags": result.tags,
                "productType": result.product_type,
                "jobId": (
                    result.job_id
                    or job_id_map.get(result.test_id)
                    or job_id_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "parseJobId": result.parse_job_id or "",
                "parseJobLogsUrl": (
                    parse_job_logs_url_map.get(result.test_id)
                    or parse_job_logs_url_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "parseJobLogsLocalPath": (
                    parse_job_logs_local_path_map.get(result.test_id)
                    or parse_job_logs_local_path_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "parseJobLogsHtmlPath": (
                    parse_job_logs_html_path_map.get(result.test_id)
                    or parse_job_logs_html_path_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "metrics": metrics_dict,
                "stats": stats_dict,
                "ruleDetails": rule_details,
                "ruleResults": rule_results_map,
                "metricDetails": metric_details_map,
                "predictedOutput": (
                    predicted_map.get(result.test_id) or predicted_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "expectedOutput": (
                    expected_map.get(result.test_id) or expected_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "predictedHtml": _render_markdown_to_html(
                    predicted_map.get(result.test_id) or predicted_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
                "expectedHtml": _render_markdown_to_html(
                    expected_map.get(result.test_id) or expected_map.get(result.test_id.rsplit("/", 1)[-1], "")
                ),
            }
        )

    completed_at_str = ""
    if summary.completed_at is not None:
        completed_at_str = summary.completed_at.isoformat()

    return {
        "summary": {
            "total": summary.total_examples,
            "successful": summary.successful,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "completedAt": completed_at_str,
        },
        "aggMetrics": agg_metrics,
        "aggStats": agg_stats,
        "metricNames": metric_names_map,
        "metricTooltips": tooltip_dict(),
        "tags": sorted(all_tags),
        "tagMetrics": {tag: dict(metrics.items()) for tag, metrics in summary.tag_metrics.items()},
        "examples": examples,
        "pdfBaseUrl": pdf_base_url,
    }


# ---------------------------------------------------------------------------
# HTML template parts
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
"""

_CSS = """\
/* ───── Reset & variables ───── */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #f8f7f4;
    --fg: #1c1917;
    --card: #ffffff;
    --border: #e7e5e4;
    --muted: #78716c;
    --muted-light: #a8a29e;
    --cream: #faf9f6;
    --emerald: #059669;
    --emerald-bg: #ecfdf5;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --blue: #2563eb;
    --blue-bg: #eff6ff;
    --font-heading: 'Newsreader', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    --shadow-sm: 0 1px 2px rgba(28,25,23,0.05);
    --shadow-md: 0 4px 6px -1px rgba(28,25,23,0.07), 0 2px 4px -2px rgba(28,25,23,0.05);
    --shadow-lg: 0 10px 15px -3px rgba(28,25,23,0.08), 0 4px 6px -4px rgba(28,25,23,0.04);
    --radius: 10px;
    --radius-sm: 6px;
}
html { font-size: 15px; }
body {
    font-family: var(--font-body);
    background: var(--bg);
    color: var(--fg);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* ───── Scrollbar ───── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--muted-light); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ───── Layout ───── */
.report-container {
    max-width: 1340px;
    margin: 0 auto;
    padding: 32px 24px 64px;
}

/* ───── Header ───── */
.report-header {
    margin-bottom: 36px;
}
.report-header h1 {
    font-family: var(--font-heading);
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--fg);
    line-height: 1.2;
}
.report-header .subtitle {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 6px;
}

/* ───── Summary cards ───── */
.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 32px;
}
.summary-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s;
}
.summary-card:hover { box-shadow: var(--shadow-md); }
.summary-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 4px;
}
.summary-card .big-number {
    font-family: var(--font-heading);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.15;
}
.summary-card.card-success .big-number { color: var(--emerald); }
.summary-card.card-failed .big-number { color: var(--red); }
.summary-card.card-skipped .big-number { color: var(--muted-light); }

/* ───── Section titles ───── */
.section-title {
    font-family: var(--font-heading);
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
}

/* ───── Aggregate metrics ───── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
    margin-bottom: 36px;
}
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
}
.metric-card .metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-card .metric-avg {
    font-family: var(--font-heading);
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.metric-card .metric-bar-track {
    height: 6px;
    background: var(--cream);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 8px;
}
.metric-card .metric-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}
.bar-emerald { background: var(--emerald); }
.bar-amber { background: var(--amber); }
.bar-red { background: var(--red); }
.color-emerald { color: var(--emerald); }
.color-amber { color: var(--amber); }
.color-red { color: var(--red); }
.metric-card .metric-range {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--muted-light);
}

/* ───── Aggregate stats (collapsible) ───── */
.stats-section { margin-bottom: 36px; }
.stats-toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: none;
    cursor: pointer;
    font-family: var(--font-heading);
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--fg);
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
    width: 100%;
    text-align: left;
    margin-bottom: 16px;
}
.stats-toggle .chevron {
    display: inline-block;
    transition: transform 0.2s;
    font-size: 0.9rem;
}
.stats-toggle.open .chevron { transform: rotate(90deg); }
.stats-body {
    display: none;
    overflow: hidden;
}
.stats-body.open { display: block; }
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
}
.stat-card .stat-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 6px;
}
.stat-card .stat-avg {
    font-family: var(--font-heading);
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.stat-card .stat-detail {
    font-size: 0.72rem;
    color: var(--muted-light);
    line-height: 1.7;
}
.stat-card .stat-detail span {
    display: inline-block;
    margin-right: 10px;
}

/* ───── Tag metrics ───── */
.tag-metrics-section { margin-bottom: 36px; }
.tag-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 14px;
}
.tag-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
}
.tag-card .tag-name {
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 8px;
    color: var(--fg);
}
.tag-card .tag-metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    padding: 3px 0;
    border-bottom: 1px solid var(--cream);
}
.tag-card .tag-metric-row:last-child { border-bottom: none; }
.tag-card .tag-metric-name { color: var(--muted); }
.tag-card .tag-metric-value { font-weight: 600; font-family: var(--font-mono); font-size: 0.75rem; }

/* ───── Controls bar ───── */
.controls-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    margin-bottom: 16px;
    padding: 14px 18px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}
.controls-bar label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.controls-bar select,
.controls-bar input[type="text"] {
    font-family: var(--font-body);
    font-size: 0.85rem;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--cream);
    color: var(--fg);
    outline: none;
    transition: border-color 0.15s;
}
.controls-bar select:focus,
.controls-bar input[type="text"]:focus {
    border-color: var(--blue);
}
.controls-bar input[type="text"] { width: 180px; }
.control-group {
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Range slider */
.range-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.range-group input[type="range"] {
    -webkit-appearance: none;
    width: 90px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
}
.range-group input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    background: var(--blue);
    border-radius: 50%;
    cursor: pointer;
}
.range-group .range-label {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted);
    min-width: 30px;
    text-align: center;
}

/* Tag pills */
.tag-filters {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.tag-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--cream);
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
}
.tag-pill:hover { border-color: var(--blue); color: var(--blue); }
.tag-pill.active {
    background: var(--blue);
    color: #fff;
    border-color: var(--blue);
}

/* Results count */
.results-count {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 8px;
}

/* ───── Examples table ───── */
.table-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    margin-bottom: 16px;
}
.examples-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}
.examples-table > thead > tr > th {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid var(--border);
    background: var(--cream);
    position: sticky;
    top: 0;
    z-index: 2;
}
.examples-table > tbody > tr > td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
    vertical-align: middle;
}
.examples-table tbody tr {
    cursor: pointer;
    transition: background 0.1s;
}
.examples-table tbody tr:hover { background: var(--cream); }
.examples-table tbody tr.selected { background: var(--blue-bg); }

/* Column widths */
.col-status { width: 36px; }
.col-id { width: 40%; }
.col-score { width: 30%; }
.col-tags { width: 30%; }

/* Status dot */
.status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.status-dot.ok { background: var(--emerald); }
.status-dot.fail { background: var(--red); }

/* Score bar inline */
.score-cell {
    display: flex;
    align-items: center;
    gap: 8px;
}
.score-bar-track {
    flex: 1;
    height: 5px;
    background: var(--cream);
    border-radius: 3px;
    overflow: hidden;
    min-width: 40px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.2s;
}
.score-value {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 500;
    min-width: 42px;
    text-align: right;
}

/* Tags in table */
.tag-badges { display: flex; flex-wrap: wrap; gap: 4px; }
.tag-badge {
    font-size: 0.65rem;
    padding: 1px 7px;
    border-radius: 999px;
    background: var(--cream);
    border: 1px solid var(--border);
    color: var(--muted);
    white-space: nowrap;
}

/* ───── Pagination ───── */
.pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    margin-top: 8px;
    margin-bottom: 36px;
}
.pagination button {
    font-family: var(--font-body);
    font-size: 0.8rem;
    padding: 5px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--card);
    color: var(--fg);
    cursor: pointer;
    transition: all 0.12s;
}
.pagination button:hover:not(:disabled) {
    border-color: var(--blue);
    color: var(--blue);
}
.pagination button:disabled {
    opacity: 0.4;
    cursor: default;
}
.pagination button.active {
    background: var(--blue);
    color: #fff;
    border-color: var(--blue);
}
.pagination .page-info {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0 8px;
}

/* ───── Detail panel ───── */
.detail-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-lg);
    margin: -1px 0 0;
    padding: 24px;
    animation: slideDown 0.2s ease;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}
.detail-panel .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}
.detail-panel .detail-title {
    font-family: var(--font-heading);
    font-size: 1.2rem;
    font-weight: 600;
    word-break: break-all;
}
.detail-close {
    background: none;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 4px 10px;
    cursor: pointer;
    font-size: 0.8rem;
    color: var(--muted);
    transition: all 0.12s;
}
.detail-close:hover { border-color: var(--red); color: var(--red); }

.detail-panel .detail-error {
    background: var(--red-bg);
    border-left: 3px solid var(--red);
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    margin-bottom: 16px;
    font-size: 0.85rem;
    color: var(--red);
    font-family: var(--font-mono);
    white-space: pre-wrap;
    word-break: break-all;
}

/* Detail sub-tables */
.detail-section { margin-bottom: 18px; }
.detail-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 8px;
}
.detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.detail-table th {
    text-align: left;
    padding: 6px 10px;
    background: var(--cream);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
}
.detail-table td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
}
.detail-table td.mono {
    font-family: var(--font-mono);
    font-size: 0.78rem;
}

/* Rule result badges */
.rule-pass {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--emerald-bg);
    color: var(--emerald);
    font-weight: 500;
}
.rule-fail {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--red-bg);
    color: var(--red);
    font-weight: 500;
}

/* PDF viewer area */
.pdf-viewer-section { margin-bottom: 18px; }
.pdf-url-bar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 10px;
}
.pdf-url-bar label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    white-space: nowrap;
}
.pdf-url-bar input {
    flex: 1;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--cream);
    color: var(--fg);
    outline: none;
}
.pdf-url-bar input:focus { border-color: var(--blue); }
.pdf-url-bar button {
    font-family: var(--font-body);
    font-size: 0.78rem;
    padding: 6px 14px;
    border: 1px solid var(--blue);
    border-radius: var(--radius-sm);
    background: var(--blue);
    color: #fff;
    cursor: pointer;
}
.pdf-canvas-wrap {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: #e5e5e5;
    height: 600px;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    overflow: auto;
    position: relative;
}
.pdf-canvas-wrap canvas {
    display: block;
}
.pdf-placeholder {
    color: var(--muted-light);
    font-size: 0.85rem;
    padding: 40px;
    text-align: center;
}
.pdf-nav {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 8px;
    font-size: 0.8rem;
    color: var(--muted);
}
.pdf-nav button {
    font-family: var(--font-body);
    font-size: 0.78rem;
    padding: 3px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--card);
    cursor: pointer;
}
.pdf-nav button:disabled { opacity: 0.4; cursor: default; }

/* Collapsible detail sections */
.detail-collapsible {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 14px;
    overflow: hidden;
}
.detail-collapsible-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 10px 14px;
    background: var(--cream);
    border: none;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    text-align: left;
}
.detail-collapsible-toggle:hover { background: var(--border); }
.detail-collapsible-toggle .chevron {
    display: inline-block;
    transition: transform 0.2s;
    font-size: 0.7rem;
}
.detail-collapsible-toggle.open .chevron { transform: rotate(90deg); }
.detail-collapsible-body {
    display: none;
    padding: 0;
}
.detail-collapsible-body.open { display: block; }
.detail-collapsible-body .detail-table { margin: 0; }

/* Output panels (side-by-side) */
.output-columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 18px;
}
@media (max-width: 900px) {
    .output-columns { grid-template-columns: 1fr; }
}
.output-panel {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
}
.output-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    padding: 10px 14px;
    background: var(--cream);
    border-bottom: 1px solid var(--border);
}
.output-view-toggle {
    display: inline-flex;
    gap: 2px;
    background: var(--border);
    border-radius: var(--radius-sm);
    padding: 2px;
}
.output-view-btn {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 10px;
    border: none;
    border-radius: 4px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    text-transform: none;
    letter-spacing: 0;
    transition: all 0.12s;
}
.output-view-btn.active {
    background: var(--card);
    color: var(--fg);
    box-shadow: var(--shadow-sm);
}
.output-copy-btn {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--card);
    color: var(--muted);
    cursor: pointer;
    margin-left: 6px;
    transition: all 0.12s;
}
.output-copy-btn:hover { background: var(--border); color: var(--fg); }
.output-copy-btn.copied { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
.output-panel-body {
    padding: 12px 14px;
    max-height: 500px;
    overflow: auto;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--card);
}
.output-panel-body.rendered-md {
    font-family: var(--font-body);
    white-space: normal;
}
.output-panel-body.rendered-md table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.8rem; }
.output-panel-body.rendered-md th,
.output-panel-body.rendered-md td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
.output-panel-body.rendered-md th { background: var(--cream); font-weight: 600; }
.output-panel-body.rendered-md pre { background: var(--cream); padding: 10px; border-radius: var(--radius-sm); overflow-x: auto; font-size: 0.78rem; }
.output-panel-body.rendered-md code { font-family: var(--font-mono); font-size: 0.85em; background: var(--cream); padding: 1px 4px; border-radius: 3px; }
.output-panel-body.rendered-md pre code { background: none; padding: 0; }
.output-panel-body.rendered-md h1, .output-panel-body.rendered-md h2, .output-panel-body.rendered-md h3,
.output-panel-body.rendered-md h4, .output-panel-body.rendered-md h5, .output-panel-body.rendered-md h6 {
    margin: 16px 0 8px; font-family: var(--font-heading);
}
.output-panel-body.rendered-md p { margin: 8px 0; }
.output-panel-body.rendered-md ul, .output-panel-body.rendered-md ol { margin: 8px 0; padding-left: 24px; }
.output-panel-body.rendered-md blockquote { border-left: 3px solid var(--border); margin: 8px 0; padding: 4px 12px; color: var(--muted); }
.output-panel-body.rendered-md img { max-width: 100%; }
.output-panel-body.rendered-md hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
.output-empty {
    color: var(--muted-light);
    font-style: italic;
    font-family: var(--font-body);
}

/* Placeholder sections */
.placeholder-note {
    font-size: 0.82rem;
    color: var(--muted-light);
    font-style: italic;
    padding: 12px 0;
}

/* ───── Responsive ───── */
@media (max-width: 768px) {
    .report-container { padding: 16px 12px; }
    .controls-bar { flex-direction: column; align-items: stretch; }
    .controls-bar input[type="text"] { width: 100%; }
    .col-tags { display: none; }
}
""" + TOOLTIP_CSS

_JS = r"""
(function() {
"use strict";

// ─── State ───
var state = {
    currentMetric: '',
    sortMode: 'score_desc',  // score_desc, score_asc, alpha
    searchQuery: '',
    activeTags: [],
    rangeMin: 0.0,
    rangeMax: 1.0,
    currentPage: 1,
    perPage: 50,
    filtered: [],
    expandedId: null
};

var metricKeys = Object.keys(DATA.metricNames);
if (DATA.aggMetrics.length > 0) {
    state.currentMetric = DATA.aggMetrics[0].name;
} else if (metricKeys.length > 0) {
    state.currentMetric = metricKeys[0];
}

// ─── Helpers ───
function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

""" + TOOLTIP_JS + r"""

// Toggle between raw/rendered markdown view
window.toggleOutputView = function(panelId, mode) {
    var panel = document.getElementById(panelId);
    if (!panel) return;
    var rawEl = panel.querySelector('.output-raw');
    var renderedEl = panel.querySelector('.output-rendered');
    var btns = panel.querySelectorAll('.output-view-btn');
    btns.forEach(function(b) { b.classList.toggle('active', b.getAttribute('data-mode') === mode); });
    if (mode === 'rendered') {
        rawEl.style.display = 'none';
        renderedEl.style.display = 'block';
    } else {
        rawEl.style.display = 'block';
        renderedEl.style.display = 'none';
    }
};

// Copy raw output to clipboard
window.copyOutput = function(panelId, btn) {
    var panel = document.getElementById(panelId);
    if (!panel) return;
    var rawEl = panel.querySelector('.output-raw');
    if (!rawEl) return;
    var text = rawEl.textContent || rawEl.innerText;
    navigator.clipboard.writeText(text).then(function() {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
    });
};

function scoreColor(v) {
    if (v >= 0.9) return 'emerald';
    if (v >= 0.7) return 'amber';
    return 'red';
}

function fmt(v, decimals) {
    if (v == null || isNaN(v)) return '-';
    return Number(v).toFixed(decimals !== undefined ? decimals : 4);
}

function fmtStat(v, unit) {
    if (v == null || isNaN(v)) return '-';
    var n = Number(v);
    if (Math.abs(n) >= 1000) return n.toFixed(0) + (unit || '');
    if (Math.abs(n) >= 1) return n.toFixed(1) + (unit || '');
    return n.toFixed(3) + (unit || '');
}

function debounce(fn, ms) {
    var timer;
    return function() {
        var args = arguments, ctx = this;
        clearTimeout(timer);
        timer = setTimeout(function() { fn.apply(ctx, args); }, ms);
    };
}

function getExampleScore(ex) {
    if (!state.currentMetric) return null;
    var v = ex.metrics[state.currentMetric];
    return (v !== undefined && v !== null) ? v : null;
}

// ─── Render summary cards ───
function renderSummary() {
    var el = document.getElementById('summary-cards');
    var s = DATA.summary;
    el.innerHTML =
        '<div class="summary-card"><div class="label">Total Examples</div><div class="big-number">' + s.total + '</div></div>' +
        '<div class="summary-card card-success"><div class="label">Successful</div><div class="big-number">' + s.successful + '</div></div>' +
        '<div class="summary-card card-failed"><div class="label">Failed</div><div class="big-number">' + s.failed + '</div></div>' +
        '<div class="summary-card card-skipped"><div class="label">Skipped</div><div class="big-number">' + s.skipped + '</div></div>';
}

// ─── Render aggregate metrics ───
function renderAggMetrics() {
    var el = document.getElementById('agg-metrics');
    if (!DATA.aggMetrics.length) { el.style.display = 'none'; return; }
    var html = '<h2 class="section-title">Aggregate Metrics</h2><div class="metrics-grid">';
    for (var i = 0; i < DATA.aggMetrics.length; i++) {
        var m = DATA.aggMetrics[i];
        var c = scoreColor(m.avg);
        html += '<div class="metric-card">' +
            '<div class="metric-label">' + esc(m.displayName) + tooltipIcon(m.name) + '</div>' +
            '<div class="metric-avg color-' + c + '">' + fmt(m.avg) + '</div>' +
            '<div class="metric-bar-track"><div class="metric-bar-fill bar-' + c + '" style="width:' + (m.avg * 100).toFixed(1) + '%"></div></div>' +
            '<div class="metric-range"><span>Min: ' + fmt(m.min) + '</span><span>Max: ' + fmt(m.max) + '</span></div>' +
            '</div>';
    }
    html += '</div>';
    el.innerHTML = html;
}

// ─── Render aggregate stats ───
function renderAggStats() {
    var el = document.getElementById('agg-stats');
    if (!DATA.aggStats.length) { el.style.display = 'none'; return; }
    var html = '<button class="stats-toggle" id="stats-toggle">' +
        '<span class="chevron">&#9654;</span> Operational Statistics</button>' +
        '<div class="stats-body" id="stats-body"><div class="stats-grid">';
    for (var i = 0; i < DATA.aggStats.length; i++) {
        var s = DATA.aggStats[i];
        var u = s.unit || '';
        html += '<div class="stat-card">' +
            '<div class="stat-label">' + esc(s.displayName) + (u ? ' (' + esc(u) + ')' : '') + '</div>' +
            '<div class="stat-avg">' + fmtStat(s.avg, u) + '</div>' +
            '<div class="stat-detail">' +
            '<span>Min: ' + fmtStat(s.min, u) + '</span>' +
            '<span>Max: ' + fmtStat(s.max, u) + '</span>' +
            '<span>P50: ' + fmtStat(s.p50, u) + '</span>' +
            '<span>P95: ' + fmtStat(s.p95, u) + '</span>' +
            '<span>P99: ' + fmtStat(s.p99, u) + '</span>' +
            '<span>Total: ' + fmtStat(s.total, u) + '</span>' +
            '<span>Count: ' + s.count + '</span>' +
            '</div></div>';
    }
    html += '</div></div>';
    el.innerHTML = html;

    document.getElementById('stats-toggle').addEventListener('click', function() {
        this.classList.toggle('open');
        document.getElementById('stats-body').classList.toggle('open');
    });
}

// ─── Render tag metrics ───
function renderTagMetrics() {
    var el = document.getElementById('tag-metrics');
    var tagKeys = Object.keys(DATA.tagMetrics);
    if (!tagKeys.length) { el.style.display = 'none'; return; }
    var html = '<h2 class="section-title">Metrics by Tag</h2><div class="tag-metrics-grid">';
    tagKeys.sort();
    for (var t = 0; t < tagKeys.length; t++) {
        var tag = tagKeys[t];
        var tm = DATA.tagMetrics[tag];
        var mKeys = Object.keys(tm).sort();
        html += '<div class="tag-card"><div class="tag-name">' + esc(tag) + '</div>';
        for (var mk = 0; mk < mKeys.length; mk++) {
            var mName = mKeys[mk];
            // try to find a display name
            var baseName = mName.replace(/^avg_/, '').replace(/^min_/, '').replace(/^max_/, '');
            var displayName = DATA.metricNames[baseName] || mName.replace(/_/g, ' ');
            html += '<div class="tag-metric-row"><span class="tag-metric-name">' + esc(displayName) +
                '</span><span class="tag-metric-value">' + fmt(tm[mKeys[mk]]) + '</span></div>';
        }
        html += '</div>';
    }
    html += '</div>';
    el.innerHTML = html;
}

// ─── Controls ───
function renderControls() {
    var el = document.getElementById('controls');
    var html = '';

    // Metric selector
    html += '<div class="control-group"><label>Metric</label><select id="metric-select">';
    for (var i = 0; i < metricKeys.length; i++) {
        var k = metricKeys[i];
        var sel = (k === state.currentMetric) ? ' selected' : '';
        html += '<option value="' + esc(k) + '"' + sel + '>' + esc(DATA.metricNames[k]) + '</option>';
    }
    html += '</select></div>';

    // Sort
    html += '<div class="control-group"><label>Sort</label><select id="sort-select">' +
        '<option value="score_desc"' + (state.sortMode === 'score_desc' ? ' selected' : '') + '>Score &#x2193;</option>' +
        '<option value="score_asc"' + (state.sortMode === 'score_asc' ? ' selected' : '') + '>Score &#x2191;</option>' +
        '<option value="alpha"' + (state.sortMode === 'alpha' ? ' selected' : '') + '>Name A-Z</option>' +
        '</select></div>';

    // Score range
    html += '<div class="control-group range-group"><label>Score Range</label>' +
        '<span class="range-label" id="range-min-label">' + state.rangeMin.toFixed(1) + '</span>' +
        '<input type="range" id="range-min" min="0" max="1" step="0.05" value="' + state.rangeMin + '">' +
        '<input type="range" id="range-max" min="0" max="1" step="0.05" value="' + state.rangeMax + '">' +
        '<span class="range-label" id="range-max-label">' + state.rangeMax.toFixed(1) + '</span>' +
        '</div>';

    // Search
    html += '<div class="control-group"><label>Search</label>' +
        '<input type="text" id="search-input" placeholder="Filter by test ID..." value="' + esc(state.searchQuery) + '"></div>';

    el.innerHTML = html;

    // Tag pills
    var tagEl = document.getElementById('tag-filters');
    if (DATA.tags.length) {
        var thtml = '';
        for (var t = 0; t < DATA.tags.length; t++) {
            var active = state.activeTags.indexOf(DATA.tags[t]) >= 0 ? ' active' : '';
            thtml += '<span class="tag-pill' + active + '" data-tag="' + esc(DATA.tags[t]) + '">' + esc(DATA.tags[t]) + '</span>';
        }
        tagEl.innerHTML = thtml;
        tagEl.style.display = '';
    } else {
        tagEl.style.display = 'none';
    }

    // Event listeners
    document.getElementById('metric-select').addEventListener('change', function() {
        state.currentMetric = this.value;
        state.currentPage = 1;
        applyFiltersAndRender();
    });
    document.getElementById('sort-select').addEventListener('change', function() {
        state.sortMode = this.value;
        state.currentPage = 1;
        applyFiltersAndRender();
    });
    document.getElementById('range-min').addEventListener('input', function() {
        state.rangeMin = parseFloat(this.value);
        if (state.rangeMin > state.rangeMax) { state.rangeMin = state.rangeMax; this.value = state.rangeMin; }
        document.getElementById('range-min-label').textContent = state.rangeMin.toFixed(1);
    });
    document.getElementById('range-min').addEventListener('change', function() {
        state.currentPage = 1;
        applyFiltersAndRender();
    });
    document.getElementById('range-max').addEventListener('input', function() {
        state.rangeMax = parseFloat(this.value);
        if (state.rangeMax < state.rangeMin) { state.rangeMax = state.rangeMin; this.value = state.rangeMax; }
        document.getElementById('range-max-label').textContent = state.rangeMax.toFixed(1);
    });
    document.getElementById('range-max').addEventListener('change', function() {
        state.currentPage = 1;
        applyFiltersAndRender();
    });
    document.getElementById('search-input').addEventListener('input', debounce(function() {
        state.searchQuery = this.value.toLowerCase();
        state.currentPage = 1;
        applyFiltersAndRender();
    }, 300));

    tagEl.addEventListener('click', function(e) {
        var pill = e.target.closest('.tag-pill');
        if (!pill) return;
        var tag = pill.getAttribute('data-tag');
        var idx = state.activeTags.indexOf(tag);
        if (idx >= 0) {
            state.activeTags.splice(idx, 1);
            pill.classList.remove('active');
        } else {
            state.activeTags.push(tag);
            pill.classList.add('active');
        }
        state.currentPage = 1;
        applyFiltersAndRender();
    });
}

// ─── Filter + sort ───
function applyFilters() {
    var results = [];
    for (var i = 0; i < DATA.examples.length; i++) {
        var ex = DATA.examples[i];
        // Search filter
        if (state.searchQuery && ex.id.toLowerCase().indexOf(state.searchQuery) < 0) continue;
        // Tag filter
        if (state.activeTags.length > 0) {
            var hasTag = false;
            for (var t = 0; t < state.activeTags.length; t++) {
                if (ex.tags.indexOf(state.activeTags[t]) >= 0) { hasTag = true; break; }
            }
            if (!hasTag) continue;
        }
        // Score range filter
        var score = getExampleScore(ex);
        if (score !== null) {
            if (score < state.rangeMin || score > state.rangeMax) continue;
        }
        results.push(ex);
    }

    // Sort
    results.sort(function(a, b) {
        if (state.sortMode === 'alpha') {
            return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
        }
        var sa = getExampleScore(a);
        var sb = getExampleScore(b);
        // Nulls go last
        if (sa === null && sb === null) return 0;
        if (sa === null) return 1;
        if (sb === null) return -1;
        if (state.sortMode === 'score_desc') return sb - sa;
        return sa - sb;
    });

    state.filtered = results;
}

function applyFiltersAndRender() {
    applyFilters();
    state.expandedId = null;
    renderResultsCount();
    renderTable();
    renderPagination();
}

// ─── Results count ───
function renderResultsCount() {
    var el = document.getElementById('results-count');
    el.textContent = state.filtered.length + ' of ' + DATA.examples.length + ' examples';
}

// ─── Table ───
function renderTable() {
    var tbody = document.getElementById('examples-tbody');
    var frag = document.createDocumentFragment();
    var start = (state.currentPage - 1) * state.perPage;
    var end = Math.min(start + state.perPage, state.filtered.length);

    // Clear
    tbody.innerHTML = '';

    for (var i = start; i < end; i++) {
        var ex = state.filtered[i];
        var score = getExampleScore(ex);
        var tr = document.createElement('tr');
        tr.setAttribute('data-id', ex.id);
        if (ex.id === state.expandedId) tr.className = 'selected';

        // Status
        var td0 = document.createElement('td');
        td0.innerHTML = '<span class="status-dot ' + (ex.success ? 'ok' : 'fail') + '"></span>';
        tr.appendChild(td0);

        // Test ID
        var td1 = document.createElement('td');
        td1.textContent = ex.id;
        td1.style.fontFamily = 'var(--font-mono)';
        td1.style.fontSize = '0.8rem';
        td1.style.wordBreak = 'break-all';
        tr.appendChild(td1);

        // Score
        var td2 = document.createElement('td');
        if (score !== null) {
            var c = scoreColor(score);
            td2.innerHTML = '<div class="score-cell">' +
                '<div class="score-bar-track"><div class="score-bar-fill bar-' + c + '" style="width:' + (score * 100).toFixed(1) + '%"></div></div>' +
                '<span class="score-value color-' + c + '">' + fmt(score) + '</span></div>';
        } else {
            td2.innerHTML = '<span style="color:var(--muted-light);font-size:0.8rem">-</span>';
        }
        tr.appendChild(td2);

        // Tags
        var td3 = document.createElement('td');
        td3.className = 'col-tags-cell';
        if (ex.tags.length) {
            var badges = '';
            for (var t = 0; t < ex.tags.length; t++) {
                badges += '<span class="tag-badge">' + esc(ex.tags[t]) + '</span>';
            }
            td3.innerHTML = '<div class="tag-badges">' + badges + '</div>';
        }
        tr.appendChild(td3);

        frag.appendChild(tr);

        // Detail panel if expanded
        if (ex.id === state.expandedId) {
            var detailTr = document.createElement('tr');
            var detailTd = document.createElement('td');
            detailTd.colSpan = 4;
            detailTd.style.padding = '0';
            detailTd.innerHTML = buildDetailPanel(ex);
            detailTr.appendChild(detailTd);
            frag.appendChild(detailTr);
        }
    }

    tbody.appendChild(frag);

    // Row click handler (delegated)
    tbody.onclick = function(e) {
        var tr = e.target.closest('tr[data-id]');
        if (!tr) return;
        // Don't toggle if clicking inside detail panel
        if (e.target.closest('.detail-panel')) return;
        var id = tr.getAttribute('data-id');
        if (state.expandedId === id) {
            state.expandedId = null;
        } else {
            state.expandedId = id;
        }
        renderTable();
        initPdfViewerIfNeeded();
    };
}

// ─── Detail panel ───
function buildDetailPanel(ex) {
    var html = '<div class="detail-panel">';
    html += '<div class="detail-header"><div class="detail-title">' + esc(ex.id) + '</div>' +
        '<button class="detail-close" onclick="event.stopPropagation();closeDetail()">Close</button></div>';

    // Job IDs
    if (ex.parseJobId || ex.jobId) {
        html += '<div style="font-size:0.8rem;color:var(--muted);margin-bottom:12px">';
        if (ex.parseJobId) {
            html += '<span><strong>Parse Job ID:</strong> <span style="font-family:var(--font-mono);font-size:0.78rem">' + esc(ex.parseJobId) + '</span></span>';
        }
        if (ex.jobId && ex.jobId !== ex.parseJobId) {
            if (ex.parseJobId) html += '<span style="margin:0 10px;color:var(--border)">|</span>';
            html += '<span><strong>Job ID:</strong> <span style="font-family:var(--font-mono);font-size:0.78rem">' + esc(ex.jobId) + '</span></span>';
        }
        html += '</div>';
    }

    if (ex.parseJobLogsUrl || ex.parseJobLogsLocalPath || ex.parseJobLogsHtmlPath) {
        html += '<div style="font-size:0.8rem;color:var(--muted);margin:-4px 0 12px 0">';
        html += '<strong>Parse Job Logs:</strong> ';
        var links = [];
        if (ex.parseJobLogsUrl) {
            links.push('<a href="' + esc(ex.parseJobLogsUrl) + '" target="_blank" rel="noopener noreferrer">jobLogs.json (presigned)</a>');
        }
        if (ex.parseJobLogsLocalPath) {
            links.push('<a href="' + esc(ex.parseJobLogsLocalPath) + '" target="_blank" rel="noopener noreferrer">jobLogs.json (local)</a>');
        }
        if (ex.parseJobLogsHtmlPath) {
            links.push('<a href="' + esc(ex.parseJobLogsHtmlPath) + '" target="_blank" rel="noopener noreferrer">Pretty Viewer</a>');
        }
        html += links.join(' <span style="color:var(--border);margin:0 8px;">|</span> ');
        html += '</div>';
    }

    // Error
    if (ex.error) {
        html += '<div class="detail-error">' + esc(ex.error) + '</div>';
    }

    // All metrics (collapsible, with inline sub-collapsible details per metric)
    var metricEntries = Object.keys(ex.metrics);
    if (metricEntries.length) {
        var metricSummary = metricEntries.length + ' metric' + (metricEntries.length > 1 ? 's' : '');
        html += '<div class="detail-collapsible">' +
            '<button class="detail-collapsible-toggle" onclick="event.stopPropagation();toggleCollapsible(this)">' +
            '<span class="chevron">&#9654;</span> Metrics (' + metricSummary + ')</button>' +
            '<div class="detail-collapsible-body">';
        metricEntries.sort();
        for (var i = 0; i < metricEntries.length; i++) {
            var mName = metricEntries[i];
            var mVal = ex.metrics[mName];
            var c = scoreColor(mVal);
            var ruleStr = '';
            if (ex.ruleDetails[mName]) {
                var rd = ex.ruleDetails[mName];
                ruleStr = ' <span class="mono" style="margin-left:8px;font-size:0.78rem;color:var(--muted)">' + rd.passed + '/' + rd.total + ' rules</span>';
            }
            var hasDetails = ex.metricDetails && ex.metricDetails[mName] && ex.metricDetails[mName].length > 0;
            if (hasDetails) {
                html += '<div class="detail-collapsible" style="margin:0;border:none;border-bottom:1px solid var(--border)">' +
                    '<button class="detail-collapsible-toggle" style="padding:5px 14px;font-size:0.82rem" onclick="event.stopPropagation();toggleCollapsible(this)">' +
                    '<span class="chevron">&#9654;</span> ' +
                    '<span style="display:inline-flex;align-items:center;min-width:220px">' + esc(DATA.metricNames[mName] || mName) + tooltipIcon(mName) + '</span>' +
                    '<span class="mono color-' + c + '" style="margin-left:8px">' + fmt(mVal) + '</span>' +
                    ruleStr + '</button>' +
                    '<div class="detail-collapsible-body" style="padding:2px 14px 8px 32px;background:var(--bg)">';
                var lines = ex.metricDetails[mName];
                // Check if lines use [SECTION:...] markers for sub-collapsibles
                var hasSections = false;
                for (var si = 0; si < lines.length; si++) {
                    if (lines[si].indexOf('[SECTION:') === 0) { hasSections = true; break; }
                }
                if (hasSections) {
                    var inSection = false;
                    for (var li = 0; li < lines.length; li++) {
                        var ln = lines[li];
                        var secMatch = ln.match(/^\[SECTION:(.+)\]$/);
                        if (secMatch) {
                            if (inSection) html += '</div></div></div>';
                            html += '<div class="detail-collapsible" style="margin:2px 0;border:none;border-bottom:1px solid var(--border)">' +
                                '<button class="detail-collapsible-toggle" style="padding:3px 10px;font-size:0.78rem" onclick="event.stopPropagation();toggleCollapsible(this)">' +
                                '<span class="chevron">&#9654;</span> ' + esc(secMatch[1]) + '</button>' +
                                '<div class="detail-collapsible-body" style="padding:2px 10px 6px 28px;background:var(--bg)">' +
                                '<div style="font-size:0.75rem;line-height:1.5;font-family:var(--font-mono);white-space:pre-wrap;color:var(--text)">';
                            inSection = true;
                        } else {
                            html += esc(ln) + '\n';
                        }
                    }
                    if (inSection) html += '</div></div></div>';
                } else {
                    html += '<div style="font-size:0.78rem;line-height:1.6;font-family:var(--font-mono);white-space:pre-wrap;color:var(--text)">';
                    for (var li = 0; li < lines.length; li++) {
                        html += esc(lines[li]) + '\n';
                    }
                    html += '</div>';
                }
                html += '</div></div>';
            } else {
                html += '<div style="padding:5px 14px;font-size:0.82rem;border-bottom:1px solid var(--border)">' +
                    '<span style="display:inline-block;width:18px"></span>' +
                    '<span style="display:inline-flex;align-items:center;min-width:220px">' + esc(DATA.metricNames[mName] || mName) + tooltipIcon(mName) + '</span>' +
                    '<span class="mono color-' + c + '" style="margin-left:8px">' + fmt(mVal) + '</span>' +
                    ruleStr + '</div>';
            }
        }
        html += '</div></div>';
    }

    // Rule results (collapsible)
    var ruleMetrics = Object.keys(ex.ruleResults);
    if (ruleMetrics.length) {
        var totalRules = 0;
        for (var r = 0; r < ruleMetrics.length; r++) { totalRules += ex.ruleResults[ruleMetrics[r]].length; }
        if (totalRules > 0) {
            html += '<div class="detail-collapsible">' +
                '<button class="detail-collapsible-toggle" onclick="event.stopPropagation();toggleCollapsible(this)">' +
                '<span class="chevron">&#9654;</span> Rule Results (' + totalRules + ' rules)</button>' +
                '<div class="detail-collapsible-body">';
            for (var r = 0; r < ruleMetrics.length; r++) {
                var rmName = ruleMetrics[r];
                var rules = ex.ruleResults[rmName];
                if (!rules.length) continue;
                html += '<div style="padding:8px 14px 4px;font-size:0.75rem;font-weight:600;color:var(--muted)">' + esc(DATA.metricNames[rmName] || rmName) + '</div>';
                html += '<table class="detail-table"><thead><tr><th>Type</th><th>ID</th><th>Status</th><th>Message</th></tr></thead><tbody>';
                for (var ri = 0; ri < rules.length; ri++) {
                    var rule = rules[ri];
                    var badge = rule.passed
                        ? '<span class="rule-pass">&#10003; Pass</span>'
                        : '<span class="rule-fail">&#10007; Fail</span>';
                    html += '<tr><td>' + esc(rule.type) + '</td><td class="mono">' + esc(rule.id) + '</td>' +
                        '<td>' + badge + '</td><td>' + esc(rule.message) + '</td></tr>';
                }
                html += '</tbody></table>';
            }
            html += '</div></div>';
        }
    }

    // Stats (collapsible)
    var statEntries = Object.keys(ex.stats);
    if (statEntries.length) {
        html += '<div class="detail-collapsible">' +
            '<button class="detail-collapsible-toggle" onclick="event.stopPropagation();toggleCollapsible(this)">' +
            '<span class="chevron">&#9654;</span> Operational Stats</button>' +
            '<div class="detail-collapsible-body">' +
            '<table class="detail-table"><thead><tr><th>Stat</th><th>Value</th></tr></thead><tbody>';
        statEntries.sort();
        for (var si = 0; si < statEntries.length; si++) {
            var sName = statEntries[si];
            html += '<tr><td>' + esc(sName.replace(/_/g, ' ')) + '</td><td class="mono">' + fmt(ex.stats[sName], 2) + '</td></tr>';
        }
        html += '</tbody></table></div></div>';
    }

    // PDF viewer
    var productType = ex.productType || 'parse';
    html += '<div class="detail-section pdf-viewer-section">' +
        '<div class="detail-section-title">PDF Viewer</div>' +
        '<div class="pdf-url-bar">' +
        '<label>Base URL</label>' +
        '<input type="text" id="pdf-base-url" value="' + esc(getPdfBaseUrl()) + '" placeholder="http://localhost:8080/data">' +
        '<button onclick="event.stopPropagation();savePdfBaseUrl();loadPdf(\'' + esc(ex.id) + '\',\'' + esc(productType) + '\')">Load PDF</button>' +
        '</div>' +
        '<div class="pdf-canvas-wrap" id="pdf-canvas-wrap"><div class="pdf-placeholder">Set base path and click Load PDF</div></div>' +
        '<div class="pdf-nav" id="pdf-nav" style="display:none">' +
        '<button id="pdf-prev" onclick="event.stopPropagation();pdfPrev()">Prev</button>' +
        '<span id="pdf-page-info">-</span>' +
        '<button id="pdf-next" onclick="event.stopPropagation();pdfNext()">Next</button>' +
        '<span style="margin-left:12px;border-left:1px solid var(--border);padding-left:12px"></span>' +
        '<button onclick="event.stopPropagation();pdfZoomOut()">−</button>' +
        '<span id="pdf-zoom-info">150%</span>' +
        '<button onclick="event.stopPropagation();pdfZoomIn()">+</button>' +
        '</div></div>';

    // Predicted / Expected output side-by-side
    html += '<div class="output-columns">';

    // Predicted output panel
    var predId = 'output-pred-' + ex.id.replace(/[^a-zA-Z0-9]/g, '_');
    html += '<div class="output-panel" id="' + predId + '">';
    html += '<div class="output-panel-header"><span>Predicted Output</span>';
    if (ex.predictedOutput) {
        html += '<div class="output-view-toggle">' +
            '<button class="output-view-btn active" data-mode="rendered" onclick="event.stopPropagation();toggleOutputView(\'' + predId + '\',\'rendered\')">Rendered</button>' +
            '<button class="output-view-btn" data-mode="raw" onclick="event.stopPropagation();toggleOutputView(\'' + predId + '\',\'raw\')">Raw</button>' +
            '</div>' +
            '<button class="output-copy-btn" onclick="event.stopPropagation();copyOutput(\'' + predId + '\',this)">Copy</button>';
    }
    html += '</div>';
    if (ex.predictedOutput) {
        html += '<div class="output-panel-body output-raw" style="display:none">' + esc(ex.predictedOutput) + '</div>';
        html += '<div class="output-panel-body rendered-md output-rendered">' + (ex.predictedHtml || esc(ex.predictedOutput)) + '</div>';
    } else {
        html += '<div class="output-panel-body"><span class="output-empty">No predicted output available</span></div>';
    }
    html += '</div>';

    // Expected output panel
    var expId = 'output-exp-' + ex.id.replace(/[^a-zA-Z0-9]/g, '_');
    html += '<div class="output-panel" id="' + expId + '">';
    html += '<div class="output-panel-header"><span>Expected Output</span>';
    if (ex.expectedOutput) {
        html += '<div class="output-view-toggle">' +
            '<button class="output-view-btn active" data-mode="rendered" onclick="event.stopPropagation();toggleOutputView(\'' + expId + '\',\'rendered\')">Rendered</button>' +
            '<button class="output-view-btn" data-mode="raw" onclick="event.stopPropagation();toggleOutputView(\'' + expId + '\',\'raw\')">Raw</button>' +
            '</div>' +
            '<button class="output-copy-btn" onclick="event.stopPropagation();copyOutput(\'' + expId + '\',this)">Copy</button>';
    }
    html += '</div>';
    if (ex.expectedOutput) {
        html += '<div class="output-panel-body output-raw" style="display:none">' + esc(ex.expectedOutput) + '</div>';
        html += '<div class="output-panel-body rendered-md output-rendered">' + (ex.expectedHtml || esc(ex.expectedOutput)) + '</div>';
    } else {
        html += '<div class="output-panel-body"><span class="output-empty">No expected output available</span></div>';
    }
    html += '</div>';

    html += '</div>';

    html += '</div>';
    return html;
}

// Close detail (global)
window.closeDetail = function() {
    state.expandedId = null;
    renderTable();
};

// Toggle collapsible sections
window.toggleCollapsible = function(btn) {
    btn.classList.toggle('open');
    var body = btn.nextElementSibling;
    if (body) body.classList.toggle('open');
};

// ─── PDF viewer ───
var pdfState = { doc: null, page: 1, total: 0, scale: 1.5 };

window.getPdfBaseUrl = function() {
    try { return DATA.pdfBaseUrl || localStorage.getItem('bench_pdf_base_url') || ''; } catch(e) { return DATA.pdfBaseUrl || ''; }
};
window.savePdfBaseUrl = function() {
    var input = document.getElementById('pdf-base-url');
    if (input) {
        try { localStorage.setItem('bench_pdf_base_url', input.value); } catch(e) {}
    }
};

window.loadPdf = function(testId, productType) {
    var baseUrl = document.getElementById('pdf-base-url').value.replace(/\/+$/, '');
    if (!baseUrl) return;
    savePdfBaseUrl();
    // Strip overlapping path segments between baseUrl and testId to avoid duplication
    // e.g. baseUrl="http://host/data/tables/v1" + testId="tables/v1/file" -> ".../data/tables/v1/file.pdf"
    var relPath = testId;
    var baseParts = baseUrl.replace(/^https?:\/\/[^\/]*/i, '').split('/').filter(Boolean);
    var idParts = testId.split('/');
    for (var overlap = Math.min(baseParts.length, idParts.length); overlap > 0; overlap--) {
        var baseTail = baseParts.slice(baseParts.length - overlap).join('/');
        var idHead = idParts.slice(0, overlap).join('/');
        if (baseTail === idHead) {
            relPath = idParts.slice(overlap).join('/');
            break;
        }
    }
    var url;
    if (/^https?:\/\//i.test(baseUrl)) {
        url = baseUrl + '/' + relPath.split('/').map(function(p) { return encodeURIComponent(p); }).join('/') + '.pdf';
    } else {
        url = baseUrl + '/' + relPath + '.pdf';
    }
    var wrap = document.getElementById('pdf-canvas-wrap');
    wrap.innerHTML = '<div class="pdf-placeholder">Loading PDF...</div>';
    document.getElementById('pdf-nav').style.display = 'none';

    if (typeof pdfjsLib === 'undefined') {
        // Load PDF.js from CDN
        var script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js';
        script.onload = function() {
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';
            doLoadPdf(url);
        };
        script.onerror = function() {
            wrap.innerHTML = '<div class="pdf-placeholder">Failed to load PDF.js library</div>';
        };
        document.head.appendChild(script);
    } else {
        doLoadPdf(url);
    }
};

function doLoadPdf(url) {
    var wrap = document.getElementById('pdf-canvas-wrap');
    pdfjsLib.getDocument(url).promise.then(function(doc) {
        pdfState.doc = doc;
        pdfState.total = doc.numPages;
        pdfState.page = 1;
        renderPdfPage();
        document.getElementById('pdf-nav').style.display = 'flex';
    }).catch(function(err) {
        wrap.innerHTML = '<div class="pdf-placeholder">Failed to load PDF: ' + esc(String(err)) + '</div>';
    });
}

function renderPdfPage() {
    if (!pdfState.doc) return;
    pdfState.doc.getPage(pdfState.page).then(function(page) {
        var viewport = page.getViewport({ scale: pdfState.scale });
        var wrap = document.getElementById('pdf-canvas-wrap');
        wrap.innerHTML = '';
        var canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        wrap.appendChild(canvas);
        page.render({ canvasContext: canvas.getContext('2d'), viewport: viewport });
        document.getElementById('pdf-page-info').textContent = pdfState.page + ' / ' + pdfState.total;
        document.getElementById('pdf-zoom-info').textContent = Math.round(pdfState.scale * 100) + '%';
        document.getElementById('pdf-prev').disabled = pdfState.page <= 1;
        document.getElementById('pdf-next').disabled = pdfState.page >= pdfState.total;
    });
}

window.pdfPrev = function() {
    if (pdfState.page > 1) { pdfState.page--; renderPdfPage(); }
};
window.pdfNext = function() {
    if (pdfState.page < pdfState.total) { pdfState.page++; renderPdfPage(); }
};
window.pdfZoomIn = function() {
    pdfState.scale = Math.min(pdfState.scale + 0.25, 5.0);
    renderPdfPage();
};
window.pdfZoomOut = function() {
    pdfState.scale = Math.max(pdfState.scale - 0.25, 0.5);
    renderPdfPage();
};

function initPdfViewerIfNeeded() {
    // Restore saved base URL into input if present
    var input = document.getElementById('pdf-base-url');
    if (input) {
        var saved = getPdfBaseUrl();
        if (saved && !input.value) input.value = saved;
        // Auto-load if we have a base URL and a test is expanded
        if (input.value && state.expandedId) {
            var ex = null;
            for (var i = 0; i < DATA.examples.length; i++) {
                if (DATA.examples[i].id === state.expandedId) { ex = DATA.examples[i]; break; }
            }
            if (ex) {
                loadPdf(ex.id, ex.productType || 'parse');
            }
        }
    }
}

// ─── Pagination ───
function totalPages() {
    return Math.max(1, Math.ceil(state.filtered.length / state.perPage));
}

function renderPagination() {
    var el = document.getElementById('pagination');
    var tp = totalPages();
    if (tp <= 1) { el.innerHTML = ''; return; }

    var html = '';
    html += '<button ' + (state.currentPage <= 1 ? 'disabled' : '') + ' data-page="' + (state.currentPage - 1) + '">&laquo; Prev</button>';

    // Page numbers (show max 7 around current)
    var startP = Math.max(1, state.currentPage - 3);
    var endP = Math.min(tp, startP + 6);
    startP = Math.max(1, endP - 6);

    if (startP > 1) {
        html += '<button data-page="1">1</button>';
        if (startP > 2) html += '<span class="page-info">...</span>';
    }
    for (var p = startP; p <= endP; p++) {
        html += '<button data-page="' + p + '"' + (p === state.currentPage ? ' class="active"' : '') + '>' + p + '</button>';
    }
    if (endP < tp) {
        if (endP < tp - 1) html += '<span class="page-info">...</span>';
        html += '<button data-page="' + tp + '">' + tp + '</button>';
    }

    html += '<button ' + (state.currentPage >= tp ? 'disabled' : '') + ' data-page="' + (state.currentPage + 1) + '">Next &raquo;</button>';
    html += '<span class="page-info">Page ' + state.currentPage + ' of ' + tp + '</span>';

    el.innerHTML = html;

    el.onclick = function(e) {
        var btn = e.target.closest('button[data-page]');
        if (!btn || btn.disabled) return;
        state.currentPage = parseInt(btn.getAttribute('data-page'));
        state.expandedId = null;
        renderTable();
        renderPagination();
        // Scroll to table
        document.getElementById('examples-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
}

// ─── Init ───
function init() {
    renderSummary();
    renderAggMetrics();
    renderAggStats();
    renderTagMetrics();
    renderControls();
    applyFilters();
    renderResultsCount();
    renderTable();
    renderPagination();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
"""

_HTML_BODY = """\
</style>
</head>
<body>
<div class="report-container">
    <div class="report-header">
        <h1>Evaluation Report</h1>
        <div class="subtitle" id="report-subtitle"></div>
    </div>
    <div class="summary-row" id="summary-cards"></div>
    <div id="agg-metrics"></div>
    <div class="stats-section" id="agg-stats"></div>
    <div class="tag-metrics-section" id="tag-metrics"></div>
    <div id="examples-section">
        <h2 class="section-title">Examples</h2>
        <div class="controls-bar" id="controls"></div>
        <div class="tag-filters" id="tag-filters"></div>
        <div class="results-count" id="results-count"></div>
        <div class="table-wrap">
            <table class="examples-table">
                <thead>
                    <tr>
                        <th class="col-status"></th>
                        <th class="col-id">Test ID</th>
                        <th class="col-score">Score</th>
                        <th class="col-tags">Tags</th>
                    </tr>
                </thead>
                <tbody id="examples-tbody"></tbody>
            </table>
        </div>
        <div class="pagination" id="pagination"></div>
    </div>
</div>
"""


def generate_detailed_html_report(
    summary: EvaluationSummary,
    report_dir: Path,
    output_dir: Path | None = None,
    test_cases_dir: Path | None = None,
    pdf_base_url: str | None = None,
    pipeline_name: str | None = None,
    group: str | None = None,
) -> Path:
    """Export evaluation summary to an interactive HTML report.

    Args:
        summary: Evaluation summary data.
        report_dir: Directory to write the HTML report.
        output_dir: Directory containing inference result files (for predicted output).
        test_cases_dir: Directory containing test case files (for expected output).
        pdf_base_url: Base URL for PDF files. If not provided but test_cases_dir is set,
            falls back to the local filesystem path.
        pipeline_name: Name of the pipeline (e.g., 'llamaparse_agentic').
        group: Evaluation category/group (e.g., 'text_content').
    """
    html_path = report_dir / "_evaluation_report_detailed.html"

    # Resolve PDF base URL: explicit > relative path from report to PDF directory
    resolved_pdf_base_url = ""
    if pdf_base_url:
        resolved_pdf_base_url = pdf_base_url.rstrip("/")
    elif test_cases_dir is not None and test_cases_dir.exists():
        import os

        # JSONL datasets store PDFs under a pdfs/ subdirectory, while sidecar
        # datasets store them directly alongside test.json files. Use the pdfs/
        # subdirectory if it exists so that {baseUrl}/{testId}.pdf resolves correctly.
        pdf_root = test_cases_dir.resolve()
        if (pdf_root / "pdfs").is_dir():
            pdf_root = pdf_root / "pdfs"
        resolved_pdf_base_url = os.path.relpath(pdf_root, report_dir.resolve())

    # Load pipeline metadata if available
    metadata: dict[str, Any] = {}
    if output_dir:
        # Try pipeline output root (one level up from group report dir)
        for candidate in [output_dir / "_metadata.json", output_dir.parent / "_metadata.json"]:
            if candidate.exists():
                try:
                    metadata = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    pass
                break

    # Extract pipeline info
    pipeline_info = metadata.get("pipeline", {})
    resolved_pipeline_name = pipeline_name or pipeline_info.get("pipeline_name", "")
    provider_name = pipeline_info.get("provider_name", "")
    product_type = pipeline_info.get("product_type", "")
    pipeline_config = pipeline_info.get("config", {})

    data_blob = _build_data_blob(
        summary,
        output_dir=output_dir,
        test_cases_dir=test_cases_dir,
        pdf_base_url=resolved_pdf_base_url,
    )

    # Add run info to data blob
    data_blob["runInfo"] = {
        "pipelineName": resolved_pipeline_name,
        "providerName": provider_name,
        "productType": product_type,
        "category": group or "",
        "config": pipeline_config,
    }

    # Serialize and escape for safe embedding inside <script>
    data_json = json.dumps(data_blob, default=str, ensure_ascii=False)
    # Prevent premature script close
    data_json = data_json.replace("</script>", "<\\/script>")
    # Prevent HTML comment issues
    data_json = data_json.replace("<!--", "<\\!--")

    completed_str = ""
    if summary.completed_at is not None:
        completed_str = summary.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build full HTML by concatenation (no f-string over JS/CSS to avoid brace issues)
    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append(_CSS)
    parts.append(_HTML_BODY)

    # Title and subtitle script — show pipeline, provider, category
    title_parts = []
    if resolved_pipeline_name:
        title_parts.append(resolved_pipeline_name.replace("_", " ").title())
    if group:
        title_parts.append(group.replace("_", " ").title())
    title_text = " — ".join(title_parts) if title_parts else "Evaluation Report"

    subtitle_parts = []
    if provider_name:
        subtitle_parts.append("Provider: " + provider_name)
    if product_type:
        subtitle_parts.append("Product: " + product_type)
    if completed_str:
        subtitle_parts.append("Generated: " + completed_str)
    subtitle_text = "  |  ".join(subtitle_parts) if subtitle_parts else ("Generated: " + completed_str if completed_str else "")

    parts.append('<script>')
    parts.append('document.querySelector(".report-header h1").textContent = ' + json.dumps(title_text) + ';')
    parts.append('document.getElementById("report-subtitle").textContent = ' + json.dumps(subtitle_text) + ';')
    parts.append('</script>\n')

    # Data blob
    parts.append("<script>\nconst DATA = ")
    parts.append(data_json)
    parts.append(";\n</script>\n")

    # Application JS
    parts.append("<script>\n")
    parts.append(_JS)
    parts.append("\n</script>\n")

    parts.append("</body>\n</html>\n")

    html_path.write_text("".join(parts), encoding="utf-8")
    return html_path
