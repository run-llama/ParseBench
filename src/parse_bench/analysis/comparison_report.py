"""HTML report generator for pipeline comparisons.

Generates a rich, interactive comparison report matching the dashboard's
warm editorial design system (Newsreader / Plus Jakarta Sans / JetBrains Mono).
"""

import base64
import html
import json
from pathlib import Path
from typing import Any

from parse_bench.analysis.metric_definitions import (
    TOOLTIP_CSS,
    TOOLTIP_JS,
    display_name_dict,
    tooltip_dict,
)


def _get_file_data_url(file_path: Path) -> str | None:
    """Convert a file to a data URL for embedding in HTML."""
    if not file_path.exists():
        return None

    try:
        # Determine MIME type
        suffix = file_path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }
        mime_type = mime_types.get(suffix, "application/octet-stream")

        # Read file and encode
        with open(file_path, "rb") as f:
            file_data = f.read()
            encoded = base64.b64encode(file_data).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


METRIC_DISPLAY_NAMES: dict[str, str] = display_name_dict()


def _format_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _embed_input_files(
    matched_results: list[dict[str, Any]],
    test_cases_dir: Path | None,
    original_base_path: str,
) -> None:
    """Embed image input files as base64 data URLs and compute relative paths (in-place)."""
    import os

    # Use original_base_path as the base for relative path computation
    # (may differ from test_cases_dir which requires local existence)
    base_for_rel = original_base_path or (str(test_cases_dir) if test_cases_dir else "")

    for result in matched_results:
        input_file = result.get("input_file")
        if not input_file:
            continue
        fp = Path(input_file)
        if not fp.exists() and test_cases_dir:
            rel = fp.name
            candidate = test_cases_dir / rel
            if candidate.exists():
                fp = candidate

        # Compute relative path for PDF.js URL resolution
        # Try file-system-based relpath first, fall back to string manipulation
        if test_cases_dir and fp.exists():
            try:
                result["input_file_rel"] = os.path.relpath(str(fp.resolve()), str(test_cases_dir.resolve()))
            except ValueError:
                result["input_file_rel"] = str(fp)
        elif base_for_rel and input_file.startswith(base_for_rel):
            # String-based relative path (for CI paths that don't exist locally)
            rel_path = input_file[len(base_for_rel) :]
            if rel_path.startswith("/"):
                rel_path = rel_path[1:]
            result["input_file_rel"] = rel_path

        # Embed images as base64 (not PDFs — they're too large, use PDF.js instead)
        if fp.exists() and fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"):
            data_url = _get_file_data_url(fp)
            if data_url:
                result["input_data_url"] = data_url


def generate_comparison_html(comparison_data: dict[str, Any], output_path: Path | None = None) -> str | Path:
    """
    Generate an interactive HTML report for pipeline comparison.

    :param comparison_data: Comparison data from compare_pipelines()
    :param output_path: Optional path to save the HTML report. If None, returns HTML string.
    :return: HTML string if output_path is None, otherwise Path to the generated file
    """
    matched_results = comparison_data["matched_results"]
    stats = comparison_data["stats"]
    product_type = comparison_data.get("product_type", "extract")
    comparison_metric = comparison_data.get("comparison_metric", "accuracy")
    original_base_path = comparison_data.get("original_base_path", "")
    pdf_base_url = comparison_data.get("pdf_base_url", "")

    metric_display_name = METRIC_DISPLAY_NAMES.get(comparison_metric, comparison_metric)

    # Embed input images as data URLs and compute relative paths
    test_cases_dir_str = comparison_data.get("original_base_path", "")
    test_cases_dir = Path(test_cases_dir_str) if test_cases_dir_str else None
    _embed_input_files(matched_results, test_cases_dir, original_base_path)

    pipeline_a_name = html.escape(stats["pipeline_a_name"])
    pipeline_b_name = html.escape(stats["pipeline_b_name"])

    # Sort matched results by test_id
    matched_results_sorted = sorted(matched_results, key=lambda r: r["test_id"])

    html_content = _build_html(
        matched_results_sorted,
        stats,
        product_type,
        comparison_metric,
        metric_display_name,
        pipeline_a_name,
        pipeline_b_name,
        original_base_path,
        pdf_base_url,
    )

    if output_path is None:
        return html_content

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path


def _build_html(
    matched_results: list[dict[str, Any]],
    stats: dict[str, Any],
    product_type: str,
    comparison_metric: str,
    metric_display_name: str,
    pipeline_a_name: str,
    pipeline_b_name: str,
    original_base_path: str,
    pdf_base_url: str = "",
) -> str:
    """Build the full HTML document."""

    # Pre-compute result rows HTML
    rows_html = []
    for i, result in enumerate(matched_results):
        rows_html.append(_build_result_row(result, stats, i))

    title = f"Pipeline Comparison: {pipeline_a_name} vs {pipeline_b_name}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    {_css()}
</head>
<body>
    <header class="page-header">
        <div class="header-inner">
            <h1>Pipeline Comparison</h1>
            <p class="subtitle">{pipeline_a_name} <span class="vs">vs</span> {pipeline_b_name}</p>
            <div class="metric-selector">
                <label class="metric-selector-label" for="metricSelect">Primary metric:</label>
                <select id="metricSelect" onchange="switchMetric(this.value)"></select>
            </div>
        </div>
    </header>

    <main class="container">
        {_build_stats_section(stats, pipeline_a_name, pipeline_b_name)}

        <div class="path-config">
            <div class="path-config-header">
                <span class="path-config-title">Data Path Configuration</span>
                <button class="path-config-toggle" onclick="togglePathConfig()">Configure</button>
            </div>
            <div class="path-config-body" id="pathConfigBody">
                <p class="hint">If you received this report from someone else, set your local path to the test data folder:</p>
                <input type="text" id="dataBasePath" placeholder="e.g., /Users/yourname/data/financial_tables" onchange="updateBasePath()" />
                <p class="current-path">Current: <span id="currentBasePath">(using original paths)</span></p>
            </div>
        </div>

        {_build_filter_bar(stats, pipeline_a_name, pipeline_b_name)}

        <div class="results-list" id="resultsList">
            <div class="results-table-header">
                <div class="col-id">Test ID</div>
                <div class="col-metric">{pipeline_a_name}</div>
                <div class="col-metric">{pipeline_b_name}</div>
                <div class="col-delta">Delta</div>
                <div class="col-category">Category</div>
            </div>
            {"".join(rows_html)}
        </div>
    </main>

    <script>
        const comparisonData = {json.dumps(matched_results)};
        const pipelineAName = {json.dumps(stats["pipeline_a_name"])};
        const pipelineBName = {json.dumps(stats["pipeline_b_name"])};
        const productType = {json.dumps(product_type)};
        let comparisonMetric = {json.dumps(comparison_metric)};
        let metricDisplayName = {json.dumps(METRIC_DISPLAY_NAMES.get(comparison_metric, comparison_metric))};
        const originalBasePath = {json.dumps(original_base_path)};
        const pdfBaseUrl = {json.dumps(pdf_base_url)};
        const metricTooltips = {json.dumps(tooltip_dict())};
    </script>
    {_javascript()}
</body>
</html>"""


def _build_result_row(result: dict[str, Any], stats: dict[str, Any], index: int) -> str:
    """Build a single result row with expandable detail area."""
    test_id = html.escape(result["test_id"])
    metric_a = result["pipeline_a"].get("metric_value")
    metric_b = result["pipeline_b"].get("metric_value")
    category = result["category"]

    def fmt(val: float | None) -> str:
        if val is None:
            return '<span class="na">N/A</span>'
        pct = val * 100
        css = _metric_color_class(val)
        return f'<span class="metric-val {css}">{pct:.1f}%</span>'

    def delta_str(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return '<span class="na">&mdash;</span>'
        d = (a - b) * 100
        sign = "+" if d > 0 else ""
        css = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "delta-zero")
        return f'<span class="{css}">{sign}{d:.1f}pp</span>'

    category_labels = {
        "a_better": f"{stats['pipeline_a_name']} Better",
        "b_better": f"{stats['pipeline_b_name']} Better",
        "both_bad": "Both Bad",
        "tie": "Tie",
    }
    cat_label = html.escape(category_labels.get(category, category))

    return f"""
            <div class="result-row" data-category="{category}" data-index="{index}">
                <div class="row-summary" onclick="toggleRow({index})">
                    <div class="col-id"><span class="expand-icon" id="icon-{index}">&#9654;</span> {test_id}</div>
                    <div class="col-metric">{fmt(metric_a)}</div>
                    <div class="col-metric">{fmt(metric_b)}</div>
                    <div class="col-delta">{delta_str(metric_a, metric_b)}</div>
                    <div class="col-category"><span class="badge badge-{category.replace("_", "-")}">{cat_label}</span></div>
                </div>
                <div class="row-detail" id="detail-{index}"></div>
            </div>"""


def _build_stats_section(stats: dict[str, Any], a_name: str, b_name: str) -> str:
    return f"""
        <section class="stats-grid">
            <div class="stat-card" data-filter="all" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["total_matched"]}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-card stat-a-better" data-filter="a_better" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["a_better"]}</div>
                <div class="stat-label">{a_name} Better</div>
            </div>
            <div class="stat-card stat-b-better" data-filter="b_better" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["b_better"]}</div>
                <div class="stat-label">{b_name} Better</div>
            </div>
            <div class="stat-card stat-tie" data-filter="tie" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["tie"]}</div>
                <div class="stat-label">Tie</div>
            </div>
            <div class="stat-card stat-bad" data-filter="both_bad" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["both_bad"]}</div>
                <div class="stat-label">Both Bad</div>
            </div>
        </section>"""


def _build_filter_bar(stats: dict[str, Any], a_name: str, b_name: str) -> str:
    return f"""
        <div class="filter-bar">
            <button class="filter-btn active" data-filter="all">All ({stats["total_matched"]})</button>
            <button class="filter-btn" data-filter="a_better">{a_name} Better ({stats["a_better"]})</button>
            <button class="filter-btn" data-filter="b_better">{b_name} Better ({stats["b_better"]})</button>
            <button class="filter-btn" data-filter="tie">Tie ({stats["tie"]})</button>
            <button class="filter-btn" data-filter="both_bad">Both Bad ({stats["both_bad"]})</button>
        </div>"""


def _metric_color_class(val: float | None) -> str:
    if val is None:
        return "metric-na"
    if val >= 0.9:
        return "metric-high"
    if val >= 0.7:
        return "metric-mid"
    if val >= 0.5:
        return "metric-low"
    return "metric-bad"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def _css() -> str:
    return """<style>
    :root {
        --bg: #f8f7f4;
        --fg: #1c1917;
        --card: #ffffff;
        --border: #e7e5e4;
        --border-light: #f0efed;
        --muted: #78716c;
        --muted-light: #a8a29e;
        --cream: #faf9f6;

        --emerald: #059669;
        --amber: #d97706;
        --yellow: #ca8a04;
        --red: #dc2626;

        --font-heading: 'Newsreader', Georgia, serif;
        --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    }
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: var(--font-body);
        background: var(--bg);
        color: var(--fg);
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
    }

    /* Header */
    .page-header {
        background: var(--cream);
        border-bottom: 1px solid var(--border);
        padding: 2.5rem 2rem 2rem;
    }
    .header-inner {
        max-width: 1400px;
        margin: 0 auto;
    }
    .page-header h1 {
        font-family: var(--font-heading);
        font-size: 2.25rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--fg);
    }
    .page-header .subtitle {
        font-size: 1.1rem;
        color: var(--muted);
        margin-top: 0.25rem;
    }
    .page-header .vs {
        color: var(--muted-light);
        font-style: italic;
        font-family: var(--font-heading);
    }
    .page-header .metric-label {
        font-size: 0.85rem;
        color: var(--muted-light);
        margin-top: 0.35rem;
        font-family: var(--font-mono);
    }
    .metric-selector {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.35rem;
    }
    .metric-selector-label {
        font-size: 0.85rem;
        color: var(--muted-light);
        font-family: var(--font-mono);
    }
    .metric-selector select {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        padding: 0.25rem 0.5rem;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--card);
        color: var(--fg);
        cursor: pointer;
    }
    .metric-selector select:hover {
        border-color: var(--muted);
    }

    /* Container */
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1.5rem 2rem 3rem;
    }

    /* Stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem 1rem;
        text-align: center;
    }
    .stat-value {
        font-family: var(--font-mono);
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--fg);
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.25rem;
    }
    .stat-card { cursor: pointer; transition: border-color 0.15s, box-shadow 0.15s; }
    .stat-card:hover { border-color: var(--muted-light); }
    .stat-card.active { border-color: var(--fg); box-shadow: 0 0 0 1px var(--fg); }
    .stat-a-better .stat-value { color: var(--emerald); }
    .stat-b-better .stat-value { color: #2563eb; }
    .stat-tie .stat-value { color: var(--muted); }
    .stat-bad .stat-value { color: var(--red); }

    /* Path config */
    .path-config {
        background: var(--cream);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.85rem;
    }
    .path-config-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .path-config-title { color: var(--muted); font-weight: 500; }
    .path-config-toggle {
        background: none;
        border: none;
        color: var(--muted);
        cursor: pointer;
        font-size: 0.8rem;
        text-decoration: underline;
        font-family: var(--font-body);
    }
    .path-config-body { display: none; margin-top: 0.75rem; }
    .path-config-body.expanded { display: block; }
    .path-config input {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid var(--border);
        border-radius: 6px;
        font-family: var(--font-mono);
        font-size: 0.8rem;
        margin-top: 0.5rem;
        background: var(--card);
    }
    .path-config .hint { color: var(--muted); font-size: 0.8rem; margin-top: 0.5rem; }
    .path-config .current-path { color: var(--muted-light); font-size: 0.75rem; font-family: var(--font-mono); margin-top: 0.25rem; }

    /* Filters */
    .filter-bar {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
        flex-wrap: wrap;
    }
    .filter-btn {
        padding: 0.5rem 1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        font-family: var(--font-body);
        background: var(--card);
        color: var(--muted);
        transition: all 0.15s;
    }
    .filter-btn:hover {
        border-color: var(--muted);
        color: var(--fg);
    }
    .filter-btn.active {
        background: var(--fg);
        color: var(--card);
        border-color: var(--fg);
    }

    /* Results table */
    .results-list {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
    .results-table-header {
        display: grid;
        grid-template-columns: 2fr 0.8fr 0.8fr 0.8fr 1.2fr;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background: var(--cream);
        border-bottom: 1px solid var(--border);
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .col-metric, .col-delta, .col-category { text-align: center; }

    .result-row {
        border-bottom: 1px solid var(--border-light);
    }
    .result-row:last-child { border-bottom: none; }
    .row-summary {
        display: grid;
        grid-template-columns: 2fr 0.8fr 0.8fr 0.8fr 1.2fr;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        cursor: pointer;
        transition: background 0.1s;
        align-items: center;
    }
    .row-summary:hover { background: var(--cream); }
    .col-id {
        font-weight: 500;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .expand-icon {
        font-size: 0.6rem;
        color: var(--muted-light);
        transition: transform 0.15s;
        display: inline-block;
        flex-shrink: 0;
    }
    .expand-icon.expanded { transform: rotate(90deg); }

    /* Metric values */
    .metric-val {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-high { color: var(--emerald); }
    .metric-mid { color: var(--amber); }
    .metric-low { color: var(--yellow); }
    .metric-bad { color: var(--red); }
    .metric-na { color: var(--muted-light); }
    .na { color: var(--muted-light); font-size: 0.8rem; }

    .delta-pos { color: var(--emerald); font-family: var(--font-mono); font-size: 0.8rem; font-weight: 500; }
    .delta-neg { color: var(--red); font-family: var(--font-mono); font-size: 0.8rem; font-weight: 500; }
    .delta-zero { color: var(--muted); font-family: var(--font-mono); font-size: 0.8rem; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-a-better { background: #ecfdf5; color: var(--emerald); }
    .badge-b-better { background: #eff6ff; color: #2563eb; }
    .badge-tie { background: #f5f5f4; color: var(--muted); }
    .badge-both-bad { background: #fef2f2; color: var(--red); }

    /* Detail pane */
    .row-detail {
        display: none;
        padding: 0 1rem 1rem;
        background: var(--cream);
        border-top: 1px solid var(--border-light);
    }
    .row-detail.open { display: block; }

    /* Two-column detail layout */
    .detail-two-col {
        display: grid;
        grid-template-columns: 35% 1fr;
        gap: 1rem;
        margin-top: 0.75rem;
    }
    .detail-left {
        position: sticky;
        top: 0;
        align-self: start;
    }
    .detail-right {
        min-width: 0;
    }

    /* Metric pills */
    .metric-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.75rem;
    }
    .pill {
        display: flex;
        flex-direction: column;
        padding: 0.35rem 0.6rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--card);
        font-size: 0.75rem;
        min-width: 0;
    }
    .pill-primary {
        border-color: var(--fg);
        background: var(--fg);
    }
    .pill-primary .pill-label {
        color: var(--muted-light);
    }
    .pill-primary .pill-values {
        color: #fff;
    }
    .pill-label {
        font-size: 0.65rem;
        color: var(--muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 1px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .pill-values {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .pill-delta {
        font-size: 0.7rem;
        margin-left: 4px;
    }

    /* Collapsible sections */
    .collapsible-section {
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-top: 0.75rem;
        background: var(--card);
        overflow: hidden;
    }
    .collapsible-section > summary {
        padding: 0.6rem 0.75rem;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--muted);
        list-style: none;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        user-select: none;
    }
    .collapsible-section > summary::-webkit-details-marker { display: none; }
    .collapsible-section > summary::before {
        content: '\\25B6';
        font-size: 0.6rem;
        transition: transform 0.15s;
        color: var(--muted-light);
    }
    .collapsible-section[open] > summary::before {
        transform: rotate(90deg);
    }
    .collapsible-section > summary:hover {
        color: var(--fg);
    }
    .collapsible-section > :not(summary) {
        padding: 0 0.75rem 0.75rem;
    }

    @media (max-width: 1100px) {
        .detail-two-col {
            grid-template-columns: 1fr;
        }
        .detail-left {
            position: static;
        }
    }

    /* Tabs */
    .tab-bar {
        display: flex;
        gap: 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
        margin-top: 0.75rem;
    }
    .tab-btn {
        padding: 0.6rem 1.25rem;
        border: none;
        background: none;
        cursor: pointer;
        font-family: var(--font-body);
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--muted);
        border-bottom: 2px solid transparent;
        transition: all 0.15s;
    }
    .tab-btn:hover { color: var(--fg); }
    .tab-btn.active {
        color: var(--fg);
        border-bottom-color: var(--fg);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* Metrics comparison table */
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .metrics-table th {
        background: var(--cream);
        padding: 0.6rem 0.75rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid var(--border);
    }
    .metrics-table td {
        padding: 0.6rem 0.75rem;
        border-bottom: 1px solid var(--border-light);
    }
    .metrics-table tr:last-child td { border-bottom: none; }
    .metrics-table .metric-name { font-weight: 500; display: flex; align-items: center; overflow: visible; }
    .metrics-table .val-cell {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        text-align: center;
    }

    /* Rule diff table */
    .rule-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    .rule-table th {
        background: var(--cream);
        padding: 0.5rem 0.6rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid var(--border);
    }
    .rule-table td {
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid var(--border-light);
        vertical-align: top;
    }
    .rule-table tr:last-child td { border-bottom: none; }
    .rule-pass { color: var(--emerald); font-weight: 600; }
    .rule-fail { color: var(--red); font-weight: 600; }
    .rule-diff-row { background: #fffbeb; }
    .rule-same-pass { background: #f0fdf4; }
    .rule-same-fail { background: #fef2f2; }
    .rule-explanation {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 0.25rem;
        cursor: pointer;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s;
    }
    .rule-explanation.open { max-height: 200px; overflow-y: auto; }
    .rule-expander {
        font-size: 0.7rem;
        color: var(--muted-light);
        cursor: pointer;
        user-select: none;
    }

    /* Output comparison */
    .output-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    .output-panel {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--card);
    }
    .output-panel-header {
        padding: 0.5rem 0.75rem;
        background: var(--cream);
        border-bottom: 1px solid var(--border);
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--muted);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .output-panel-body {
        padding: 0.75rem;
        max-height: 500px;
        overflow-y: auto;
        font-size: 0.85rem;
    }
    .output-panel-body.json-view {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .output-panel-body.md-view h1 { font-size: 1.3rem; margin: 0.75rem 0 0.4rem; }
    .output-panel-body.md-view h2 { font-size: 1.1rem; margin: 0.6rem 0 0.3rem; }
    .output-panel-body.md-view h3 { font-size: 1rem; margin: 0.5rem 0 0.25rem; }
    .output-panel-body.md-view p { margin-bottom: 0.4rem; }
    .output-panel-body.md-view table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
    .output-panel-body.md-view th, .output-panel-body.md-view td {
        border: 1px solid var(--border);
        padding: 4px 8px;
        text-align: left;
    }
    .output-panel-body.md-view th { background: var(--cream); font-weight: 600; }
    .output-panel-body.md-view code {
        background: #f5f5f4;
        padding: 1px 4px;
        border-radius: 3px;
        font-family: var(--font-mono);
        font-size: 0.85em;
    }
    .output-panel-body.md-view pre {
        background: #f5f5f4;
        padding: 0.75rem;
        border-radius: 6px;
        overflow-x: auto;
    }
    .output-panel-body.md-view pre code { background: none; padding: 0; }

    .view-toggle {
        display: flex;
        gap: 2px;
    }
    .view-toggle button {
        padding: 2px 8px;
        border: 1px solid var(--border);
        background: var(--card);
        cursor: pointer;
        font-family: var(--font-body);
        font-size: 0.7rem;
        color: var(--muted);
    }
    .view-toggle button:first-child { border-radius: 4px 0 0 4px; }
    .view-toggle button:last-child { border-radius: 0 4px 4px 0; }
    .view-toggle button.active { background: var(--fg); color: var(--card); border-color: var(--fg); }

    /* Diff view */
    .diff-container {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--card);
        margin-top: 1rem;
    }
    .diff-header {
        padding: 0.5rem 0.75rem;
        background: var(--cream);
        border-bottom: 1px solid var(--border);
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--muted);
    }
    .diff-body {
        max-height: 500px;
        overflow-y: auto;
        font-family: var(--font-mono);
        font-size: 0.8rem;
        line-height: 1.5;
    }
    .diff-line { padding: 1px 12px; white-space: pre-wrap; word-break: break-all; }
    .diff-add { background: #ecfdf5; color: #065f46; }
    .diff-del { background: #fef2f2; color: #991b1b; }
    .diff-ctx { color: var(--muted); }
    .diff-hunk { background: #f5f3ff; color: #6d28d9; font-weight: 500; padding: 4px 12px; }

    /* Input preview */
    .input-preview {
        margin-top: 0.5rem;
    }
    .input-preview img {
        max-width: 100%;
        max-height: 600px;
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    .input-preview .file-path {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--muted);
        margin-bottom: 0.5rem;
        word-break: break-all;
    }
    .input-preview iframe {
        width: 100%;
        height: 600px;
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* Layout detection styles */
    .layoutdet-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
    }
    .layoutdet-panel {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--card);
    }
    .layoutdet-panel-header {
        padding: 0.5rem 0.75rem;
        background: var(--cream);
        border-bottom: 1px solid var(--border);
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--muted);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .layoutdet-panel-body {
        position: relative;
        display: inline-block;
        width: 100%;
    }
    .layoutdet-panel-body img {
        max-width: 100%;
        height: auto;
        display: block;
    }
    .layoutdet-canvas {
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
    }
    .bbox-btn {
        padding: 2px 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: var(--card);
        cursor: pointer;
        font-family: var(--font-body);
        font-size: 0.7rem;
        color: var(--muted);
    }
    .bbox-btn.active {
        background: var(--fg);
        color: var(--card);
        border-color: var(--fg);
    }
    .comparison-legend {
        display: flex;
        gap: 16px;
        margin-top: 0.75rem;
        padding: 0.5rem 0.75rem;
        background: var(--cream);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 0.75rem;
        color: var(--muted);
        flex-wrap: wrap;
    }
    .legend-item { display: flex; align-items: center; gap: 4px; }
    .legend-swatch {
        width: 16px;
        height: 10px;
        border-radius: 2px;
    }

    /* Responsive */
    @media (max-width: 900px) {
        .stats-grid { grid-template-columns: repeat(3, 1fr); }
        .output-grid { grid-template-columns: 1fr; }
        .layoutdet-grid { grid-template-columns: 1fr; }
        .results-table-header, .row-summary {
            grid-template-columns: 2fr 0.6fr 0.6fr 0.6fr 1fr;
            font-size: 0.8rem;
        }
    }
""" + TOOLTIP_CSS + """</style>"""


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------


def _javascript() -> str:
    return (
        """<script>
    // ---- Path configuration ----
    let customBasePath = localStorage.getItem('comparisonDataBasePath') || '';

    function togglePathConfig() {
        const body = document.getElementById('pathConfigBody');
        body.classList.toggle('expanded');
    }
    function updateBasePath() {
        customBasePath = document.getElementById('dataBasePath').value.trim();
        localStorage.setItem('comparisonDataBasePath', customBasePath);
        document.getElementById('currentBasePath').textContent = customBasePath || '(using original paths)';
    }
    function resolveFilePath(originalPath) {
        if (!originalPath) return '';
        if (!customBasePath || !originalBasePath) return originalPath;
        if (originalPath.startsWith(originalBasePath)) {
            return customBasePath + originalPath.slice(originalBasePath.length);
        }
        return originalPath;
    }
    document.addEventListener('DOMContentLoaded', () => {
        if (customBasePath) {
            document.getElementById('dataBasePath').value = customBasePath;
            document.getElementById('currentBasePath').textContent = customBasePath;
        }
        populateMetricSelector();
    });

    // ---- Metric selector ----
    const metricDisplayNames = """
        + json.dumps(METRIC_DISPLAY_NAMES)
        + """;
    let currentFilter = 'all';

    function populateMetricSelector() {
        const select = document.getElementById('metricSelect');
        if (!select) return;
        // Discover all metric names across all results
        const names = new Set();
        comparisonData.forEach(r => {
            (r.pipeline_a.all_metrics || []).forEach(m => names.add(m.metric_name));
            (r.pipeline_b.all_metrics || []).forEach(m => names.add(m.metric_name));
        });
        // Sort with current primary first, then alphabetical
        const sorted = [...names].sort((a, b) => {
            if (a === comparisonMetric) return -1;
            if (b === comparisonMetric) return 1;
            return a.localeCompare(b);
        });
        select.innerHTML = '';
        sorted.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = metricDisplayNames[name] || name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
            if (name === comparisonMetric) opt.selected = true;
            select.appendChild(opt);
        });
    }

    function getMetricValue(allMetrics, metricName) {
        if (!allMetrics) return null;
        const m = allMetrics.find(m => m.metric_name === metricName);
        return m ? m.value : null;
    }

    function switchMetric(newMetric) {
        comparisonMetric = newMetric;
        metricDisplayName = metricDisplayNames[newMetric] || newMetric;

        // Recompute categories and metric values for each row
        const counts = { total: comparisonData.length, a_better: 0, b_better: 0, tie: 0, both_bad: 0 };

        comparisonData.forEach((r, i) => {
            const vA = getMetricValue(r.pipeline_a.all_metrics, newMetric);
            const vB = getMetricValue(r.pipeline_b.all_metrics, newMetric);

            // Update metric_value on the data so detail panes use it
            r.pipeline_a.metric_value = vA;
            r.pipeline_b.metric_value = vB;

            // Recompute category
            let cat;
            if (vA != null && vB != null) {
                cat = vA > vB ? 'a_better' : vB > vA ? 'b_better' : 'tie';
            } else if (vA == null && vB == null) {
                cat = 'both_bad';
            } else if (vA == null) {
                cat = 'b_better';
            } else {
                cat = 'a_better';
            }
            r.category = cat;
            counts[cat] = (counts[cat] || 0) + 1;

            // Update DOM for this row
            const row = document.querySelector(`.result-row[data-index="${i}"]`);
            if (!row) return;
            row.dataset.category = cat;

            const cols = row.querySelector('.row-summary');
            if (!cols) return;
            const metricCols = cols.querySelectorAll('.col-metric');
            const deltaCol = cols.querySelector('.col-delta');
            const catCol = cols.querySelector('.col-category');

            if (metricCols[0]) metricCols[0].innerHTML = fmtMetric(vA);
            if (metricCols[1]) metricCols[1].innerHTML = fmtMetric(vB);
            if (deltaCol) deltaCol.innerHTML = fmtDelta(vA, vB);
            if (catCol) {
                const catLabels = {
                    a_better: pipelineAName + ' Better',
                    b_better: pipelineBName + ' Better',
                    tie: 'Tie',
                    both_bad: 'Both Bad',
                };
                catCol.innerHTML = `<span class="badge badge-${cat.replace('_', '-')}">${esc(catLabels[cat] || cat)}</span>`;
            }

            // If detail pane is open, rebuild it
            const detail = document.getElementById('detail-' + i);
            if (detail && detail.classList.contains('open')) {
                detail.innerHTML = buildDetailContent(i);
                initDetailInteractions(i);
            }
        });

        // Update stat cards
        const statCards = document.querySelectorAll('.stat-card');
        statCards.forEach(card => {
            const filter = card.dataset.filter;
            const valueEl = card.querySelector('.stat-value');
            if (!valueEl) return;
            if (filter === 'all') valueEl.textContent = counts.total;
            else if (counts[filter] !== undefined) valueEl.textContent = counts[filter];
        });

        // Update filter bar
        document.querySelectorAll('.filter-btn').forEach(btn => {
            const filter = btn.dataset.filter;
            if (filter === 'all') btn.textContent = `All (${counts.total})`;
            else if (filter === 'a_better') btn.textContent = `${pipelineAName} Better (${counts.a_better})`;
            else if (filter === 'b_better') btn.textContent = `${pipelineBName} Better (${counts.b_better})`;
            else if (filter === 'tie') btn.textContent = `Tie (${counts.tie})`;
            else if (filter === 'both_bad') btn.textContent = `Both Bad (${counts.both_bad})`;
        });

        // Re-apply current filter
        applyFilter(currentFilter);
    }

    function fmtMetric(val) {
        if (val === null || val === undefined) return '<span class="na">N/A</span>';
        const pct = val * 100;
        return `<span class="metric-val ${metricColorClass(val)}">${pct.toFixed(1)}%</span>`;
    }

    function fmtDelta(a, b) {
        if (a === null || a === undefined || b === null || b === undefined) return '<span class="na">&mdash;</span>';
        const d = (a - b) * 100;
        const sign = d > 0 ? '+' : '';
        const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-zero';
        return `<span class="${cls}">${sign}${d.toFixed(1)}pp</span>`;
    }

    // ---- Filter ----
    function applyFilter(filter) {
        currentFilter = filter;
        // Update filter buttons
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        const matchBtn = document.querySelector(`.filter-btn[data-filter="${filter}"]`);
        if (matchBtn) matchBtn.classList.add('active');
        // Update stat cards
        document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
        const matchCard = document.querySelector(`.stat-card[data-filter="${filter}"]`);
        if (matchCard) matchCard.classList.add('active');
        // Filter rows
        document.querySelectorAll('.result-row').forEach(row => {
            row.style.display = (filter === 'all' || row.dataset.category === filter) ? '' : 'none';
        });
    }

    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => applyFilter(btn.dataset.filter));
    });

    function filterFromCard(card) {
        applyFilter(card.dataset.filter);
    }

    // ---- Expand / collapse rows ----
    const openRows = new Set();

    function toggleRow(index) {
        const detail = document.getElementById('detail-' + index);
        const icon = document.getElementById('icon-' + index);
        if (openRows.has(index)) {
            detail.classList.remove('open');
            icon.classList.remove('expanded');
            openRows.delete(index);
        } else {
            // Build detail content if empty
            if (!detail.innerHTML.trim()) {
                detail.innerHTML = buildDetailContent(index);
                initDetailInteractions(index);
            }
            detail.classList.add('open');
            icon.classList.add('expanded');
            openRows.add(index);
        }
    }

    // ---- Build detail content ----
    function buildDetailContent(index) {
        const r = comparisonData[index];
        if (!r) return '<p>No data</p>';

        if (productType === 'layout_detection') {
            return buildLayoutDetectionDetail(r, index);
        }

        let html = '<div class="detail-two-col">';

        // --- Left column: input preview (sticky) ---
        html += '<div class="detail-left">';
        html += buildInputPanel(r);
        html += buildMetricPills(r);
        html += '</div>';

        // --- Right column: output + metrics + rules ---
        html += '<div class="detail-right">';
        html += buildOutputTab(r, index);

        // Collapsible full metrics
        const fullMetrics = buildFullMetricsSection(r);
        if (fullMetrics) {
            html += `<details class="collapsible-section"><summary>All Metrics</summary>${fullMetrics}</details>`;
        }

        // Collapsible rules
        const rulesHtml = buildRulesSection(r);
        if (rulesHtml) {
            html += `<details class="collapsible-section"><summary>${rulesHtml.summary}</summary>${rulesHtml.body}</details>`;
        }

        html += '</div>';
        html += '</div>';

        return html;
    }

    // Compact metric pills for the left column
    function buildMetricPills(r) {
        const metricsA = r.pipeline_a.all_metrics || [];
        const metricsB = r.pipeline_b.all_metrics || [];
        const statsA = r.pipeline_a.all_stats || [];
        const statsB = r.pipeline_b.all_stats || [];
        if (metricsA.length === 0 && metricsB.length === 0 && statsA.length === 0 && statsB.length === 0) return '';

        const metricDisplayNames = """
        + json.dumps(METRIC_DISPLAY_NAMES)
        + """;

        // Collect all metric names, primary first
        const names = new Set();
        metricsA.forEach(m => names.add(m.metric_name));
        metricsB.forEach(m => names.add(m.metric_name));
        const sorted = [...names].sort((a, b) => {
            if (a === comparisonMetric) return -1;
            if (b === comparisonMetric) return 1;
            return a.localeCompare(b);
        });

        let html = '<div class="metric-pills">';
        sorted.forEach(name => {
            const mA = metricsA.find(m => m.metric_name === name);
            const mB = metricsB.find(m => m.metric_name === name);
            const vA = mA ? mA.value : null;
            const vB = mB ? mB.value : null;
            const displayName = metricDisplayNames[name] || name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
            const isPrimary = name === comparisonMetric;

            const fmtVal = v => v === null || v === undefined ? 'N/A' : (v * 100).toFixed(1) + '%';
            let deltaHtml = '';
            if (vA != null && vB != null) {
                const d = (vA - vB) * 100;
                const sign = d > 0 ? '+' : '';
                const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-zero';
                deltaHtml = `<span class="pill-delta ${cls}">${sign}${d.toFixed(1)}</span>`;
            }

            html += `<div class="pill${isPrimary ? ' pill-primary' : ''}">`;
            html += `<span class="pill-label">${esc(displayName)}${tooltipIcon(name)}</span>`;
            html += `<span class="pill-values"><span class="${metricColorClass(vA)}">${fmtVal(vA)}</span> / <span class="${metricColorClass(vB)}">${fmtVal(vB)}</span>${deltaHtml}</span>`;
            html += '</div>';
        });

        // Stats pills (raw value + unit, not percentage)
        const statNames = new Set();
        statsA.forEach(s => statNames.add(s.name));
        statsB.forEach(s => statNames.add(s.name));
        [...statNames].sort().forEach(name => {
            const sA = statsA.find(s => s.name === name);
            const sB = statsB.find(s => s.name === name);
            const vA = sA ? sA.value : null;
            const vB = sB ? sB.value : null;
            const unit = (sA || sB || {}).unit || '';
            const displayName = name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
            const fmtStat = v => v === null || v === undefined ? 'N/A' : v.toFixed(0) + unit;
            let deltaHtml = '';
            if (vA != null && vB != null) {
                const d = vA - vB;
                const sign = d > 0 ? '+' : '';
                const cls = d > 0 ? 'delta-neg' : d < 0 ? 'delta-pos' : 'delta-zero';
                deltaHtml = `<span class="pill-delta ${cls}">${sign}${d.toFixed(0)}${unit}</span>`;
            }
            html += `<div class="pill">`;
            html += `<span class="pill-label">${esc(displayName)}${tooltipIcon(name)}</span>`;
            html += `<span class="pill-values">${fmtStat(vA)} / ${fmtStat(vB)}${deltaHtml}</span>`;
            html += '</div>';
        });

        html += '</div>';
        return html;
    }

    // Input panel for left column
    function buildInputPanel(r) {
        return buildInputTab(r);
    }

    // Full metrics table (for collapsible section)
    function buildFullMetricsSection(r) {
        const metricsA = r.pipeline_a.all_metrics || [];
        const metricsB = r.pipeline_b.all_metrics || [];
        const statsA = r.pipeline_a.all_stats || [];
        const statsB = r.pipeline_b.all_stats || [];
        const metricNames = new Set();
        metricsA.forEach(m => metricNames.add(m.metric_name));
        metricsB.forEach(m => metricNames.add(m.metric_name));
        const statNames = new Set();
        statsA.forEach(s => statNames.add(s.name));
        statsB.forEach(s => statNames.add(s.name));
        if (metricNames.size === 0 && statNames.size === 0) return '';

        const metricDisplayNames = """
        + json.dumps(METRIC_DISPLAY_NAMES)
        + """;
        const sorted = [...metricNames].sort((a, b) => {
            if (a === comparisonMetric) return -1;
            if (b === comparisonMetric) return 1;
            return a.localeCompare(b);
        });

        let html = `<table class="metrics-table"><thead><tr><th>Metric</th><th style="text-align:center">${esc(pipelineAName)}</th><th style="text-align:center">${esc(pipelineBName)}</th><th style="text-align:center">Delta</th></tr></thead><tbody>`;

        sorted.forEach(name => {
            const mA = metricsA.find(m => m.metric_name === name);
            const mB = metricsB.find(m => m.metric_name === name);
            const vA = mA ? mA.value : null;
            const vB = mB ? mB.value : null;
            const displayName = metricDisplayNames[name] || name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
            const fmtVal = v => v === null || v === undefined ? '<span class="na">N/A</span>' : `<span class="${metricColorClass(v)}">${(v * 100).toFixed(1)}%</span>`;
            const delta = (vA != null && vB != null)
                ? (() => { const d = (vA - vB) * 100; const sign = d > 0 ? '+' : ''; const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-zero'; return `<span class="${cls}">${sign}${d.toFixed(1)}pp</span>`; })()
                : '<span class="na">&mdash;</span>';
            const isPrimary = name === comparisonMetric ? ' style="font-weight:600;"' : '';
            html += `<tr${isPrimary}><td class="metric-name">${esc(displayName)}${tooltipIcon(name)}</td><td class="val-cell">${fmtVal(vA)}</td><td class="val-cell">${fmtVal(vB)}</td><td class="val-cell">${delta}</td></tr>`;
        });

        html += '</tbody></table>';

        // Stats table (raw value + unit)
        if (statNames.size > 0) {
            html += `<table class="metrics-table" style="margin-top:10px"><thead><tr><th>Stat</th><th style="text-align:center">${esc(pipelineAName)}</th><th style="text-align:center">${esc(pipelineBName)}</th><th style="text-align:center">Delta</th></tr></thead><tbody>`;
            [...statNames].sort().forEach(name => {
                const sA = statsA.find(s => s.name === name);
                const sB = statsB.find(s => s.name === name);
                const vA = sA ? sA.value : null;
                const vB = sB ? sB.value : null;
                const unit = (sA || sB || {}).unit || '';
                const displayName = name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
                const fmtStat = v => v === null || v === undefined ? '<span class="na">N/A</span>' : v.toFixed(0) + unit;
                const delta = (vA != null && vB != null)
                    ? (() => { const d = vA - vB; const sign = d > 0 ? '+' : ''; const cls = d > 0 ? 'delta-neg' : d < 0 ? 'delta-pos' : 'delta-zero'; return `<span class="${cls}">${sign}${d.toFixed(0)}${unit}</span>`; })()
                    : '<span class="na">&mdash;</span>';
                html += `<tr><td class="metric-name">${esc(displayName)}</td><td class="val-cell">${fmtStat(vA)}</td><td class="val-cell">${fmtStat(vB)}</td><td class="val-cell">${delta}</td></tr>`;
            });
            html += '</tbody></table>';
        }

        return html;
    }

    // Rules section (returns {summary, body} or null)
    function buildRulesSection(r) {
        const metricsA = r.pipeline_a.all_metrics || [];
        const metricsB = r.pipeline_b.all_metrics || [];

        function getRules(metrics) {
            for (const m of metrics) {
                if (m.metric_name === 'rule_pass_rate' && m.metadata && m.metadata.rule_results) return m.metadata.rule_results;
            }
            for (const m of metrics) {
                if (m.metadata && m.metadata.rule_results) return m.metadata.rule_results;
            }
            return [];
        }

        const rulesA = getRules(metricsA);
        const rulesB = getRules(metricsB);
        if (rulesA.length === 0 && rulesB.length === 0) return null;

        const maxLen = Math.max(rulesA.length, rulesB.length);
        let diffCount = 0;
        for (let i = 0; i < maxLen; i++) {
            const pA = (rulesA[i] || {}).passed;
            const pB = (rulesB[i] || {}).passed;
            if (pA !== undefined && pB !== undefined && pA !== pB) diffCount++;
        }

        const summary = `Rules (${maxLen} total${diffCount > 0 ? `, <strong>${diffCount} differ</strong>` : ''})`;

        let body = `<table class="rule-table"><thead><tr><th>#</th><th>Type</th><th>Name</th><th>Page</th><th style="text-align:center">${esc(pipelineAName)}</th><th style="text-align:center">${esc(pipelineBName)}</th><th></th></tr></thead><tbody>`;

        for (let i = 0; i < maxLen; i++) {
            const rA = rulesA[i] || {};
            const rB = rulesB[i] || {};
            const passA = rA.passed;
            const passB = rB.passed;
            let rowClass = '';
            if (passA !== undefined && passB !== undefined) {
                rowClass = passA === passB ? (passA ? 'rule-same-pass' : 'rule-same-fail') : 'rule-diff-row';
            }
            const type = rA.type || rB.type || '';
            const name = rA.name || rB.name || '';
            const page = rA.page || rB.page || '';
            const fmtPass = v => v === true ? '<span class="rule-pass">PASS</span>' : v === false ? '<span class="rule-fail">FAIL</span>' : '<span class="na">—</span>';
            const explA = rA.explanation || '';
            const explB = rB.explanation || '';
            const hasExpl = explA || explB;
            body += `<tr class="${rowClass}"><td>${i+1}</td><td>${esc(type)}</td><td>${esc(name)}</td><td>${page||''}</td><td style="text-align:center">${fmtPass(passA)}</td><td style="text-align:center">${fmtPass(passB)}</td><td>${hasExpl ? '<span class="rule-expander" onclick="toggleRuleExpl(this)">details</span>' : ''}</td></tr>`;
            if (hasExpl) {
                body += `<tr class="${rowClass}" style="display:none;" data-expl-row="1"><td colspan="7" style="padding:0.25rem 0.6rem;"><div style="font-size:0.75rem;color:var(--muted);">`;
                if (explA) body += `<div><strong>${esc(pipelineAName)}:</strong> ${esc(explA)}</div>`;
                if (explB) body += `<div><strong>${esc(pipelineBName)}:</strong> ${esc(explB)}</div>`;
                body += '</div></td></tr>';
            }
        }
        body += '</tbody></table>';

        return { summary, body };
    }

    function toggleRuleExpl(el) {
        const tr = el.closest('tr');
        const next = tr.nextElementSibling;
        if (next && next.dataset.explRow) {
            next.style.display = next.style.display === 'none' ? '' : 'none';
        }
    }

    // ---- Output tab ----
    function buildOutputTab(r, index) {
        if (productType === 'parse') {
            return buildParseOutputTab(r, index);
        } else if (productType === 'extract') {
            return buildExtractOutputTab(r, index);
        }
        return '<p class="na" style="padding:1rem;">No output comparison available for this product type.</p>';
    }

    function buildParseOutputTab(r, index) {
        const outA = r.pipeline_a.output || '';
        const outB = r.pipeline_b.output || '';

        let html = '';

        // View mode toggle
        html += `<div style="display:flex;gap:0.5rem;margin-bottom:0.75rem;">`;
        html += `<div class="view-toggle">`;
        html += `<button class="active" onclick="setParseView(${index}, 'rendered', this)">Rendered</button>`;
        html += `<button onclick="setParseView(${index}, 'raw', this)">Raw</button>`;
        html += `<button onclick="setParseView(${index}, 'diff', this)">Diff</button>`;
        html += `</div></div>`;

        // Side-by-side rendered
        html += `<div id="parse-rendered-${index}" class="output-grid">`;
        html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineAName)}</div>`;
        html += `<div class="output-panel-body md-view" id="md-a-${index}">${outA ? marked.parse(outA) : '<em>No output</em>'}</div></div>`;
        html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineBName)}</div>`;
        html += `<div class="output-panel-body md-view" id="md-b-${index}">${outB ? marked.parse(outB) : '<em>No output</em>'}</div></div>`;
        html += `</div>`;

        // Side-by-side raw
        html += `<div id="parse-raw-${index}" class="output-grid" style="display:none;">`;
        html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineAName)}</div>`;
        html += `<div class="output-panel-body json-view">${esc(outA) || '<em>No output</em>'}</div></div>`;
        html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineBName)}</div>`;
        html += `<div class="output-panel-body json-view">${esc(outB) || '<em>No output</em>'}</div></div>`;
        html += `</div>`;

        // Diff view
        html += `<div id="parse-diff-${index}" style="display:none;">`;
        html += `<div class="diff-container"><div class="diff-header">Unified Diff</div>`;
        html += `<div class="diff-body">${computeLineDiff(outA, outB)}</div></div>`;
        html += `</div>`;

        return html;
    }

    function setParseView(index, mode, btn) {
        const toggle = btn.closest('.view-toggle');
        toggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        ['rendered', 'raw', 'diff'].forEach(m => {
            const el = document.getElementById(`parse-${m}-${index}`);
            if (el) el.style.display = m === mode ? '' : 'none';
        });
        if (mode === 'rendered' || mode === 'raw') {
            setupSyncScroll(index, mode);
        }
    }

    function buildExtractOutputTab(r, index) {
        const outA = r.pipeline_a.output;
        const outB = r.pipeline_b.output;
        const jsonA = outA != null ? JSON.stringify(outA, null, 2) : '';
        const jsonB = outB != null ? JSON.stringify(outB, null, 2) : '';

        let html = `<div style="display:flex;gap:0.5rem;margin-bottom:0.75rem;">`;
        html += `<div class="view-toggle">`;
        html += `<button class="active" onclick="setExtractView(${index}, 'side', this)">Side by Side</button>`;
        html += `<button onclick="setExtractView(${index}, 'diff', this)">Diff</button>`;
        html += `</div></div>`;

        // Side by side JSON
        html += `<div id="extract-side-${index}" class="output-grid">`;
        html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineAName)}</div>`;
        html += `<div class="output-panel-body json-view">${esc(jsonA) || '<em>No output</em>'}</div></div>`;
        html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineBName)}</div>`;
        html += `<div class="output-panel-body json-view">${esc(jsonB) || '<em>No output</em>'}</div></div>`;
        html += `</div>`;

        // Diff
        html += `<div id="extract-diff-${index}" style="display:none;">`;
        html += `<div class="diff-container"><div class="diff-header">Unified Diff</div>`;
        html += `<div class="diff-body">${computeLineDiff(jsonA, jsonB)}</div></div>`;
        html += `</div>`;

        return html;
    }

    function setExtractView(index, mode, btn) {
        const toggle = btn.closest('.view-toggle');
        toggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        ['side', 'diff'].forEach(m => {
            const el = document.getElementById(`extract-${m}-${index}`);
            if (el) el.style.display = m === mode ? '' : 'none';
        });
    }

    // ---- Input tab ----
    function buildInputTab(r) {
        const inputFile = r.input_file;
        const inputFileRel = r.input_file_rel;
        const dataUrl = r.input_data_url;

        if (!inputFile && !dataUrl) {
            return '<p class="na" style="padding:1rem;">No input file available.</p>';
        }

        const ext = inputFile ? inputFile.split('.').pop().toLowerCase() : '';
        const isPdf = ext === 'pdf';

        let html = '<div class="input-preview">';

        if (inputFile) {
            html += `<p class="file-path">${esc(inputFile)}</p>`;
        }

        if (dataUrl && !isPdf) {
            // Embedded image
            html += `<img src="${dataUrl}" />`;
        } else if (isPdf) {
            // Resolve PDF URL: use pdfBaseUrl + relative path, like the detailed report
            let pdfSrc = '';
            if (pdfBaseUrl && inputFileRel) {
                const base = pdfBaseUrl.endsWith('/') ? pdfBaseUrl.slice(0, -1) : pdfBaseUrl;
                const rel = inputFileRel.startsWith('/') ? inputFileRel.slice(1) : inputFileRel;
                pdfSrc = base + '/' + rel;
            } else if (inputFile) {
                pdfSrc = resolveFilePath(inputFile);
            }
            if (pdfSrc) {
                const viewerId = 'pdfviewer-' + Math.random().toString(36).slice(2, 10);
                html += `<div id="${viewerId}" class="pdfjs-viewer" data-pdf-src="${esc(pdfSrc)}" data-pdf-pending="true" style="max-height:700px;overflow-y:auto;border:1px solid var(--border);border-radius:8px;padding:8px;background:#f5f5f4;"><p class="na">Loading PDF...</p></div>`;
            } else {
                html += `<p class="na">PDF preview not available — no base URL configured.</p>`;
            }
        } else if (inputFile) {
            if (['png', 'jpg', 'jpeg', 'gif'].includes(ext)) {
                // Resolve image URL via file server, same as PDFs
                let imgSrc = '';
                if (pdfBaseUrl && inputFileRel) {
                    const base = pdfBaseUrl.endsWith('/') ? pdfBaseUrl.slice(0, -1) : pdfBaseUrl;
                    const rel = inputFileRel.startsWith('/') ? inputFileRel.slice(1) : inputFileRel;
                    imgSrc = base + '/' + rel;
                } else {
                    imgSrc = resolveFilePath(inputFile);
                }
                html += `<img src="${esc(imgSrc)}" />`;
            } else {
                html += `<p class="na">Preview not available for .${esc(ext)} files</p>`;
            }
        }

        html += '</div>';
        return html;
    }

    // ---- Layout detection detail ----
    const LAYOUTDET_COLORS = {
        'Caption': '#E91E63', 'Footnote': '#9C27B0', 'Formula': '#673AB7',
        'List-item': '#3F51B5', 'Page-footer': '#2196F3', 'Page-header': '#00BCD4',
        'Picture': '#4CAF50', 'Section-header': '#FF9800', 'Table': '#FF5722',
        'Text': '#795548', 'Title': '#F44336'
    };

    function buildLayoutDetectionDetail(r, index) {
        const predA = r.pipeline_a.predictions || [];
        const predB = r.pipeline_b.predictions || [];
        const gt = r.gt_annotations || [];
        const imgPath = r.input_data_url || resolveFilePath(r.input_file || '');

        // Store in window for drawing
        window[`predA_${index}`] = predA;
        window[`predB_${index}`] = predB;
        window[`gt_${index}`] = gt;
        window[`bboxState_${index}`] = { showA: true, showB: true, showGT: true };

        const metricA = r.pipeline_a.metric_value != null ? (r.pipeline_a.metric_value * 100).toFixed(1) + '%' : 'N/A';
        const metricB = r.pipeline_b.metric_value != null ? (r.pipeline_b.metric_value * 100).toFixed(1) + '%' : 'N/A';

        let html = '';

        // Metric pills at top
        html += buildMetricPills(r);

        // 3-column grid
        html += `<div class="layoutdet-grid">`;

        // Panel A
        html += `<div class="layoutdet-panel"><div class="layoutdet-panel-header"><span>${esc(pipelineAName)} (${metricA})</span>`;
        html += `<button class="bbox-btn active" onclick="toggleLayoutBbox(${index}, 'A', this)">Bboxes</button>`;
        html += `</div><div class="layoutdet-panel-body"><img id="ld-img-a-${index}" src="${imgPath}" onload="drawLayoutPanel(${index}, 'A')" /><canvas id="ld-canvas-a-${index}" class="layoutdet-canvas"></canvas></div></div>`;

        // Panel B
        html += `<div class="layoutdet-panel"><div class="layoutdet-panel-header"><span>${esc(pipelineBName)} (${metricB})</span>`;
        html += `<button class="bbox-btn active" onclick="toggleLayoutBbox(${index}, 'B', this)">Bboxes</button>`;
        html += `</div><div class="layoutdet-panel-body"><img id="ld-img-b-${index}" src="${imgPath}" onload="drawLayoutPanel(${index}, 'B')" /><canvas id="ld-canvas-b-${index}" class="layoutdet-canvas"></canvas></div></div>`;

        // GT Panel
        html += `<div class="layoutdet-panel"><div class="layoutdet-panel-header"><span>Ground Truth</span>`;
        html += `<span style="display:flex;gap:4px;">`;
        html += `<button class="bbox-btn active" onclick="toggleLayoutOverlay(${index}, 'GT', this)">GT</button>`;
        html += `<button class="bbox-btn active" onclick="toggleLayoutOverlay(${index}, 'A', this)">A</button>`;
        html += `<button class="bbox-btn active" onclick="toggleLayoutOverlay(${index}, 'B', this)">B</button>`;
        html += `</span></div><div class="layoutdet-panel-body"><img id="ld-img-gt-${index}" src="${imgPath}" onload="drawLayoutOverlay(${index})" /><canvas id="ld-canvas-gt-${index}" class="layoutdet-canvas"></canvas></div></div>`;

        html += `</div>`;

        // Legend
        html += `<div class="comparison-legend">`;
        html += `<div class="legend-item"><div class="legend-swatch" style="background:#4CAF50;border:2px dashed #4CAF50;"></div><span>Ground Truth</span></div>`;
        html += `<div class="legend-item"><div class="legend-swatch" style="background:#2196F3;"></div><span>${esc(pipelineAName)}</span></div>`;
        html += `<div class="legend-item"><div class="legend-swatch" style="background:#9C27B0;border:2px dashed #9C27B0;"></div><span>${esc(pipelineBName)}</span></div>`;
        html += `</div>`;

        return html;
    }

    function drawLayoutPanel(index, which) {
        const canvas = document.getElementById(`ld-canvas-${which.toLowerCase()}-${index}`);
        const img = document.getElementById(`ld-img-${which.toLowerCase()}-${index}`);
        const preds = window[`pred${which}_${index}`];
        const state = window[`bboxState_${index}`];
        if (!canvas || !img) return;
        const ctx = canvas.getContext('2d');
        const scale = img.clientWidth / img.naturalWidth;
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!state[`show${which}`] || !preds) return;
        preds.forEach(p => {
            const bbox = p.bbox.map(v => v * scale);
            const color = LAYOUTDET_COLORS[p.class] || '#999';
            ctx.fillStyle = color + '1a';
            ctx.fillRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
            const label = p.score != null ? `${p.class} (${(p.score * 100).toFixed(0)}%)` : p.class;
            ctx.font = '10px Arial';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(bbox[0], bbox[1] - 13, tw + 4, 13);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, bbox[0] + 2, bbox[1] - 2);
        });
    }

    function drawLayoutOverlay(index) {
        const canvas = document.getElementById(`ld-canvas-gt-${index}`);
        const img = document.getElementById(`ld-img-gt-${index}`);
        if (!canvas || !img) return;
        const ctx = canvas.getContext('2d');
        const scale = img.clientWidth / img.naturalWidth;
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const state = window[`bboxState_${index}`];

        // GT (dashed green)
        const gt = window[`gt_${index}`] || [];
        if (state.showGT && gt.length) {
            ctx.setLineDash([5, 3]);
            gt.forEach(item => {
                const bbox = [item.bbox[0], item.bbox[1], item.bbox[0] + item.bbox[2], item.bbox[1] + item.bbox[3]].map(v => v * scale);
                ctx.strokeStyle = '#4CAF50';
                ctx.lineWidth = 2;
                ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
            });
        }
        // Pred A (solid blue)
        const predA = window[`predA_${index}`] || [];
        if (state.showA && predA.length) {
            ctx.setLineDash([]);
            predA.forEach(item => {
                const bbox = item.bbox.map(v => v * scale);
                ctx.strokeStyle = '#2196F3';
                ctx.lineWidth = 2;
                ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
            });
        }
        // Pred B (dashed purple)
        const predB = window[`predB_${index}`] || [];
        if (state.showB && predB.length) {
            ctx.setLineDash([3, 2]);
            predB.forEach(item => {
                const bbox = item.bbox.map(v => v * scale);
                ctx.strokeStyle = '#9C27B0';
                ctx.lineWidth = 2;
                ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
            });
        }
        ctx.setLineDash([]);
    }

    function toggleLayoutBbox(index, which, btn) {
        const state = window[`bboxState_${index}`];
        state[`show${which}`] = !state[`show${which}`];
        btn.classList.toggle('active', state[`show${which}`]);
        drawLayoutPanel(index, which);
    }

    function toggleLayoutOverlay(index, which, btn) {
        const state = window[`bboxState_${index}`];
        state[`show${which}`] = !state[`show${which}`];
        btn.classList.toggle('active', state[`show${which}`]);
        drawLayoutOverlay(index);
        // Also redraw individual panels if A or B toggled
        if (which === 'A' || which === 'B') {
            drawLayoutPanel(index, which);
        }
    }

    // ---- Diff computation (simple line diff) ----
    function computeLineDiff(textA, textB) {
        if (!textA && !textB) return '<div class="diff-line diff-ctx">(both empty)</div>';
        const linesA = (textA || '').split('\\n');
        const linesB = (textB || '').split('\\n');

        // Simple LCS-based diff
        const m = linesA.length, n = linesB.length;

        // For very large files, fall back to simple comparison
        if (m + n > 4000) {
            let html = '';
            const maxLen = Math.max(m, n);
            for (let i = 0; i < maxLen; i++) {
                const a = i < m ? linesA[i] : undefined;
                const b = i < n ? linesB[i] : undefined;
                if (a === b) {
                    html += `<div class="diff-line diff-ctx"> ${esc(a)}</div>`;
                } else {
                    if (a !== undefined) html += `<div class="diff-line diff-del">-${esc(a)}</div>`;
                    if (b !== undefined) html += `<div class="diff-line diff-add">+${esc(b)}</div>`;
                }
            }
            return html;
        }

        // Build LCS table
        const dp = Array.from({length: m + 1}, () => new Uint16Array(n + 1));
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                dp[i][j] = linesA[i-1] === linesB[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }

        // Backtrack
        const ops = [];
        let i = m, j = n;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && linesA[i-1] === linesB[j-1]) {
                ops.push({type: 'ctx', line: linesA[i-1]});
                i--; j--;
            } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
                ops.push({type: 'add', line: linesB[j-1]});
                j--;
            } else {
                ops.push({type: 'del', line: linesA[i-1]});
                i--;
            }
        }
        ops.reverse();

        // Render with context collapsing
        let html = '';
        let ctxCount = 0;
        const CTX_LIMIT = 3;

        ops.forEach((op, idx) => {
            if (op.type === 'ctx') {
                // Show context lines near changes
                const nearChange = ops.slice(Math.max(0, idx - CTX_LIMIT), idx).some(o => o.type !== 'ctx')
                    || ops.slice(idx + 1, idx + CTX_LIMIT + 1).some(o => o.type !== 'ctx');
                if (nearChange) {
                    if (ctxCount > 0) {
                        html += `<div class="diff-line diff-hunk">@@ ${ctxCount} unchanged lines @@</div>`;
                        ctxCount = 0;
                    }
                    html += `<div class="diff-line diff-ctx"> ${esc(op.line)}</div>`;
                } else {
                    ctxCount++;
                }
            } else {
                if (ctxCount > 0) {
                    html += `<div class="diff-line diff-hunk">@@ ${ctxCount} unchanged lines @@</div>`;
                    ctxCount = 0;
                }
                const cls = op.type === 'add' ? 'diff-add' : 'diff-del';
                const prefix = op.type === 'add' ? '+' : '-';
                html += `<div class="diff-line ${cls}">${prefix}${esc(op.line)}</div>`;
            }
        });

        if (ctxCount > 0) {
            html += `<div class="diff-line diff-hunk">@@ ${ctxCount} unchanged lines @@</div>`;
        }

        return html || '<div class="diff-line diff-ctx">(no differences)</div>';
    }

    // ---- Sync scroll ----
    function setupSyncScroll(index, mode) {
        const container = document.getElementById(`parse-${mode}-${index}`);
        if (!container) return;
        const panels = container.querySelectorAll('.output-panel-body');
        if (panels.length < 2) return;

        let syncing = false;
        panels.forEach((pane, i) => {
            pane.addEventListener('scroll', () => {
                if (syncing) return;
                syncing = true;
                const other = panels[1 - i];
                const pct = pane.scrollTop / (pane.scrollHeight - pane.clientHeight || 1);
                other.scrollTop = pct * (other.scrollHeight - other.clientHeight);
                setTimeout(() => { syncing = false; }, 10);
            });
        });
    }

    // ---- Detail interactions initialization ----
    function initDetailInteractions(index) {
        // Setup sync scroll for rendered view
        if (productType === 'parse') {
            setTimeout(() => setupSyncScroll(index, 'rendered'), 100);
        }
        // Render any pending PDF viewers
        const detail = document.getElementById('detail-' + index);
        if (detail) {
            detail.querySelectorAll('[data-pdf-pending]').forEach(el => {
                el.removeAttribute('data-pdf-pending');
                renderEmbeddedPdf(el.id);
            });
        }
    }

    // ---- Utilities ----
    function metricColorClass(v) {
        if (v === null || v === undefined) return 'metric-na';
        if (v >= 0.9) return 'metric-high';
        if (v >= 0.7) return 'metric-mid';
        if (v >= 0.5) return 'metric-low';
        return 'metric-bad';
    }

    function esc(str) {
        if (str === null || str === undefined) return '';
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

"""
        + TOOLTIP_JS
        + """

    // ---- PDF.js rendering ----
    const pdfJsCdnList = [
        { src: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js',
          worker: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js' },
        { src: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.min.js',
          worker: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.worker.min.js' },
    ];
    let pdfJsLoadPromise = null;

    function ensurePdfJsLoaded() {
        if (window.pdfjsLib) return Promise.resolve(true);
        if (pdfJsLoadPromise) return pdfJsLoadPromise;
        pdfJsLoadPromise = new Promise(resolve => {
            let idx = 0;
            const tryLoad = () => {
                if (idx >= pdfJsCdnList.length) { resolve(false); return; }
                const { src, worker } = pdfJsCdnList[idx++];
                const s = document.createElement('script');
                s.src = src; s.async = true;
                s.onload = () => {
                    if (window.pdfjsLib) {
                        pdfjsLib.GlobalWorkerOptions.workerSrc = worker;
                        resolve(true);
                    } else tryLoad();
                };
                s.onerror = tryLoad;
                document.head.appendChild(s);
            };
            tryLoad();
        });
        return pdfJsLoadPromise;
    }

    async function renderEmbeddedPdf(viewerId) {
        const viewer = document.getElementById(viewerId);
        if (!viewer) return;
        const pdfSrc = viewer.getAttribute('data-pdf-src');
        if (!pdfSrc) { viewer.innerHTML = '<p class="na">No PDF data</p>'; return; }

        const ready = await ensurePdfJsLoaded();
        if (!ready) { viewer.innerHTML = '<p class="na">Failed to load PDF.js</p>'; return; }

        try {
            const loadingTask = pdfjsLib.getDocument(pdfSrc);
            const pdfDoc = await loadingTask.promise;
            viewer.innerHTML = '';

            for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
                const page = await pdfDoc.getPage(pageNum);
                const containerWidth = viewer.clientWidth - 16;
                const unscaledViewport = page.getViewport({ scale: 1 });
                const scale = containerWidth / unscaledViewport.width;
                const viewport = page.getViewport({ scale });

                const canvas = document.createElement('canvas');
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                canvas.style.display = 'block';
                canvas.style.marginBottom = '8px';
                canvas.style.borderRadius = '4px';
                viewer.appendChild(canvas);

                await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
            }
        } catch (e) {
            viewer.innerHTML = `<p class="na">Failed to render PDF: ${esc(e.message || '')}</p>`;
        }
    }
</script>"""
    )
