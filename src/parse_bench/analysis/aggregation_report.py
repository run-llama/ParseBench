"""Aggregation dashboard report for multi-category benchmark runs.

Generates a self-contained HTML dashboard showing all categories side-by-side,
with per-category metric selectors, pipeline metadata, and links to detailed reports.

Uses the same design system (Newsreader / Plus Jakarta Sans / JetBrains Mono,
warm editorial palette) as the detailed evaluation reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parse_bench.analysis.metric_definitions import (
    TOOLTIP_CSS,
    TOOLTIP_JS,
    display_name,
    tooltip_dict,
)
from parse_bench.schemas.evaluation import EvaluationSummary


def _load_category_summary(report_json: Path) -> EvaluationSummary | None:
    """Load an EvaluationSummary from a per-category report JSON."""
    try:
        data = json.loads(report_json.read_text(encoding="utf-8"))
        return EvaluationSummary.model_validate(data)
    except Exception:
        return None


# Default "main metric" per category type.  Everything else falls back to rule_pass_rate.
_DEFAULT_METRICS: dict[str, str] = {
    "table": "grits_trm_composite",
    "layout": "layout_element_rule_pass_rate",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
}


def _extract_category_data(name: str, summary: EvaluationSummary) -> dict[str, Any]:
    """Extract display data for a single category from its EvaluationSummary."""
    metrics = summary.aggregate_metrics

    # Build metric list from avg_* keys only
    metric_list: list[dict[str, Any]] = []
    for key in sorted(metrics.keys()):
        if not key.startswith("avg_"):
            continue
        metric_name = key[len("avg_"):]
        # Skip _predicted duplicates and _judge duplicates
        if "_predicted" in metric_name or "_judge" in metric_name:
            continue
        metric_list.append(
            {
                "name": metric_name,
                "displayName": display_name(metric_name),
                "value": metrics[key],  # raw 0-1 float
            }
        )

    # Determine default metric for this category
    default_metric = _DEFAULT_METRICS.get(name, "rule_pass_rate")
    # Fall back if default isn't available in the metrics list
    metric_names_set = {m["name"] for m in metric_list}
    if default_metric not in metric_names_set:
        default_metric = "rule_pass_rate" if "rule_pass_rate" in metric_names_set else (metric_list[0]["name"] if metric_list else "")

    return {
        "name": name,
        "displayName": name.replace("_", " ").title(),
        "files": summary.total_examples,
        "defaultMetric": default_metric,
        "metrics": metric_list,
    }


def generate_aggregation_report(
    pipeline_output_dir: Path,
    groups: list[str],
    pipeline_name: str = "",
) -> Path:
    """Generate an aggregation dashboard HTML showing all categories side-by-side.

    Args:
        pipeline_output_dir: Directory containing per-category subdirectories with
            _evaluation_report.json files.
        groups: List of category/group names to include.
        pipeline_name: Pipeline name for display in the report header.

    Returns:
        Path to the generated HTML file.
    """
    # Load pipeline metadata
    pipeline_metadata: dict[str, Any] = {}
    metadata_path = pipeline_output_dir / "_metadata.json"
    if metadata_path.exists():
        try:
            pipeline_metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("pipeline", {})
        except Exception:
            pass

    if not pipeline_name and pipeline_metadata.get("pipeline_name"):
        pipeline_name = pipeline_metadata["pipeline_name"]

    categories: list[dict[str, Any]] = []
    for group_name in groups:
        report_path = pipeline_output_dir / group_name / "_evaluation_report.json"
        summary = _load_category_summary(report_path)
        if summary is not None:
            cat_data = _extract_category_data(group_name, summary)
            categories.append(cat_data)

    total_files = sum(c["files"] for c in categories)

    data_blob = {
        "pipelineName": pipeline_name,
        "pipelineMetadata": pipeline_metadata,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "totalFiles": total_files,
        "categories": categories,
        "metricTooltips": tooltip_dict(),
    }

    data_json = json.dumps(data_blob, default=str, ensure_ascii=False)
    data_json = data_json.replace("</script>", "<\\/script>")
    data_json = data_json.replace("<!--", "<\\!--")

    # Build full HTML by concatenation (same pattern as detailed_report.py)
    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append(_CSS)
    parts.append("</style>\n</head>\n")
    parts.append(_HTML_BODY)

    # Data blob
    parts.append("\n<script>\nconst DATA = ")
    parts.append(data_json)
    parts.append(";\n</script>\n")

    # Application JS
    parts.append("<script>\n")
    parts.append(_JS)
    parts.append("\n</script>\n")
    parts.append("</body>\n</html>\n")

    html = "".join(parts)
    output_path = pipeline_output_dir / "_evaluation_report_dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# HTML template parts — uses same design system as detailed_report.py
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
/* ───── Reset & variables (shared with detailed_report.py) ───── */
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

/* ───── Section titles ───── */
.section-title {
    font-family: var(--font-heading);
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
}

/* ───── Categories grid ───── */
.categories-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}

.category-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s;
    cursor: pointer;
    display: block;
    color: inherit;
}
.category-card:hover { box-shadow: var(--shadow-md); }
.category-card h3 {
    font-family: var(--font-heading);
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--fg);
}
.category-card h3 .file-count {
    font-size: 0.8rem;
    font-weight: 400;
    color: var(--muted);
}
.category-card .main-score {
    font-family: var(--font-heading);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 8px;
}
.color-emerald { color: var(--emerald); }
.color-amber { color: var(--amber); }
.color-red { color: var(--red); }

/* ───── Progress bar ───── */
.progress-bar-track {
    height: 6px;
    background: var(--cream);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 12px;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}
.bar-emerald { background: var(--emerald); }
.bar-amber { background: var(--amber); }
.bar-red { background: var(--red); }

/* ───── Metric selector dropdown ───── */
.metric-selector {
    width: 100%;
    margin-bottom: 10px;
    padding: 6px 8px;
    font-family: var(--font-body);
    font-size: 0.8rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--cream);
    color: var(--fg);
    cursor: pointer;
    outline: none;
    transition: border-color 0.15s;
}
.metric-selector:focus { border-color: var(--blue); }

/* ───── Metric rows ───── */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--cream);
}
.metric-row:last-child { border-bottom: none; }
.metric-row .metric-name {
    color: var(--muted);
    flex: 1;
    min-width: 0;
    line-height: 1.4;
}
.metric-row .metric-value {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 500;
    margin-left: 8px;
    white-space: nowrap;
    flex-shrink: 0;
}
.metric-row.selected .metric-name {
    font-weight: 700;
    color: var(--fg);
}

/* ───── Responsive ───── */
@media (max-width: 900px) {
    .summary-row { grid-template-columns: repeat(2, 1fr); }
    .categories-grid { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
    .report-container { padding: 16px 12px 48px; }
    .summary-row { grid-template-columns: 1fr; }
}
""" + TOOLTIP_CSS

_HTML_BODY = """\
<body>
<div class="report-container">
  <header class="report-header">
    <h1 id="report-title">Evaluation Report</h1>
    <p class="subtitle" id="subtitle"></p>
  </header>

  <div class="summary-row" id="summary-cards"></div>

  <h2 class="section-title">Categories</h2>
  <div class="categories-grid" id="categories-grid"></div>
</div>
"""

_JS = """\
(function() {
  function colorClass(rate) {
    if (rate >= 80) return 'emerald';
    if (rate >= 50) return 'amber';
    return 'red';
  }

  function pct(val, d) {
    d = d !== undefined ? d : 1;
    return val.toFixed(d) + '%';
  }

  function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

""" + TOOLTIP_JS + """

  // ─── State: selected metric per category ───
  var selectedMetrics = {};
  DATA.categories.forEach(function(cat) {
    selectedMetrics[cat.name] = cat.defaultMetric;
  });

  // ─── Header ───
  var titleText = DATA.pipelineName
    ? DATA.pipelineName.replace(/_/g, ' ').replace(/\\b\\w/g, function(c) { return c.toUpperCase(); })
    : 'Evaluation Report';
  document.getElementById('report-title').textContent = titleText;
  document.getElementById('subtitle').textContent = 'Generated: ' + DATA.generatedAt;

  // ─── Helpers ───
  function getSelectedValue(cat) {
    var sel = selectedMetrics[cat.name];
    for (var i = 0; i < cat.metrics.length; i++) {
      if (cat.metrics[i].name === sel) return cat.metrics[i].value * 100;
    }
    return 0;
  }

  function computeAvgScore() {
    var sum = 0, count = 0;
    DATA.categories.forEach(function(cat) {
      sum += getSelectedValue(cat);
      count++;
    });
    return count > 0 ? sum / count : 0;
  }

  // ─── Render summary cards ───
  function renderSummary() {
    var sc = document.getElementById('summary-cards');
    sc.innerHTML = '';
    var avg = computeAvgScore();
    var ac = colorClass(avg);

    // Total Files card
    var d1 = document.createElement('div');
    d1.className = 'summary-card';
    d1.innerHTML = '<div class="label">TOTAL FILES</div><div class="big-number">' + DATA.totalFiles + '</div>';
    sc.appendChild(d1);

    // Avg Score card
    var d2 = document.createElement('div');
    d2.className = 'summary-card';
    d2.innerHTML = '<div class="label">AVG SCORE</div><div class="big-number color-' + ac + '">' + pct(avg) + '</div>';
    sc.appendChild(d2);
  }

  // ─── Render category cards ───
  function renderCategories() {
    var grid = document.getElementById('categories-grid');
    grid.innerHTML = '';
    // Set grid columns to match number of categories
    var numCats = DATA.categories.length;
    grid.style.gridTemplateColumns = 'repeat(' + numCats + ', 1fr)';

    DATA.categories.forEach(function(cat) {
      var card = document.createElement('div');
      card.className = 'category-card';

      var selMetric = selectedMetrics[cat.name];
      var mainVal = getSelectedValue(cat);
      var c = colorClass(mainVal);

      var html = '<h3>' + esc(cat.displayName) + ' <span class="file-count">(' + cat.files + ' files)</span></h3>';
      html += '<div class="main-score color-' + c + '">' + pct(mainVal) + '</div>';
      html += '<div class="progress-bar-track"><div class="progress-bar-fill bar-' + c + '" style="width:' + Math.min(mainVal, 100) + '%"></div></div>';

      // Metric selector dropdown
      html += '<select class="metric-selector" data-cat="' + esc(cat.name) + '">';
      for (var i = 0; i < cat.metrics.length; i++) {
        var m = cat.metrics[i];
        var selected = m.name === selMetric ? ' selected' : '';
        html += '<option value="' + esc(m.name) + '"' + selected + '>' + esc(m.displayName) + '</option>';
      }
      html += '</select>';

      // Metric list: selected metric first, then the rest in original order
      var sorted = [];
      var rest = [];
      for (var j = 0; j < cat.metrics.length; j++) {
        if (cat.metrics[j].name === selMetric) {
          sorted.unshift(cat.metrics[j]);
        } else {
          rest.push(cat.metrics[j]);
        }
      }
      sorted = sorted.concat(rest);

      for (var k = 0; k < sorted.length; k++) {
        var sm = sorted[k];
        var mc = colorClass(sm.value * 100);
        var selClass = sm.name === selMetric ? ' selected' : '';
        html += '<div class="metric-row' + selClass + '">';
        html += '<span class="metric-name">' + esc(sm.displayName) + '</span>' + tooltipIcon(sm.name);
        html += '<span class="metric-value color-' + mc + '">' + pct(sm.value * 100) + '</span>';
        html += '</div>';
      }

      card.innerHTML = html;

      // Click on card navigates to detailed report (but not when clicking dropdown)
      card.addEventListener('click', function(e) {
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'OPTION') return;
        if (e.target.closest && e.target.closest('.metric-hint')) return;
        window.location.href = cat.name + '/_evaluation_report_detailed.html';
      });

      // Dropdown change updates selected metric and re-renders
      var select = card.querySelector('.metric-selector');
      if (select) {
        select.addEventListener('change', function(e) {
          e.stopPropagation();
          selectedMetrics[cat.name] = e.target.value;
          renderCategories();
          renderSummary();
        });
      }

      grid.appendChild(card);
    });
  }

  renderSummary();
  renderCategories();
})();
"""
