"""Multi-pipeline leaderboard report.

Generates a self-contained HTML leaderboard comparing all pipelines in the
output directory side-by-side, with per-category metric selectors, best-score
highlighting, and links to individual pipeline dashboards.

Uses the same design system (Newsreader / Plus Jakarta Sans / JetBrains Mono,
warm editorial palette) as the other reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parse_bench.analysis.aggregation_report import _DEFAULT_METRICS
from parse_bench.analysis.metric_definitions import display_name as _display_name
from parse_bench.schemas.evaluation import EvaluationSummary


def _load_pipeline_data(pipeline_dir: Path) -> dict[str, Any] | None:
    """Load pipeline metadata and per-category avg metrics from a pipeline output dir."""
    metadata_path = pipeline_dir / "_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    pm = metadata.get("pipeline", {})
    pipeline_name = pm.get("pipeline_name", pipeline_dir.name)

    # Discover categories (subdirs with _evaluation_report.json)
    categories: list[dict[str, Any]] = []
    for subdir in sorted(pipeline_dir.iterdir()):
        if not subdir.is_dir():
            continue
        report_path = subdir / "_evaluation_report.json"
        if not report_path.exists():
            continue
        try:
            summary = EvaluationSummary.model_validate(
                json.loads(report_path.read_text(encoding="utf-8"))
            )
        except Exception:
            continue

        # Extract avg metrics, same filtering as aggregation_report
        metrics_dict: dict[str, float] = {}
        for key in sorted(summary.aggregate_metrics.keys()):
            if not key.startswith("avg_"):
                continue
            metric_name = key[len("avg_"):]
            if "_predicted" in metric_name or "_judge" in metric_name:
                continue
            metrics_dict[metric_name] = summary.aggregate_metrics[key]

        categories.append(
            {
                "name": subdir.name,
                "files": summary.total_examples,
                "metrics": metrics_dict,
            }
        )

    if not categories:
        return None

    return {
        "name": pipeline_name,
        "dirName": pipeline_dir.name,
        "displayName": pipeline_name.replace("_", " ").title(),
        "provider": pm.get("provider_name", ""),
        "productType": pm.get("product_type", ""),
        "config": pm.get("config", {}),
        "categories": categories,
    }


def generate_leaderboard_report(
    output_dir: Path,
    pipeline_names: list[str] | None = None,
    output_file: Path | None = None,
) -> Path:
    """Generate a leaderboard HTML comparing multiple pipelines.

    Args:
        output_dir: Parent directory containing pipeline subdirectories.
        pipeline_names: Optional list of pipeline dir names to include.
            If None, auto-discovers all subdirs with _metadata.json.
        output_file: Path for the output HTML. Defaults to output_dir/_leaderboard.html.

    Returns:
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)

    # Discover or filter pipelines
    if pipeline_names:
        dirs = [output_dir / name for name in pipeline_names]
    else:
        dirs = sorted(
            d for d in output_dir.iterdir()
            if d.is_dir() and (d / "_metadata.json").exists()
        )

    pipelines: list[dict[str, Any]] = []
    for d in dirs:
        data = _load_pipeline_data(d)
        if data is not None:
            pipelines.append(data)

    if not pipelines:
        raise ValueError(f"No valid pipeline results found in {output_dir}")

    # Collect union of categories and metrics
    all_categories: list[str] = []
    seen_cats: set[str] = set()
    for p in pipelines:
        for cat in p["categories"]:
            if cat["name"] not in seen_cats:
                all_categories.append(cat["name"])
                seen_cats.add(cat["name"])

    # Build scores matrix and collect per-category metrics
    scores: dict[str, dict[str, dict[str, float]]] = {}
    category_files: dict[str, dict[str, int]] = {}
    category_metrics: dict[str, list[str]] = {}
    all_metric_names: set[str] = set()

    for cat_name in all_categories:
        scores[cat_name] = {}
        category_files[cat_name] = {}
        metric_set: set[str] = set()
        for p in pipelines:
            cat_data = next((c for c in p["categories"] if c["name"] == cat_name), None)
            if cat_data:
                scores[cat_name][p["name"]] = cat_data["metrics"]
                category_files[cat_name][p["name"]] = cat_data["files"]
                metric_set.update(cat_data["metrics"].keys())
                all_metric_names.update(cat_data["metrics"].keys())
            else:
                scores[cat_name][p["name"]] = {}
                category_files[cat_name][p["name"]] = 0
        category_metrics[cat_name] = sorted(metric_set)

    # Build metric display names
    metric_names_map: dict[str, str] = {}
    for m in all_metric_names:
        metric_names_map[m] = _display_name(m)

    # Build default metrics per category
    default_metrics: dict[str, str] = {}
    for cat_name in all_categories:
        default = _DEFAULT_METRICS.get(cat_name, "rule_pass_rate")
        if default not in category_metrics.get(cat_name, []):
            default = "rule_pass_rate" if "rule_pass_rate" in category_metrics.get(cat_name, []) else (category_metrics[cat_name][0] if category_metrics[cat_name] else "")
        default_metrics[cat_name] = default

    data_blob = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "defaultMetrics": default_metrics,
        "pipelines": [
            {
                "name": p["name"],
                "dirName": p["dirName"],
                "displayName": p["displayName"],
                "provider": p["provider"],
                "productType": p["productType"],
                "config": p["config"],
                "dashboardUrl": p["dirName"] + "/_evaluation_report_dashboard.html",
            }
            for p in pipelines
        ],
        "categories": all_categories,
        "categoryDisplayNames": {c: c.replace("_", " ").title() for c in all_categories},
        "categoryFiles": category_files,
        "scores": scores,
        "metricNames": metric_names_map,
        "categoryMetrics": category_metrics,
    }

    data_json = json.dumps(data_blob, default=str, ensure_ascii=False)
    data_json = data_json.replace("</script>", "<\\/script>")
    data_json = data_json.replace("<!--", "<\\!--")

    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append(_CSS)
    parts.append("</style>\n</head>\n")
    parts.append(_HTML_BODY)
    parts.append("\n<script>\nconst DATA = ")
    parts.append(data_json)
    parts.append(";\n</script>\n")
    parts.append("<script>\n")
    parts.append(_JS)
    parts.append("\n</script>\n")
    parts.append("</body>\n</html>\n")

    html = "".join(parts)
    if output_file is None:
        output_file = output_dir / "_leaderboard.html"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    return output_file


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
"""

_CSS = """\
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
    --emerald-light: #d1fae5;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --blue: #2563eb;
    --blue-bg: #eff6ff;
    --gold: #b8860b;
    --gold-bg: #fef9e7;
    --gold-border: #e6c547;
    --font-heading: 'Newsreader', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    --shadow-sm: 0 1px 2px rgba(28,25,23,0.05);
    --shadow-md: 0 4px 6px -1px rgba(28,25,23,0.07), 0 2px 4px -2px rgba(28,25,23,0.05);
    --shadow-lg: 0 10px 25px -5px rgba(28,25,23,0.1), 0 4px 10px -4px rgba(28,25,23,0.06);
    --radius: 12px;
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

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--muted-light); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

.report-container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 40px 32px 80px;
}

/* ───── Header ───── */
.report-header { margin-bottom: 40px; }
.report-header h1 {
    font-family: var(--font-heading);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--fg);
    line-height: 1.15;
}
.report-header .subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 8px;
    letter-spacing: 0.01em;
}

/* ───── Table wrapper ───── */
.leaderboard-wrap {
    overflow-x: auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-lg);
}
.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 600px;
}

/* ───── Cells ───── */
.leaderboard-table th,
.leaderboard-table td {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    text-align: center;
    vertical-align: middle;
    transition: background 0.12s ease;
}
.leaderboard-table th {
    background: var(--cream);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 20px 24px;
    border-bottom: 2px solid var(--border);
    vertical-align: bottom;
}

/* Sticky first column */
.leaderboard-table th:first-child,
.leaderboard-table td:first-child {
    text-align: left;
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--card);
    border-right: 1px solid var(--border);
    min-width: 220px;
    padding-left: 28px;
}
.leaderboard-table th:first-child {
    background: var(--cream);
    z-index: 3;
    vertical-align: bottom;
}
.category-header-label {
    font-family: var(--font-heading);
    font-size: 1rem;
    font-weight: 600;
    color: var(--fg);
    text-transform: none;
    letter-spacing: -0.01em;
}
.leaderboard-table tbody tr:last-child td {
    border-bottom: none;
}

/* Row hover (category label column only) */
.leaderboard-table tbody tr:not(.overall-row):hover td:first-child {
    background: #f5f4f1;
}

/* ───── Pipeline header ───── */
.pipeline-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    min-width: 140px;
}
.pipeline-header .pipeline-crown {
    font-size: 1.3rem;
    line-height: 1;
    filter: drop-shadow(0 1px 2px rgba(184,134,11,0.3));
}
.pipeline-header .pipeline-name {
    font-family: var(--font-heading);
    font-size: 1rem;
    font-weight: 700;
    color: var(--fg);
    text-transform: none;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.pipeline-header .pipeline-name a {
    color: inherit;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.15s, color 0.15s;
}
.pipeline-header .pipeline-name a:hover {
    color: var(--blue);
    border-bottom-color: var(--blue);
}
.pipeline-header .pipeline-tier {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--muted-light);
    text-transform: none;
    letter-spacing: 0.02em;
    background: var(--bg);
    padding: 2px 8px;
    border-radius: 99px;
    border: 1px solid var(--border);
}

/* Winner column header glow */
.pipeline-header.is-winner .pipeline-name a {
    color: var(--gold);
}
.pipeline-header.is-winner .pipeline-tier {
    background: var(--gold-bg);
    border-color: var(--gold-border);
    color: var(--gold);
}

/* ───── Category cell ───── */
.category-cell {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.category-name {
    font-family: var(--font-heading);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--fg);
    line-height: 1.3;
}
.category-name .file-count {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted-light);
    margin-left: 2px;
}
.category-selector {
    width: 100%;
    max-width: 210px;
    padding: 5px 8px;
    font-family: var(--font-body);
    font-size: 0.73rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--cream);
    color: var(--muted);
    cursor: pointer;
    outline: none;
    transition: border-color 0.15s;
}
.category-selector:focus { border-color: var(--blue); }
.category-selector:hover { border-color: var(--muted-light); }

/* ───── Column hover: simple bounding box ───── */
.leaderboard-table th[data-col],
.leaderboard-table td[data-col] {
    cursor: pointer;
}
.leaderboard-table th[data-col].col-hover {
    box-shadow: inset 2px 0 0 var(--muted), inset -2px 0 0 var(--muted), inset 0 2px 0 var(--muted);
}
.leaderboard-table .overall-row td[data-col].col-hover {
    box-shadow: inset 2px 0 0 var(--muted), inset -2px 0 0 var(--muted), inset 0 -2px 0 var(--muted);
}
.leaderboard-table tbody tr:not(.overall-row) td[data-col].col-hover {
    box-shadow: inset 2px 0 0 var(--muted), inset -2px 0 0 var(--muted);
}

/* ───── Score cell ───── */
.score-wrap {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 90px;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    transition: background 0.15s ease;
}
.score-wrap.is-best {
    background: var(--emerald-bg);
}
.score-number {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 500;
    white-space: nowrap;
    line-height: 1;
}
.score-wrap.is-best .score-number {
    font-weight: 700;
    font-size: 1.05rem;
}
.score-bar-track {
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}
.bar-emerald { background: var(--emerald); }
.bar-amber { background: var(--amber); }
.bar-red { background: var(--red); }
.score-badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--emerald);
    line-height: 1;
}
.score-na {
    color: var(--muted-light);
    font-size: 0.8rem;
    font-style: italic;
}
.color-emerald { color: var(--emerald); }
.color-amber { color: var(--amber); }
.color-red { color: var(--red); }

/* ───── Overall row ───── */
.overall-row td {
    border-top: 2px solid var(--border);
    border-bottom: none;
    background: var(--cream);
    padding-top: 20px;
    padding-bottom: 20px;
}
.overall-row td:first-child {
    background: var(--cream) !important;
    border-right-color: var(--border);
}
.overall-row:hover td,
.overall-row:hover td:first-child {
    background: #f3f1ec !important;
}
.overall-label {
    font-family: var(--font-heading);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--fg);
    letter-spacing: -0.01em;
}
.overall-sublabel {
    font-size: 0.7rem;
    font-weight: 400;
    color: var(--muted);
    display: block;
    margin-top: 2px;
}
/* Overall score cells */
.overall-row .score-wrap {
    background: transparent;
}
.overall-row .score-wrap.is-best {
    background: var(--emerald-bg);
}
.overall-row .score-number {
    color: var(--fg);
}
.overall-row .score-wrap.is-best .score-number {
    color: var(--emerald);
    font-weight: 600;
}
.overall-row .score-bar-track {
    background: var(--border);
}
.overall-row .score-badge {
    color: var(--emerald);
}
.overall-row .score-na {
    color: var(--muted-light);
}

@media (max-width: 768px) {
    .report-container { padding: 20px 16px 48px; }
    .report-header h1 { font-size: 1.8rem; }
}
"""

_HTML_BODY = """\
<body>
<div class="report-container">
  <header class="report-header">
    <h1>Benchmark Leaderboard</h1>
    <p class="subtitle" id="subtitle"></p>
  </header>

  <div class="leaderboard-wrap">
    <table class="leaderboard-table" id="leaderboard-table">
      <thead id="table-head"></thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
</div>
"""

_JS = """\
(function() {
  function colorClass(rate) {
    if (rate >= 80) return 'emerald';
    if (rate >= 50) return 'amber';
    return 'red';
  }

  function pct(val) { return val.toFixed(1) + '%'; }

  function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  // ─── State ───
  var selectedMetrics = {};
  for (var cat in DATA.defaultMetrics) {
    selectedMetrics[cat] = DATA.defaultMetrics[cat];
  }

  // Subtitle
  document.getElementById('subtitle').textContent =
    DATA.pipelines.length + ' pipeline' + (DATA.pipelines.length !== 1 ? 's' : '') +
    ' across ' + DATA.categories.length + ' categories';

  // ─── Helpers ───
  function getScore(category, pipelineName) {
    var cs = DATA.scores[category];
    if (!cs) return null;
    var ps = cs[pipelineName];
    if (!ps) return null;
    var v = ps[selectedMetrics[category]];
    return (v !== undefined && v !== null) ? v : null;
  }

  function findBest(category) {
    var bestVal = -1, bestNames = [];
    for (var i = 0; i < DATA.pipelines.length; i++) {
      var v = getScore(category, DATA.pipelines[i].name);
      if (v === null) continue;
      if (v > bestVal) { bestVal = v; bestNames = [DATA.pipelines[i].name]; }
      else if (v === bestVal) { bestNames.push(DATA.pipelines[i].name); }
    }
    return bestNames;
  }

  function getOverallScore(pipelineName) {
    var sum = 0, count = 0;
    for (var i = 0; i < DATA.categories.length; i++) {
      var v = getScore(DATA.categories[i], pipelineName);
      if (v !== null) { sum += v; count++; }
    }
    return count > 0 ? sum / count : null;
  }

  function getOverallWinners() {
    var best = -1, names = [];
    for (var i = 0; i < DATA.pipelines.length; i++) {
      var v = getOverallScore(DATA.pipelines[i].name);
      if (v === null) continue;
      if (v > best) { best = v; names = [DATA.pipelines[i].name]; }
      else if (v === best) { names.push(DATA.pipelines[i].name); }
    }
    return names;
  }

  function getMaxFiles(category) {
    var files = DATA.categoryFiles[category] || {};
    var max = 0;
    for (var p in files) { if (files[p] > max) max = files[p]; }
    return max;
  }

  // Extract a clean tier/model label from config
  function getTierLabel(p) {
    var cfg = p.config || {};
    if (cfg.tier) return cfg.tier;
    if (cfg.model) return cfg.model;
    if (cfg.ocr_system) return cfg.ocr_system;
    // Fallback: use first config value that's a short string
    for (var k in cfg) {
      var v = cfg[k];
      if (typeof v === 'string' && v.length < 30) return v;
    }
    return p.productType || '';
  }

  // Build a score cell with progress bar
  function buildScoreCell(v, isBest, isOverall) {
    if (v === null) return '<span class="score-na">N/A</span>';
    var pctVal = v * 100;
    var c = colorClass(pctVal);
    var cls = 'score-wrap' + (isBest ? ' is-best' : '');
    var h = '<div class="' + cls + '">';
    h += '<span class="score-number color-' + c + '">' + pct(pctVal) + '</span>';
    h += '<div class="score-bar-track"><div class="score-bar-fill bar-' + c + '" style="width:' + Math.min(pctVal, 100).toFixed(1) + '%"></div></div>';
    if (isBest) h += '<span class="score-badge">Best</span>';
    h += '</div>';
    return h;
  }

  // ─── Render ───
  function renderHead() {
    var winners = getOverallWinners();
    var thead = document.getElementById('table-head');
    var html = '<tr><th><span class="category-header-label">Category</span></th>';
    for (var i = 0; i < DATA.pipelines.length; i++) {
      var p = DATA.pipelines[i];
      var isWinner = winners.indexOf(p.name) >= 0;
      var tierLabel = getTierLabel(p);
      html += '<th data-col="' + i + '" data-url="' + esc(p.dashboardUrl) + '"><div class="pipeline-header' + (isWinner ? ' is-winner' : '') + '">';
      if (isWinner) html += '<span class="pipeline-crown">\\ud83d\\udc51</span>';
      html += '<span class="pipeline-name">' + esc(p.displayName) + '</span>';
      var sub = p.provider || '';
      if (tierLabel && tierLabel !== p.provider) sub += (sub ? ' / ' : '') + tierLabel;
      if (sub) html += '<span class="pipeline-tier">' + esc(sub) + '</span>';
      html += '</div></th>';
    }
    html += '</tr>';
    thead.innerHTML = html;
  }

  function renderBody() {
    var tbody = document.getElementById('table-body');
    var html = '';

    // Category rows
    for (var ci = 0; ci < DATA.categories.length; ci++) {
      var cat = DATA.categories[ci];
      var bestPipelines = findBest(cat);
      var files = getMaxFiles(cat);
      var metrics = DATA.categoryMetrics[cat] || [];

      html += '<tr>';
      html += '<td><div class="category-cell">';
      html += '<span class="category-name">' + esc(DATA.categoryDisplayNames[cat] || cat);
      html += ' <span class="file-count">' + files + ' files</span></span>';
      html += '<select class="category-selector" data-cat="' + esc(cat) + '">';
      for (var mi = 0; mi < metrics.length; mi++) {
        var m = metrics[mi];
        var sel = m === selectedMetrics[cat] ? ' selected' : '';
        html += '<option value="' + esc(m) + '"' + sel + '>' + esc(DATA.metricNames[m] || m) + '</option>';
      }
      html += '</select></div></td>';

      for (var pi = 0; pi < DATA.pipelines.length; pi++) {
        var pName = DATA.pipelines[pi].name;
        var v = getScore(cat, pName);
        var isBest = bestPipelines.indexOf(pName) >= 0;
        html += '<td data-col="' + pi + '" data-url="' + esc(DATA.pipelines[pi].dashboardUrl) + '">' + buildScoreCell(v, isBest, false) + '</td>';
      }
      html += '</tr>';
    }

    // Overall row
    var overallWinners = getOverallWinners();

    html += '<tr class="overall-row">';
    html += '<td><span class="overall-label">Overall<span class="overall-sublabel">Average across categories</span></span></td>';
    for (var opi = 0; opi < DATA.pipelines.length; opi++) {
      var opName = DATA.pipelines[opi].name;
      var ov = getOverallScore(opName);
      var oIsBest = overallWinners.indexOf(opName) >= 0;
      html += '<td data-col="' + opi + '" data-url="' + esc(DATA.pipelines[opi].dashboardUrl) + '">' + buildScoreCell(ov, oIsBest, true) + '</td>';
    }
    html += '</tr>';

    tbody.innerHTML = html;

    // Bind dropdowns
    var selects = tbody.querySelectorAll('.category-selector');
    for (var si = 0; si < selects.length; si++) {
      selects[si].addEventListener('change', function(e) {
        selectedMetrics[e.target.getAttribute('data-cat')] = e.target.value;
        render();
      });
    }
  }

  function bindColumnInteractions() {
    var table = document.getElementById('leaderboard-table');
    var lastCol = null;

    function highlightCol(colIdx) {
      if (colIdx === lastCol) return;
      clearCol();
      if (colIdx === null) return;
      lastCol = colIdx;
      var cells = table.querySelectorAll('[data-col="' + colIdx + '"]');
      for (var i = 0; i < cells.length; i++) cells[i].classList.add('col-hover');
    }

    function clearCol() {
      if (lastCol === null) return;
      var cells = table.querySelectorAll('[data-col="' + lastCol + '"]');
      for (var i = 0; i < cells.length; i++) cells[i].classList.remove('col-hover');
      lastCol = null;
    }

    table.addEventListener('mouseover', function(e) {
      var cell = e.target.closest('[data-col]');
      if (cell) {
        highlightCol(cell.getAttribute('data-col'));
      }
    });

    table.addEventListener('mouseleave', function() {
      clearCol();
    });

    table.addEventListener('click', function(e) {
      // Don't navigate if clicking a dropdown
      if (e.target.tagName === 'SELECT' || e.target.tagName === 'OPTION') return;
      var cell = e.target.closest('[data-col]');
      if (cell && cell.getAttribute('data-url')) {
        window.location.href = cell.getAttribute('data-url');
      }
    });
  }

  function render() {
    renderHead();
    renderBody();
    bindColumnInteractions();
  }

  render();
})();
"""
