"""Shared metric display names and tooltip explanations.

Single source of truth for metric metadata used across all report generators
(aggregation dashboard, detailed report, comparison report).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class MetricInfo:
    """Display metadata for a single evaluation metric."""

    display_name: str
    tooltip: str


# ---------------------------------------------------------------------------
# Master metric definitions — superset of all report display-name dicts.
# Tooltip text is derived from the actual evaluation / metric code.
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS: dict[str, MetricInfo] = {
    # ── Parse: TEDS ──
    "teds": MetricInfo(
        "TEDS (All)",
        "Tree Edit Distance Similarity. Compares table HTML as trees using APTED algorithm: "
        "1 \u2212 (edit_distance / max_nodes). Evaluates both structure and cell text content. "
        "Score is averaged across matched table pairs.",
    ),
    "teds_predicted": MetricInfo(
        "TEDS (Predicted)",
        "Same as TEDS, but computed only among examples where tables were actually predicted "
        "(excludes examples with zero predicted tables).",
    ),
    "teds_struct": MetricInfo(
        "TEDS-Struct (All)",
        "TEDS structure-only variant. Compares table HTML tree structure while ignoring "
        "cell text content entirely.",
    ),
    "teds_struct_predicted": MetricInfo(
        "TEDS-Struct (Predicted)",
        "Same as TEDS-Struct, but only among examples with predicted tables.",
    ),
    "teds_struct_bool": MetricInfo(
        "TEDS-Struct+Bool (All)",
        "TEDS structure variant with boolean content awareness. Penalizes when one cell is "
        "empty and the other is not, but ignores actual text differences.",
    ),
    "teds_struct_bool_predicted": MetricInfo(
        "TEDS-Struct+Bool (Predicted)",
        "Same as TEDS-Struct+Bool, but only among examples with predicted tables.",
    ),
    # ── Parse: GriTS ──
    "grits_top": MetricInfo(
        "GriTS Top (All)",
        "Grid Table Similarity for topology. F-score from 2D Most-Similar Substructures "
        "algorithm, using IoU on cell spans for structural comparison.",
    ),
    "grits_con": MetricInfo(
        "GriTS Con (All)",
        "Grid Table Similarity for content. F-score from 2D Most-Similar Substructures "
        "algorithm, using Longest Common Subsequence for cell text comparison.",
    ),
    "grits_trm_composite": MetricInfo(
        "GTRM",
        "Composite GriTS score combining topology, recognition, and matching components.",
    ),
    "grits_top_predicted": MetricInfo(
        "GriTS Top (Predicted)",
        "Same as GriTS Top, but only among examples with predicted tables.",
    ),
    "grits_con_predicted": MetricInfo(
        "GriTS Con (Predicted)",
        "Same as GriTS Con, but only among examples with predicted tables.",
    ),
    "grits_precision_top": MetricInfo(
        "GriTS Precision (Topology)",
        "Precision component of GriTS topology score.",
    ),
    "grits_recall_top": MetricInfo(
        "GriTS Recall (Topology)",
        "Recall component of GriTS topology score.",
    ),
    "grits_precision_con": MetricInfo(
        "GriTS Precision (Content)",
        "Precision component of GriTS content score.",
    ),
    "grits_recall_con": MetricInfo(
        "GriTS Recall (Content)",
        "Recall component of GriTS content score.",
    ),
    "grits_top_upper_bound": MetricInfo(
        "GriTS Top Upper Bound",
        "Upper bound for GriTS topology score based on table matching.",
    ),
    "grits_con_upper_bound": MetricInfo(
        "GriTS Con Upper Bound",
        "Upper bound for GriTS content score based on table matching.",
    ),
    # ── Parse: reference GriTS ──
    "ref_grits_top": MetricInfo(
        "Ref GriTS Top (All)",
        "Reference GriTS topology score computed against reference tables.",
    ),
    "ref_grits_con": MetricInfo(
        "Ref GriTS Con (All)",
        "Reference GriTS content score computed against reference tables.",
    ),
    "ref_grits_top_predicted": MetricInfo(
        "Ref GriTS Top (Predicted)",
        "Reference GriTS topology score, only among examples with predicted tables.",
    ),
    "ref_grits_con_predicted": MetricInfo(
        "Ref GriTS Con (Predicted)",
        "Reference GriTS content score, only among examples with predicted tables.",
    ),
    # ── Parse: header accuracy ──
    "header_composite": MetricInfo(
        "Header Composite",
        "Mean of 8 header submetrics: cell count, GriTS content, content bag, perfect match, "
        "structure, block order, block extent, and block relative position.",
    ),
    "header_composite_v3": MetricInfo(
        "Header Composite v3",
        "Version 3 of the header composite metric with updated submetric weights and components.",
    ),
    "header_cell_count": MetricInfo(
        "Header Cell Count",
        "Ratio of predicted header cells to expected. Penalizes both missing and extra "
        "header cells symmetrically.",
    ),
    "header_grits": MetricInfo(
        "Header GriTS",
        "GriTS content score applied to contiguous header blocks only.",
    ),
    "header_content_bag": MetricInfo(
        "Header Content Bag",
        "Bag-of-cells exact content overlap: measures how many header cell texts match "
        "regardless of position.",
    ),
    "header_perfect": MetricInfo(
        "Header Perfect",
        "Binary metric: 1.0 if the header structure matches the ground truth exactly, "
        "0.0 otherwise.",
    ),
    "header_structure": MetricInfo(
        "Header Structure",
        "GriTS topology score applied to the header region, measuring grid structure accuracy.",
    ),
    "header_block_order": MetricInfo(
        "Header Block Order",
        "Relative position preservation when multiple header blocks exist in a table.",
    ),
    "header_block_extent": MetricInfo(
        "Header Block Extent",
        "Location and size accuracy of each header block within the full table.",
    ),
    "header_block_proximity": MetricInfo(
        "Header Block Proximity",
        "Nearest-edge distance between matched header blocks, measuring spatial closeness.",
    ),
    "header_block_relative_direction": MetricInfo(
        "Header Block Relative Direction",
        "Cosine similarity of relative direction vectors between matched header blocks.",
    ),
    "header_block_relative_position": MetricInfo(
        "Header Block Relative Position",
        "Product of proximity (nearest-edge distance) and direction (cosine similarity) "
        "between matched header blocks.",
    ),
    # ── Parse: structural consistency ──
    "structural_consistency": MetricInfo(
        "Structural Consistency",
        "Self-consistency check on predicted tables (no ground truth comparison). "
        "Binary: 1.0 if every row has the same column count and every column has the same "
        "row count after resolving colspan/rowspan.",
    ),
    # ── Parse: table composite ──
    "table_composite": MetricInfo(
        "Table Composite",
        "Weighted combination of table metrics: "
        "0.8 \u00d7 (header_composite \u00d7 grits_con) + 0.2 \u00d7 structural_consistency. "
        "Balances content accuracy with structural integrity.",
    ),
    "table_composite_v3": MetricInfo(
        "Table Composite v3",
        "Version 3 of the table composite metric with updated component weights.",
    ),
    "table_composite_v3_harmonic": MetricInfo(
        "Table Composite v3 Harmonic",
        "Harmonic mean variant of table composite v3, penalizing low outlier scores more heavily.",
    ),
    # ── Parse: experimental composites ──
    "exp_header_composite_v3_generous": MetricInfo(
        "Exp Header Composite v3 Generous",
        "Experimental header composite v3 with generous matching criteria.",
    ),
    "exp_table_composite_v3_generous": MetricInfo(
        "Exp Table Composite v3 Generous",
        "Experimental table composite v3 with generous matching criteria.",
    ),
    "exp_table_composite_v3_generous_harmonic": MetricInfo(
        "Exp Table Composite v3 Generous (Harmonic)",
        "Harmonic mean variant of the experimental generous table composite v3.",
    ),
    # ── Parse: normalized text metrics ──
    "normalized_text_styling": MetricInfo(
        "Normalized Text Styling",
        "Normalized score for text styling accuracy (bold, italic, underline, etc.).",
    ),
    "normalized_text_correctness": MetricInfo(
        "Normalized Text Correctness",
        "Normalized score for text content correctness.",
    ),
    "normalized_order": MetricInfo(
        "Normalized Order",
        "Normalized score for reading order accuracy of text elements.",
    ),
    "normalized_title_accuracy": MetricInfo(
        "Normalized Title Accuracy",
        "Normalized score for title detection and content accuracy.",
    ),
    "normalized_code_block": MetricInfo(
        "Normalized Code Block",
        "Normalized score for code block detection and content accuracy.",
    ),
    "normalized_latex": MetricInfo(
        "Normalized LaTeX",
        "Normalized score for LaTeX equation rendering accuracy.",
    ),
    "normalized_text_score": MetricInfo(
        "Normalized Text Score",
        "Overall normalized text score combining multiple text quality dimensions.",
    ),
    # ── Parse: semantic metrics ──
    "content_faithfulness": MetricInfo(
        "Content Faithfulness",
        "Measures how faithfully the predicted content represents the source document.",
    ),
    "semantic_formatting": MetricInfo(
        "Semantic Formatting",
        "Measures accuracy of semantic formatting elements (headings, lists, emphasis, etc.).",
    ),
    # ── Parse: text similarity ──
    "text_similarity": MetricInfo(
        "Text Similarity",
        "Normalized Levenshtein distance between expected and predicted text, "
        "scaled to 0\u20131 where 1.0 is a perfect match.",
    ),
    # ── Parse: rule-based ──
    "rule_pass_rate": MetricInfo(
        "Rule Pass Rate",
        "Fraction of test rules that pass for each example: passed / total across all "
        "rule types.",
    ),
    # ── Parse: rule subtypes ──
    "chart_data_point": MetricInfo(
        "Chart Data Point",
        "Pass rate for chart data point extraction rules.",
    ),
    "order": MetricInfo(
        "Order",
        "Pass rate for reading order rules, checking that elements appear in the "
        "expected sequence.",
    ),
    "is_bold": MetricInfo(
        "Is Bold",
        "Pass rate for bold formatting detection rules.",
    ),
    "is_footer": MetricInfo(
        "Is Footer",
        "Pass rate for footer section detection rules.",
    ),
    "is_header": MetricInfo(
        "Is Header",
        "Pass rate for header section detection rules.",
    ),
    "is_sup": MetricInfo(
        "Is Sup",
        "Pass rate for superscript formatting detection rules.",
    ),
    "is_underline": MetricInfo(
        "Is Underline",
        "Pass rate for underline formatting detection rules.",
    ),
    "missing_sentence": MetricInfo(
        "Missing Sentence",
        "Pass rate for missing sentence rules. Checks that expected sentences "
        "appear in the output.",
    ),
    "missing_specific_sentence": MetricInfo(
        "Missing Specific Sentence",
        "Pass rate for specific required sentence presence rules.",
    ),
    "missing_specific_word": MetricInfo(
        "Missing Specific Word",
        "Pass rate for specific required word presence rules.",
    ),
    "missing_word": MetricInfo(
        "Missing Word",
        "Pass rate for missing word rules. Checks that expected words appear "
        "in the output.",
    ),
    "too_many_sentence_occurence": MetricInfo(
        "Too Many Sentence Occurence",
        "Pass rate for sentence frequency rules. Penalizes when sentences appear "
        "more times than expected.",
    ),
    "too_many_word_occurence": MetricInfo(
        "Too Many Word Occurence",
        "Pass rate for word frequency rules. Penalizes when words appear more "
        "times than expected.",
    ),
    "unexpected_sentence": MetricInfo(
        "Unexpected Sentence",
        "Pass rate for unexpected sentence rules. Penalizes extra sentences not "
        "in the ground truth.",
    ),
    "unexpected_word": MetricInfo(
        "Unexpected Word",
        "Pass rate for unexpected word rules. Penalizes extra words not in the "
        "ground truth.",
    ),
    "table_adjacent_down": MetricInfo(
        "Table Adjacent Down",
        "Pass rate for table adjacency rules checking cells directly below.",
    ),
    "table_adjacent_right": MetricInfo(
        "Table Adjacent Right",
        "Pass rate for table adjacency rules checking cells directly to the right.",
    ),
    "table_colspan": MetricInfo(
        "Table Colspan",
        "Pass rate for column span detection rules in tables.",
    ),
    "table_rowspan": MetricInfo(
        "Table Rowspan",
        "Pass rate for row span detection rules in tables.",
    ),
    "table_no_above": MetricInfo(
        "Table No Above",
        "Pass rate for rules verifying no content exists above a given table cell.",
    ),
    "table_no_below": MetricInfo(
        "Table No Below",
        "Pass rate for rules verifying no content exists below a given table cell.",
    ),
    "table_no_left": MetricInfo(
        "Table No Left",
        "Pass rate for rules verifying no content exists to the left of a given table cell.",
    ),
    "table_no_right": MetricInfo(
        "Table No Right",
        "Pass rate for rules verifying no content exists to the right of a given table cell.",
    ),
    "table_same_column": MetricInfo(
        "Table Same Column",
        "Pass rate for rules verifying that specific cells share the same column.",
    ),
    "table_same_row": MetricInfo(
        "Table Same Row",
        "Pass rate for rules verifying that specific cells share the same row.",
    ),
    "table_top_header": MetricInfo(
        "Table Top Header",
        "Pass rate for rules checking top header row identification in tables.",
    ),
    # ── Extract ──
    "accuracy": MetricInfo(
        "Accuracy",
        "JSON subset match comparing expected vs actual extracted data. Supports date "
        "normalization and weighted scoring by leaf node count.",
    ),
    # ── Layout detection: attribution ──
    "af1": MetricInfo(
        "Attribution F1",
        "Harmonic mean of LAP and LAR. Measures overall content attribution "
        "accuracy in spatial regions.",
    ),
    "lap": MetricInfo(
        "Local Attribution Precision",
        "For each predicted block, checks whether its text tokens are found in "
        "ground truth elements that spatially overlap with it.",
    ),
    "lar": MetricInfo(
        "Local Attribution Recall",
        "For each ground truth element, checks whether its text tokens are "
        "recovered by predicted blocks that spatially overlap with it.",
    ),
    # ── Layout detection: COCO metrics ──
    "mAP@[.50:.95]": MetricInfo(
        "mAP@[.50:.95]",
        "Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 "
        "(step 0.05). Standard COCO object detection metric.",
    ),
    "AP50": MetricInfo(
        "AP@50",
        "Average Precision at IoU threshold 0.50. Measures detection accuracy "
        "with a lenient overlap requirement.",
    ),
    "AP75": MetricInfo(
        "AP@75",
        "Average Precision at IoU threshold 0.75. Measures detection accuracy "
        "with a strict overlap requirement.",
    ),
    "mean_f1": MetricInfo(
        "Mean F1",
        "Average F1 score across all layout element classes.",
    ),
    # ── Layout detection: rule pass rates ──
    "layout_element_rule_pass_rate": MetricInfo(
        "Layout Element Rule Pass Rate",
        "Overall per-element rule pass rate combining localization, classification, "
        "attribution, and reading order checks.",
    ),
    "layout_localization_rule_pass_rate": MetricInfo(
        "Layout Localization Rule Pass Rate",
        "Pass rate for bounding box localization rules. Checks spatial accuracy "
        "of predicted element positions.",
    ),
    "layout_classification_rule_pass_rate": MetricInfo(
        "Layout Classification Rule Pass Rate",
        "Pass rate for class label prediction rules. Checks whether predicted "
        "element types match ground truth.",
    ),
    "layout_attribution_rule_pass_rate": MetricInfo(
        "Layout Attribution Rule Pass Rate",
        "Pass rate for content attribution rules. Checks whether predicted blocks "
        "contain the correct text content.",
    ),
    "layout_reading_order_pass_rate": MetricInfo(
        "Layout Reading Order Pass Rate",
        "Pass rate for reading order rules. Checks whether layout elements are "
        "ordered correctly.",
    ),
    # ── QA ──
    "qa_answer_match": MetricInfo(
        "QA Match",
        "Binary pass/fail for each question. Supports single-choice, multiple-choice, "
        "numerical (with tolerance), and free-text answer types.",
    ),
    "qa_anls_star": MetricInfo(
        "QA ANLS*",
        "Average Normalized Levenshtein Similarity for free-text answers. "
        "Ranges from 0 (completely different) to 1 (perfect match).",
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def display_name(metric_key: str) -> str:
    """Return human-friendly display name for a metric.

    Falls back to title-cased key with underscores replaced by spaces.
    """
    info = METRIC_DEFINITIONS.get(metric_key)
    if info is not None:
        return info.display_name
    return metric_key.replace("_", " ").title()


def tooltip(metric_key: str) -> str:
    """Return tooltip explanation for a metric.

    Returns empty string when no tooltip is available (dynamic fallbacks
    are handled by the JS ``tooltipIcon()`` function in each report).
    """
    info = METRIC_DEFINITIONS.get(metric_key)
    if info is not None:
        return info.tooltip
    return ""


def display_name_dict() -> dict[str, str]:
    """Return ``{metric_key: display_name}`` for embedding in report JS."""
    return {k: v.display_name for k, v in METRIC_DEFINITIONS.items()}


def tooltip_dict() -> dict[str, str]:
    """Return ``{metric_key: tooltip_text}`` for embedding in report JS."""
    return {k: v.tooltip for k, v in METRIC_DEFINITIONS.items()}


# ---------------------------------------------------------------------------
# Shared CSS for metric tooltips (injected into each report's <style> block).
# ---------------------------------------------------------------------------

TOOLTIP_CSS = """\
/* ───── Metric Tooltips ───── */
.metric-hint {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    margin-left: 5px;
    flex-shrink: 0;
    cursor: help;
}
.metric-hint::before {
    content: '?';
    display: flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1.5px solid var(--muted-light);
    font-family: var(--font-body);
    font-size: 0.55rem;
    font-weight: 700;
    color: var(--muted-light);
    line-height: 1;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.metric-hint:hover::before {
    border-color: var(--muted);
    color: var(--card);
    background: var(--muted);
}
/* Tooltip popup — rendered as a singleton fixed-position element via JS */
#metric-tooltip {
    position: fixed;
    z-index: 10000;
    width: 280px;
    padding: 10px 13px;
    background: var(--fg, #1c1917);
    color: #f5f5f4;
    font-family: var(--font-body, 'Plus Jakarta Sans', sans-serif);
    font-size: 0.85rem;
    font-weight: 400;
    line-height: 1.6;
    border-radius: 6px;
    box-shadow: 0 8px 24px rgba(28,25,23,0.18), 0 2px 6px rgba(28,25,23,0.08);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s ease;
}
#metric-tooltip.visible {
    opacity: 1;
}
#metric-tooltip::after {
    content: '';
    position: absolute;
    border: 5px solid transparent;
}
#metric-tooltip.arrow-bottom::after {
    top: 100%;
    left: var(--arrow-left, 50%);
    transform: translateX(-50%);
    border-top-color: var(--fg, #1c1917);
}
#metric-tooltip.arrow-top::after {
    bottom: 100%;
    left: var(--arrow-left, 50%);
    transform: translateX(-50%);
    border-bottom-color: var(--fg, #1c1917);
}
"""

# ---------------------------------------------------------------------------
# Shared JS helper for building tooltip icon HTML (injected into each report).
# ---------------------------------------------------------------------------

TOOLTIP_JS = """\
  // ─── Metric tooltip system (fixed-position, never clipped) ───
  var _tipEl = null;
  var _tipHideTimer = null;
  function _ensureTip() {
    if (_tipEl) return _tipEl;
    _tipEl = document.createElement('div');
    _tipEl.id = 'metric-tooltip';
    document.body.appendChild(_tipEl);
    return _tipEl;
  }
  function _getTooltipText(metricKey) {
    var tips = (typeof DATA !== 'undefined' && DATA.metricTooltips) ? DATA.metricTooltips : (typeof metricTooltips !== 'undefined' ? metricTooltips : {});
    var text = tips[metricKey] || '';
    if (!text) {
      if (metricKey.indexOf('field_accuracy_') === 0) {
        var field = metricKey.slice(15).replace(/_/g, ' ');
        text = 'JSON subset match accuracy for the \\u201c' + field + '\\u201d field.';
      } else if (metricKey.indexOf('rule_') === 0 && metricKey.lastIndexOf('_pass_rate') === metricKey.length - 10) {
        var ruleType = metricKey.slice(5, metricKey.length - 10).replace(/_/g, ' ');
        text = 'Fraction of \\u201c' + ruleType + '\\u201d rules that pass.';
      } else if (metricKey.indexOf('f1_') === 0) {
        var cls = metricKey.slice(3).replace(/_/g, ' ');
        text = 'F1 score for the \\u201c' + cls + '\\u201d layout class: 2\\u00d7P\\u00d7R/(P+R).';
      } else if (metricKey.indexOf('precision_') === 0) {
        var cls2 = metricKey.slice(10).replace(/_/g, ' ');
        text = 'Precision for the \\u201c' + cls2 + '\\u201d layout class: TP/(TP+FP).';
      } else if (metricKey.indexOf('recall_') === 0) {
        var cls3 = metricKey.slice(7).replace(/_/g, ' ');
        text = 'Recall for the \\u201c' + cls3 + '\\u201d layout class: TP/(TP+FN).';
      }
    }
    return text;
  }
  function _showTip(icon) {
    clearTimeout(_tipHideTimer);
    var key = icon.getAttribute('data-metric');
    var text = _getTooltipText(key);
    if (!text) return;
    var tip = _ensureTip();
    tip.textContent = text;
    // Reset: position offscreen to measure, remove classes
    tip.className = '';
    tip.style.cssText = 'position:fixed;display:block;visibility:hidden;top:-9999px;left:-9999px';
    // Measure after text is set
    var iconRect = icon.getBoundingClientRect();
    var tipW = tip.offsetWidth;
    var tipH = tip.offsetHeight;
    var gap = 8;
    // Default: above
    var top = iconRect.top - tipH - gap;
    var arrowDir = 'arrow-bottom';
    // If not enough room above, go below
    if (top < 4) {
      top = iconRect.bottom + gap;
      arrowDir = 'arrow-top';
    }
    // Horizontal: center on icon, but clamp to viewport
    var left = iconRect.left + iconRect.width / 2 - tipW / 2;
    var maxLeft = window.innerWidth - tipW - 8;
    if (left < 8) left = 8;
    if (left > maxLeft) left = maxLeft;
    // Arrow position relative to tooltip
    var arrowLeft = (iconRect.left + iconRect.width / 2 - left);
    arrowLeft = Math.max(12, Math.min(arrowLeft, tipW - 12));
    // Apply final position — clear inline styles so CSS classes take effect
    tip.style.cssText = '';
    tip.style.top = top + 'px';
    tip.style.left = left + 'px';
    tip.style.setProperty('--arrow-left', arrowLeft + 'px');
    tip.className = arrowDir + ' visible';
  }
  function _hideTip() {
    _tipHideTimer = setTimeout(function() {
      if (_tipEl) { _tipEl.className = ''; }
    }, 80);
  }
  // Attach listeners via event delegation (mouseover/mouseout bubble, mouseenter/mouseleave do NOT)
  var _currentHint = null;
  document.addEventListener('mouseover', function(e) {
    var icon = e.target.closest ? e.target.closest('.metric-hint') : null;
    if (icon && icon !== _currentHint) {
      _currentHint = icon;
      _showTip(icon);
    } else if (!icon && _currentHint) {
      _currentHint = null;
      _hideTip();
    }
  });
  document.addEventListener('mouseout', function(e) {
    if (!_currentHint) return;
    var related = e.relatedTarget;
    if (!related || !(related.closest && related.closest('.metric-hint') === _currentHint)) {
      _currentHint = null;
      _hideTip();
    }
  });

  function tooltipIcon(metricKey) {
    var text = _getTooltipText(metricKey);
    if (!text) return '';
    return '<span class="metric-hint" data-metric="' + esc(metricKey) + '"></span>';
  }
"""
