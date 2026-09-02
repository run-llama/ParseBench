"""Cross-page TABLE formatting-consistency metric (parse).

Scores whether a parser formats tables consistently across pages and segments
them correctly. Reads GT from the test case's
``metadata["cross_page_table_consistency"]`` (clusters -> per-member page+header,
the per-page-fair expected table count, and the logical count); operates on the
parser's own per-page tables from ``ParseOutput``.

Metrics (all 0..1, higher better):
  - cross_page_table_col_consistency        : fraction of multi-page clusters whose
                                              tables agree on column count
  - cross_page_table_header_row_consistency : same, for header-row count
  - cross_page_table_header_similarity       : mean pairwise header similarity in clusters
  - cross_page_table_count_accuracy          : 1 - |parser_tables - expected| / max(...),
                                              where `expected` is the per-page-fair
                                              count an ideal per-page parser can emit.

This is table-scoped (vs. future cross_page_section_* metrics).
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from bs4 import BeautifulSoup

from parse_bench.schemas.evaluation import MetricValue

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip()).lower()


def _row_width(tr: Any) -> int:
    w = 0
    for c in tr.find_all(["td", "th"], recursive=False):
        try:
            w += int(c.get("colspan", 1) or 1)
        except (TypeError, ValueError):
            w += 1
    return w


def _describe(tbl: Any) -> tuple[int, int, str]:
    rows = tbl.find_all("tr")
    thead = tbl.find("thead")
    if thead is not None:
        hrows = thead.find_all("tr", recursive=False)
    else:
        hrows = []
        for tr in rows:
            cells = tr.find_all(["td", "th"], recursive=False)
            if cells and all(c.name == "th" for c in cells):
                hrows.append(tr)
            else:
                break
    header_ids = {id(r) for r in hrows}
    body = [r for r in rows if id(r) not in header_ids]
    widths = [w for w in (_row_width(r) for r in (body or rows)) if w > 0]
    n_cols = 0
    if widths:
        c = Counter(widths)
        n_cols = max(c, key=lambda w: (c[w], w))
    header = " | ".join(
        _norm(c.get_text(" ", strip=True)) for tr in hrows for c in tr.find_all(["td", "th"], recursive=False)
    )
    return n_cols, len(hrows), header


def _tables_by_page(parse_output: Any) -> dict[int, list[tuple[int, int, str]]]:
    out: dict[int, list[tuple[int, int, str]]] = {}
    for i, page in enumerate(getattr(parse_output, "pages", []) or []):
        pg = int(getattr(page, "page_index", i)) + 1
        md = getattr(page, "markdown", "") or ""
        out[pg] = [_describe(t) for t in BeautifulSoup(md, "lxml").find_all("table")]
    return out


def _select_cluster_tables(
    members: list[Any], by_page: dict[int, list[tuple[int, int, str]]]
) -> list[tuple[int, int, str]]:
    """Pick one parser table per cluster member — the best header match on that
    member's page.

    We do NOT drop poor matches: a parser that mangles a recurring table's header
    is exactly what this metric must catch, so the mangled table stays in and its
    low similarity / differing structure shows the inconsistency. A member is
    skipped only when the parser produced no tables at all on its page. Legacy GT
    (members are page numbers) pools every table on those pages.
    """
    selected: list[tuple[int, int, str]] = []
    for m in members:
        if isinstance(m, dict):
            cands = by_page.get(int(m["page"]), [])
            if not cands:
                continue
            gt_header = str(m.get("header", ""))
            selected.append(max(cands, key=lambda t: SequenceMatcher(None, t[2], gt_header).ratio()))
        else:
            selected.extend(by_page.get(int(m), []))
    return selected


def compute_cross_page_table_metrics(parse_output: Any, gt: dict[str, Any]) -> list[MetricValue]:
    by_page = _tables_by_page(parse_output)
    # Count parser tables only on the GT's annotated pages.
    pages_of_interest = {int(p) for p in (gt.get("logical_tables_per_page") or {})}
    if pages_of_interest:
        total = sum(len(by_page.get(p, [])) for p in pages_of_interest)
    else:
        total = sum(len(v) for v in by_page.values())
    # Per-page-fair target: an ideal *per-page* parser cannot merge tables across a
    # page break, and side-by-side fragments are naturally separate, so the
    # achievable count is the de-segmented total plus the splits the GT merged
    # back (logical + merge_fragments). Falls back to the logical count.
    expected = int(gt.get("expected_page_table_count") or gt.get("logical_table_count") or 0)
    count_acc = 1.0 - abs(total - expected) / max(total, expected, 1)

    clusters = gt.get("clusters") or {}
    col_ok = hrow_ok = n_multi = 0
    sims: list[float] = []
    details: list[str] = []
    for name, members in clusters.items():
        tbls = _select_cluster_tables(members, by_page)
        if len(tbls) < 2:
            continue
        n_multi += 1
        cols = {t[0] for t in tbls}
        hrows = {t[1] for t in tbls}
        col_ok += len(cols) == 1
        hrow_ok += len(hrows) == 1
        hdrs = [t[2] for t in tbls]
        ratios = [
            SequenceMatcher(None, hdrs[i], hdrs[j]).ratio() for i in range(len(hdrs)) for j in range(i + 1, len(hdrs))
        ]
        sim = sum(ratios) / len(ratios) if ratios else 1.0
        sims.append(sim)
        if len(cols) > 1 or len(hrows) > 1:
            details.append(f"{name}: cols={sorted(cols)} hrows={sorted(hrows)} sim={sim:.2f}")

    md = {"parser_tables": total, "expected_page_tables": expected, "n_multi_clusters": n_multi}
    return [
        MetricValue(
            metric_name="cross_page_table_col_consistency",
            value=col_ok / n_multi if n_multi else 1.0,
            metadata=md,
            details=details,
        ),
        MetricValue(
            metric_name="cross_page_table_header_row_consistency",
            value=hrow_ok / n_multi if n_multi else 1.0,
            metadata=md,
        ),
        MetricValue(
            metric_name="cross_page_table_header_similarity",
            value=sum(sims) / len(sims) if sims else 1.0,
            metadata=md,
        ),
        MetricValue(metric_name="cross_page_table_count_accuracy", value=count_acc, metadata=md),
    ]
