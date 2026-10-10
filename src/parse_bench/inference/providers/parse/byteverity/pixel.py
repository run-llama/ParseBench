"""Pixel completeness for chart proposals (analysis-by-synthesis residual).

The proposer's chart table implies how many series are drawn. The page
raster says how many distinct series inks are actually present in the chart
region. More inks than series is a *residual*: ink the proposal does not
explain (a missed series, a second chart). The residual is a fact for the
sealed oracle, never a correction by itself.
"""

from __future__ import annotations

import re
from typing import Any

_TH_ROW_RE = re.compile(r"<tr>([\s\S]*?)</tr>", re.IGNORECASE)
_CELL_RE = re.compile(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", re.IGNORECASE)


def table_shape(html: str) -> tuple[int, int]:
    """(series columns, category rows) of a chart table: columns after the label column."""
    rows = _TH_ROW_RE.findall(html)
    if not rows:
        return 0, 0
    cols = max(len(_CELL_RE.findall(r)) for r in rows)
    return max(0, cols - 1), max(0, len(rows) - 1)


def series_inks(
    page: Any, bbox_1000: list[float], dpi: int = 72, min_share: float = 0.006
) -> list[tuple[int, int, int]]:
    """Distinct non-background, non-text inks covering >= min_share of the chart region."""
    import numpy as np

    W, H = page.rect.width, page.rect.height
    x0, y0, x1, y1 = bbox_1000
    import pymupdf

    clip = pymupdf.Rect(x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H) & page.rect
    if clip.is_empty or clip.width < 20 or clip.height < 20:
        return []
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    px = a.reshape(-1, 3)
    mx, mn = px.max(1), px.min(1)
    keep = (mn < 235) & (mx > 45)  # drop paper white and text black
    px = px[keep]
    if len(px) == 0:
        return []
    q = (px // 24) * 24 + 12  # quantize
    keys, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)
    total = a.shape[0] * a.shape[1]
    clusters: list[tuple[np.ndarray, int]] = []
    for i in order:
        c = keys[i]
        n = int(counts[i])
        for j, (cc, cn) in enumerate(clusters):
            if np.abs(cc - c).max() <= 48:
                clusters[j] = (cc, cn + n)
                break
        else:
            clusters.append((c, n))
    return [tuple(int(v) for v in c) for c, n in clusters if n / total >= min_share]


def residual_class(n_inks: int, n_series: int) -> str:
    """explained / extra_ink / missing_ink — does the table account for the drawn series?"""
    if n_series == 0:
        return "no_table"
    if n_inks > n_series + 1:  # +1: one neutral ink (axes/gridlines/background band) is normal
        return "extra_ink"
    if n_inks < n_series and n_series > 2:
        return "missing_ink"
    return "explained"


def text_residual(page: Any, bbox_1000: list[float], proposal_text: str) -> tuple[float, list[str]]:
    """Share of the chart region's printed tokens (labels, legend, data labels) absent from the proposal."""
    from .floor import norm_tokens

    W, H = page.rect.width, page.rect.height
    x0, y0, x1, y1 = (
        bbox_1000[0] / 1000 * W,
        bbox_1000[1] / 1000 * H,
        bbox_1000[2] / 1000 * W,
        bbox_1000[3] / 1000 * H,
    )
    have = set(norm_tokens(proposal_text))
    toks: list[str] = []
    for wx0, wy0, wx1, wy1, w, *_ in page.get_text("words"):
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            toks += [t for t in norm_tokens(w) if len(t) >= 2 or t.isdigit()]
    if len(toks) < 5:
        return 0.0, []
    miss = [t for t in toks if t not in have]
    return len(miss) / len(toks), miss[:12]
