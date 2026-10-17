"""Deterministic chart-value refinement from vector geometry.

The VLM proposes a chart's data table with *estimated* values. When the chart
is vector art, the PDF already contains the exact bar extents and line
vertices, and the numeric tick labels give the axis scale. This module
calibrates value axes (linear fit over aligned numeric tick labels, rejected
unless the fit is near-exact), turns bar/segment/vertex geometry into
candidate values, and snaps each proposed value to a candidate only when the
match is close and unambiguous. No calibration, no change: the proposer's
value stands.
"""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"^[−\-–(]?\$?€?£?(\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\.(\d+))?%?\)?[kKmMbB]?$")


def _parse_num(t: str) -> float | None:
    t = t.strip().replace("−", "-")
    if not _NUM_RE.match(t):
        return None
    neg = t.startswith(("-", "–", "(", "−"))
    core = re.sub(r"[^\d.,]", "", t).replace(",", "")
    try:
        v = float(core)
    except ValueError:
        return None
    return -v if neg else v


def _fit(ps: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Least-squares v = a*p + b; returns (a, b, max relative residual vs span)."""
    n = len(ps)
    if n < 3:
        return None
    mp = sum(p for p, _ in ps) / n
    mv = sum(v for _, v in ps) / n
    sxx = sum((p - mp) ** 2 for p, _ in ps)
    if sxx <= 0:
        return None
    a = sum((p - mp) * (v - mv) for p, v in ps) / sxx
    b = mv - a * mp
    span = max(v for _, v in ps) - min(v for _, v in ps)
    if span <= 0 or a == 0:
        return None
    res = max(abs(a * p + b - v) for p, v in ps) / span
    return a, b, res


def calibrate_axes(page: Any) -> list[dict[str, Any]]:
    """Find value axes: >=3 numeric labels aligned on a shared x (vertical axis) or y (horizontal axis)."""
    words = page.get_text("words")
    nums = []
    for x0, y0, x1, y1, w, *_ in words:
        v = _parse_num(w)
        if v is not None:
            nums.append({"v": v, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    axes: list[dict[str, Any]] = []
    used: set[int] = set()
    # vertical axes: labels right-aligned (shared x1) with distinct y
    for key, pos, orient in (("x1", "cy", "v"), ("cy", "cx", "h")):
        groups: dict[int, list[int]] = {}
        for i, n in enumerate(nums):
            groups.setdefault(round(n[key] / 3), []).append(i)
        for _, idxs in sorted(groups.items()):
            idxs = [i for i in idxs if i not in used]
            if len(idxs) < 3:
                continue
            idxs.sort(key=lambda i: nums[i][pos])
            ps = [(nums[i][pos], nums[i]["v"]) for i in idxs]
            # tick labels are evenly spaced with evenly stepped values
            steps = [b[1] - a[1] for a, b in zip(ps, ps[1:], strict=False)]
            if len({round(s, 6) for s in steps}) > max(1, len(steps) // 3):
                continue
            f = _fit(ps)
            if f is None or f[2] > 0.01:
                continue
            a, b, _ = f
            lo = min(nums[i][pos] for i in idxs)
            hi = max(nums[i][pos] for i in idxs)
            ext = [
                min(nums[i]["x0"] for i in idxs),
                min(nums[i]["y0"] for i in idxs),
                max(nums[i]["x1"] for i in idxs),
                max(nums[i]["y1"] for i in idxs),
            ]
            axes.append(
                {
                    "orient": orient,
                    "a": a,
                    "b": b,
                    "lo": lo,
                    "hi": hi,
                    "ext": ext,
                    "pct": any("%" in w[4] for w in words if _parse_num(w[4]) is not None),
                }
            )
            used.update(idxs)
    return axes


def candidate_values(page: Any, axes: list[dict[str, Any]]) -> list[float]:
    """Values implied by filled rects (bars, stacked segments) and path vertices, per calibrated axis."""
    if not axes:
        return []
    cands: list[float] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for ax in axes:
        a, b = ax["a"], ax["b"]
        span = abs(ax["hi"] - ax["lo"])
        pad = span * 0.08 + 4
        for d in drawings:
            r = d["rect"]
            if ax["orient"] == "v":
                # plot lies to the right of the labels, within their vertical range
                if r.x0 < ax["ext"][0] - 2 or r.y0 < ax["lo"] - pad or r.y1 > ax["hi"] + pad:
                    continue
                if d.get("fill") is not None and r.width > 1.5 and r.height > 0.3:
                    cands += [a * r.y0 + b, a * r.y1 + b, abs(a) * r.height]
                for it in d.get("items", []):
                    if it[0] == "l":
                        for p in (it[1], it[2]):
                            if ax["lo"] - pad <= p.y <= ax["hi"] + pad:
                                cands.append(a * p.y + b)
            else:
                if r.y1 > ax["ext"][3] + 2 or r.x0 < ax["lo"] - pad or r.x1 > ax["hi"] + pad:
                    continue
                if d.get("fill") is not None and r.height > 1.5 and r.width > 0.3:
                    cands += [a * r.x0 + b, a * r.x1 + b, abs(a) * r.width]
                for it in d.get("items", []):
                    if it[0] == "l":
                        for p in (it[1], it[2]):
                            if ax["lo"] - pad <= p.x <= ax["hi"] + pad:
                                cands.append(a * p.x + b)
    return cands


_CELL_NUM_RE = re.compile(r"(<td[^>]*>)\s*(-?\d+(?:\.\d+)?)\s*(</td>)")


def _nice(v: float, like: str) -> str:
    dec = len(like.split(".")[1]) if "." in like else 0
    dec = max(dec, 1 if abs(v) < 10 else 0)
    return f"{v:.{dec}f}".rstrip("0").rstrip(".") if dec else f"{round(v)}"


def refine_table_values(html: str, cands: list[float], printed: set[float], max_rel: float = 0.10) -> tuple[str, int]:
    """Snap proposed numbers to geometry candidates when close and unambiguous.

    Values that exactly match a number printed on the page are never touched
    (a printed data label beats geometry).
    """
    if not cands:
        return html, 0
    n = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal n
        s = m.group(2)
        v = float(s)
        if v in printed or v == 0:
            return m.group(0)
        near = sorted(cands, key=lambda c: abs(c - v))
        best = near[0]
        rel = abs(best - v) / max(abs(v), 1e-9)
        if rel > max_rel or rel < 0.002:
            return m.group(0)
        # ambiguity: a clearly different candidate almost as close -> leave it
        for c in near[1:6]:
            if abs(c - best) / max(abs(best), 1e-9) > 0.02 and abs(c - v) < 1.5 * abs(best - v):
                return m.group(0)
        n += 1
        return f"{m.group(1)}{_nice(best, s)}{m.group(3)}"

    return _CELL_NUM_RE.sub(sub, html), n


def printed_numbers(page: Any) -> set[float]:
    out = set()
    for w in page.get_text("words"):
        v = _parse_num(w[4])
        if v is not None:
            out.add(v)
    return out
