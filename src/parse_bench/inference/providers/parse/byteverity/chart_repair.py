"""Chart residual measurement shared by the live hybrid repair and offline replays."""

from __future__ import annotations

import re
from typing import Any

from . import pixel

NUM_RE = re.compile(r"^[−\-(]?\d[\d.,]*%?\)?$")


def band(r: float | None) -> str:
    return "no_table" if r is None else ("low" if r < 0.05 else "mid" if r < 0.15 else "high")


def residual(page: Any, items: list[dict[str, Any]]) -> tuple[float | None, list[str]]:
    """Worst chart residual on the page; None = a numeric figure with no data table."""
    allt = " ".join(i.get("text", "") for i in items)
    worst, missing = 0.0, []
    for it in items:
        if (it.get("label") or "").lower() not in ("picture", "figure") or not isinstance(it.get("bbox"), list):
            continue
        if "<table" in it.get("text", ""):
            r, miss = pixel.text_residual(page, it["bbox"], allt)
            if r > worst:
                worst, missing = r, miss
        else:
            W, H = page.rect.width, page.rect.height
            x0, y0, x1, y1 = [v / 1000 for v in it["bbox"]]
            nums = [
                w[4]
                for w in page.get_text("words")
                if x0 * W <= (w[0] + w[2]) / 2 <= x1 * W
                and y0 * H <= (w[1] + w[3]) / 2 <= y1 * H
                and NUM_RE.match(w[4])
            ]
            if len(nums) >= 3:
                return None, nums[:12]
    return worst, missing


def repair_note(missing: list[str]) -> str:
    return (
        "\n\nIMPORTANT — a previous transcription of this page left printed chart text unexplained: "
        + ", ".join(repr(m) for m in missing)
        + ". Every chart/graph on the page must get its OWN data table inside its Picture div, "
        "preceded by that chart's title. Every legend entry is a series column; every axis "
        "category is a row. Use printed data labels exactly."
    )
