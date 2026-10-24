"""Render chart2json data as HTML tables for the existing chart rules."""

from __future__ import annotations

import json
import math
import re
from html import escape
from typing import Any


def _json_objects(text: str) -> list[dict]:
    """Decode fenced or prefixed JSON, skipping malformed blocks as a whole."""
    if not isinstance(text, str) or not text.strip():
        return []
    blocks = re.findall(r"```(?:json)?[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    objects: list[dict] = []
    if blocks:
        for block in blocks:
            try:
                obj = json.loads(block)
            except (ValueError, RecursionError):
                continue
            if isinstance(obj, dict):
                objects.append(obj)
        return objects

    start = text.find("{")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except (ValueError, RecursionError):
        return []
    return [obj] if isinstance(obj, dict) else []


def _is_value(value: Any) -> bool:
    return (
        isinstance(value, (str, int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _fmt(value: Any) -> str:
    if not _is_value(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _title(value: Any) -> str:
    if not isinstance(value, str) or value.strip().lower() in {"", "none", "null", "n/a"}:
        return ""
    return value.strip()


def _series_points(series: Any) -> dict[str, Any]:
    if isinstance(series, dict):
        return {str(key): value for key, value in series.items() if _is_value(value) or value is None}
    return {}


def _render_table(headers: list[str], rows: list[list[str]], titles: list[str]) -> str:
    if not rows:
        return ""
    caption_text = " / ".join(title for title in titles if title)
    caption = f"<caption>{escape(caption_text)}</caption>" if caption_text else ""
    head = "".join(f"<th>{escape(cell)}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    # Encode pipes so the Markdown parser does not also interpret these HTML rows as a table.
    return f"<table>{caption}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>".replace("|", "&#124;")


def _panel_to_table(titles: list[str], values: dict[str, Any]) -> str:
    if not values:
        return ""
    if all(_is_value(value) or value is None for value in values.values()):
        return _render_table(["", ""], [[str(key), _fmt(value)] for key, value in values.items()], titles)

    tables = []
    series = {}
    for name, data in values.items():
        if isinstance(data, list):
            rows = [
                [_fmt(point["x"]), _fmt(point["y"])]
                for point in data
                if isinstance(point, dict) and _is_value(point.get("x")) and _is_value(point.get("y"))
            ]
            tables.append(_render_table(["x", "y"], rows, [*titles, str(name)]))
        elif points := _series_points(data):
            series[str(name)] = points
    if series:
        categories = list(dict.fromkeys(category for points in series.values() for category in points))
        rows = [[category, *(_fmt(points.get(category)) for points in series.values())] for category in categories]
        tables.append(_render_table(["", *series], rows, titles))
    return "\n\n".join(table for table in tables if table)


def chart_json_to_html(chart: dict) -> str:
    """Render each panel separately, preserving its title, series and categories."""
    if not isinstance(chart, dict):
        return ""
    figure_title = _title(chart.get("title"))
    additional_info = _title(chart.get("additional_info"))
    figure_context = [figure_title, additional_info]
    tables: list[str] = []
    panels = chart.get("panels")
    if isinstance(panels, dict) and panels:
        for name, panel in panels.items():
            if not isinstance(panel, dict):
                continue
            values = panel.get("values") or panel.get("series")
            if isinstance(values, dict):
                # Each panel retains the figure context in its caption.
                tables.append(_panel_to_table([*figure_context, _title(name)], values))
    else:
        values = chart.get("values")
        if isinstance(values, dict) and values:
            nested_panels = all(
                isinstance(panel, dict) and panel and all(isinstance(series, (dict, list)) for series in panel.values())
                for panel in values.values()
            )
            if nested_panels:
                for name, panel in values.items():
                    tables.append(_panel_to_table([*figure_context, _title(name)], panel))
            else:
                tables.append(_panel_to_table(figure_context, values))
    return "\n\n".join(table for table in tables if table)


def chart_description_to_html(description: str) -> str:
    """Convert all complete chart JSON blocks in one figure description."""
    tables = [chart_json_to_html(obj) for obj in _json_objects(description)]
    return "\n\n".join(table for table in tables if table)
