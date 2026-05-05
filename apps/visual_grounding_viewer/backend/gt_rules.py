from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast

from dateutil import parser as date_parser
from rapidfuzz.distance import JaroWinkler

from .models import GroundingBbox, GroundingPage, GroundTruthRuleMatch

_FIELD_GROUPING_TOUCH_MARGIN = 0.005
_FIELD_TEXT_PASS_THRESHOLD = 0.9
_FIELD_STRING_PASS_THRESHOLD = 0.9
_FIELD_NUMERIC_ABSOLUTE_TOLERANCE = 1e-6
_FIELD_NUMERIC_RELATIVE_TOLERANCE = 1e-6

_IGNORED_INVISIBLE_CODEPOINTS = {
    0x00AD,  # soft hyphen
    0x200B,  # zero width space
    0x2060,  # word joiner
    0xFEFF,  # zero width no-break space / BOM
}
_FIELD_TRUE_STRINGS = frozenset({"true", "yes", "y", "1", "checked"})
_FIELD_FALSE_STRINGS = frozenset({"false", "no", "n", "0", "unchecked"})
_FIELD_DATE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    re.compile(r"\d{1,2}-\d{1,2}-\d{2,4}"),
    re.compile(r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"),
    re.compile(r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"),
)
_FIELD_PATH_SEGMENT_RE = re.compile(r"([^.\[]+)(?:\[(\d+)\])?")
_FIELD_NAME_DATE_TOKEN_RE = re.compile(r"(?:^|_)date(?:$|_)")
_DESCRIPTION_DATE_TOKEN_RE = re.compile(r"\bdate\b")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_EVALUATION_REPORT_CACHE: dict[Path, tuple[int, int, dict[str, dict[str, Any]]]] = {}
_MISSING_FIELD_VALUE = object()


@dataclass(frozen=True)
class _FieldValueMatch:
    score: float
    passed: bool
    reason: str
    mode: str


@dataclass(frozen=True)
class _SupportUnit:
    unit_id: str
    granularity: Literal["line", "word"]
    order_index: int | None
    text: str
    bbox_page_xyxy: tuple[float, float, float, float]
    bbox_page_xywh: GroundingBbox


@dataclass(frozen=True)
class _FieldGroupMatch:
    unit_ids: tuple[str, ...]
    granularity: Literal["line", "word"]
    component_bboxes: tuple[GroundingBbox, ...]
    bbox_page_xyxy: tuple[float, float, float, float]
    text: str
    iou: float
    bbox_recall: float
    text_score: float


@dataclass(frozen=True)
class _FieldCitationMatch:
    item_id: str
    component_bboxes: tuple[GroundingBbox, ...]
    bbox_page_xyxy: tuple[float, float, float, float]
    text: str | None
    iou: float
    bbox_recall: float
    text_score: float
    value_match: _FieldValueMatch


def normalize_granular_text(text: str | None) -> str:
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized_chars: list[str] = []
    for char in normalized:
        if ord(char) in _IGNORED_INVISIBLE_CODEPOINTS:
            continue
        if unicodedata.category(char) == "Cc":
            continue
        normalized_chars.append(" " if char.isspace() else char)

    normalized = "".join(normalized_chars)
    normalized = " ".join(normalized.split())
    return normalized.casefold().strip()


def normalize_field_string_for_jaro(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).lower().strip()


def _field_path_array_index_and_leaf(field_path: str | None) -> tuple[int | None, str | None]:
    if not field_path:
        return None, None

    row_index: int | None = None
    leaf_name: str | None = None
    for match in _FIELD_PATH_SEGMENT_RE.finditer(field_path):
        leaf_name = match.group(1)
        index = match.group(2)
        if row_index is None and index is not None:
            try:
                row_index = int(index)
            except ValueError:
                row_index = None
    return row_index, leaf_name


def _parse_field_path_tokens(field_path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for segment in field_path.split("."):
        if not segment:
            continue
        cursor = 0
        name_buffer: list[str] = []
        while cursor < len(segment):
            char = segment[cursor]
            if char != "[":
                name_buffer.append(char)
                cursor += 1
                continue

            if name_buffer:
                tokens.append("".join(name_buffer))
                name_buffer = []

            close_index = segment.find("]", cursor)
            if close_index < 0:
                name_buffer.append(segment[cursor:])
                break

            index_text = segment[cursor + 1 : close_index]
            try:
                tokens.append(int(index_text))
            except ValueError:
                tokens.append(index_text)
            cursor = close_index + 1

        if name_buffer:
            tokens.append("".join(name_buffer))
    return tokens


def _result_extracted_data(result_payload: dict[str, Any] | None) -> Any:
    if not isinstance(result_payload, dict):
        return None

    output = result_payload.get("output")
    if isinstance(output, dict):
        extracted_data = output.get("extracted_data")
        if extracted_data is not None:
            return extracted_data
        data = output.get("data")
        if data is not None:
            return data

    extracted_data = result_payload.get("extracted_data")
    if extracted_data is not None:
        return extracted_data
    return result_payload.get("data")


def _result_field_value(result_payload: dict[str, Any] | None, field_path: str) -> Any:
    current = _result_extracted_data(result_payload)
    for token in _parse_field_path_tokens(field_path):
        if isinstance(token, int):
            if not isinstance(current, list) or token < 0 or token >= len(current):
                return _MISSING_FIELD_VALUE
            current = current[token]
            continue
        if not isinstance(current, dict) or token not in current:
            return _MISSING_FIELD_VALUE
        current = current[token]
    return current


def _field_value_to_prediction_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _field_path_from_item(item: Any) -> str | None:
    raw_payload = getattr(item, "raw_payload", None)
    if not isinstance(raw_payload, dict):
        return None
    field_path = raw_payload.get("field_path")
    return field_path if isinstance(field_path, str) and field_path else None


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _markdown_cell_to_text(cell: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", cell, flags=re.IGNORECASE)
    text = text.replace("\\_", "_")
    text = text.replace("\\|", "|")
    text = re.sub(r"[*`]+", "", text)
    return " ".join(text.split()).strip()


def _is_markdown_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(_MARKDOWN_TABLE_SEPARATOR_RE.match(cell.strip()) for cell in cells)


def _header_field_score(header: str, leaf_name: str) -> tuple[int, int, int]:
    header_tokens = set(re.findall(r"[a-z0-9]+", _markdown_cell_to_text(header).lower()))
    field_tokens = re.findall(r"[a-z0-9]+", leaf_name.lower())
    aliases = {
        "employee": ("employee", "emp"),
        "number": ("number", "no", "num"),
    }

    matched = 0
    for token in field_tokens:
        candidates = aliases.get(token, (token,))
        if any(candidate in header_tokens for candidate in candidates):
            matched += 1

    normalized_header = "_".join(re.findall(r"[a-z0-9]+", _markdown_cell_to_text(header).lower()))
    contiguous_hint = 1 if leaf_name.lower() in normalized_header else 0
    return matched, contiguous_hint, -abs(len(header_tokens) - len(field_tokens))


def _extract_field_text_from_markdown_table(markdown: str, field_path: str | None) -> str | None:
    row_index, leaf_name = _field_path_array_index_and_leaf(field_path)
    if row_index is None or not leaf_name:
        return None

    rows = [_split_markdown_table_row(line) for line in markdown.splitlines() if "|" in line]
    rows = [row for row in rows if row and not _is_markdown_separator_row(row)]
    if len(rows) < 2:
        return None

    header = rows[0]
    data_rows = rows[1:]
    if row_index < 0 or row_index >= len(data_rows):
        return None

    scored_headers = [(_header_field_score(cell, leaf_name), index) for index, cell in enumerate(header)]
    best_score, best_index = max(scored_headers, key=lambda item: item[0])
    if best_score[0] <= 0 or best_index >= len(data_rows[row_index]):
        return None

    cell_text = _markdown_cell_to_text(data_rows[row_index][best_index])
    return cell_text or None


def _normalize_schema_type(raw_type: Any, schema_node: dict[str, Any]) -> str | None:
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), raw_type[0] if raw_type else None)
    if not isinstance(raw_type, str):
        return None
    if raw_type == "string":
        field_name = str(schema_node.get("_field_name", "")).lower()
        description = str(schema_node.get("description", "")).lower()
        field_format = str(schema_node.get("format", "")).lower()
        if field_format in {"date", "date-time"}:
            return "date"
        if _FIELD_NAME_DATE_TOKEN_RE.search(field_name) or _DESCRIPTION_DATE_TOKEN_RE.search(description):
            return "date"
    return raw_type


def _resolve_field_schema_type(data_schema: dict[str, Any] | None, field_path: str) -> str | None:
    if not data_schema:
        return None

    current: Any = data_schema
    for segment, _index in _FIELD_PATH_SEGMENT_RE.findall(field_path):
        if not isinstance(current, dict):
            return None
        properties = current.get("properties")
        if not isinstance(properties, dict) or segment not in properties:
            return None
        current = dict(properties[segment])
        current["_field_name"] = segment
        raw_type = current.get("type")
        if isinstance(raw_type, list):
            raw_type = next((item for item in raw_type if item != "null"), raw_type[0] if raw_type else None)
        if raw_type == "array":
            current = current.get("items")

    if not isinstance(current, dict):
        return None
    return _normalize_schema_type(current.get("type"), current)


def compare_field_value(
    expected: str | int | float | bool | None,
    actual: str | None,
    *,
    field_type: str | None = None,
) -> _FieldValueMatch:
    normalized_field_type = (field_type or "").lower()

    if expected is None:
        actual_norm = normalize_granular_text(actual)
        passed = actual_norm == ""
        return _FieldValueMatch(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="pass" if passed else "expected_null_but_found_text",
            mode="null_exact_match",
        )

    if normalized_field_type == "boolean" or isinstance(expected, bool):
        actual_bool = _parse_field_bool(actual)
        expected_bool = expected if isinstance(expected, bool) else _parse_field_bool(str(expected))
        passed = actual_bool is not None and expected_bool is not None and actual_bool is expected_bool
        return _FieldValueMatch(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="pass" if passed else "boolean_exact_mismatch",
            mode="boolean_exact_match",
        )

    if normalized_field_type == "integer" or (isinstance(expected, int) and not isinstance(expected, bool)):
        actual_number = _parse_field_number(actual)
        expected_int = (
            expected
            if isinstance(expected, int) and not isinstance(expected, bool)
            else _parse_field_number(str(expected))
        )
        passes_integer = expected_int is not None and actual_number is not None and _is_integer_like(actual_number)
        expected_int_value = int(round(float(expected_int))) if expected_int is not None else 0
        actual_int_value = int(round(actual_number)) if actual_number is not None else 0
        passed = bool(passes_integer and actual_int_value == expected_int_value)
        return _FieldValueMatch(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="pass" if passed else "integer_exact_mismatch",
            mode="integer_exact_match",
        )

    if normalized_field_type == "number" or isinstance(expected, float):
        actual_number = _parse_field_number(actual)
        expected_number = (
            float(expected)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool)
            else _parse_field_number(str(expected))
        )
        passed = actual_number is not None and math.isclose(
            actual_number,
            float(expected_number) if expected_number is not None else math.inf,
            rel_tol=_FIELD_NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=_FIELD_NUMERIC_ABSOLUTE_TOLERANCE,
        )
        return _FieldValueMatch(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="pass" if passed else "numeric_tolerance_mismatch",
            mode="numeric_tolerance_match",
        )

    if normalized_field_type == "date" or isinstance(expected, (date, datetime)):
        actual_date = _parse_field_date(actual)
        if isinstance(expected, datetime):
            expected_date = expected.date()
        elif isinstance(expected, date):
            expected_date = expected
        else:
            expected_date = _parse_field_date(str(expected))
        passed = actual_date is not None and actual_date == expected_date
        return _FieldValueMatch(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="pass" if passed else "date_ymd_mismatch",
            mode="date_ymd_match",
        )

    expected_norm = normalize_field_string_for_jaro(str(expected))
    actual_norm = normalize_field_string_for_jaro(actual)
    score = float(JaroWinkler.normalized_similarity(expected_norm, actual_norm))
    passed = score >= _FIELD_STRING_PASS_THRESHOLD
    return _FieldValueMatch(
        score=score,
        passed=passed,
        reason="pass" if passed else "jaro_winkler_below_threshold",
        mode="jaro_winkler_normalized_string",
    )


def _parse_field_bool(value: str | None) -> bool | None:
    normalized = normalize_granular_text(value)
    if normalized in _FIELD_TRUE_STRINGS:
        return True
    if normalized in _FIELD_FALSE_STRINGS:
        return False
    return None


def _is_integer_like(value: float) -> bool:
    return math.isclose(value, round(value), abs_tol=_FIELD_NUMERIC_ABSOLUTE_TOLERANCE)


def _parse_field_number(value: str | int | float | bool | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    normalized = normalize_granular_text(value)
    if not normalized:
        return None

    negative = False
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
        negative = True

    normalized = re.sub(r"^[~≈]", "", normalized).strip()
    normalized = re.sub(r"^[$€£¥₹]\s*", "", normalized)
    normalized = re.sub(r"\s*[$€£¥₹]$", "", normalized)
    normalized = normalized.rstrip("%")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace(" ", "")

    multiplier = 1.0
    suffix_patterns = (
        (r"(?i)(trillion|trill|trn)$", 1e12),
        (r"(?i)(billion|bill|bln)$", 1e9),
        (r"(?i)(million|mill|mln)$", 1e6),
        (r"(?i)t$", 1e12),
        (r"(?i)g$", 1e9),
        (r"(?i)b$", 1e9),
        (r"(?i)m$", 1e6),
        (r"(?i)k$", 1e3),
    )
    for pattern, pattern_multiplier in suffix_patterns:
        if re.search(pattern, normalized):
            normalized = re.sub(pattern, "", normalized)
            multiplier = pattern_multiplier
            break

    try:
        parsed = float(normalized) * multiplier
    except ValueError:
        return None
    return -parsed if negative else parsed


def _parse_field_date(value: str | None) -> date | None:
    normalized = normalize_granular_text(value)
    if not normalized:
        return None
    if not any(pattern.search(normalized) for pattern in _FIELD_DATE_PATTERNS):
        return None
    try:
        parsed = cast(datetime, date_parser.parse(normalized, fuzzy=False))
        return parsed.date()
    except (ValueError, OverflowError, TypeError):
        return None


def _bbox_xywh_to_xyxy(bbox: GroundingBbox) -> tuple[float, float, float, float]:
    return (bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)


def _bbox_area(bbox_xyxy: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = bbox_xyxy
    return max(0.0, right - left) * max(0.0, bottom - top)


def _bbox_intersection_area(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> float:
    left = max(left_bbox[0], right_bbox[0])
    top = max(left_bbox[1], right_bbox[1])
    right = min(left_bbox[2], right_bbox[2])
    bottom = min(left_bbox[3], right_bbox[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _bbox_iou(left_bbox: tuple[float, float, float, float], right_bbox: tuple[float, float, float, float]) -> float:
    intersection = _bbox_intersection_area(left_bbox, right_bbox)
    if intersection <= 0.0:
        return 0.0
    union = _bbox_area(left_bbox) + _bbox_area(right_bbox) - intersection
    return intersection / union if union > 0 else 0.0


def _union_bbox(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(left_bbox[0], right_bbox[0]),
        min(left_bbox[1], right_bbox[1]),
        max(left_bbox[2], right_bbox[2]),
        max(left_bbox[3], right_bbox[3]),
    )


def _union_bboxes(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    union_bbox = bboxes[0]
    for bbox in bboxes[1:]:
        union_bbox = _union_bbox(union_bbox, bbox)
    return union_bbox


def _bbox_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)


def _bbox_contains_point(bbox_xyxy: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    x, y = point
    return bbox_xyxy[0] <= x <= bbox_xyxy[2] and bbox_xyxy[1] <= y <= bbox_xyxy[3]


def _expand_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    margin_x: float,
    margin_y: float,
) -> tuple[float, float, float, float]:
    return (
        bbox_xyxy[0] - margin_x,
        bbox_xyxy[1] - margin_y,
        bbox_xyxy[2] + margin_x,
        bbox_xyxy[3] + margin_y,
    )


def _clip_bbox_to_bbox(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    left = max(left_bbox[0], right_bbox[0])
    top = max(left_bbox[1], right_bbox[1])
    right = min(left_bbox[2], right_bbox[2])
    bottom = min(left_bbox[3], right_bbox[3])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _rect_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0

    xs = sorted({coord for rect in rectangles for coord in (rect[0], rect[2])})
    ys = sorted({coord for rect in rectangles for coord in (rect[1], rect[3])})
    total_area = 0.0

    for left, right in zip(xs, xs[1:], strict=False):
        if right <= left:
            continue
        for bottom, top in zip(ys, ys[1:], strict=False):
            if top <= bottom:
                continue
            for rect in rectangles:
                if rect[0] <= left and rect[2] >= right and rect[1] <= bottom and rect[3] >= top:
                    total_area += (right - left) * (top - bottom)
                    break

    return total_area


def _covered_area_within_gt(
    gt_bbox_xyxy: tuple[float, float, float, float],
    pred_bboxes_xyxy: list[tuple[float, float, float, float]],
) -> float:
    clipped_rectangles = [
        clipped
        for pred_bbox_xyxy in pred_bboxes_xyxy
        if (clipped := _clip_bbox_to_bbox(pred_bbox_xyxy, gt_bbox_xyxy)) is not None
    ]
    return _rect_union_area(clipped_rectangles)


def _bbox_from_normalized_coco(
    bbox: list[float],
    *,
    page_width: float,
    page_height: float,
    label: str,
) -> GroundingBbox:
    return GroundingBbox(
        x=float(bbox[0]) * page_width,
        y=float(bbox[1]) * page_height,
        w=float(bbox[2]) * page_width,
        h=float(bbox[3]) * page_height,
        label=label,
    )


def _bbox_from_normalized_xyxy(
    bbox: list[float],
    *,
    page_width: float,
    page_height: float,
    label: str,
) -> GroundingBbox:
    left, top, right, bottom = [float(value) for value in bbox]
    return GroundingBbox(
        x=left * page_width,
        y=top * page_height,
        w=max(0.0, right - left) * page_width,
        h=max(0.0, bottom - top) * page_height,
        label=label,
    )


def _candidate_matches(
    gt_bbox_page_xyxy: tuple[float, float, float, float],
    pred_bbox_page_xyxy: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    if _bbox_intersection_area(gt_bbox_page_xyxy, pred_bbox_page_xyxy) > 0.0:
        return True

    margin_x = page_width * _FIELD_GROUPING_TOUCH_MARGIN
    margin_y = page_height * _FIELD_GROUPING_TOUCH_MARGIN
    expanded_gt = _expand_bbox(gt_bbox_page_xyxy, margin_x, margin_y)
    pred_center = _bbox_center(pred_bbox_page_xyxy)
    gt_center = _bbox_center(gt_bbox_page_xyxy)
    return _bbox_contains_point(expanded_gt, pred_center) or _bbox_contains_point(pred_bbox_page_xyxy, gt_center)


def _ordered_support_units(page: GroundingPage, granularity: Literal["line", "word"]) -> list[_SupportUnit]:
    layer = next((candidate for candidate in page.granular_layers if candidate.granularity == granularity), None)
    if layer is None or layer.availability != "available":
        return []

    support_units = [
        _SupportUnit(
            unit_id=unit.unit_id,
            granularity=granularity,
            order_index=unit.order_index,
            text=unit.text,
            bbox_page_xyxy=_bbox_xywh_to_xyxy(unit.bbox),
            bbox_page_xywh=unit.bbox,
        )
        for unit in layer.units
    ]
    support_units.sort(
        key=lambda unit: (
            unit.order_index if unit.order_index is not None else 10**9,
            unit.bbox_page_xyxy[1],
            unit.bbox_page_xyxy[0],
            unit.unit_id,
        )
    )
    return support_units


def _best_group_for_granularity(
    *,
    expected_value: str | int | float | bool | None,
    field_type: str | None,
    gt_bbox_page_xyxy: tuple[float, float, float, float],
    page: GroundingPage,
    granularity: Literal["line", "word"],
) -> tuple[_FieldGroupMatch | None, tuple[float, float, float, float, float, float] | None]:
    candidate_units = [
        unit
        for unit in _ordered_support_units(page, granularity)
        if _candidate_matches(
            gt_bbox_page_xyxy, unit.bbox_page_xyxy, page_width=page.page_width, page_height=page.page_height
        )
    ]
    if not candidate_units:
        return None, None

    gt_area = max(_bbox_area(gt_bbox_page_xyxy), 1e-12)
    best_match: _FieldGroupMatch | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None

    for start in range(len(candidate_units)):
        component_units: list[_SupportUnit] = []
        component_bboxes_page_xyxy: list[tuple[float, float, float, float]] = []
        union_bbox = candidate_units[start].bbox_page_xyxy

        for end in range(start, len(candidate_units)):
            unit = candidate_units[end]
            component_units.append(unit)
            component_bboxes_page_xyxy.append(unit.bbox_page_xyxy)
            union_bbox = _union_bbox(union_bbox, unit.bbox_page_xyxy)

            predicted_text = " ".join(candidate.text for candidate in component_units if candidate.text).strip()
            value_match = compare_field_value(expected_value, predicted_text, field_type=field_type)
            covered_area = _covered_area_within_gt(gt_bbox_page_xyxy, component_bboxes_page_xyxy)
            bbox_recall = covered_area / gt_area
            best_box_covered_area = max(
                (
                    _bbox_intersection_area(gt_bbox_page_xyxy, candidate_bbox)
                    for candidate_bbox in component_bboxes_page_xyxy
                ),
                default=0.0,
            )
            score_key = (
                1.0 if value_match.passed else 0.0,
                value_match.score,
                bbox_recall,
                best_box_covered_area / gt_area,
                -float(len(component_units)),
                -_bbox_area(union_bbox),
            )
            if best_key is not None and score_key <= best_key:
                continue

            best_key = score_key
            best_match = _FieldGroupMatch(
                unit_ids=tuple(candidate.unit_id for candidate in component_units),
                granularity=granularity,
                component_bboxes=tuple(candidate.bbox_page_xywh for candidate in component_units),
                bbox_page_xyxy=union_bbox,
                text=predicted_text,
                iou=_bbox_iou(gt_bbox_page_xyxy, union_bbox),
                bbox_recall=bbox_recall,
                text_score=value_match.score,
            )

    return best_match, best_key


def _best_match_for_rule(
    *,
    expected_value: str | int | float | bool | None,
    field_type: str | None,
    gt_bbox_page_xyxy: tuple[float, float, float, float],
    page: GroundingPage,
) -> _FieldGroupMatch | None:
    best_match: _FieldGroupMatch | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None

    for granularity in ("word", "line"):
        match, score_key = _best_group_for_granularity(
            expected_value=expected_value,
            field_type=field_type,
            gt_bbox_page_xyxy=gt_bbox_page_xyxy,
            page=page,
            granularity=granularity,
        )
        if match is None or score_key is None:
            continue
        if best_key is not None and score_key <= best_key:
            continue
        best_key = score_key
        best_match = match

    return best_match


def _best_citation_match_for_rule(
    *,
    expected_value: str | int | float | bool | None,
    field_type: str | None,
    gt_bbox_page_xyxy: tuple[float, float, float, float],
    page: GroundingPage,
    field_path: str,
    result_payload: dict[str, Any] | None,
) -> _FieldCitationMatch | None:
    predicted_value = _result_field_value(result_payload, field_path)
    has_predicted_value = predicted_value is not _MISSING_FIELD_VALUE
    predicted_text_from_value = _field_value_to_prediction_text(predicted_value) if has_predicted_value else None
    gt_area = max(_bbox_area(gt_bbox_page_xyxy), 1e-12)

    best_match: _FieldCitationMatch | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for item in page.items:
        if _field_path_from_item(item) != field_path or not item.bboxes:
            continue

        component_bboxes_page_xyxy = [_bbox_xywh_to_xyxy(bbox) for bbox in item.bboxes]
        union_bbox = _union_bboxes(component_bboxes_page_xyxy)
        if union_bbox is None:
            continue

        predicted_text = predicted_text_from_value if has_predicted_value else item.value or ""
        value_match = compare_field_value(expected_value, predicted_text, field_type=field_type)
        covered_area = _covered_area_within_gt(gt_bbox_page_xyxy, component_bboxes_page_xyxy)
        bbox_recall = covered_area / gt_area
        iou = _bbox_iou(gt_bbox_page_xyxy, union_bbox)
        score_key = (
            iou,
            bbox_recall,
            1.0 if value_match.passed else 0.0,
            value_match.score,
            -_bbox_area(union_bbox),
        )
        if best_key is not None and score_key <= best_key:
            continue

        best_key = score_key
        best_match = _FieldCitationMatch(
            item_id=item.item_id,
            component_bboxes=tuple(item.bboxes),
            bbox_page_xyxy=union_bbox,
            text=predicted_text,
            iou=iou,
            bbox_recall=bbox_recall,
            text_score=value_match.score,
            value_match=value_match,
        )

    return best_match


def _find_nearest_evaluation_report_path(result_path: Path | None) -> Path | None:
    if result_path is None or not result_path.is_file():
        return None

    current = result_path.parent
    while True:
        candidate = current / "_evaluation_report.json"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def _load_evaluation_examples(report_path: Path) -> dict[str, dict[str, Any]]:
    try:
        stat_result = report_path.stat()
    except OSError:
        _EVALUATION_REPORT_CACHE.pop(report_path, None)
        return {}

    cached = _EVALUATION_REPORT_CACHE.get(report_path)
    if cached is not None:
        cached_mtime_ns, cached_size, cached_examples = cached
        if cached_mtime_ns == stat_result.st_mtime_ns and cached_size == stat_result.st_size:
            return cached_examples

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        _EVALUATION_REPORT_CACHE.pop(report_path, None)
        return {}
    if not isinstance(payload, dict):
        _EVALUATION_REPORT_CACHE.pop(report_path, None)
        return {}

    per_example_results = payload.get("per_example_results")
    if not isinstance(per_example_results, list):
        _EVALUATION_REPORT_CACHE.pop(report_path, None)
        return {}

    examples_by_key: dict[str, dict[str, Any]] = {}
    for example in per_example_results:
        if not isinstance(example, dict):
            continue
        for key_name in ("example_id", "test_id"):
            key = example.get(key_name)
            if isinstance(key, str) and key and key not in examples_by_key:
                examples_by_key[key] = example

    _EVALUATION_REPORT_CACHE[report_path] = (
        stat_result.st_mtime_ns,
        stat_result.st_size,
        examples_by_key,
    )
    return examples_by_key


def _resolve_example_id(
    result_payload: dict[str, Any] | None, result_path: Path | None, report_path: Path
) -> str | None:
    if isinstance(result_payload, dict):
        request = result_payload.get("request")
        if isinstance(request, dict):
            example_id = request.get("example_id")
            if isinstance(example_id, str) and example_id:
                return example_id

    if result_path is None:
        return None

    try:
        relative = result_path.relative_to(report_path.parent)
    except ValueError:
        return None

    suffix = ".result.json"
    relative_name = str(relative)
    if relative_name.endswith(suffix):
        return relative_name[: -len(suffix)]
    return relative_name


def _find_layout_metric_result(example_result: dict[str, Any]) -> dict[str, Any] | None:
    metrics = example_result.get("metrics")
    if not isinstance(metrics, list):
        return None

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        if metric.get("metric_name") == "layout_element_rule_pass_rate":
            return metric
    return None


def _attribute_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _layout_rule_sort_key(raw_rule: dict[str, Any]) -> tuple[int, int, str]:
    ro_index = raw_rule.get("ro_index")
    return (
        int(ro_index) if isinstance(ro_index, int) else 10**9,
        int(raw_rule.get("page")) if isinstance(raw_rule.get("page"), int) else 10**9,
        str(raw_rule.get("id") or ""),
    )


def _layout_rule_eval_index(raw_rules: list[dict[str, Any]]) -> dict[str, int]:
    non_ignored_rules: list[dict[str, Any]] = []
    for raw_rule in raw_rules:
        if raw_rule.get("type") != "layout":
            continue
        attributes = raw_rule.get("attributes")
        if isinstance(attributes, dict) and _attribute_truthy(attributes.get("ignore")):
            continue
        non_ignored_rules.append(raw_rule)

    non_ignored_rules.sort(key=_layout_rule_sort_key)
    return {
        str(raw_rule.get("id") or ""): index for index, raw_rule in enumerate(non_ignored_rules) if raw_rule.get("id")
    }


def _load_layout_rule_matches(
    *,
    raw_rules: list[dict[str, Any]],
    pages: list[GroundingPage],
    result_path: Path | None,
    result_payload: dict[str, Any] | None,
) -> dict[int, list[GroundTruthRuleMatch]]:
    report_path = _find_nearest_evaluation_report_path(result_path)
    evaluation_results_by_key = _load_evaluation_examples(report_path) if report_path is not None else {}
    example_id = _resolve_example_id(result_payload, result_path, report_path) if report_path is not None else None
    example_result = evaluation_results_by_key.get(example_id or "") if example_id else None
    layout_metric_result = _find_layout_metric_result(example_result) if isinstance(example_result, dict) else None
    metric_metadata = layout_metric_result.get("metadata") if isinstance(layout_metric_result, dict) else None
    rule_results = metric_metadata.get("rule_results") if isinstance(metric_metadata, dict) else None

    rule_result_by_id: dict[str, dict[str, Any]] = {}
    rule_result_by_index: dict[int, dict[str, Any]] = {}
    if isinstance(rule_results, list):
        for rule_result in rule_results:
            if not isinstance(rule_result, dict):
                continue
            element_id = rule_result.get("element_id")
            if isinstance(element_id, str) and element_id and element_id not in rule_result_by_id:
                rule_result_by_id[element_id] = rule_result
            element_index = rule_result.get("element_index")
            if isinstance(element_index, int) and element_index not in rule_result_by_index:
                rule_result_by_index[element_index] = rule_result

    eval_index_by_rule_id = _layout_rule_eval_index(raw_rules)
    pages_by_number = {page.page_number: page for page in pages}
    rules_by_page: dict[int, list[GroundTruthRuleMatch]] = {}

    for raw_rule in raw_rules:
        if raw_rule.get("type") != "layout":
            continue

        attributes = raw_rule.get("attributes")
        if isinstance(attributes, dict) and _attribute_truthy(attributes.get("ignore")):
            continue

        page_number = raw_rule.get("page")
        try:
            normalized_page_number = int(page_number)
        except (TypeError, ValueError):
            continue
        page = pages_by_number.get(normalized_page_number)
        if page is None:
            continue

        raw_bbox = raw_rule.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            continue

        try:
            gt_bbox = _bbox_from_normalized_coco(
                [float(value) for value in raw_bbox],
                page_width=page.page_width,
                page_height=page.page_height,
                label="GT",
            )
        except (TypeError, ValueError):
            continue

        rule_id = str(raw_rule.get("id") or "")
        rule_result = rule_result_by_id.get(rule_id)
        if rule_result is None:
            eval_index = eval_index_by_rule_id.get(rule_id)
            if eval_index is not None:
                rule_result = rule_result_by_index.get(eval_index)

        predicted_bbox = None
        predicted_bboxes: list[GroundingBbox] = []
        if isinstance(rule_result, dict):
            best_pred_bbox = rule_result.get("best_pred_bbox")
            if isinstance(best_pred_bbox, list) and len(best_pred_bbox) == 4:
                try:
                    predicted_bbox = _bbox_from_normalized_xyxy(
                        [float(value) for value in best_pred_bbox],
                        page_width=page.page_width,
                        page_height=page.page_height,
                        label="Pred",
                    )
                    predicted_bboxes = [predicted_bbox]
                except (TypeError, ValueError):
                    predicted_bbox = None
                    predicted_bboxes = []

        localization_pass = rule_result.get("localization_pass") if isinstance(rule_result, dict) else None
        classification_pass = rule_result.get("classification_pass") if isinstance(rule_result, dict) else None
        attribution_applicable = rule_result.get("attribution_applicable") if isinstance(rule_result, dict) else None
        attribution_pass = rule_result.get("attribution_pass") if isinstance(rule_result, dict) else None

        overall_pass: bool | None = None
        if isinstance(localization_pass, bool) and isinstance(classification_pass, bool):
            if isinstance(attribution_applicable, bool) and attribution_applicable:
                if isinstance(attribution_pass, bool):
                    overall_pass = localization_pass and classification_pass and attribution_pass
            else:
                overall_pass = localization_pass and classification_pass

        predicted_text = None
        if isinstance(rule_result, dict):
            predicted_text_value = str(rule_result.get("pred_text_norm") or "").strip()
            predicted_text = predicted_text_value or None

        gt_text_norm = None
        if isinstance(rule_result, dict):
            gt_text_norm_value = str(rule_result.get("gt_text_norm") or "").strip()
            gt_text_norm = gt_text_norm_value or None

        predicted_class = None
        if isinstance(rule_result, dict):
            predicted_class_value = str(rule_result.get("best_pred_class") or "").strip()
            predicted_class = predicted_class_value or None

        predicted_class_norm = None
        if isinstance(rule_result, dict):
            predicted_class_norm_value = str(rule_result.get("best_pred_class_norm") or "").strip()
            predicted_class_norm = predicted_class_norm_value or None

        localization_reason = None
        if isinstance(rule_result, dict):
            localization_reason_value = str(rule_result.get("localization_reason") or "").strip()
            localization_reason = localization_reason_value or None

        classification_reason = None
        if isinstance(rule_result, dict):
            classification_reason_value = str(rule_result.get("classification_reason") or "").strip()
            classification_reason = classification_reason_value or None

        attribution_reason = None
        if isinstance(rule_result, dict):
            attribution_reason_value = str(rule_result.get("attribution_reason") or "").strip()
            attribution_reason = attribution_reason_value or None

        attribution_method = None
        if isinstance(rule_result, dict):
            attribution_method_value = str(rule_result.get("attribution_method") or "").strip()
            attribution_method = attribution_method_value or None

        rules_by_page.setdefault(page.page_number, []).append(
            GroundTruthRuleMatch(
                rule_id=rule_id,
                rule_type="layout",
                page_number=page.page_number,
                gt_bbox=gt_bbox,
                predicted_bbox=predicted_bbox,
                predicted_bboxes=predicted_bboxes,
                predicted_text=predicted_text,
                iou=float(rule_result["best_pred_iou"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("best_pred_iou"), (int, float))
                else None,
                bbox_recall=float(rule_result["best_pred_ioa_gt"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("best_pred_ioa_gt"), (int, float))
                else None,
                canonical_class=str(raw_rule.get("canonical_class") or "") or None,
                normalized_attributes=rule_result.get("normalized_attributes")
                if isinstance(rule_result, dict) and isinstance(rule_result.get("normalized_attributes"), dict)
                else {},
                gt_ro_index=raw_rule.get("ro_index") if isinstance(raw_rule.get("ro_index"), int) else None,
                gt_text_norm=gt_text_norm,
                predicted_class=predicted_class,
                predicted_class_norm=predicted_class_norm,
                best_pred_index=rule_result.get("best_pred_index")
                if isinstance(rule_result, dict) and isinstance(rule_result.get("best_pred_index"), int)
                else None,
                best_pred_ioa_gt=float(rule_result["best_pred_ioa_gt"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("best_pred_ioa_gt"), (int, float))
                else None,
                localization_pass=localization_pass if isinstance(localization_pass, bool) else None,
                localization_reason=localization_reason,
                classification_pass=classification_pass if isinstance(classification_pass, bool) else None,
                classification_reason=classification_reason,
                attribution_applicable=attribution_applicable if isinstance(attribution_applicable, bool) else None,
                attribution_pass=attribution_pass if isinstance(attribution_pass, bool) else None,
                attribution_reason=attribution_reason,
                attribution_method=attribution_method,
                attribution_threshold=float(rule_result["attribution_threshold"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("attribution_threshold"), (int, float))
                else None,
                token_precision=float(rule_result["token_precision"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("token_precision"), (int, float))
                else None,
                token_recall=float(rule_result["token_recall"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("token_recall"), (int, float))
                else None,
                token_f1=float(rule_result["token_f1"])
                if isinstance(rule_result, dict) and isinstance(rule_result.get("token_f1"), (int, float))
                else None,
                missing_tokens=[str(token) for token in rule_result.get("missing_tokens", [])]
                if isinstance(rule_result, dict) and isinstance(rule_result.get("missing_tokens"), list)
                else [],
                extra_tokens=[str(token) for token in rule_result.get("extra_tokens", [])]
                if isinstance(rule_result, dict) and isinstance(rule_result.get("extra_tokens"), list)
                else [],
                overall_pass=overall_pass,
            )
        )

    for page_rules in rules_by_page.values():
        page_rules.sort(key=lambda rule: (rule.gt_ro_index if rule.gt_ro_index is not None else 10**9, rule.rule_id))

    return rules_by_page


def _compute_field_match(
    *,
    raw_bbox: list[Any],
    page: GroundingPage,
    expected_value: Any,
    field_path: str,
    data_schema: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
) -> (
    tuple[
        GroundingBbox,
        GroundingBbox | None,
        list[GroundingBbox],
        str | None,
        Literal["line", "word", "extract_field"] | None,
        list[str],
        float | None,
        float | None,
        float | None,
        dict[str, Any],
    ]
    | None
):
    """Convert a normalized COCO bbox into a GT bbox and try to locate the best
    supporting prediction on the page. Returns None when the bbox is malformed.

    This helper is display-only: it may find local evidence bboxes/text for
    overlays, but evaluator verdicts must come from ``rule_results`` metadata.
    """
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    try:
        gt_bbox = _bbox_from_normalized_coco(
            [float(value) for value in raw_bbox],
            page_width=page.page_width,
            page_height=page.page_height,
            label="GT",
        )
    except (TypeError, ValueError):
        return None

    gt_bbox_page_xyxy = _bbox_xywh_to_xyxy(gt_bbox)
    field_type = _resolve_field_schema_type(data_schema, field_path)
    best_match = _best_match_for_rule(
        expected_value=expected_value,
        field_type=field_type,
        gt_bbox_page_xyxy=gt_bbox_page_xyxy,
        page=page,
    )
    citation_match: _FieldCitationMatch | None = None
    if best_match is None:
        citation_match = _best_citation_match_for_rule(
            expected_value=expected_value,
            field_type=field_type,
            gt_bbox_page_xyxy=gt_bbox_page_xyxy,
            page=page,
            field_path=field_path,
            result_payload=result_payload,
        )

    predicted_bbox: GroundingBbox | None = None
    predicted_bboxes: list[GroundingBbox] = []
    predicted_text: str | None = None
    predicted_granularity: Literal["line", "word", "extract_field"] | None = None
    matched_unit_ids: list[str] = []
    iou: float | None = None
    bbox_recall: float | None = None
    text_score: float | None = None
    computed_updates: dict[str, Any] = {}

    if best_match is not None:
        predicted_bbox_xyxy = best_match.bbox_page_xyxy
        predicted_bbox = GroundingBbox(
            x=predicted_bbox_xyxy[0],
            y=predicted_bbox_xyxy[1],
            w=max(0.0, predicted_bbox_xyxy[2] - predicted_bbox_xyxy[0]),
            h=max(0.0, predicted_bbox_xyxy[3] - predicted_bbox_xyxy[1]),
            label="Pred",
        )
        predicted_bboxes = [
            GroundingBbox(
                x=bbox.x,
                y=bbox.y,
                w=bbox.w,
                h=bbox.h,
                label=best_match.granularity,
            )
            for bbox in best_match.component_bboxes
        ]
        predicted_text = best_match.text or None
        predicted_granularity = best_match.granularity
        matched_unit_ids = list(best_match.unit_ids)
        iou = best_match.iou
        bbox_recall = best_match.bbox_recall
        text_score = best_match.text_score
    elif citation_match is not None:
        predicted_bbox_xyxy = citation_match.bbox_page_xyxy
        predicted_bbox = GroundingBbox(
            x=predicted_bbox_xyxy[0],
            y=predicted_bbox_xyxy[1],
            w=max(0.0, predicted_bbox_xyxy[2] - predicted_bbox_xyxy[0]),
            h=max(0.0, predicted_bbox_xyxy[3] - predicted_bbox_xyxy[1]),
            label="Pred",
        )
        predicted_bboxes = [
            GroundingBbox(
                x=bbox.x,
                y=bbox.y,
                w=bbox.w,
                h=bbox.h,
                label="extract_field",
            )
            for bbox in citation_match.component_bboxes
        ]
        predicted_text = citation_match.text or None
        predicted_granularity = "extract_field"
        matched_unit_ids = [citation_match.item_id]
        iou = citation_match.iou
        bbox_recall = citation_match.bbox_recall
        text_score = citation_match.text_score

    return (
        gt_bbox,
        predicted_bbox,
        predicted_bboxes,
        predicted_text,
        predicted_granularity,
        matched_unit_ids,
        iou,
        bbox_recall,
        text_score,
        computed_updates,
    )


_PARSE_FIELD_RULE_RESULT_METRIC = "parse_field_element_pass_rate"
_EXTRACT_RULE_RESULT_METRIC = "extract_element_pass_rate"
_FIELD_RULE_RESULT_METRIC_FALLBACKS = (
    _PARSE_FIELD_RULE_RESULT_METRIC,
    _EXTRACT_RULE_RESULT_METRIC,
)


def _extract_field_metric_names_for_example(example_result: dict[str, Any]) -> tuple[str, ...]:
    product_type = example_result.get("product_type")
    if not isinstance(product_type, str):
        product_type = ""

    normalized_product_type = product_type.lower()
    if normalized_product_type == "extract":
        return (_EXTRACT_RULE_RESULT_METRIC,)
    if normalized_product_type == "parse":
        return (_PARSE_FIELD_RULE_RESULT_METRIC,)
    return _FIELD_RULE_RESULT_METRIC_FALLBACKS


def _metric_has_rule_results(metric: dict[str, Any]) -> bool:
    metadata = metric.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return isinstance(metadata.get("rule_results"), list)


def _find_extract_field_metric_result(example_result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the metric entry carrying extract-field ``rule_results``.

    Parse evaluations expose this metadata under
    ``parse_field_element_pass_rate``. Native extract evaluations expose the
    same per-field verdict rows under ``extract_element_pass_rate``. When the
    product type is unavailable, probe both final carriers.
    """
    metrics = example_result.get("metrics")
    if not isinstance(metrics, list):
        return None

    for metric_name in _extract_field_metric_names_for_example(example_result):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            if metric.get("metric_name") == metric_name and _metric_has_rule_results(metric):
                return metric
    return None


def _build_extract_field_rule_result_index(
    *,
    result_path: Path | None,
    result_payload: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Load extract-field ``rule_results`` metadata and index by ``field_path``.

    The metric emits one entry per rule (not per GT bbox), so all evidence
    rows from the same rule share the same loc/cls/attr outcomes. The viz
    explicitly renders one match per GT bbox — each inherits the same
    rule-level verdict. Returns an empty dict when the report or metric is
    missing (pre-Wave-1 outputs).
    """
    report_path = _find_nearest_evaluation_report_path(result_path)
    if report_path is None:
        return {}

    evaluation_results_by_key = _load_evaluation_examples(report_path)
    example_id = _resolve_example_id(result_payload, result_path, report_path)
    example_result = evaluation_results_by_key.get(example_id or "") if example_id else None
    if not isinstance(example_result, dict):
        return {}

    metric_result = _find_extract_field_metric_result(example_result)
    if metric_result is None:
        return {}

    metadata = metric_result.get("metadata")
    if not isinstance(metadata, dict):
        return {}

    rule_results = metadata.get("rule_results")
    if not isinstance(rule_results, list):
        return {}

    index: dict[str, dict[str, Any]] = {}
    for entry in rule_results:
        if not isinstance(entry, dict):
            continue
        field_path = entry.get("field_path")
        if isinstance(field_path, str) and field_path and field_path not in index:
            index[field_path] = entry
    return index


def _metric_updates_from_entry(
    entry: dict[str, Any],
    *,
    page: GroundingPage,
    field_path: str | None = None,
    preserve_prediction_evidence: bool = False,
) -> dict[str, Any]:
    """Project a per-rule metric entry into a ``model_copy(update=...)`` dict.

    Copies the Wave-1 attribution outcomes (loc_pass / cls_pass / attr_pass /
    element_pass) plus the Phase-1-added metadata (localization_reason,
    matched_pred_bboxes, matched_pred_text). Unknown / missing fields fall
    back to the match's existing defaults so pre-Phase-1 reports remain
    backward-compatible.
    """
    loc_pass = entry.get("loc_pass")
    cls_pass = entry.get("cls_pass")
    attr_pass = entry.get("attr_pass")
    element_pass = entry.get("element_pass")

    updates: dict[str, Any] = {
        "localization_pass": loc_pass if isinstance(loc_pass, bool) else None,
        "classification_pass": cls_pass if isinstance(cls_pass, bool) else None,
        "attribution_pass": attr_pass if isinstance(attr_pass, bool) else None,
        "overall_pass": element_pass if isinstance(element_pass, bool) else None,
    }

    localization_reason = entry.get("localization_reason")
    if isinstance(localization_reason, str) and localization_reason:
        updates["localization_reason"] = localization_reason

    reason = entry.get("reason")
    if isinstance(reason, str) and reason:
        updates["attribution_reason"] = reason

    mode = entry.get("mode")
    if isinstance(mode, str) and mode:
        updates["attribution_method"] = mode

    score = entry.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        updates["text_score"] = float(score)

    if not preserve_prediction_evidence:
        granularity = entry.get("granularity")
        if isinstance(granularity, str) and granularity in ("word", "line"):
            updates["predicted_granularity"] = granularity
        # "layout_item" granularity doesn't fit the Literal["line", "word"] slot;
        # the attribution_method field carries the comparator mode, which is
        # sufficient for the UI to disambiguate.

        matched_pred_text = entry.get("matched_pred_text")
        if isinstance(matched_pred_text, str) and matched_pred_text:
            updates["predicted_text"] = (
                _extract_field_text_from_markdown_table(matched_pred_text, field_path) or matched_pred_text
            )

    iou = entry.get("iou")
    if isinstance(iou, (int, float)) and not isinstance(iou, bool):
        updates["iou"] = float(iou)

    matched_pred_bboxes = entry.get("matched_pred_bboxes")
    if not preserve_prediction_evidence and isinstance(matched_pred_bboxes, list):
        predicted_bboxes: list[GroundingBbox] = []
        for raw_bbox in matched_pred_bboxes:
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            try:
                normalized = [float(value) for value in raw_bbox]
            except (TypeError, ValueError):
                continue
            predicted_bboxes.append(
                _bbox_from_normalized_coco(
                    normalized,
                    page_width=page.page_width,
                    page_height=page.page_height,
                    label="Pred",
                )
            )
        if predicted_bboxes:
            updates["predicted_bboxes"] = predicted_bboxes
            updates["predicted_bbox"] = predicted_bboxes[0]

    return updates


def _append_extract_field_rule(
    *,
    raw_rule: dict[str, Any],
    pages_by_number: dict[int, GroundingPage],
    rules_by_page: dict[int, list[GroundTruthRuleMatch]],
    data_schema: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
    metric_rule_result_by_field_path: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Expand an extract_field rule with evidence bboxes into one
    GroundTruthRuleMatch per evidence bbox. Skips rules with no bboxes so
    unlocated fields don't render as ghost 0,0 overlays. Propagates the
    rule-level ``verified`` flag and ``tags`` (including ``stray_evidence``)
    onto each expanded match so the frontend can style strays distinctly.
    """
    raw_bboxes = raw_rule.get("bboxes")
    if not isinstance(raw_bboxes, list) or not raw_bboxes:
        return

    base_rule_id = str(raw_rule.get("id") or "")
    field_path = str(raw_rule.get("field_path") or "")
    expected_value = raw_rule.get("expected_value")
    verified_raw = raw_rule.get("verified")
    verified = bool(verified_raw) if isinstance(verified_raw, bool) else None
    tags_raw = raw_rule.get("tags")
    tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []

    for bbox_index, raw_bbox_entry in enumerate(raw_bboxes):
        if not isinstance(raw_bbox_entry, dict):
            continue

        page_number = raw_bbox_entry.get("page")
        try:
            normalized_page_number = int(page_number)
        except (TypeError, ValueError):
            continue
        page = pages_by_number.get(normalized_page_number)
        if page is None:
            continue

        raw_bbox = raw_bbox_entry.get("bbox")
        match = _compute_field_match(
            raw_bbox=raw_bbox if isinstance(raw_bbox, list) else [],
            page=page,
            expected_value=expected_value,
            field_path=field_path,
            data_schema=data_schema,
            result_payload=result_payload,
        )
        if match is None:
            continue
        (
            gt_bbox,
            predicted_bbox,
            predicted_bboxes,
            predicted_text,
            predicted_granularity,
            matched_unit_ids,
            iou,
            bbox_recall,
            text_score,
            computed_updates,
        ) = match

        source_bbox_index_raw = raw_bbox_entry.get("source_bbox_index")
        source_bbox_index = (
            source_bbox_index_raw
            if isinstance(source_bbox_index_raw, int) and not isinstance(source_bbox_index_raw, bool)
            else None
        )

        # Keep base_rule_id addressable when there is only one evidence bbox;
        # suffix multi-bbox expansions so React keys and selection state remain
        # unique per bbox.
        if len(raw_bboxes) == 1 and base_rule_id:
            rule_id = base_rule_id
        elif base_rule_id:
            rule_id = f"{base_rule_id}#{bbox_index}"
        else:
            rule_id = f"extract_field#{field_path}#{bbox_index}"

        rule = GroundTruthRuleMatch(
            rule_id=rule_id,
            rule_type="extract_field",
            page_number=page.page_number,
            field_path=field_path,
            expected_value=expected_value,
            evidence_index=bbox_index,
            gt_bbox=gt_bbox,
            predicted_bbox=predicted_bbox,
            predicted_bboxes=predicted_bboxes,
            predicted_text=predicted_text,
            predicted_granularity=predicted_granularity,
            matched_unit_ids=matched_unit_ids,
            iou=iou,
            bbox_recall=bbox_recall,
            text_score=text_score,
            verified=verified,
            tags=tags,
            source_bbox_index=source_bbox_index,
        )
        if computed_updates:
            rule = rule.model_copy(update=computed_updates)

        # Project the Wave-1 / Phase-1 metric outcomes onto the rule. The
        # metric emits one entry per rule (not per GT bbox), so all evidence
        # rows from the same rule share the same loc/cls/attr verdict — this
        # is intended (plan "Indexing nuance"). When the eval report is
        # missing or predates Phase 1, the None defaults remain.
        metric_index = metric_rule_result_by_field_path or {}
        metric_entry = metric_index.get(field_path) if field_path else None
        if isinstance(metric_entry, dict):
            rule = rule.model_copy(
                update=_metric_updates_from_entry(
                    metric_entry,
                    page=page,
                    field_path=field_path,
                    preserve_prediction_evidence=bool(rule.matched_unit_ids),
                )
            )

        rules_by_page.setdefault(page.page_number, []).append(rule)


def load_page_gt_rules(
    *,
    test_case_path: Path | None,
    pages: list[GroundingPage],
    result_path: Path | None = None,
    result_payload: dict[str, Any] | None = None,
) -> dict[int, list[GroundTruthRuleMatch]]:
    if test_case_path is None or not test_case_path.is_file():
        return {}

    try:
        payload = json.loads(test_case_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    raw_rules = payload.get("test_rules")
    if not isinstance(raw_rules, list):
        return {}

    data_schema = payload.get("data_schema") if isinstance(payload.get("data_schema"), dict) else None
    rules_by_page: dict[int, list[GroundTruthRuleMatch]] = {}
    pages_by_number = {page.page_number: page for page in pages}

    metric_rule_result_by_field_path = _build_extract_field_rule_result_index(
        result_path=result_path,
        result_payload=result_payload,
    )

    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        raw_type = raw_rule.get("type")
        if raw_type == "extract_field":
            _append_extract_field_rule(
                raw_rule=raw_rule,
                pages_by_number=pages_by_number,
                rules_by_page=rules_by_page,
                data_schema=data_schema,
                result_payload=result_payload,
                metric_rule_result_by_field_path=metric_rule_result_by_field_path,
            )

    layout_rules_by_page = _load_layout_rule_matches(
        raw_rules=[raw_rule for raw_rule in raw_rules if isinstance(raw_rule, dict)],
        pages=pages,
        result_path=result_path,
        result_payload=result_payload,
    )

    for page_number, layout_rules in layout_rules_by_page.items():
        rules_by_page.setdefault(page_number, []).extend(layout_rules)

    for page_rules in rules_by_page.values():
        page_rules.sort(
            key=lambda rule: (
                rule.rule_type,
                rule.gt_ro_index if rule.gt_ro_index is not None else 10**9,
                rule.field_path or "",
                rule.evidence_index if rule.evidence_index is not None else 10**9,
                rule.rule_id,
            )
        )

    return rules_by_page
