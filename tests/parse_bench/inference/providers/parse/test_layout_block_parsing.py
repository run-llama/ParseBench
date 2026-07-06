"""Tests for layout div parsing robustness.

``parse_layout_blocks`` only checked that ``data-bbox`` decoded to a
4-element JSON list, so a model emitting ``[1, 2, 3, null]`` or string
elements produced items that raised ``TypeError`` deep inside
``build_layout_pages`` (``null / 1000``), failing the whole example.
Coordinates outside the prompt's 0-1000 range likewise flowed through
unclamped, producing layout segments outside the unit square.
"""

from __future__ import annotations

import unittest

from parse_bench.inference.providers.parse._layout_utils import (
    build_layout_pages,
    parse_layout_blocks,
)


def _div(bbox: str, label: str = "Text", text: str = "hello") -> str:
    return f'<div data-bbox="{bbox}" data-label="{label}">{text}</div>'


class TestBboxValidation(unittest.TestCase):
    def test_valid_int_bbox_parsed(self) -> None:
        blocks = parse_layout_blocks(_div("[100, 50, 900, 120]"))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["bbox"], [100, 50, 900, 120])

    def test_float_bbox_parsed(self) -> None:
        blocks = parse_layout_blocks(_div("[100.5, 50, 900, 120.5]"))
        self.assertEqual(len(blocks), 1)

    def test_null_element_skipped(self) -> None:
        blocks = parse_layout_blocks(_div("[1, 2, 3, null]"))
        self.assertEqual(blocks, [])

    def test_string_elements_skipped(self) -> None:
        blocks = parse_layout_blocks(_div('["a", "b", "c", "d"]'))
        self.assertEqual(blocks, [])

    def test_boolean_elements_skipped(self) -> None:
        blocks = parse_layout_blocks(_div("[true, false, true, false]"))
        self.assertEqual(blocks, [])

    def test_nan_and_infinity_skipped(self) -> None:
        # json.loads accepts NaN/Infinity; both must be rejected, not
        # clamped into fabricated coordinates.
        self.assertEqual(parse_layout_blocks(_div("[NaN, 0, 100, 100]")), [])
        self.assertEqual(parse_layout_blocks(_div("[-Infinity, 0, Infinity, 100]")), [])

    def test_wrong_length_skipped(self) -> None:
        blocks = parse_layout_blocks(_div("[1, 2, 3]"))
        self.assertEqual(blocks, [])

    def test_malformed_block_does_not_drop_valid_siblings(self) -> None:
        content = _div("[1, 2, 3, null]") + "\n" + _div("[0, 0, 500, 500]", text="ok")
        blocks = parse_layout_blocks(content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], "ok")

    def test_pipeline_no_longer_crashes_on_null_element(self) -> None:
        # End to end: previously the null survived parse_layout_blocks and
        # raised TypeError in build_layout_pages (null / 1000).
        content = _div("[1, 2, 3, null]") + _div("[0, 0, 500, 500]", text="ok")
        blocks = parse_layout_blocks(content)
        pages = build_layout_pages(blocks, 850, 1100, "ok")
        self.assertEqual(len(pages), 1)
        self.assertEqual(len(pages[0].items), 1)


class TestBuildLayoutPagesRevalidation(unittest.TestCase):
    def test_persisted_malformed_item_skipped_not_crashed(self) -> None:
        # Raw outputs persisted before the parse-time filter can carry
        # malformed items into build_layout_pages via renormalization.
        items = [
            {"bbox": [1, 2, 3, None], "label": "Text", "text": "bad"},
            {"bbox": [0, 0, 500, 500], "label": "Text", "text": "ok"},
        ]
        pages = build_layout_pages(items, 850, 1100, "ok")
        self.assertEqual(len(pages), 1)
        self.assertEqual(len(pages[0].items), 1)
        self.assertEqual(pages[0].items[0].value, "ok")


class TestCoordinateClamping(unittest.TestCase):
    def test_out_of_range_coordinates_clamped(self) -> None:
        blocks = parse_layout_blocks(_div("[-50, 0, 1100, 500]"))
        pages = build_layout_pages(blocks, 850, 1100, "hello")
        seg = pages[0].items[0].bbox
        self.assertEqual(seg.x, 0.0)
        self.assertEqual(seg.w, 1.0)
        self.assertEqual(seg.y, 0.0)
        self.assertEqual(seg.h, 0.5)

    def test_in_range_coordinates_unchanged(self) -> None:
        blocks = parse_layout_blocks(_div("[100, 200, 600, 700]"))
        pages = build_layout_pages(blocks, 850, 1100, "hello")
        seg = pages[0].items[0].bbox
        self.assertAlmostEqual(seg.x, 0.1)
        self.assertAlmostEqual(seg.y, 0.2)
        self.assertAlmostEqual(seg.w, 0.5)
        self.assertAlmostEqual(seg.h, 0.5)


if __name__ == "__main__":
    unittest.main()
