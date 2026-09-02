"""Unit tests for InfinityParser2 table-header heuristics.

These tests pin down the rule-driven behavior of the post-processing helpers
in ``infinity_parser2.py`` so future model/format changes don't silently
regress them.
"""

from __future__ import annotations

import json
import unittest

from bs4 import BeautifulSoup

from parse_bench.inference.providers.parse.infinity_parser2 import (
    _convert_nonstandard_table,
    _convert_table_header,
    _determine_header_row_count,
    _find_column_number,
    _is_gender_cell,
    _is_nonstandard_table,
    _is_pure_number_cell,
    _is_pure_text_cell,
    _is_year_cell,
)


class TestCellClassifiers(unittest.TestCase):
    """Cell-level predicates used by the header-row heuristics."""

    def test_year_cell(self) -> None:
        self.assertTrue(_is_year_cell("2024"))
        self.assertTrue(_is_year_cell("202401"))
        self.assertTrue(_is_year_cell("2024-01-15"))
        self.assertFalse(_is_year_cell("Revenue"))

    def test_gender_cell(self) -> None:
        self.assertTrue(_is_gender_cell("Male"))
        self.assertTrue(_is_gender_cell("female"))
        self.assertFalse(_is_gender_cell("Total"))

    def test_pure_text_vs_pure_number(self) -> None:
        # pure text: has alpha, no all-numeric requirement
        self.assertTrue(_is_pure_text_cell("Revenue"))
        self.assertFalse(_is_pure_text_cell("123"))
        self.assertFalse(_is_pure_text_cell(""))

        # pure number: digits + permitted symbols only
        self.assertTrue(_is_pure_number_cell("1,234.56"))
        self.assertTrue(_is_pure_number_cell("$(45.00)"))
        self.assertTrue(_is_pure_number_cell("-12%"))
        self.assertFalse(_is_pure_number_cell("12 apples"))
        self.assertFalse(_is_pure_number_cell(""))


class TestNonstandardTable(unittest.TestCase):
    """Detection and conversion of '&'-separated tables emitted by the model."""

    def test_is_nonstandard_table(self) -> None:
        # Has '&' and does not start with '|' → nonstandard
        self.assertTrue(_is_nonstandard_table("a | b | c & 1 | 2 | 3"))
        # Already a proper markdown table → not nonstandard
        self.assertFalse(_is_nonstandard_table("| a | b |\n| - | - |"))
        # No '&' → not nonstandard
        self.assertFalse(_is_nonstandard_table("plain text"))
        self.assertFalse(_is_nonstandard_table(""))

    def test_find_column_number(self) -> None:
        # 3 columns → header has 2 pipes between cells
        self.assertEqual(_find_column_number("a | b | c & 1 | 2 | 3"), 3)
        self.assertEqual(_find_column_number("no ampersand here"), 0)

    def test_convert_nonstandard_table_roundtrip(self) -> None:
        raw = "Year | Revenue | Profit & 2023 | 100 | 20 & 2024 | 150 | 35"
        out = _convert_nonstandard_table(raw)
        lines = out.splitlines()
        # Header + separator + 2 data rows
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("|") and lines[0].endswith("|"))
        self.assertEqual(lines[1], "| --- | --- | --- |")
        self.assertIn("2023", lines[2])
        self.assertIn("2024", lines[3])

    def test_convert_nonstandard_table_passthrough(self) -> None:
        # Already-valid markdown tables must be returned unchanged.
        already_md = "| a | b |\n| - | - |\n| 1 | 2 |"
        self.assertEqual(_convert_nonstandard_table(already_md), already_md)


class TestDetermineHeaderRowCount(unittest.TestCase):
    """Header-row count is determined by year/gender/value rules, then rowspan."""

    @staticmethod
    def _rows(html: str) -> list:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        return table.find_all("tr", recursive=False)

    def test_year_rule_single_header_row(self) -> None:
        html = """
        <table>
          <tr><td>2022</td><td>2023</td><td>2024</td></tr>
          <tr><td>10</td><td>20</td><td>30</td></tr>
          <tr><td>11</td><td>21</td><td>31</td></tr>
        </table>
        """
        self.assertEqual(_determine_header_row_count(self._rows(html)), 1)

    def test_value_rule_text_then_numbers(self) -> None:
        # First row pure text, rest pure numbers → 1 header row by value rule.
        html = """
        <table>
          <tr><td>Region</td><td>Revenue</td><td>Profit</td></tr>
          <tr><td>100</td><td>200</td><td>30</td></tr>
          <tr><td>110</td><td>210</td><td>35</td></tr>
        </table>
        """
        self.assertEqual(_determine_header_row_count(self._rows(html)), 1)

    def test_rowspan_fallback(self) -> None:
        # No year/gender/value signal → fallback to rowspan of first row.
        html = """
        <table>
          <tr><td rowspan="2">A</td><td rowspan="2">B</td></tr>
          <tr></tr>
          <tr><td>x</td><td>y</td></tr>
        </table>
        """
        self.assertEqual(_determine_header_row_count(self._rows(html)), 2)


class TestConvertTableHeader(unittest.TestCase):
    """End-to-end: <td> in detected header rows is rewritten to <th>."""

    def test_td_to_th_in_header_row(self) -> None:
        html = "<table><tr><td>2022</td><td>2023</td></tr><tr><td>10</td><td>20</td></tr></table>"
        out = _convert_table_header(html)
        soup = BeautifulSoup(out, "html.parser")
        rows = soup.find_all("tr")
        # Row 0: both cells become <th>
        self.assertEqual(len(rows[0].find_all("th")), 2)
        self.assertEqual(len(rows[0].find_all("td")), 0)
        # Row 1: data cells remain <td>
        self.assertEqual(len(rows[1].find_all("td")), 2)
        self.assertEqual(len(rows[1].find_all("th")), 0)

    def test_non_table_html_unchanged(self) -> None:
        self.assertEqual(_convert_table_header(""), "")
        self.assertEqual(_convert_table_header("plain text"), "plain text")


if __name__ == "__main__":
    unittest.main()


# =============================================================================
# Multi-page rendering
# =============================================================================


def _infinity_provider(monkeypatch, parser):
    """Build an InfinityParser2Provider with the SDK replaced by ``parser``."""
    import sys
    import types

    from parse_bench.inference.providers.parse.infinity_parser2 import InfinityParser2Provider

    stub = types.ModuleType("infinity_parser2")
    stub.InfinityParser2 = lambda **_kwargs: parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infinity_parser2", stub)
    return InfinityParser2Provider("infinity_parser2", {"deep_parsing_mode": False})


class _FakeParser:
    def __init__(self) -> None:
        self.calls: list = []

    def parse(self, image, **kwargs):
        self.calls.append(image)
        number = len(self.calls)
        return json.dumps([{"category": "text", "bbox": [0, 0, 10, 10], "text": f"page {number}", "page": 1}])


def test_infinity_parser2_default_timeout_is_900(monkeypatch) -> None:
    provider = _infinity_provider(monkeypatch, _FakeParser())
    assert provider._timeout == 900


def test_infinity_parser2_runs_pdf_pages_in_order_and_normalizes_each_page(tmp_path, monkeypatch) -> None:
    import parse_bench.inference.providers.parse.infinity_parser2 as mod
    from parse_bench.schemas.pipeline import PipelineSpec
    from parse_bench.schemas.pipeline_io import InferenceRequest
    from parse_bench.schemas.product import ProductType

    parser = _FakeParser()
    provider = _infinity_provider(monkeypatch, parser)

    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(mod, "load_images", lambda _path: [("img1", 100.0, 200.0), ("img2", 300.0, 400.0)])

    pipeline = PipelineSpec(
        pipeline_name="infinity_parser2", provider_name="infinity_parser2", product_type=ProductType.PARSE, config={}
    )
    request = InferenceRequest(
        example_id="infinity-multipage", source_file_path=str(source_pdf), product_type=ProductType.PARSE
    )

    raw = provider.run_inference(pipeline, request)
    normalized = provider.normalize(raw)

    assert parser.calls == ["img1", "img2"]
    assert len(raw.raw_output["page_results"]) == 2
    assert raw.raw_output["_config"]["page_width"] == 100.0
    assert raw.raw_output["page_results"][1]["_config"]["page_height"] == 400.0

    assert normalized.output.markdown == "page 1\n\npage 2"
    assert [p.page_index for p in normalized.output.pages] == [0, 1]
    assert [lp.page_number for lp in normalized.output.layout_pages] == [1, 2]
    assert [(lp.width, lp.height) for lp in normalized.output.layout_pages] == [(100.0, 200.0), (300.0, 400.0)]

    # Single page keeps the legacy raw shape.
    parser.calls.clear()
    single = provider._parse_pages([("only", 5.0, 6.0)])
    assert parser.calls == ["only"]
    assert "page_results" not in single
    assert single["_config"]["page_width"] == 5.0
