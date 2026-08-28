"""Unit tests for InfinityParser2 table-header heuristics.

These tests pin down the rule-driven behavior of the post-processing helpers
in ``infinity_parser2.py`` so future model/format changes don't silently
regress them.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.infinity_parser2 import (
    InfinityParser2Provider,
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
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _provider() -> InfinityParser2Provider:
    provider = object.__new__(InfinityParser2Provider)
    provider._base_config = {}
    return provider


def _raw_result(raw_output: dict[str, object]) -> RawInferenceResult:
    now = datetime.now()
    pipeline = PipelineSpec(
        pipeline_name="infinity-test",
        provider_name="infinity_parser2",
        product_type=ProductType.PARSE,
    )
    request = InferenceRequest(
        example_id="document",
        source_file_path=str(Path("document.pdf")),
        product_type=ProductType.PARSE,
    )
    return RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _page_output(result: str) -> dict[str, object]:
    return {
        "result": result,
        "_config": {"page_width": 100, "page_height": 200},
    }


def test_structured_empty_layout_normalizes_as_blank_page() -> None:
    output = _provider()._normalize(_raw_result(_page_output("[]")))

    assert [(page.page_index, page.markdown) for page in output.pages] == [(0, "")]
    assert [(page.page_number, page.items) for page in output.layout_pages] == [(1, [])]
    assert output.markdown == ""


def test_blank_middle_page_preserves_document_page_identities() -> None:
    page_one = json.dumps([{"page": 1, "category": "text", "bbox": [0, 0, 10, 10], "text": "one"}])
    page_three = json.dumps([{"page": 1, "category": "text", "bbox": [0, 0, 10, 10], "text": "three"}])
    raw_result = _raw_result(
        {
            "_parse_bench_multipage": {
                "version": 1,
                "num_pages": 3,
                "pages": [
                    {"page_index": 0, "raw_output": _page_output(page_one)},
                    {"page_index": 1, "raw_output": _page_output("[]")},
                    {"page_index": 2, "raw_output": _page_output(page_three)},
                ],
            }
        }
    )

    output = _provider().normalize(raw_result).output

    assert [page.page_index + 1 for page in output.pages] == [1, 2, 3]
    assert [page.page_number for page in output.layout_pages] == [1, 2, 3]
    assert [page.markdown for page in output.pages] == ["one", "", "three"]
    assert output.markdown == "one\n\n\n\nthree"


@pytest.mark.parametrize(
    ("raw_output", "message"),
    [
        ({"_config": {"page_width": 100, "page_height": 200}}, "Empty result"),
        (_page_output(""), "Empty result"),
        (_page_output("not json"), "not valid JSON"),
        (_page_output(json.dumps({"error": "model diagnostic"})), "must decode to a list"),
        (_page_output(json.dumps(["bad element"])), "non-object layout element"),
    ],
    ids=["missing", "empty-text", "malformed-json", "diagnostic-dict", "invalid-element"],
)
def test_empty_and_malformed_results_remain_distinct_from_blank_layout(
    raw_output: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ProviderPermanentError, match=message):
        _provider()._normalize(_raw_result(raw_output))


def test_deep_parsing_passes_through_structured_blank_layout() -> None:
    provider = _provider()
    provider._parser = SimpleNamespace(parse=lambda *args, **kwargs: pytest.fail("blank page must not deep-parse"))
    from PIL import Image

    with Image.new("RGB", (8, 8), "white") as image:
        assert provider._apply_deep_parsing("[]", image) == "[]"


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
