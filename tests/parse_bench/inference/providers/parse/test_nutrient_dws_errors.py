from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import httpx

from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.nutrient_dws import NutrientDwsProvider


class TestErrorClassification(unittest.TestCase):
    """Which error each failure raises, on a single attempt.

    The shared runner owns retries and retries only transient and rate-limit
    errors. A status wrongly classified as permanent drops that document from
    the run, which is invisible in the summary beyond a single failure count.
    """

    def setUp(self) -> None:
        self.provider = NutrientDwsProvider("nutrient_dws", {"mode": "structure", "api_key": "test"})
        self._tmp = TemporaryDirectory()
        self.doc = Path(self._tmp.name) / "doc.pdf"
        self.doc.write_bytes(b"%PDF-1.7\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _parse_with(self, **post_kwargs) -> mock.MagicMock:
        with mock.patch("parse_bench.inference.providers.parse.nutrient_dws.httpx.post", **post_kwargs) as post:
            try:
                self.provider._parse(self.doc)
            finally:
                self.assertEqual(post.call_count, 1)  # the provider never retries itself
        return post

    def _assert_status_raises(self, status: int, error: type[Exception]) -> None:
        with self.assertRaises(error):
            self._parse_with(return_value=httpx.Response(status, text="stub"))

    def test_200_returns_payload(self) -> None:
        with mock.patch(
            "parse_bench.inference.providers.parse.nutrient_dws.httpx.post",
            return_value=httpx.Response(200, json={"output": {"markdown": "ok"}}),
        ):
            self.assertEqual(self.provider._parse(self.doc)["output"]["markdown"], "ok")

    def test_408_is_transient(self) -> None:
        # The server giving up on a slow document is transient, not a bad request.
        self._assert_status_raises(408, ProviderTransientError)

    def test_429_is_rate_limit(self) -> None:
        self._assert_status_raises(429, ProviderRateLimitError)

    def test_5xx_is_transient(self) -> None:
        self._assert_status_raises(500, ProviderTransientError)
        self._assert_status_raises(503, ProviderTransientError)

    def test_400_and_401_are_permanent(self) -> None:
        self._assert_status_raises(400, ProviderPermanentError)
        self._assert_status_raises(401, ProviderPermanentError)

    def test_timeout_is_transient(self) -> None:
        with self.assertRaises(ProviderTransientError):
            self._parse_with(side_effect=httpx.ReadTimeout("slow"))

    def test_transport_error_is_transient(self) -> None:
        with self.assertRaises(ProviderTransientError):
            self._parse_with(side_effect=httpx.ConnectError("refused"))


class TestPricing(unittest.TestCase):
    """USD cost is priced at the leaderboard's common Free plan PAYG basis."""

    def _provider(self, **config) -> NutrientDwsProvider:
        env = {k: v for k, v in os.environ.items() if k != "NUTRIENT_DWS_CREDIT_RATE_USD"}
        with mock.patch.dict(os.environ, env, clear=True):
            return NutrientDwsProvider("nutrient_dws", {"mode": "agentic", "api_key": "test", **config})

    def test_default_rate_is_free_plan_payg(self) -> None:
        self.assertEqual(self._provider().credit_rate_usd, 0.001)

    def test_agentic_page_costs_18_credits_at_free_plan_rate(self) -> None:
        payload = {"metrics": {"pagesProcessed": 2}, "usage": {"data_extraction_credits": {"cost": 36}}}
        out = self._provider()._attach_usage(payload)
        self.assertEqual(out["credits_per_page"], 18.0)
        self.assertAlmostEqual(out["cost_per_page_usd"], 0.018)
        self.assertAlmostEqual(out["cost_usd"], 0.036)

    def test_rate_is_overridable(self) -> None:
        self.assertEqual(self._provider(credit_rate_usd=0.0012).credit_rate_usd, 0.0012)


class TestInstructions(unittest.TestCase):
    """The options each mode sends."""

    def test_default_body_is_the_hosted_markdown(self) -> None:
        for mode in ("structure", "understand", "agentic"):
            with mock.patch.dict(os.environ):
                os.environ.pop("NUTRIENT_DWS_BODY_SOURCE", None)
                provider = NutrientDwsProvider("nutrient_dws", {"mode": mode, "api_key": "test"})
            self.assertEqual(provider._body_source, "hosted")
            self.assertEqual(
                json.loads(provider._instructions()),
                {
                    "mode": mode,
                    "output": {
                        "formats": ["markdown", "spatial"],
                        "includeWords": True,
                        "enableInlineFormatting": True,
                        "includeHeadersAndFooters": True,
                        "extractWordsFromPictures": True,
                    },
                },
            )

    def test_graph_body_asks_for_words_and_inline_formatting_only(self) -> None:
        for mode in ("structure", "understand", "agentic"):
            provider = NutrientDwsProvider("nutrient_dws", {"mode": mode, "body_source": "graph", "api_key": "test"})
            self.assertEqual(
                json.loads(provider._instructions()),
                {
                    "mode": mode,
                    "output": {
                        "formats": ["markdown", "spatial"],
                        "includeWords": True,
                        "enableInlineFormatting": True,
                    },
                },
            )

    def test_graph_body_from_environment(self) -> None:
        with mock.patch.dict(os.environ, {"NUTRIENT_DWS_BODY_SOURCE": "graph"}):
            provider = NutrientDwsProvider("nutrient_dws", {"mode": "understand", "api_key": "test"})
        self.assertEqual(provider._body_source, "graph")
        output = json.loads(provider._instructions())["output"]
        self.assertNotIn("includeHeadersAndFooters", output)
        self.assertNotIn("extractWordsFromPictures", output)

    def test_text_mode_sends_only_the_markdown_format(self) -> None:
        provider = NutrientDwsProvider("nutrient_dws", {"mode": "text", "api_key": "test"})
        self.assertEqual(json.loads(provider._instructions()), {"mode": "text", "output": {"format": "markdown"}})


class TestChartCaption(unittest.TestCase):
    """A chart table's caption comes from the page, never from the generated summary."""

    _TABLE = "<table><tr><th>Year</th><th>Sales</th></tr><tr><td>2024</td><td>12</td></tr></table>"

    def test_printed_caption_is_attached(self) -> None:
        el = {"type": "chart", "htmlTable": self._TABLE, "captionIds": ["c1"], "summary": "Bar chart of sales"}
        html = NutrientDwsProvider._chart_table_html(el, {"c1": {"text": "Figure 3. Sales by year"}})
        self.assertIn("<caption>Figure 3. Sales by year</caption>", html)
        self.assertNotIn("Bar chart of sales", html)

    def test_summary_is_not_used_without_a_printed_caption(self) -> None:
        el = {"type": "chart", "htmlTable": self._TABLE, "summary": "Bar chart of sales"}
        self.assertEqual(NutrientDwsProvider._chart_table_html(el, {}), self._TABLE)


if __name__ == "__main__":
    unittest.main()
