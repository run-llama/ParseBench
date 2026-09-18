from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse import nutrient_dws
from parse_bench.inference.providers.parse.nutrient_dws import NutrientDwsProvider


class TestRetryClassification(unittest.TestCase):
    """Which HTTP statuses get another attempt.

    A status wrongly classified as permanent drops that document from the run,
    which is invisible in the summary beyond a single failure count.
    """

    def setUp(self) -> None:
        self.provider = NutrientDwsProvider(
            "nutrient_dws",
            {"mode": "structure", "api_key": "test", "retry_count": 2, "retry_delay_s": 0},
        )
        self._tmp = TemporaryDirectory()
        self.doc = Path(self._tmp.name) / "doc.pdf"
        self.doc.write_bytes(b"%PDF-1.7\n")
        self.calls = 0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stub(self, status: int, then_ok: bool = False):
        def post(*_args, **_kwargs):
            self.calls += 1
            if then_ok and self.calls > 1:
                return httpx.Response(200, json={"output": {"markdown": "ok"}})
            return httpx.Response(status, text="stub")

        return post

    def test_408_is_retried_and_can_succeed(self) -> None:
        # The server giving up on a slow document is transient, not a bad request.
        nutrient_dws.httpx.post = self._stub(408, then_ok=True)
        payload = self.provider._parse(self.doc)
        self.assertEqual(payload["output"]["markdown"], "ok")
        self.assertEqual(self.calls, 2)

    def test_408_exhausted_raises_transient(self) -> None:
        nutrient_dws.httpx.post = self._stub(408)
        with self.assertRaises(ProviderTransientError):
            self.provider._parse(self.doc)
        self.assertEqual(self.calls, 3)  # initial attempt + retry_count

    def test_500_is_retried(self) -> None:
        nutrient_dws.httpx.post = self._stub(503, then_ok=True)
        self.provider._parse(self.doc)
        self.assertEqual(self.calls, 2)

    def test_400_is_permanent_and_not_retried(self) -> None:
        nutrient_dws.httpx.post = self._stub(400)
        with self.assertRaises(ProviderPermanentError):
            self.provider._parse(self.doc)
        self.assertEqual(self.calls, 1)

    def test_401_is_permanent_and_not_retried(self) -> None:
        nutrient_dws.httpx.post = self._stub(401)
        with self.assertRaises(ProviderPermanentError):
            self.provider._parse(self.doc)
        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
