"""Pin the charts/images wording in the layout prompts.

Chart rules (``ChartDataPointRule`` and friends) only read tables, so the
prompt must give charts exactly one path — tabulation. The images bullet
says ``[Picture: description]`` rather than ``[Figure: ...]`` so "figure"
cannot pull charts into the describe path.
"""

from __future__ import annotations

import unittest

from parse_bench.inference.providers.parse._layout_utils import (
    SYSTEM_PROMPT_LAYOUT,
    SYSTEM_PROMPT_LAYOUT_ABS,
    SYSTEM_PROMPT_LAYOUT_GEMINI,
    SYSTEM_PROMPT_LAYOUT_GEMINI_ABS,
)

_ALL_VARIANTS = {
    "base": SYSTEM_PROMPT_LAYOUT,
    "abs": SYSTEM_PROMPT_LAYOUT_ABS,
    "gemini": SYSTEM_PROMPT_LAYOUT_GEMINI,
    "gemini_abs": SYSTEM_PROMPT_LAYOUT_GEMINI_ABS,
}


class TestChartWording(unittest.TestCase):
    def test_charts_bullet_is_imperative(self) -> None:
        # A conditional phrasing ("For charts/graphs being converted...")
        # leaves describing the chart as a valid alternative, which chart
        # rules cannot score.
        for name, prompt in _ALL_VARIANTS.items():
            with self.subTest(variant=name):
                self.assertIn("Convert charts/graphs/figures to tables", prompt)
                self.assertNotIn("being converted to tables", prompt)

    def test_images_bullet_does_not_say_figure(self) -> None:
        for name, prompt in _ALL_VARIANTS.items():
            with self.subTest(variant=name):
                self.assertIn("[Picture: description]", prompt)
                self.assertNotIn("[Figure:", prompt)


if __name__ == "__main__":
    unittest.main()
