"""scripts/update_readme.py: the ParseBench 2 leaderboard table, and the README matching the CSVs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_readme.py"
spec = importlib.util.spec_from_file_location("update_readme", SCRIPT)
assert spec and spec.loader
update_readme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update_readme)


def _row(provider: str, overall: str, cost: str = "") -> dict[str, str]:
    dims = dict.fromkeys(("Text", "Formatting", "Tables", "Charts", "Visual_Grounding", "Forms"), "50.00")
    return {"Provider": provider, "Category": "Vision Models", "Overall": overall, **dims, "Cost_Per_Page": cost}


def test_v2_table_ranks_by_overall_and_formats_cost() -> None:
    table = update_readme.build_v2_table([_row("B", "70.10", "1.250"), _row("A", "80.20"), _row("C", "60.00")])
    lines = table.splitlines()
    assert lines[0].startswith("| Rank | Provider | Category | Overall | Text | Formatting |")
    assert lines[2] == "| 1 | A | Vision Models | 80.20 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 | — |"
    assert lines[3].endswith("| 1.25¢ |") and lines[3].startswith("| 2 | B |")


def test_missing_markers_stop_the_update() -> None:
    with pytest.raises(SystemExit):
        update_readme.replace_block("no markers here", "<!-- A -->", "<!-- B -->", "x")


def test_readme_matches_the_leaderboard_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "README.md"
    copy.write_text(update_readme.README_PATH.read_text())
    monkeypatch.setattr(update_readme, "README_PATH", copy)
    update_readme.main()
    assert copy.read_text() == update_readme.README_PATH.read_text(), "run scripts/update_readme.py"
