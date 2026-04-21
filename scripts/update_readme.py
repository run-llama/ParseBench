"""Regenerate the Top-N leaderboard table in README.md from leaderboard.csv.

Run: uv run python scripts/update_readme.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "leaderboard.csv"
README_PATH = REPO_ROOT / "README.md"
TOP_N = 10

START_MARKER = "<!-- LEADERBOARD:START -->"
END_MARKER = "<!-- LEADERBOARD:END -->"


def fmt_score(v: str) -> str:
    return v if v else "—"


def fmt_cost(v: str) -> str:
    if not v:
        return "—"
    return f"{float(v):.2f}¢"


def build_table(rows: list[dict]) -> str:
    rows_sorted = sorted(rows, key=lambda r: float(r["Overall"]), reverse=True)[:TOP_N]

    header = (
        "| Rank | Provider | Category | Overall | Tables | Charts | "
        "Content Faith. | Sem. Format. | Visual Ground. | ¢ / Page |\n"
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for i, r in enumerate(rows_sorted, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    r["Provider"],
                    r["Category"],
                    fmt_score(r["Overall"]),
                    fmt_score(r["Tables"]),
                    fmt_score(r["Charts"]),
                    fmt_score(r["Content_Faithfulness"]),
                    fmt_score(r["Semantic_Formatting"]),
                    fmt_score(r["Visual_Grounding"]),
                    fmt_cost(r["Cost_Per_Page"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    with CSV_PATH.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("Provider")]

    table = build_table(rows)
    block = (
        f"{START_MARKER}\n"
        f"_Top {TOP_N} by Overall score. For the full sortable, filterable leaderboard, "
        f"see [parsebench.ai](https://parsebench.ai/#leaderboard); for raw data, "
        f"see [leaderboard.csv](leaderboard.csv)._\n\n"
        f"{table}\n"
        f"{END_MARKER}"
    )

    readme = README_PATH.read_text()
    start = readme.find(START_MARKER)
    end = readme.find(END_MARKER)
    if start == -1 or end == -1:
        raise SystemExit(f"Markers not found in README.md. Add {START_MARKER} and {END_MARKER}.")
    new_readme = readme[:start] + block + readme[end + len(END_MARKER) :]
    README_PATH.write_text(new_readme)
    print(f"Updated README.md with top {TOP_N} from {CSV_PATH.name}")


if __name__ == "__main__":
    main()
