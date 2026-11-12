"""Regenerate the Top-N leaderboard tables in README.md from leaderboard.csv and leaderboard-v2.csv.

Run: uv run python scripts/update_readme.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "leaderboard.csv"
V2_CSV_PATH = REPO_ROOT / "leaderboard-v2.csv"
README_PATH = REPO_ROOT / "README.md"
TOP_N = 10

START_MARKER = "<!-- LEADERBOARD:START -->"
END_MARKER = "<!-- LEADERBOARD:END -->"
V2_START_MARKER = "<!-- LEADERBOARD-V2:START -->"
V2_END_MARKER = "<!-- LEADERBOARD-V2:END -->"


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
                    r["Provider"] + (" †" if r.get("Verified") == "no" else ""),
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


def build_v2_table(rows: list[dict]) -> str:
    rows_sorted = sorted(rows, key=lambda r: float(r["Overall"]), reverse=True)[:TOP_N]

    header = (
        "| Rank | Provider | Category | Overall | Text | Formatting | Tables | Charts | "
        "Visual Ground. | Forms | ¢ / Page |\n"
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for i, r in enumerate(rows_sorted, 1):
        cells = [str(i), r["Provider"], r["Category"], r["Overall"]]
        cells += [fmt_score(r[c]) for c in ("Text", "Formatting", "Tables", "Charts", "Visual_Grounding", "Forms")]
        cells.append(fmt_cost(r["Cost_Per_Page"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_block(readme: str, start_marker: str, end_marker: str, block: str) -> str:
    start = readme.find(start_marker)
    end = readme.find(end_marker)
    if start == -1 or end == -1:
        raise SystemExit(f"Markers not found in README.md. Add {start_marker} and {end_marker}.")
    return readme[:start] + block + readme[end + len(end_marker) :]


def main() -> None:
    with CSV_PATH.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("Provider")]

    table = build_table(rows)
    if any(r.get("Verified") == "no" for r in sorted(rows, key=lambda r: float(r["Overall"]), reverse=True)[:TOP_N]):
        table += (
            "\n\n† Unverified: no published outputs in "
            "[ParseBench-results](https://huggingface.co/datasets/llamaindex/ParseBench-results); scores as submitted."
        )
    block = (
        f"{START_MARKER}\n"
        f"_Top {TOP_N} by Overall score. For the full sortable, filterable leaderboard, "
        f"see [parsebench.ai](https://parsebench.ai); for raw data, "
        f"see [leaderboard.csv](leaderboard.csv)._\n\n"
        f"{table}\n"
        f"{END_MARKER}"
    )
    readme = replace_block(README_PATH.read_text(), START_MARKER, END_MARKER, block)

    with V2_CSV_PATH.open() as f:
        v2_rows = [r for r in csv.DictReader(f) if r.get("Provider")]
    v2_block = (
        f"{V2_START_MARKER}\n"
        f"_Top {TOP_N} by Overall score. For the full sortable, filterable leaderboard, "
        f"see [parsebench.ai](https://parsebench.ai); for raw data, "
        f"see [leaderboard-v2.csv](leaderboard-v2.csv)._\n\n"
        f"{build_v2_table(v2_rows)}\n"
        f"{V2_END_MARKER}"
    )
    readme = replace_block(readme, V2_START_MARKER, V2_END_MARKER, v2_block)

    README_PATH.write_text(readme)
    print(f"Updated README.md with top {TOP_N} from {CSV_PATH.name} and {V2_CSV_PATH.name}")


if __name__ == "__main__":
    main()
