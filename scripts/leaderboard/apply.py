"""Apply or verify re-scored leaderboard numbers.

``apply``: write the scores from a rescore JSON (scripts/leaderboard/rescore.py)
into leaderboard.csv for the verified rows it covers, record the ParseBench
commit in the results map, and write a Markdown report of every change. Rows
whose re-score errored keep their old values and are listed in the report.

``compare``: check that the scores in a (PR) leaderboard.csv match a re-score
of the same pipelines within a tolerance; exit 1 otherwise.

Under GitHub Actions both write ``changed``, ``large_changes`` and ``errors``
counts to ``$GITHUB_OUTPUT``.

    python scripts/leaderboard/apply.py apply rescore.json --report report.md --threshold 2
    python scripts/leaderboard/apply.py compare rescore.json --tolerance 0.5 --report report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BOARDS,
    V1,
    Board,
    fmt_score,
    load_csv,
    load_map,
    overall,
    save_map,
    write_csv,
)

COLUMNS = ["Overall", *V1.dimensions]  # v1; see columns()
POLICY = (
    "Pages the provider produced nothing usable for score 0. A dimension where the evaluator left pages out of its "
    "average (a harness error or missing layout data) is not re-scored and keeps its old value."
)
SHORT = V1.short


def columns(board: Board = V1) -> list[str]:
    return ["Overall", *board.dimensions]


def _board(args: argparse.Namespace) -> Board:
    return BOARDS[getattr(args, "board", None) or "v1"]


def _short(sha: str) -> str:
    return sha[:12] + ("+dirty" if sha.endswith("+dirty") else "")


def _num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def new_values(result: dict[str, Any], old: dict[str, float | None], board: Board = V1) -> dict[str, float | None]:
    """Values to publish: re-scored dimensions, old values where a dimension errored.

    Each dimension is rounded before averaging so Overall matches the published
    columns; Overall is recomputed only when every dimension of the board has a value.
    """
    values: dict[str, float | None] = {}
    for c in board.dimensions:
        score = result["scores"].get(c)
        values[c] = round(score, 2) if score is not None else old[c]
    dims = [values[c] for c in board.dimensions]
    values["Overall"] = (
        round(overall(dict(zip(board.dimensions, dims, strict=True)), board), 2)  # type: ignore[arg-type]
        if all(v is not None for v in dims)
        else old["Overall"]
    )
    return values


def diff_rows(
    rows: list[dict[str, str]], rescore: dict[str, Any], pipeline_by_provider: dict[str, str], board: Board = V1
) -> list[dict]:
    """One record per re-scored row: old and new values, per-dimension errors and coverage."""
    out = []
    for r in rows:
        pipeline = pipeline_by_provider.get(r["Provider"])
        result = rescore["results"].get(pipeline) if pipeline else None
        if result is None:
            continue
        if "error" in result:
            out.append({"provider": r["Provider"], "pipeline": pipeline, "error": result["error"]})
            continue
        cols = columns(board)
        old = {c: _num(r.get(c, "")) for c in cols}
        new = new_values(result, old, board)
        delta = {c: (new[c] - old[c]) if old[c] is not None and new[c] is not None else None for c in cols}
        details = result.get("details", {})
        dim_errors = {c: d["error"] for c, d in details.items() if d.get("error")}
        coverage = {
            c: (d["published"], d["total"])
            for c, d in details.items()
            if isinstance(d.get("published"), int) and isinstance(d.get("total"), int) and d["published"] < d["total"]
        }
        out.append(
            {
                "provider": r["Provider"],
                "pipeline": pipeline,
                "old": old,
                "new": new,
                "delta": delta,
                "dim_errors": dim_errors,
                "coverage": coverage,
            }
        )
    return out


def max_abs_delta(rec: dict) -> float:
    return max((abs(d) for d in rec["delta"].values() if d is not None), default=0.0)


def _cell(rec: dict, col: str, threshold: float) -> str:
    old, new, d = rec["old"][col], rec["new"][col], rec["delta"][col]
    if col in rec["dim_errors"]:
        return f"{old:.2f} ❌" if old is not None else "— ❌"
    if new is None:
        return "—"
    if old is None:
        return f"{new:.2f} (new)"
    note = f" ᶜ{rec['coverage'][col][0]}/{rec['coverage'][col][1]}" if col in rec["coverage"] else ""
    if abs(d) < 0.005:
        return f"{new:.2f}{note}"
    mark = " ⚠️" if abs(d) >= threshold else ""
    return f"{old:.2f} → {new:.2f} ({d:+.2f}){mark}{note}"


def render_report(title: str, records: list[dict], threshold: float, intro: str, board: Board = V1) -> str:
    cols, short = columns(board), board.short
    scored = [r for r in records if "error" not in r]
    changed = [r for r in scored if max_abs_delta(r) >= 0.005 or r["dim_errors"]]
    errors = [r for r in records if "error" in r]
    with_dim_errors = [r for r in scored if r["dim_errors"]]
    large = [r for r in scored if max_abs_delta(r) >= threshold]
    incomplete = [r for r in scored if r["coverage"]]
    lines = [f"## {title}", "", intro, ""]
    lines.append(
        f"- Re-scored: **{len(records)}** rows · changed: **{sum(max_abs_delta(r) >= 0.005 for r in scored)}** · "
        f"≥ {threshold:g} pt change: **{len(large)}** · rows with a dimension that could not be re-scored: "
        f"**{len(with_dim_errors)}** · rows that failed entirely: **{len(errors)}** · "
        f"rows with incomplete published outputs: **{len(incomplete)}**"
    )
    lines.append("")
    lines.append(
        "⚠️ change ≥ threshold · ❌ dimension not re-scored, old value kept (see below) · "
        "ᶜ published pages / ground-truth pages (missing pages score 0)"
    )
    lines.append("")
    if changed:
        changed.sort(key=lambda r: (max_abs_delta(r), len(r["dim_errors"])), reverse=True)
        lines.append("| Provider | " + " | ".join(short[c] for c in cols) + " |")
        lines.append("|---|" + "---:|" * len(cols))
        for r in changed:
            lines.append(f"| {r['provider']} | " + " | ".join(_cell(r, c, threshold) for c in cols) + " |")
        lines.append("")
    if with_dim_errors:
        lines.append("### Dimensions that could not be re-scored (old values kept)")
        lines.append("")
        for r in with_dim_errors:
            for col, err in r["dim_errors"].items():
                lines.append(f"- **{r['provider']}** (`{r['pipeline']}`) {short[col]}: {err[:300]}")
        lines.append("")
    if errors:
        lines.append("### Rows that could not be re-scored (values left unchanged)")
        lines.append("")
        for r in errors:
            first = r["error"].strip().splitlines()[-1] if r["error"].strip() else "unknown error"
            lines.append(f"- **{r['provider']}** (`{r['pipeline']}`): {first[:300]}")
        lines.append("")
    return "\n".join(lines)


def _github_output(**values: int) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for k, v in values.items():
                f.write(f"{k}={v}\n")


def _check_board(rescore: dict[str, Any], board: Board) -> None:
    if rescore.get("board", "v1") != board.name:
        raise SystemExit(f"{rescore.get('board', 'v1')} re-score given for --board {board.name}")


def cmd_apply(args: argparse.Namespace) -> int:
    board = _board(args)
    rescore = json.loads(args.rescore.read_text())
    _check_board(rescore, board)
    fieldnames, rows = load_csv(args.csv)
    rmap = load_map(args.map)
    pipeline_by_provider = {e.provider: e.pipeline for e in rmap.verified() if e.pipeline}
    records = diff_rows(rows, rescore, pipeline_by_provider, board)

    by_provider = {r["provider"]: r for r in records if "error" not in r}
    for row in rows:
        rec = by_provider.get(row["Provider"])
        if rec:
            for c in columns(board):
                if rec["new"][c] is not None:
                    row[c] = fmt_score(rec["new"][c])
    write_csv(fieldnames, rows, args.csv)

    # Recorded even when some rows errored (they are listed in the report and
    # keep their old values); otherwise the daily job would re-score everything
    # every day until one broken folder is fixed.
    rmap.scored_with = rescore["scored_with"]
    rmap.data_revision = rescore["data_revision"]
    save_map(rmap, args.map)

    intro = (
        f"Re-scored from `{rescore['results_repo']}` outputs with ParseBench "
        f"`{_short(rescore['scored_with'])}` against `{rescore['data_repo']}@{rescore['data_revision'][:12]}`. "
        f"{POLICY}"
    )
    title = "Leaderboard re-score" if board.name == "v1" else "ParseBench 2 leaderboard re-score"
    report = render_report(title, records, args.threshold, intro, board)
    if args.report:
        args.report.write_text(report + "\n")
    print(report)
    changed = [r for r in records if "error" not in r and max_abs_delta(r) >= 0.005]
    _github_output(
        changed=len(changed),
        large_changes=sum(max_abs_delta(r) >= args.threshold for r in changed),
        errors=sum("error" in r or bool(r.get("dim_errors")) for r in records),
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    board = _board(args)
    rescore = json.loads(args.rescore.read_text())
    _check_board(rescore, board)
    _, rows = load_csv(args.csv)
    rmap = load_map(args.map)
    pipeline_by_provider = {e.provider: e.pipeline for e in rmap.verified() if e.pipeline}
    records = diff_rows(rows, rescore, pipeline_by_provider, board)

    bad = [r for r in records if "error" in r or r["dim_errors"] or max_abs_delta(r) > args.tolerance]
    intro = (
        f"Submitted values compared with a re-score of the published outputs using ParseBench "
        f"`{_short(rescore['scored_with'])}` (tolerance ±{args.tolerance:g} pt). {POLICY}"
    )
    title = "Leaderboard verification" if board.name == "v1" else "ParseBench 2 leaderboard verification"
    report = render_report(title, records, args.tolerance, intro, board)
    report += "\n" + (
        "**Result: pass** — every changed row matches its re-score."
        if not bad
        else f"**Result: fail** — {len(bad)} row(s) differ from the re-score or could not be re-scored. "
        f"Update {board.csv_path.name} to the re-scored values above or fix the published outputs."
    )
    if args.report:
        args.report.write_text(report + "\n")
    print(report)
    _github_output(
        changed=len(records),
        large_changes=sum("error" not in r and max_abs_delta(r) > args.tolerance for r in bad),
        errors=sum("error" in r or bool(r.get("dim_errors")) for r in records),
    )
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("apply", "compare"):
        p = sub.add_parser(name)
        p.add_argument("rescore", type=Path)
        p.add_argument("--board", choices=sorted(BOARDS), default="v1")
        p.add_argument("--csv", type=Path, default=None, help="default: the board's csv")
        p.add_argument("--map", type=Path, default=None, help="default: the board's results map")
        p.add_argument("--report", type=Path, default=None)
        if name == "apply":
            p.add_argument("--threshold", type=float, default=2.0, help="flag changes at least this large (points)")
        else:
            p.add_argument("--tolerance", type=float, default=0.5, help="allowed |submitted - re-scored| (points)")
    args = parser.parse_args()
    board = BOARDS[args.board]
    args.csv = args.csv or board.csv_path
    args.map = args.map or board.map_path
    return cmd_apply(args) if args.command == "apply" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
