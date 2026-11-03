"""Validate leaderboard.csv against leaderboard/results.yaml.

Static checks (no network):
  - every leaderboard row has exactly one results-map entry and vice versa
  - verified entries name a pipeline and a results revision; no pipeline is mapped twice
  - the csv ``Verified`` column agrees with the map (``--fix`` rewrites it)

With ``--hf``: each verified pipeline folder exists in the results repo at its
pinned revision (needs a token with read access while the repo is private).

With ``--base-map``: the map's header (results repo, data repo, data revision)
must match the base branch's, so a pull request cannot point verification at
another repository or ground truth. The scheduled re-score moves the data revision.

With ``--base-csv``: rows added or changed relative to that csv must be
verified. The changed verified pipelines are printed and, under GitHub Actions,
written to ``$GITHUB_OUTPUT`` as ``pipelines`` for re-scoring.

    python scripts/leaderboard/check.py
    python scripts/leaderboard/check.py --hf --base-csv /tmp/base_leaderboard.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CSV_PATH,
    DIMENSIONS,
    MAP_PATH,
    STATUSES,
    VERIFIED,
    ResultsMap,
    load_csv,
    load_map,
    write_csv,
)

VERIFIED_COLUMN = "Verified"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
SAFE_REVISION = re.compile(r"[0-9a-f]{40}|refs/pr/[0-9]+")
SCORE_COLUMNS = ["Overall", *DIMENSIONS]


def static_errors(rows: list[dict[str, str]], rmap: ResultsMap) -> list[str]:
    errors: list[str] = []
    providers = [r["Provider"] for r in rows]
    for dup in sorted({p for p in providers if providers.count(p) > 1}):
        errors.append(f"leaderboard.csv has duplicate Provider {dup!r}")
    mapped = [e.provider for e in rmap.rows]
    for dup in sorted({p for p in mapped if mapped.count(p) > 1}):
        errors.append(f"results map has duplicate provider {dup!r}")
    for p in sorted(set(providers) - set(mapped)):
        errors.append(f"leaderboard row {p!r} has no entry in {MAP_PATH.name}")
    for p in sorted(set(mapped) - set(providers)):
        errors.append(f"results map entry {p!r} matches no leaderboard row")
    pipelines = [e.pipeline for e in rmap.verified()]
    for dup in sorted({p for p in pipelines if p and pipelines.count(p) > 1}):
        errors.append(f"pipeline {dup!r} is mapped to more than one row")
    for e in rmap.rows:
        # Map values reach shell commands in CI; keep them to identifier characters.
        if e.pipeline and not SAFE_NAME.fullmatch(e.pipeline):
            errors.append(f"{e.provider!r}: pipeline {e.pipeline!r} must match {SAFE_NAME.pattern}")
        if e.results_revision and not SAFE_REVISION.fullmatch(e.results_revision):
            errors.append(f"{e.provider!r}: results_revision {e.results_revision!r} must match {SAFE_REVISION.pattern}")
        if e.status not in STATUSES:
            errors.append(f"{e.provider!r}: status must be one of {STATUSES}, got {e.status!r}")
        elif e.status == VERIFIED and not (e.pipeline and e.results_revision):
            errors.append(f"{e.provider!r}: verified entries need `pipeline` and `results_revision`")
    return errors


def verified_column_errors(rows: list[dict[str, str]], rmap: ResultsMap) -> list[str]:
    by = rmap.by_provider()
    errors = []
    for r in rows:
        entry = by.get(r["Provider"])
        if entry is None:
            continue
        want = "yes" if entry.status == VERIFIED else "no"
        if r.get(VERIFIED_COLUMN, "") != want:
            errors.append(f"{r['Provider']!r}: {VERIFIED_COLUMN} column should be {want!r} (run check.py --fix)")
    return errors


def fix_verified_column(fieldnames: list[str], rows: list[dict[str, str]], rmap: ResultsMap) -> list[str]:
    by = rmap.by_provider()
    if VERIFIED_COLUMN not in fieldnames:
        fieldnames = [*fieldnames, VERIFIED_COLUMN]
    for r in rows:
        entry = by.get(r["Provider"])
        r[VERIFIED_COLUMN] = "yes" if entry and entry.status == VERIFIED else "no"
    return fieldnames


def hf_errors(rmap: ResultsMap) -> list[str]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    api = HfApi()
    errors = []
    for e in rmap.verified():
        try:
            files = [
                f.path
                for f in api.list_repo_tree(
                    rmap.results_repo,
                    path_in_repo=f"results/{e.pipeline}",
                    repo_type="dataset",
                    revision=e.results_revision,
                )
            ]
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
            errors.append(
                f"{e.provider!r}: results/{e.pipeline} not found at {e.results_revision}: {type(exc).__name__}"
            )
            continue
        if not any(f.endswith(".jsonl") for f in files):
            errors.append(f"{e.provider!r}: results/{e.pipeline}@{e.results_revision} has no <category>.jsonl files")
    return errors


def changed_rows(rows: list[dict[str, str]], base_rows: list[dict[str, str]]) -> list[str]:
    base = {r["Provider"]: r for r in base_rows}
    changed = []
    for r in rows:
        old = base.get(r["Provider"])
        if old is None or any(r.get(c, "") != old.get(c, "") for c in SCORE_COLUMNS):
            changed.append(r["Provider"])
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--map", type=Path, default=MAP_PATH)
    parser.add_argument("--hf", action="store_true", help="check verified folders exist in the results repo")
    parser.add_argument("--base-csv", type=Path, default=None, help="leaderboard.csv of the PR base")
    parser.add_argument("--base-map", type=Path, default=None, help="results.yaml of the PR base")
    parser.add_argument("--fix", action="store_true", help="rewrite the Verified column from the map")
    args = parser.parse_args()

    fieldnames, rows = load_csv(args.csv)
    rmap = load_map(args.map)

    if args.fix:
        write_csv(fix_verified_column(fieldnames, rows, rmap), rows, args.csv)
        print(f"rewrote {VERIFIED_COLUMN} column in {args.csv}")
        fieldnames, rows = load_csv(args.csv)

    errors = static_errors(rows, rmap) + verified_column_errors(rows, rmap)
    if args.base_map:
        base_map = load_map(args.base_map)
        for field in ("results_repo", "data_repo", "data_revision"):
            if getattr(rmap, field) != getattr(base_map, field):
                errors.append(f"{field} must stay {getattr(base_map, field)!r} (got {getattr(rmap, field)!r})")
    if args.hf and not errors:
        errors += hf_errors(rmap)

    changed_pipelines: list[str] = []
    if args.base_csv:
        _, base_rows = load_csv(args.base_csv)
        by = rmap.by_provider()
        for provider in changed_rows(rows, base_rows):
            entry = by.get(provider)
            if entry is None:
                continue  # already reported above
            if entry.status != VERIFIED:
                errors.append(
                    f"{provider!r} is added or changed but unverified: publish its outputs to "
                    f"{rmap.results_repo} and mark it verified in {args.map.name}"
                )
            else:
                changed_pipelines.append(entry.pipeline or "")
        print(f"changed verified pipelines: {', '.join(changed_pipelines) or '(none)'}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"pipelines={','.join(changed_pipelines)}\n")

    for e in errors:
        print(f"ERROR: {e}")
    print(f"{len(rows)} rows, {len(rmap.verified())} verified, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
