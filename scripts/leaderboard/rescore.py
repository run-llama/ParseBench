"""Re-score leaderboard pipelines from their published outputs in the HF results repo.

For each pipeline, downloads ``results/<pipeline>/*.jsonl`` (and the repo's
``tools/``) at its pinned revision, explodes them into the per-document layout
the evaluator reads, and runs ``parse-bench evaluation run`` once per leaderboard
dimension against the ground truth at the map's ``data_revision`` (downloaded
once into ``--gt-cache``). Pipelines and revisions come from
``leaderboard/results.yaml``; only verified rows are re-scored.

    python scripts/leaderboard/rescore.py run --out rescore.json                    # every verified row
    python scripts/leaderboard/rescore.py run --pipelines a,b --out rescore.json     # selected rows
    python scripts/leaderboard/rescore.py merge shard-*.json --out rescore.json     # combine CI shards
    python scripts/leaderboard/rescore.py ground-truth                              # only fetch the ground truth

``HF_TOKEN`` is read from the environment when the results repo is private.
The output is what ``apply.py`` reads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIMENSIONS, MAP_PATH, git_head, load_map, overall  # noqa: E402

from parse_bench.evaluation.runner import _is_infra_failure, _is_skipped_result  # noqa: E402
from parse_bench.schemas.evaluation import EvaluationResult  # noqa: E402

# Published results category (results/<pipeline>/<category>.jsonl) behind each evaluation group.
CATEGORY_FOR_GROUP = {
    "table": "table",
    "chart": "chart",
    "text_content": "text",
    "text_formatting": "text",
    "layout": "layout",
}


def _download(results_repo: str, revision: str, pipeline: str, dest: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=results_repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"results/{pipeline}/*", "tools/*"],
        local_dir=str(dest),
        token=token,
    )
    return dest


def _evaluate(output_dir: Path, data_dir: Path, group: str, report_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "parse_bench.cli",
        "evaluation",
        "run",
        str(output_dir),
        "-t",
        str(data_dir),
        "-g",
        group,
        "-r",
        str(report_dir),
        "--force",
        "True",
        "--export_csv",
        "False",
        "--export_rule_csv",
        "False",
        "--export_markdown",
        "False",
        "--export_html",
        "False",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = report_dir / "_evaluation_report.json"
    if proc.returncode != 0 or not report.exists():
        tail = (proc.stdout + proc.stderr)[-2000:]
        raise RuntimeError(f"evaluation of group {group!r} failed (exit {proc.returncode}):\n{tail}")
    return json.loads(report.read_text())


def rescore_pipeline(
    pipeline: str,
    *,
    results_repo: str,
    results_revision: str,
    data_dir: Path,
    work_dir: Path,
    token: str | None = None,
) -> dict[str, Any]:
    hf_dir = _download(results_repo, results_revision, pipeline, work_dir / "hf", token)
    results_dir = hf_dir / "results" / pipeline
    categories = sorted(p.stem for p in results_dir.glob("*.jsonl"))
    if not categories:
        raise RuntimeError(f"no results/{pipeline}/*.jsonl at {results_repo}@{results_revision}")

    output_dir = work_dir / "output" / pipeline
    subprocess.run(
        [
            sys.executable,
            str(hf_dir / "tools" / "explode.py"),
            "--benchmark",
            "parsebench",
            str(results_dir),
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    published = {c: _count_lines(results_dir / f"{c}.jsonl") for c in categories}

    scores: dict[str, float | None] = {}
    details: dict[str, Any] = {}
    for column, (group, metric) in DIMENSIONS.items():
        report = _evaluate(output_dir, data_dir, group, work_dir / "reports" / group)
        value = (report.get("aggregate_metrics") or {}).get(f"avg_{metric}")
        # Failures the evaluator counts as the provider's (no output, no layout)
        # are already 0 in the average. Failures it leaves out of the
        # denominator (harness crashes, skipped pages) would make the average
        # cover fewer pages than the ground truth, so the dimension is reported
        # as an error instead of a score. The classification is the evaluator's
        # own, so this always agrees with `parse-bench run`.
        dropped = Counter(
            (r.error or "")[:200]
            for r in (EvaluationResult.model_validate(x) for x in report.get("per_example_results") or [])
            if not r.success and (_is_infra_failure(r) or _is_skipped_result(r))
        )
        error = None
        if dropped:
            error = f"{sum(dropped.values())} page(s) left out of the average: {dropped.most_common(3)}"
        scores[column] = None if error else (100.0 * float(value) if value is not None else 0.0)
        details[column] = {
            "group": group,
            "metric": metric,
            "published": published.get(CATEGORY_FOR_GROUP[group], 0),
            "total": report.get("total_examples"),
            "successful": report.get("successful"),
            "failed": report.get("failed"),
            "error": error,
        }
    complete = all(v is not None for v in scores.values())
    return {
        "pipeline": pipeline,
        "results_revision": results_revision,
        "categories_published": categories,
        "scores": scores,
        "overall": overall(scores) if complete else None,  # type: ignore[arg-type]
        "details": details,
    }


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


GT_DOWNLOAD_ATTEMPTS = 5
GT_DOWNLOAD_WORKERS = 4


def ground_truth(data_repo: str, data_revision: str, cache: Path, token: str | None) -> Path:
    """Ground truth at ``data_revision``, downloaded once into ``cache``.

    The snapshot is ~2,100 files. The Hub drops connections when many arrive at
    once from one address (shared CI runners show up as HTTP 499/429), so the
    download uses a few workers and retries with backoff; files already on disk
    are skipped on each retry.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    dest = cache / data_revision
    marker = dest / ".complete"
    if marker.exists():
        return dest
    for attempt in range(1, GT_DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=data_repo,
                repo_type="dataset",
                revision=data_revision,
                local_dir=str(dest),
                token=token,
                max_workers=GT_DOWNLOAD_WORKERS,
            )
            break
        except (HfHubHTTPError, OSError) as exc:
            if attempt == GT_DOWNLOAD_ATTEMPTS:
                raise
            wait = 15 * 2 ** (attempt - 1)
            print(
                f"ground truth download failed ({type(exc).__name__}: {str(exc)[:120]}); retrying in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    marker.write_text(data_revision)
    return dest


def rescore_rows(rmap: Any, pipelines: list[str], data_dir: Path, token: str | None) -> dict[str, Any]:
    """Re-score ``pipelines`` (verified rows of ``rmap``); one failing folder does not stop the rest."""
    revisions = {e.pipeline: e.results_revision for e in rmap.verified()}
    results: dict[str, Any] = {}
    for pipeline in pipelines:
        error = ""
        for _attempt in range(2):  # one retry covers transient Hub disconnects
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    results[pipeline] = rescore_pipeline(
                        pipeline,
                        results_repo=rmap.results_repo,
                        results_revision=revisions[pipeline],
                        data_dir=data_dir,
                        work_dir=Path(tmp),
                        token=token,
                    )
                break
            except Exception as exc:  # reported per pipeline
                error = str(exc)[-4000:]
        else:
            results[pipeline] = {"pipeline": pipeline, "results_revision": revisions[pipeline], "error": error}
        res = results[pipeline]
        status = (
            "ERROR" if "error" in res else ", ".join(f"{c} {v:.2f}" for c, v in res["scores"].items() if v is not None)
        )
        print(f"  {pipeline}: {status}", flush=True)
    return results


def _payload(rmap: Any, results: dict[str, Any]) -> dict[str, Any]:
    return {
        "scored_with": git_head(),
        "data_repo": rmap.data_repo,
        "data_revision": rmap.data_revision,
        "results_repo": rmap.results_repo,
        "results": dict(sorted(results.items())),
    }


def cmd_run(args: argparse.Namespace) -> int:
    rmap = load_map(args.map)
    if args.data_revision:
        rmap.data_revision = args.data_revision
    verified = [e.pipeline for e in rmap.verified() if e.pipeline]
    wanted = [p.strip() for p in args.pipelines.split(",") if p.strip()] if args.pipelines else verified
    unknown = sorted(set(wanted) - set(verified))
    if unknown:
        raise SystemExit(f"not verified rows in {args.map.name}: {unknown}")
    token = os.environ.get("HF_TOKEN") or None
    data_dir = ground_truth(rmap.data_repo, rmap.data_revision, args.gt_cache, token)
    print(f"re-scoring {len(wanted)} pipeline(s) against {rmap.data_repo}@{rmap.data_revision[:12]}", flush=True)
    payload = _payload(rmap, rescore_rows(rmap, wanted, data_dir, token))
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    errors = sum("error" in r for r in payload["results"].values())
    print(f"wrote {args.out}: {len(wanted) - errors} scored, {errors} error(s)")
    return 0


def cmd_ground_truth(args: argparse.Namespace) -> int:
    rmap = load_map(args.map)
    revision = args.data_revision or rmap.data_revision
    path = ground_truth(rmap.data_repo, revision, args.gt_cache, os.environ.get("HF_TOKEN") or None)
    print(f"ground truth {rmap.data_repo}@{revision[:12]} ready at {path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    shards = [json.loads(path.read_text()) for path in args.shards]
    if not shards:
        raise SystemExit("no shards to merge")
    keys = ("scored_with", "data_repo", "data_revision", "results_repo")
    for shard in shards[1:]:
        if any(shard[k] != shards[0][k] for k in keys):
            raise SystemExit(f"shards disagree on {keys}; they must come from one run")
    merged = {k: shards[0][k] for k in keys}
    merged["results"] = dict(sorted(r for shard in shards for r in shard["results"].items()))
    args.out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"merged {len(shards)} shard(s), {len(merged['results'])} pipeline(s) into {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="re-score verified rows")
    run.add_argument("--pipelines", default="", help="comma-separated pipelines (default: every verified row)")
    run.add_argument("--map", type=Path, default=MAP_PATH)
    run.add_argument("--data-revision", default="", help="score against this ground-truth revision instead")
    run.add_argument("--gt-cache", type=Path, default=Path(".cache/leaderboard-gt"))
    run.add_argument("--out", type=Path, required=True)
    gt = sub.add_parser("ground-truth", help="download the ground truth into the cache (CI prepare step)")
    gt.add_argument("--map", type=Path, default=MAP_PATH)
    gt.add_argument("--data-revision", default="")
    gt.add_argument("--gt-cache", type=Path, default=Path(".cache/leaderboard-gt"))
    merge = sub.add_parser("merge", help="combine rescore JSON shards from one run")
    merge.add_argument("shards", type=Path, nargs="+")
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"run": cmd_run, "merge": cmd_merge, "ground-truth": cmd_ground_truth}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
