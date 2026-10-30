"""Re-score one pipeline from its published outputs in the HF results repo.

Downloads ``results/<pipeline>/*.jsonl`` (and the repo's ``tools/``) at a pinned
revision, explodes them into the per-document layout the evaluator reads, and
runs ``parse-bench evaluation run`` once per leaderboard dimension against the
ground truth in ``--data-dir``. Prints (or writes) a JSON summary.

Runs anywhere the package is installed; scripts/leaderboard/modal_app.py runs it
on Modal so nothing is downloaded locally.

    python scripts/leaderboard/rescore.py --pipeline llamaparse_agentic --data-dir data --out score.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIMENSIONS, overall  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--results-repo", default="llamaindex/ParseBench-results")
    parser.add_argument("--results-revision", default="main")
    parser.add_argument("--data-dir", type=Path, required=True, help="ground truth (parse-bench download)")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        result = rescore_pipeline(
            args.pipeline,
            results_repo=args.results_repo,
            results_revision=args.results_revision,
            data_dir=args.data_dir.resolve(),
            work_dir=args.work_dir or Path(tmp),
        )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
