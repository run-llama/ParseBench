"""Shared helpers for the leaderboard accountability tooling.

Every row of ``leaderboard.csv`` is tied to an entry in ``leaderboard/results.yaml``.
A *verified* entry names the pipeline folder in the HF results repo
(``results/<pipeline>/<category>.jsonl``) and the repo revision it was published
at, so the row can be re-scored from those outputs with the current harness.
An *unverified* entry has no published outputs; its scores are kept as submitted.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "leaderboard.csv"
MAP_PATH = REPO_ROOT / "leaderboard" / "results.yaml"

# leaderboard.csv column -> (evaluation group, headline metric). The metric is
# read as ``avg_<metric>`` from the group's evaluation report; the evaluator
# already counts missing and failed pages as 0 (see EvaluationRunner).
DIMENSIONS: dict[str, tuple[str, str]] = {
    "Tables": ("table", "grits_trm_composite"),
    "Charts": ("chart", "rule_pass_rate"),
    "Content_Faithfulness": ("text_content", "content_faithfulness"),
    "Semantic_Formatting": ("text_formatting", "semantic_formatting"),
    "Visual_Grounding": ("layout", "layout_element_rule_pass_rate"),
}

VERIFIED = "verified"
UNVERIFIED = "unverified"
STATUSES = (VERIFIED, UNVERIFIED)


@dataclass
class MapEntry:
    provider: str
    status: str
    pipeline: str | None = None
    results_revision: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"provider": self.provider, "status": self.status}
        if self.pipeline:
            d["pipeline"] = self.pipeline
        if self.results_revision:
            d["results_revision"] = self.results_revision
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class ResultsMap:
    results_repo: str
    data_repo: str
    data_revision: str
    scored_with: str | None
    rows: list[MapEntry] = field(default_factory=list)

    def by_provider(self) -> dict[str, MapEntry]:
        return {e.provider: e for e in self.rows}

    def verified(self) -> list[MapEntry]:
        return [e for e in self.rows if e.status == VERIFIED]


_MAP_HEADER = """\
# Maps every leaderboard.csv row to the published outputs it was scored from.
#
#   provider          exact `Provider` value of the leaderboard.csv row
#   status            verified   -> outputs are in `results_repo` and the row is re-scored from them
#                     unverified -> no published outputs; scores are kept as submitted
#   pipeline          folder under results/ in `results_repo` (verified rows only)
#   results_revision  `results_repo` commit (or refs/pr/N for a pending submission)
#
# `scored_with` is the ParseBench commit the verified rows were last re-scored at.
# Maintained by scripts/leaderboard/; see leaderboard/README.md.
"""


def load_map(path: Path = MAP_PATH) -> ResultsMap:
    raw = yaml.safe_load(path.read_text())
    return ResultsMap(
        results_repo=raw["results_repo"],
        data_repo=raw["data_repo"],
        data_revision=raw["data_revision"],
        scored_with=raw.get("scored_with"),
        rows=[MapEntry(**r) for r in raw.get("rows", [])],
    )


def save_map(rmap: ResultsMap, path: Path = MAP_PATH) -> None:
    body = {
        "results_repo": rmap.results_repo,
        "data_repo": rmap.data_repo,
        "data_revision": rmap.data_revision,
        "scored_with": rmap.scored_with,
        "rows": [e.to_dict() for e in rmap.rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_MAP_HEADER + "\n" + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=120))


def load_csv(path: Path = CSV_PATH) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("Provider")]
        return list(reader.fieldnames or []), rows


def write_csv(fieldnames: list[str], rows: list[dict[str, str]], path: Path = CSV_PATH) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_score(value: float) -> str:
    return f"{value:.2f}"


def overall(scores: dict[str, float]) -> float:
    """Overall is the unweighted mean of the five dimensions."""
    return sum(scores[c] for c in DIMENSIONS) / len(DIMENSIONS)


HARNESS_PATHS = ("src/parse_bench", "scripts/leaderboard", "pyproject.toml", "uv.lock")


def git_head(repo: Path = REPO_ROOT) -> str:
    """HEAD commit, suffixed ``+dirty`` when the scoring code has uncommitted changes.

    The scheduled re-score compares ``scored_with`` against HEAD; a ``+dirty``
    value is not a commit, so the next run re-scores rather than trusting it.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()

    sha = git("rev-parse", "HEAD")
    return f"{sha}+dirty" if git("status", "--porcelain", "--", *HARNESS_PATHS) else sha
