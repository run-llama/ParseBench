"""Sync community results from the HF ParseBench leaderboard into leaderboard.csv.

Fetches the HF dataset leaderboard, pulls per-dimension scores from each model's
`.eval_results/parsebench.yaml`, and upserts into leaderboard.csv.

Dedup key: `HF_Model_ID`. If a row with the same `HF_Model_ID` already exists,
the existing row wins — our own runs always take precedence over community
submissions (they have cost data and more decimals of precision).

A non-empty `HF_Model_ID` indicates the row came from the HF community leaderboard.

Run: uv run python scripts/sync_hf_leaderboard.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "leaderboard.csv"

LEADERBOARD_API = "https://huggingface.co/api/datasets/llamaindex/ParseBench/leaderboard?limit=100"
YAML_URL_MAIN = "https://huggingface.co/{model_id}/raw/main/.eval_results/parsebench.yaml"
YAML_URL_PR = "https://huggingface.co/{model_id}/raw/refs%2Fpr%2F{pr}/.eval_results/parsebench.yaml"

TASK_TO_COLUMN = {
    "mean": "Overall",
    "table": "Tables",
    "chart": "Charts",
    "text_content": "Content_Faithfulness",
    "text_formatting": "Semantic_Formatting",
    "layout": "Visual_Grounding",
}

FIELDNAMES = [
    "Provider",
    "Category",
    "Overall",
    "Tables",
    "Charts",
    "Content_Faithfulness",
    "Semantic_Formatting",
    "Visual_Grounding",
    "Cost_Per_Page",
    "Cost_Charts",
    "Cost_Tables",
    "Cost_Text",
    "Cost_Layout",
    "HF_Model_ID",
]


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_text(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fmt(v) -> str:
    if v is None:
        return ""
    return f"{float(v):g}"


def parse_yaml_to_scores(yaml_text: str) -> dict[str, float]:
    entries = yaml.safe_load(yaml_text) or []
    scores: dict[str, float] = {}
    for entry in entries:
        task_id = (entry.get("dataset") or {}).get("task_id")
        col = TASK_TO_COLUMN.get(task_id)
        if col is not None:
            scores[col] = entry.get("value")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print diff, don't write")
    args = parser.parse_args()

    with CSV_PATH.open() as f:
        existing = list(csv.DictReader(f))
    existing_by_hf_id = {r["HF_Model_ID"]: r for r in existing if r["HF_Model_ID"]}

    print(f"GET {LEADERBOARD_API}")
    lb = fetch_json(LEADERBOARD_API)
    print(f"  {len(lb)} entries\n")

    added: list[dict] = []
    skipped: list[str] = []
    missing_yaml: list[str] = []

    for entry in lb:
        model_id = entry["modelId"]
        if model_id in existing_by_hf_id:
            skipped.append(model_id)
            continue

        pr = entry.get("pullRequest")
        url = YAML_URL_PR.format(model_id=model_id, pr=pr) if pr else YAML_URL_MAIN.format(model_id=model_id)
        yaml_text = fetch_text(url)
        if yaml_text is None:
            missing_yaml.append(f"{model_id} ({url})")
            continue

        scores = parse_yaml_to_scores(yaml_text)
        if not scores.get("Overall"):
            missing_yaml.append(f"{model_id} (no mean score)")
            continue

        name = model_id.split("/")[-1]
        if name and name[0].islower():
            name = name[0].upper() + name[1:]
        row = dict.fromkeys(FIELDNAMES, "")
        row["Provider"] = name
        row["Category"] = "VLM - Open Weight"
        for col, val in scores.items():
            row[col] = fmt(val)
        row["HF_Model_ID"] = model_id
        added.append(row)

    print(f"Added ({len(added)}):")
    for r in added:
        print(f"  + {r['HF_Model_ID']:<45} Overall {r['Overall']}")
    print(f"\nSkipped — already in CSV ({len(skipped)}):")
    for m in skipped:
        print(f"  = {m}")
    if missing_yaml:
        print(f"\nNo parsebench.yaml ({len(missing_yaml)}):")
        for m in missing_yaml:
            print(f"  ? {m}")

    if args.dry_run:
        print("\n(dry-run — no changes written)")
        return

    if added:
        all_rows = existing + added
        with CSV_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {CSV_PATH.name}")
    else:
        print("\nNo new rows.")

    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "update_readme.py")],
        check=True,
    )


if __name__ == "__main__":
    main()
