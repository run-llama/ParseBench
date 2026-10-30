"""Re-score leaderboard pipelines on Modal.

Each verified pipeline in ``leaderboard/results.yaml`` is re-scored in its own
container from the published outputs in the HF results repo, against the
ground truth pinned by ``data_revision`` (cached in a Modal volume). The image
is built from this checkout, so the harness that scores is the one you run from.

    modal run scripts/leaderboard/modal_app.py --out rescore.json                       # every verified row
    modal run scripts/leaderboard/modal_app.py --pipelines a,b --out rescore.json        # selected pipelines
    modal run scripts/leaderboard/modal_app.py --data-revision <sha> --out rescore.json  # newer ground truth

Needs a Modal secret holding a Hugging Face token with read access to the
results repo (``HF_TOKEN``); its name defaults to ``huggingface-secret`` and can
be overridden with the ``LEADERBOARD_HF_SECRET`` environment variable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# Inside the container this module is /root/modal_app.py and the checkout is /repo.
REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO_ROOT / "scripts" / "leaderboard"))

HF_SECRET = os.environ.get("LEADERBOARD_HF_SECRET", "huggingface-secret")
GT_ROOT = "/gt"

# Upload an explicit allowlist, never the whole checkout: the working tree can
# hold a .env with provider API keys, local outputs and large data.
_UPLOAD_FILES = ("pyproject.toml", "README.md", "LICENSE")
_UPLOAD_DIRS = ("src", "scripts/leaderboard")

image = modal.Image.debian_slim(python_version="3.12").pip_install("uv")
for _name in _UPLOAD_FILES:
    image = image.add_local_file(str(REPO_ROOT / _name), f"/repo/{_name}", copy=True)
for _name in _UPLOAD_DIRS:
    image = image.add_local_dir(str(REPO_ROOT / _name), f"/repo/{_name}", copy=True, ignore=["**/__pycache__"])
image = image.run_commands("cd /repo && uv pip install --system '.[runners,fast]'")

app = modal.App("parsebench-leaderboard-rescore", image=image)
gt_volume = modal.Volume.from_name("parsebench-leaderboard-gt", create_if_missing=True)


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    return None


@app.function(volumes={GT_ROOT: gt_volume}, secrets=[modal.Secret.from_name(HF_SECRET)], timeout=3600)
def prepare_ground_truth(data_repo: str, data_revision: str) -> str:
    """Download the ground truth once per revision into the shared volume."""
    from huggingface_hub import snapshot_download

    dest = Path(GT_ROOT) / data_revision
    marker = dest / ".complete"
    if not marker.exists():
        snapshot_download(
            repo_id=data_repo, repo_type="dataset", revision=data_revision, local_dir=str(dest), token=_hf_token()
        )
        marker.write_text(data_revision)
        gt_volume.commit()
    return str(dest)


@app.function(
    volumes={GT_ROOT: gt_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
    cpu=4,
    memory=16384,
    timeout=3 * 3600,
    max_containers=64,
)
def rescore_one(job: dict) -> dict:
    import tempfile

    sys.path.insert(0, "/repo/scripts/leaderboard")
    from rescore import rescore_pipeline

    gt_volume.reload()
    error = ""
    for _attempt in range(2):  # one retry covers transient Hub disconnects
        try:
            with tempfile.TemporaryDirectory() as tmp:
                return rescore_pipeline(
                    job["pipeline"],
                    results_repo=job["results_repo"],
                    results_revision=job["results_revision"],
                    data_dir=Path(GT_ROOT) / job["data_revision"],
                    work_dir=Path(tmp),
                    token=_hf_token(),
                )
        except Exception as exc:  # reported per pipeline; one bad folder must not sink the batch
            error = str(exc)[-4000:]
    return {"pipeline": job["pipeline"], "results_revision": job["results_revision"], "error": error}


@app.local_entrypoint()
def main(pipelines: str = "", out: str = "rescore.json", map_path: str = "", data_revision: str = "") -> None:
    from common import MAP_PATH, git_head, load_map

    rmap = load_map(Path(map_path) if map_path else MAP_PATH)
    if data_revision:
        rmap.data_revision = data_revision
    wanted = {p.strip() for p in pipelines.split(",") if p.strip()}
    entries = [e for e in rmap.verified() if not wanted or e.pipeline in wanted]
    missing = wanted - {e.pipeline for e in entries}
    if missing:
        raise SystemExit(f"not verified rows in the results map: {sorted(missing)}")

    gt = prepare_ground_truth.remote(rmap.data_repo, rmap.data_revision)
    print(f"ground truth ready at {gt}; re-scoring {len(entries)} pipeline(s)")
    jobs = [
        {
            "pipeline": e.pipeline,
            "results_repo": rmap.results_repo,
            "results_revision": e.results_revision,
            "data_revision": rmap.data_revision,
        }
        for e in entries
    ]
    results = {}
    for res in rescore_one.map(jobs, order_outputs=False):
        if "error" in res:
            status = "ERROR"
        else:
            failed = [c for c, d in res["details"].items() if d.get("error")]
            status = f"overall {res['overall']:.2f}" if not failed else f"not re-scored: {', '.join(failed)}"
        print(f"  {res['pipeline']}: {status}")
        results[res["pipeline"]] = res

    payload = {
        "scored_with": git_head(),
        "data_repo": rmap.data_repo,
        "data_revision": rmap.data_revision,
        "results_repo": rmap.results_repo,
        "results": dict(sorted(results.items())),
    }
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    errors = [p for p, r in results.items() if "error" in r]
    print(f"wrote {out}: {len(results) - len(errors)} scored, {len(errors)} error(s)")
