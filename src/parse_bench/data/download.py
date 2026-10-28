"""Download parse-bench datasets from HuggingFace.

Datasets (``--dataset``):
    parsebench  https://huggingface.co/datasets/llamaindex/ParseBench (default)
    org/name    any other HuggingFace dataset repo in the same layout: ``<category>.jsonl`` rows and ``docs/``,
                downloaded to ``./data/<name, lowercased>``

Structure after download:
    <local_dir>/
    ├── chart.jsonl
    ├── layout.jsonl
    ├── table.jsonl
    ├── text.jsonl
    ├── expected_markdown.json
    └── pdfs/{chart,layout,table,text}/*.pdf
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DATASET_REPO = "llamaindex/ParseBench"
DATASET_REPO_TYPE = "dataset"
TEST_DATA_REVISION = "test-data"

# Default on-disk locations. Test data lives in a sibling subdirectory so the
# two datasets coexist and `--test` does not silently overlay or get masked
# by an existing full download.
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_TEST_DATA_DIR = Path("./data/test")


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset the CLI can download: where it lives on HuggingFace and on disk, and what a complete copy holds."""

    repo: str
    data_dir: Path
    required_files: tuple[str, ...]
    doc_dirs: tuple[str, ...]
    test_data_dir: Path | None = None
    test_revision: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "parsebench": DatasetSpec(
        repo=DATASET_REPO,
        data_dir=DEFAULT_DATA_DIR,
        required_files=(
            "chart.jsonl",
            "layout.jsonl",
            "table.jsonl",
            "text_content.jsonl",
            "text_formatting.jsonl",
        ),
        doc_dirs=("docs/chart", "docs/layout", "docs/table", "docs/text"),
        test_data_dir=DEFAULT_TEST_DATA_DIR,
        test_revision=TEST_DATA_REVISION,
    ),
}

_REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")


def dataset_spec(dataset: str = "parsebench", test: bool = False) -> DatasetSpec:
    """Look up a dataset by name or HuggingFace repo id; ``test`` requires one that has a test subset.

    A repo id (``org/name``) not in ``DATASETS`` is any dataset in the ParseBench layout. Its categories are not
    known in advance, so a complete copy is one with at least one ``*.jsonl`` and a non-empty ``docs/``.
    """
    if dataset in DATASETS:
        spec = DATASETS[dataset]
    elif _REPO_ID.fullmatch(dataset):
        name = dataset.split("/", 1)[1].lower()
        spec = DatasetSpec(repo=dataset, data_dir=DEFAULT_DATA_DIR / name, required_files=(), doc_dirs=("docs",))
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASETS)}, or a HuggingFace repo id (org/name)"
        )
    if test and spec.test_data_dir is None:
        raise ValueError(f"The {dataset!r} dataset has no test subset; drop --test")
    return spec


def default_data_dir(test: bool = False, dataset: str = "parsebench") -> Path:
    """Return the default on-disk dataset path for the given mode.

    Used by every CLI surface that takes ``--test`` so the routing is
    consistent between ``download``, ``run`` and ``status``.
    """
    spec = dataset_spec(dataset, test)
    return spec.test_data_dir if test and spec.test_data_dir else spec.data_dir


def is_dataset_ready(data_dir: Path, dataset: str = "parsebench") -> bool:
    """Check if the dataset is already downloaded and complete.

    Args:
        data_dir: Path to the data directory.
        dataset: Which dataset to check for (see ``dataset_spec``).

    Returns:
        True if all required files and at least one PDF per category exist.
    """
    spec = dataset_spec(dataset)
    if not data_dir.exists():
        return False

    for f in spec.required_files:
        if not (data_dir / f).exists():
            return False
    if not spec.required_files and not any(data_dir.glob("*.jsonl")):
        return False

    for d in spec.doc_dirs:
        doc_dir = data_dir / d
        if not doc_dir.exists():
            return False
        # Check for any supported document file (PDF, image, etc.)
        if not any(doc_dir.rglob("*.*")):
            return False

    return True


def download_dataset(
    data_dir: Path | None = None,
    force: bool = False,
    test: bool = False,
    dataset: str = "parsebench",
) -> Path:
    """Download a parse-bench dataset from HuggingFace.

    Uses huggingface_hub's snapshot_download to fetch the full dataset,
    including JSONL files and PDFs.

    Args:
        data_dir: Local directory to store the dataset.
            Defaults to ./data in the current working directory.
        force: Force re-download even if data already exists.
        test: Download the small test dataset (3 files per category)
            instead of the full dataset.
        dataset: Which dataset to download: a name in ``DATASETS`` or a HuggingFace repo id.

    Returns:
        Path to the downloaded dataset directory.
    """
    from huggingface_hub import snapshot_download

    spec = dataset_spec(dataset, test)
    if data_dir is None:
        data_dir = Path.cwd() / default_data_dir(test=test, dataset=dataset)

    revision = spec.test_revision if test else None

    if not force and is_dataset_ready(data_dir, dataset):
        print(f"Dataset already downloaded at: {data_dir}")
        return data_dir

    label = "test dataset" if test else "dataset"
    print(f"Downloading {label} from HuggingFace: {spec.repo}")
    if revision:
        print(f"Branch: {revision}")
    print(f"Destination: {data_dir}")

    snapshot_download(
        repo_id=spec.repo,
        repo_type=DATASET_REPO_TYPE,
        local_dir=str(data_dir),
        revision=revision,
    )

    if not is_dataset_ready(data_dir, dataset):
        raise RuntimeError(f"Dataset download completed but validation failed. Check {data_dir} for missing files.")

    print(f"Dataset ready at: {data_dir}")
    return data_dir
