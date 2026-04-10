"""Download parse-bench dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/llamaindex/ParseBench

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

from pathlib import Path

DATASET_REPO = "llamaindex/ParseBench"
DATASET_REPO_TYPE = "dataset"
TEST_DATA_REVISION = "test-data"

# Files that must exist for the dataset to be considered complete
_REQUIRED_FILES = [
    "chart.jsonl",
    "layout.jsonl",
    "table.jsonl",
    "text_content.jsonl",
    "text_formatting.jsonl",
]

# At least one document must exist per category
_REQUIRED_DOC_DIRS = ["docs/chart", "docs/layout", "docs/table", "docs/text"]


def is_dataset_ready(data_dir: Path) -> bool:
    """Check if the dataset is already downloaded and complete.

    Args:
        data_dir: Path to the data directory.

    Returns:
        True if all required files and at least one PDF per category exist.
    """
    if not data_dir.exists():
        return False

    for f in _REQUIRED_FILES:
        if not (data_dir / f).exists():
            return False

    for d in _REQUIRED_DOC_DIRS:
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
) -> Path:
    """Download the parse-bench dataset from HuggingFace.

    Uses huggingface_hub's snapshot_download to fetch the full dataset,
    including JSONL files and PDFs.

    Args:
        data_dir: Local directory to store the dataset.
            Defaults to ./data in the current working directory.
        force: Force re-download even if data already exists.
        test: Download the small test dataset (3 files per category)
            instead of the full dataset.

    Returns:
        Path to the downloaded dataset directory.
    """
    from huggingface_hub import snapshot_download

    if data_dir is None:
        data_dir = Path.cwd() / "data"

    revision = TEST_DATA_REVISION if test else None

    if not force and is_dataset_ready(data_dir):
        print(f"Dataset already downloaded at: {data_dir}")
        return data_dir

    label = "test dataset" if test else "dataset"
    print(f"Downloading {label} from HuggingFace: {DATASET_REPO}")
    if test:
        print(f"Branch: {TEST_DATA_REVISION}")
    print(f"Destination: {data_dir}")

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type=DATASET_REPO_TYPE,
        local_dir=str(data_dir),
        revision=revision,
    )

    if not is_dataset_ready(data_dir):
        raise RuntimeError(
            f"Dataset download completed but validation failed. "
            f"Check {data_dir} for missing files."
        )

    print(f"Dataset ready at: {data_dir}")
    return data_dir
