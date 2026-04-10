"""Command-line interface for data management."""

import sys
from pathlib import Path

from parse_bench.data.download import download_dataset, is_dataset_ready


class DataCLI:
    """Command-line interface for managing benchmark datasets."""

    def download(
        self,
        data_dir: str | Path | None = None,
        force: bool = False,
        test: bool = False,
    ) -> int:
        """Download the parse-bench dataset from HuggingFace.

        Args:
            data_dir: Local directory to store the dataset (default: ./data)
            force: Force re-download even if data already exists
            test: Download the small test dataset (3 files per category)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            data_path = Path(data_dir) if data_dir else None
            download_dataset(data_dir=data_path, force=force, test=test)
            return 0
        except Exception as e:
            print(f"Error downloading dataset: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    def status(
        self,
        data_dir: str | Path | None = None,
    ) -> int:
        """Check if the dataset is downloaded and show summary statistics.

        Args:
            data_dir: Data directory to check (default: ./data)

        Returns:
            Exit code (0 if ready, 1 if not)
        """
        import json

        data_path = Path(data_dir) if data_dir else Path.cwd() / "data"
        ready = is_dataset_ready(data_path)

        if not ready:
            print(f"Dataset is NOT ready at: {data_path}")
            print("Run 'parse-bench download' to download it.")
            return 1

        print(f"Dataset: {data_path}")
        print()

        # Gather per-category stats from JSONL files
        jsonl_files = sorted(data_path.glob("*.jsonl"))
        total_cases = 0
        total_pdfs = 0
        all_pdfs: set[str] = set()  # track unique PDFs across all categories
        rows: list[tuple[str, int, int]] = []

        for jf in jsonl_files:
            category = jf.stem
            lines = jf.read_text().strip().splitlines()
            n_cases = len(lines)
            pdfs: set[str] = set()
            for line in lines:
                rec = json.loads(line)
                pdfs.add(rec.get("pdf", ""))
            n_pdfs = len(pdfs)
            rows.append((category, n_cases, n_pdfs))
            total_cases += n_cases
            total_pdfs += n_pdfs
            all_pdfs.update(pdfs)

        # Count docs on disk per category
        doc_counts: dict[str, int] = {}
        docs_dir = data_path / "docs"
        if docs_dir.exists():
            for cat_dir in sorted(docs_dir.iterdir()):
                if cat_dir.is_dir():
                    doc_counts[cat_dir.name] = sum(
                        1 for _ in cat_dir.rglob("*") if _.is_file()
                    )

        # Print table
        hdr = f"{'Category':<20} {'Test Cases':>12} {'PDFs':>8}"
        print(hdr)
        print("-" * len(hdr))
        for category, n_cases, n_pdfs in rows:
            print(f"{category:<20} {n_cases:>12,} {n_pdfs:>8,}")
        print("-" * len(hdr))
        print(f"{'Total':<20} {total_cases:>12,} {total_pdfs:>8,}")
        n_unique = len(all_pdfs)
        if n_unique < total_pdfs:
            print(f"{'Unique documents':<20} {'':>12} {n_unique:>8,}")
            print(f"  (text_content and text_formatting share the same PDF files)")
        print()

        # Docs on disk
        if doc_counts:
            print("Documents on disk:")
            for cat, count in doc_counts.items():
                print(f"  {cat:<18} {count:>6,} files")
            print(f"  {'total':<18} {sum(doc_counts.values()):>6,} files")

        return 0
