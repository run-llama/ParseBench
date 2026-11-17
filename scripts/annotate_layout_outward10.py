"""Add opt-in outward allowances to a local HF layout dataset; keep sidecars and JSONL synchronized.

Usage: python scripts/annotate_layout_outward10.py DATASET_ROOT
Requires PyMuPDF for source-page dimensions. Does not change any existing bbox.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf as fitz

from parse_bench.test_cases.layout_bbox_tolerance import word_outward_tolerances
from parse_bench.test_cases.schema import LayoutTestRule


def annotate(root: Path) -> None:
    """Validate and stage all modified objects before replacing local dataset files."""
    rows = [json.loads(line) for line in (root / "layout.jsonl").read_text(encoding="utf-8").splitlines()]
    staged: dict[Path, str] = {}
    for row in rows:
        path = root / (row["id"] + ".test.json")
        obj = json.loads(path.read_text(encoding="utf-8"))
        with fitz.open(root / row["document"]) as source:
            doc = source if source.is_pdf else fitz.open("pdf", source.convert_to_pdf())
            for number in sorted({r["page"] for r in obj["test_rules"] if r.get("type") == "layout"}):
                page = doc[number - 1]
                if page.rotation:
                    page.remove_rotation()
                raw = [
                    r
                    for r in obj["test_rules"]
                    if r.get("type") == "layout" and r.get("granularity") == "word" and r["page"] == number
                ]
                rules = [LayoutTestRule.model_validate(r) for r in raw]
                for rule, tolerance in zip(
                    raw, word_outward_tolerances(rules, (page.rect.width, page.rect.height)), strict=True
                ):
                    rule["bbox_tolerance"] = tolerance.model_dump(mode="json")
            if doc is not source:
                doc.close()
        staged[path] = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        row["test_rules"] = json.dumps(obj["test_rules"], ensure_ascii=False)
    staged[root / "layout.jsonl"] = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    for path, text in staged.items():
        path.write_text(text, encoding="utf-8")
    print(f"Annotated {len(rows)} pages; source boxes unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    annotate(parser.parse_args().dataset)
