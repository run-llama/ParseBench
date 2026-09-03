"""Regression tests for JSONL test-case loading in ``_load_jsonl_dataset``."""

from __future__ import annotations

import json
from pathlib import Path

from parse_bench.test_cases.loader import _load_jsonl_dataset
from parse_bench.test_cases.schema import LayoutDetectionTestCase, ParseTestCase


def _write_jsonl(root: Path, rows: list[dict]) -> None:
    (root / "layout.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_layout_doc_with_order_rule_keeps_layout_rules(tmp_path: Path) -> None:
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "doc1.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_jsonl(
        tmp_path,
        [
            {
                "pdf": "pdfs/doc1.pdf",
                "category": "layout",
                "type": "layout",
                "id": "el-1",
                "rule": {"id": "el-1", "page": 1, "bbox": [0, 0, 1, 1], "canonical_class": "title"},
            },
            {
                "pdf": "pdfs/doc1.pdf",
                "category": "layout",
                "type": "order",
                "id": "ord-1",
                "rule": {"layout_bindings": {"before": "el-1", "after": "el-1"}},
            },
        ],
    )

    test_cases = _load_jsonl_dataset(tmp_path)

    assert len(test_cases) == 1
    tc = test_cases[0]
    assert isinstance(tc, LayoutDetectionTestCase)
    assert len(tc.get_layout_rules()) == 1
    assert len(tc.test_rules) == 2


def test_layout_only_doc_still_builds_layout_test_case(tmp_path: Path) -> None:
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "doc2.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_jsonl(
        tmp_path,
        [
            {
                "pdf": "pdfs/doc2.pdf",
                "category": "layout",
                "type": "layout",
                "id": "el-1",
                "rule": {"id": "el-1", "page": 1, "bbox": [0, 0, 1, 1], "canonical_class": "title"},
            },
        ],
    )

    test_cases = _load_jsonl_dataset(tmp_path)

    assert len(test_cases) == 1
    tc = test_cases[0]
    assert isinstance(tc, LayoutDetectionTestCase)
    assert len(tc.get_layout_rules()) == 1


def test_doc_with_only_parse_rules_still_builds_parse_test_case(tmp_path: Path) -> None:
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "doc3.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_jsonl(
        tmp_path,
        [
            {
                "pdf": "pdfs/doc3.pdf",
                "category": "text",
                "type": "present",
                "id": "p-1",
                "rule": {"text": "hello"},
            },
        ],
    )

    test_cases = _load_jsonl_dataset(tmp_path)

    assert len(test_cases) == 1
    assert isinstance(test_cases[0], ParseTestCase)
