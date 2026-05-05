from __future__ import annotations

import json
import os
from pathlib import Path

from backend.indexer import build_index


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_index_basic_visualizable(tmp_path: Path) -> None:
    doc_dir = tmp_path / "suite" / "candidate_model" / "default"
    doc_dir.mkdir(parents=True)
    (doc_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    _write_json(doc_dir / "sample.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].base_name == "sample"
    assert result.response.tree.total_document_count == 1


def test_build_index_handles_malformed_pdf_stem(tmp_path: Path) -> None:
    doc_dir = tmp_path / "tables_core" / "candidate_layout" / "v1.0"
    doc_dir.mkdir(parents=True)

    source_name = "sample.2020.page_26.pdf_000001_page1.pdf"
    v2_name = "sample.2020.page_26_000001_page1.pdf.v2.items.json"

    (doc_dir / source_name).write_bytes(b"%PDF-1.4\n")
    _write_json(doc_dir / v2_name, {"pages": [{"page_number": 1, "items": []}]})

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].base_name == "sample.2020.page_26_000001_page1"


def test_build_index_accepts_raw_v2_items_payload(tmp_path: Path) -> None:
    doc_dir = tmp_path / "text_core" / "candidate_model" / "default"
    doc_dir.mkdir(parents=True)

    (doc_dir / "doc.png").write_bytes(b"PNG")
    _write_json(
        doc_dir / "doc.raw.json",
        {"raw_output": {"v2_items": {"pages": [{"page_number": 1, "items": []}]}}},
    )

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].artifact_flags.has_v2_items_payload is True


def test_build_index_accepts_result_layout_pages_payload(tmp_path: Path) -> None:
    doc_dir = tmp_path / "text_core" / "azure_di_layout" / "v0.1"
    doc_dir.mkdir(parents=True)

    (doc_dir / "doc.png").write_bytes(b"PNG")
    _write_json(
        doc_dir / "doc.result.json",
        {
            "output": {
                "layout_pages": [
                    {
                        "page_number": 1,
                        "width": 1000,
                        "height": 1000,
                        "items": [],
                    }
                ]
            }
        },
    )

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].artifact_flags.has_v2_items_payload is True


def test_build_index_attaches_per_document_evaluation_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    doc_a_dir = run_root / "annotated_v0.4"
    doc_b_dir = run_root / "tables_core_v1.0"
    doc_a_dir.mkdir(parents=True)
    doc_b_dir.mkdir(parents=True)

    (doc_a_dir / "doc-a.pdf").write_bytes(b"%PDF-1.4\n")
    (doc_b_dir / "doc-b.pdf").write_bytes(b"%PDF-1.4\n")
    _write_json(doc_a_dir / "doc-a.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})
    _write_json(doc_b_dir / "doc-b.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})
    _write_json(
        run_root / "_evaluation_report.json",
        {
            "per_example_results": [
                {
                    "example_id": "annotated_v0.4/doc-a",
                    "metrics": [{"metric_name": "f1_Text", "value": 0.25}],
                },
                {
                    "example_id": "tables_core_v1.0/doc-b",
                    "metrics": [{"metric_name": "f1_Text", "value": 0.75}],
                },
            ]
        },
    )

    result = build_index(str(run_root), page=1, page_size=100)

    metrics_by_name = {doc.base_name: doc.evaluation_metrics for doc in result.response.documents}
    assert metrics_by_name["doc-a"]["f1_Text"] == 0.25
    assert metrics_by_name["doc-b"]["f1_Text"] == 0.75


def test_build_index_accepts_raw_layout_pages_payload(tmp_path: Path) -> None:
    doc_dir = tmp_path / "text_core" / "dots_parse" / "v0.1"
    doc_dir.mkdir(parents=True)

    (doc_dir / "doc.png").write_bytes(b"PNG")
    _write_json(
        doc_dir / "doc.raw.json",
        {
            "output": {
                "layout_pages": [
                    {
                        "page_number": 1,
                        "width": 3508,
                        "height": 4961,
                        "items": [],
                    }
                ]
            }
        },
    )

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].artifact_flags.has_v2_items_payload is True


def test_build_index_accepts_raw_items_pages_payload(tmp_path: Path) -> None:
    doc_dir = tmp_path / "text_core" / "candidate_model" / "default"
    doc_dir.mkdir(parents=True)

    (doc_dir / "doc.png").write_bytes(b"PNG")
    _write_json(
        doc_dir / "doc.raw.json",
        {"raw_output": {"items": {"pages": [{"page_number": 1, "items": []}]}}},
    )

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].artifact_flags.has_v2_items_payload is True


def test_build_index_accepts_extract_result_grounded_items_payload(tmp_path: Path) -> None:
    doc_dir = tmp_path / "extract_core" / "extract_product" / "default"
    doc_dir.mkdir(parents=True)

    (doc_dir / "doc.png").write_bytes(b"PNG")
    _write_json(
        doc_dir / "doc.result.json",
        {
            "raw_output": {
                "data": {"vendor": "Acme Corp"},
                "v2_grounded_items": [
                    {
                        "page_number": 1,
                        "page_width": 640,
                        "page_height": 480,
                        "items": [
                            {
                                "type": "text",
                                "md": "Acme Corp",
                                "bbox": [{"x": 64, "y": 48, "w": 120, "h": 20}],
                            }
                        ],
                    }
                ],
            }
        },
    )

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].artifact_flags.has_v2_items_payload is True


def test_build_index_sorts_documents_by_newest_artifact_mtime(tmp_path: Path) -> None:
    older_dir = tmp_path / "suite" / "older"
    newer_dir = tmp_path / "suite" / "newer"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)

    older_source = older_dir / "doc-old.pdf"
    newer_source = newer_dir / "doc-new.pdf"
    older_v2 = older_dir / "doc-old.v2.items.json"
    newer_v2 = newer_dir / "doc-new.v2.items.json"

    older_source.write_bytes(b"%PDF-1.4\n")
    newer_source.write_bytes(b"%PDF-1.4\n")
    _write_json(older_v2, {"pages": [{"page_number": 1, "items": []}]})
    _write_json(newer_v2, {"pages": [{"page_number": 1, "items": []}]})

    os.utime(older_source, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.utime(older_v2, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.utime(newer_source, ns=(1_700_000_050_000_000_000, 1_700_000_050_000_000_000))
    os.utime(newer_v2, ns=(1_700_000_100_000_000_000, 1_700_000_100_000_000_000))

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert [doc.base_name for doc in result.response.documents] == ["doc-new", "doc-old"]
    assert result.response.documents[0].last_modified_ms > result.response.documents[1].last_modified_ms


def test_build_index_prefers_pdf_when_pdf_and_image_exist(tmp_path: Path) -> None:
    doc_dir = tmp_path / "suite" / "candidate_model" / "default"
    doc_dir.mkdir(parents=True)

    (doc_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    (doc_dir / "sample.png").write_bytes(b"PNG")
    _write_json(doc_dir / "sample.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].source_kind == "pdf"


def test_build_index_skips_multiple_image_sources(tmp_path: Path) -> None:
    doc_dir = tmp_path / "suite" / "candidate_model" / "default"
    doc_dir.mkdir(parents=True)

    (doc_dir / "sample.png").write_bytes(b"PNG")
    (doc_dir / "sample.jpg").write_bytes(b"JPG")
    _write_json(doc_dir / "sample.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})

    result = build_index(str(tmp_path), page=1, page_size=100)

    assert result.response.document_total == 0
    assert result.response.counts.skipped == 1


def test_build_index_uses_explicit_test_cases_path_for_results_only_folder(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "group_a"
    test_cases_dir = tmp_path / "test_cases" / "group_a"
    results_dir.mkdir(parents=True)
    test_cases_dir.mkdir(parents=True)

    _write_json(results_dir / "doc.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})
    (test_cases_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")

    result = build_index(
        str(tmp_path / "results"),
        page=1,
        page_size=100,
        test_cases_path=str(tmp_path / "test_cases"),
    )

    assert result.response.document_total == 1
    assert result.response.documents[0].base_name == "doc"
    assert any("test cases path override" in warning.lower() for warning in result.response.warnings)


def test_build_index_uses_metadata_test_cases_dir_with_ci_path_remap(tmp_path: Path) -> None:
    run_root = tmp_path / "shared-data" / "bench-data"
    results_root = run_root / "results" / "2026-02-26" / "run123" / "candidate_pipeline"
    test_cases_root = run_root / "data" / "visual_grounding" / "v1.3"

    result_doc_dir = results_root / "tables_core" / "candidate_layout" / "v1.0"
    source_doc_dir = test_cases_root / "tables_core" / "candidate_layout" / "v1.0"

    result_doc_dir.mkdir(parents=True)
    source_doc_dir.mkdir(parents=True)

    _write_json(
        results_root / "_metadata.json",
        {"test_cases_dir": "/datasets/bench-data/data/visual_grounding/v1.3"},
    )
    _write_json(
        result_doc_dir / "sample.2020.page_26_000001_page1.pdf.v2.items.json",
        {"pages": [{"page_number": 1, "items": []}]},
    )
    (source_doc_dir / "sample.2020.page_26.pdf_000001_page1.pdf").write_bytes(b"%PDF-1.4\n")

    result = build_index(str(results_root), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].base_name == "sample.2020.page_26_000001_page1"
    assert any("via metadata" in warning.lower() for warning in result.response.warnings)


def test_build_index_skips_ambiguous_stem_only_match_in_test_cases_override(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    test_cases_root = tmp_path / "test-cases"
    results_root.mkdir()
    _write_json(results_root / "doc.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})

    (test_cases_root / "folder_a").mkdir(parents=True)
    (test_cases_root / "folder_b").mkdir(parents=True)
    (test_cases_root / "folder_a" / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    (test_cases_root / "folder_b" / "doc.pdf").write_bytes(b"%PDF-1.4\n")

    result = build_index(
        str(results_root),
        page=1,
        page_size=100,
        test_cases_path=str(test_cases_root),
    )

    assert result.response.document_total == 0
    assert result.response.counts.skipped == 1
    assert any("stem matches" in warning.lower() for warning in result.response.warnings)


def test_build_index_accepts_files_url_path(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-data"
    results_root = shared_root / "bench-data" / "results" / "2026-02-26" / "run123" / "candidate_pipeline"
    results_root.mkdir(parents=True)
    (results_root / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    _write_json(results_root / "doc.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))

    result = build_index(
        "http://localhost/files/bench-data/results/2026-02-26/run123/candidate_pipeline",
        page=1,
        page_size=100,
    )

    assert result.response.document_total == 1
    assert result.response.resolved_root_path == str(results_root.resolve(strict=True))
    assert any("mapped files url" in warning.lower() for warning in result.response.warnings)


def test_build_index_extracts_scalar_evaluation_metrics_only(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    doc_dir = run_root / "suite"
    doc_dir.mkdir(parents=True)

    (doc_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    _write_json(doc_dir / "doc.v2.items.json", {"pages": [{"page_number": 1, "items": []}]})
    _write_json(
        run_root / "_evaluation_report.json",
        {
            "per_example_results": [
                {
                    "example_id": "suite/doc",
                    "test_id": "suite/doc",
                    "metrics": [
                        {"metric_name": "rule_pass_rate", "value": 0.75},
                        {"metric_name": "layout_element_rule_pass_rate", "value": 0.5},
                        {"metric_name": "non_numeric", "value": "skip-me"},
                    ],
                }
            ]
        },
    )

    result = build_index(str(run_root), page=1, page_size=100)

    assert result.response.document_total == 1
    assert result.response.documents[0].evaluation_metrics == {
        "rule_pass_rate": 0.75,
        "layout_element_rule_pass_rate": 0.5,
    }
