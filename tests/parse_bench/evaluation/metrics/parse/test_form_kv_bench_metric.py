"""Bench bridge pins: both arms from one result, md-only pipelines, loud zeros, spot GT, names."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from parse_bench.evaluation.metrics.parse.form_kv_scoring import render_guide
from parse_bench.evaluation.metrics.parse.form_kv_scoring.bench_metric import (
    compute_form_kv_metrics,
    scores_markdown_only,
)

NODES: list[dict[str, Any]] = [
    {"type": "field", "field": "text", "id": "a", "label": "Name", "value": "Ada Lovelace"},
    {"type": "field", "field": "checkbox", "id": "b", "label": "Retired", "value": True},
    {"type": "field", "field": "text", "id": "c", "label": "Suffix", "isEmpty": True},
]
GUIDE_MD = render_guide(NODES)
TODAYS_MD = "**a Name**: Ada Lovelace\n\n**b Retired**: [x]\n\n**c Suffix**:\n"


def gt_doc(items: list[dict[str, Any]] = NODES) -> dict[str, Any]:
    return {"doc_id": "d", "pages": [{"page_number": 1, "page_width": 612.0, "page_height": 792.0, "items": items}]}


def forms_result(*page_mds: str | None) -> dict[str, Any]:
    pages = []
    for n, md in enumerate(page_mds, start=1):
        if md is None:
            pages.append({"page_number": n, "error": "boom", "success": False})
        else:
            pages.append(
                {
                    "page_number": n,
                    "forms": [{"json": [], "list": {"type": "list", "items": [], "ordered": False, "md": md}}],
                    "success": True,
                }
            )
    return {"pages": pages}


def result_with(
    forms: dict[str, Any] | None = None, page_mds: list[str] | None = None, doc_md: str = ""
) -> SimpleNamespace:
    raw_output = {"forms": forms} if forms is not None else {}
    pages = [SimpleNamespace(page_index=i, markdown=md) for i, md in enumerate(page_mds or [])]
    return SimpleNamespace(raw_output=raw_output, output=SimpleNamespace(pages=pages, markdown=doc_md))


def by_name(metrics: list[Any]) -> dict[str, Any]:
    return {m.metric_name: m for m in metrics}


def test_both_arms_from_one_result() -> None:
    metrics = by_name(
        compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), [TODAYS_MD]), {"gt": gt_doc(), "doc_id": "d"})
    )
    assert metrics["form_kv_list_composite"].value == 1.0
    assert metrics["form_kv_list_core_kv_f1"].value == 1.0
    assert metrics["form_kv_list_value_f1"].value == 1.0 and metrics["form_kv_list_id_f1"].value == 1.0
    assert "form_kv_list_signature_f1" not in metrics
    # the page markdown is read through the dialect adapter: bold-colon paragraphs are
    # records, with the printed id baked into the label (the lenient id dual credits it)
    md = metrics["form_kv_md_composite"]
    assert md.value > 0.8 and metrics["form_kv_md_core_kv_f1"].value == 1.0
    assert metrics["form_kv_md_id_f1"].value == 0.0 and metrics["form_kv_md_id_f1_lenient"].value == 1.0
    assert md.metadata["pred_bold_colon_unparsed"] == 0 and md.metadata["pred_md_adapted"] == 3
    assert md.details == ["3 page-markdown lines read through the markdown-dialect adapter (3 key paragraphs)"]
    assert metrics["form_kv_list_composite"].metadata["doc_id"] == "d"
    assert metrics["form_kv_list_composite"].details == []


def test_the_list_arm_is_not_adapted() -> None:
    """The enriched list is scored as written: the paragraph dialect stays a loss there."""
    metrics = by_name(compute_form_kv_metrics(result_with(forms_result(TODAYS_MD), []), {"gt": gt_doc()}))
    assert metrics["form_kv_list_composite"].value == 0.0
    assert metrics["form_kv_list_composite"].metadata["pred_bold_colon_unparsed"] == 3
    assert "non-list lines unparsed" in metrics["form_kv_list_composite"].details[0]


def test_md_only_pipeline_emits_the_md_arm_only() -> None:
    metrics = by_name(compute_form_kv_metrics(result_with(None, [GUIDE_MD]), {"gt": gt_doc()}))
    assert metrics["form_kv_md_composite"].value == 1.0
    assert not any(name.startswith("form_kv_list_") for name in metrics)


def test_markdown_scope_withholds_the_list_arm_even_with_an_enriched_result() -> None:
    meta = {"gt": gt_doc(), "scope": "markdown"}
    assert scores_markdown_only(meta) and not scores_markdown_only({"gt": gt_doc()})
    metrics = by_name(compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), [GUIDE_MD]), meta))
    assert metrics["form_kv_md_composite"].value == 1.0
    assert not any(name.startswith("form_kv_list_") for name in metrics)
    (row,) = compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), []), meta)
    assert row.metric_name == "form_kv_md_composite" and row.value == 0.0 and "nothing to score" in row.details[0]


def test_missing_or_broken_gt_is_loud() -> None:
    metrics = compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), [GUIDE_MD]), {"gt": {"pages": []}})
    assert [m.metric_name for m in metrics] == ["form_kv_list_composite", "form_kv_md_composite"]
    assert all(m.value == 0.0 and "broken dataset export" in m.details[0] for m in metrics)
    (row,) = compute_form_kv_metrics(result_with(None, []), {"gt": gt_doc()})
    assert row.metric_name == "form_kv_md_composite" and row.value == 0.0 and "nothing to score" in row.details[0]


def test_failed_form_page_scores_as_misses() -> None:
    metrics = by_name(compute_form_kv_metrics(result_with(forms_result(None), [GUIDE_MD]), {"gt": gt_doc()}))
    assert metrics["form_kv_list_composite"].value < 1.0
    assert metrics["form_kv_list_composite"].metadata["failed_pages"] == 1
    assert "failed in the product output" in metrics["form_kv_list_composite"].details[0]


def test_markdown_gt_with_a_spot_page_emits_spot_rows_only() -> None:
    meta = {"gt_pages_md": [{"page_number": 1, "markdown": "- \\[a\\] **Name**: Ada Lovelace", "coverage": "spot"}]}
    metrics = by_name(compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), []), meta))
    assert "form_kv_list_composite" not in metrics
    assert metrics["form_kv_list_spot_composite"].value == 1.0
    assert metrics["form_kv_list_spot_core_kv_recall"].value == 1.0
    assert metrics["form_kv_list_spot_composite"].metadata["spot_pages"] == 1


def test_document_level_markdown_scores_as_page_one_against_a_one_page_gt() -> None:
    metrics = by_name(compute_form_kv_metrics(result_with(None, [], doc_md=GUIDE_MD), {"gt": gt_doc()}))
    assert metrics["form_kv_md_composite"].value == 1.0
    assert "document-level markdown scored as page 1" in metrics["form_kv_md_composite"].details
    two_pages = {"gt": {"pages": [gt_doc()["pages"][0], dict(gt_doc()["pages"][0], page_number=2)]}}
    (row,) = compute_form_kv_metrics(result_with(None, [], doc_md=GUIDE_MD), two_pages)
    assert row.value == 0.0 and "cannot be paired with a 2-page GT" in row.details[0]


def test_malformed_coverage_in_markdown_gt_is_loud() -> None:
    meta = {"gt_pages_md": [{"page_number": 1, "markdown": "- **A**: 1", "coverage": "sopt"}]}
    (row,) = compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), []), meta)
    assert row.value == 0.0 and "unknown coverage 'sopt'" in row.details[0]


def test_gt_round_trip_loss_is_surfaced() -> None:
    lossy = [
        {
            "type": "field",
            "field": "single_select",
            "label": "Type",
            "valueItems": [
                {"type": "field", "field": "checkbox", "label": "Checking", "value": True},
                {"type": "field", "field": "text", "label": "Other", "value": "trust"},
            ],
        }
    ]
    metrics = by_name(compute_form_kv_metrics(result_with(forms_result(GUIDE_MD), []), {"gt": gt_doc(lossy)}))
    row = metrics["form_kv_list_composite"]
    assert row.metadata["gt_roundtrip_loss"] > 0
    assert any("GT render round trip changed" in d for d in row.details)


def test_coverage_comment_in_prediction_markdown_is_ignored() -> None:
    tagged = "<!-- coverage: partial -->\n" + GUIDE_MD
    metrics = by_name(compute_form_kv_metrics(result_with(forms_result(tagged), [tagged]), {"gt": gt_doc()}))
    assert metrics["form_kv_list_composite"].value == 1.0 and metrics["form_kv_md_composite"].value == 1.0
