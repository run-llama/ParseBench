"""Coverage for the OCRia-1 provider (``ocria_1`` pipeline): its post-processing and box contract."""

from __future__ import annotations

from datetime import datetime

from parse_bench.inference.providers.parse.ocria_1 import OCRia1Provider, ocria_postprocess
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _provider(**cfg) -> OCRia1Provider:
    return OCRia1Provider("ocria_1", {"endpoint_url": "http://localhost:8000/v1", **cfg})


def _raw_result(raw_output: dict, example_id: str = "layout/doc") -> RawInferenceResult:
    now = datetime.now()
    pipeline = PipelineSpec(pipeline_name="ocria_1", provider_name="ocria_1", product_type=ProductType.PARSE)
    return RawInferenceResult(
        request=InferenceRequest(example_id=example_id, source_file_path="doc.pdf", product_type=ProductType.PARSE),
        pipeline=pipeline,
        pipeline_name="ocria_1",
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def test_box_server_pixels_become_pdf_points_on_the_layout_page() -> None:
    """The box server answers in 300-DPI pixels; the layout page is in PDF points (x 72/300). Missing the scale
    puts every box ~4x off the page, and the grounding dimension scores 0."""
    raw = {
        "dpi": 300,
        "pages": [
            {
                "answer": "```markdown\n# Title\n\nBody text.\n```",
                "width_px": 2550,
                "height_px": 3300,
                "layout": {
                    "width": 2550,
                    "height": 3300,
                    "items": [
                        {
                            "type": "Section-header",
                            "label": "paragraph_title",
                            "md": "# Title",
                            "box": [300, 600, 1500, 750],
                        }
                    ],
                },
            }
        ],
    }
    out = _provider().normalize(_raw_result(raw)).output
    page = out.layout_pages[0]
    assert (page.width, page.height) == (612.0, 792.0)
    seg = page.items[0].bbox
    assert (seg.x, seg.y, seg.w, seg.h) == (72.0, 144.0, 288.0, 36.0)
    assert page.items[0].layout_segments[0].label == "paragraph_title"


def test_postprocess_unwraps_the_answer_and_keeps_furniture_text() -> None:
    """The raw answer arrives in a ```markdown wrapper with page-furniture tags; left in, both cost text scores."""
    raw = (
        "```markdown\n<!-- layout: r1 -->\n<page_header>ACME 2024</page_header>\n\n"
        "# Title\n\nSee ![icon: logo] here.\n```"
    )
    md = ocria_postprocess(raw)
    assert "```" not in md and "<page_header>" not in md and "<!--" not in md
    assert "ACME 2024" in md and "# Title" in md
    assert "![" not in md


def test_postprocess_turns_pipe_tables_into_html() -> None:
    """ParseBench scores tables against HTML; a markdown pipe table left as is scores about 0."""
    md = ocria_postprocess("| a | b |\n|---|---|\n| 1 | 2 |")
    assert md.startswith("<table>") and "<td>1</td>" in md


def test_only_layout_pages_are_sent_to_the_box_server_by_default() -> None:
    """Only the layout subset reads boxes; boxing all 4,317 pages would add hours to a run."""
    p = _provider(grounding_url="http://boxes:8100")

    def req(eid: str) -> InferenceRequest:
        return InferenceRequest(example_id=eid, source_file_path="doc.pdf", product_type=ProductType.PARSE)

    assert p._wants_boxes(req("layout/annotated__page_p1"))
    assert not p._wants_boxes(req("text/text_simple__page"))
    assert _provider(grounding_url="http://boxes:8100", grounding="all")._wants_boxes(req("text/text_simple__page"))
    assert not _provider()._wants_boxes(req("layout/annotated__page_p1"))
