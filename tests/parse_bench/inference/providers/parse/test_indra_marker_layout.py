"""Indra and local Marker outputs -> layout_pages."""

from parse_bench.inference.providers.parse import indra
from parse_bench.inference.providers.parse.marker_local import _build_layout_pages as marker_layout


def test_indra_layout_pages_scale_render_pixels_to_page() -> None:
    page = {
        "page_index": 0,
        "width": 72.0,
        "height": 144.0,
        "scale": 2.0,  # 144 dpi render: 2 px per point
        "blocks": [
            {"label": "Section-Header", "bbox": [14.4, 28.8, 72.0, 43.2], "text": "Title"},
            {"label": "Picture", "bbox": [0, 144, 144, 288]},
            {"label": "Page-Footer", "bbox": [60, 270, 84, 280], "text": "3"},
            {"label": "Unknown", "bbox": [0, 0, 10, 10]},
        ],
    }
    items = indra._build_layout_pages([page])[0].items
    assert [i.type for i in items] == ["Section-header", "Picture", "Page-footer"]
    b = items[0].bbox
    assert abs(b.x - 0.1) < 1e-9 and abs(b.y - 0.1) < 1e-9 and abs(b.w - 0.4) < 1e-9 and abs(b.h - 0.05) < 1e-9
    assert abs(items[1].bbox.y - 0.5) < 1e-9 and abs(items[1].bbox.h - 0.5) < 1e-9


def test_indra_page_frames_follow_dpi_and_pixel_cap(tmp_path) -> None:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(612, 792)
    pdf.new_page(20000, 20000)  # huge: render capped at MAX_PAGE_PIXELS
    path = tmp_path / "doc.pdf"
    pdf.save(str(path))
    (w0, h0, k0), (_, _, k1) = indra._page_frames(path, 200.0)
    assert (w0, h0) == (612.0, 792.0) and abs(k0 - 200 / 72) < 1e-9
    assert abs(k1 - (100e6 / 20000**2) ** 0.5) < 1e-9


def test_marker_layout_pages_flatten_groups_and_map_types() -> None:
    doc = {
        "children": [
            {
                "block_type": "Page",
                "bbox": [0, 0, 200, 100],
                "children": [
                    {"block_type": "PageHeader", "bbox": [10, 0, 100, 5], "html": "<p>hdr</p>"},
                    {
                        "block_type": "FigureGroup",
                        "bbox": [0, 10, 200, 60],
                        "children": [
                            {"block_type": "Picture", "bbox": [0, 10, 200, 50], "html": ""},
                            {"block_type": "Caption", "bbox": [0, 50, 200, 60], "html": "<p>Fig 1</p>"},
                        ],
                    },
                    {"block_type": "TextInlineMath", "bbox": [0, 60, 200, 80], "html": "<p>x</p>"},
                    {"block_type": "Line", "bbox": [0, 80, 200, 90]},
                ],
            }
        ]
    }
    page = marker_layout(doc)[0]
    assert [i.layout_segments[0].label for i in page.items] == ["Page-header", "Picture", "Caption", "Text"]
    assert abs(page.items[1].bbox.w - 1.0) < 1e-9 and abs(page.items[1].bbox.y - 0.1) < 1e-9
    assert page.items[0].value == "hdr" and page.items[2].value == "Fig 1"


def test_marker_label_map_covers_datalab_map() -> None:
    import pytest

    datalab = pytest.importorskip("parse_bench.inference.providers.parse.datalab")
    from parse_bench.inference.providers.parse.marker_local import MARKER_LABEL_MAP

    assert {k: MARKER_LABEL_MAP[k] for k in datalab.DATALAB_LABEL_MAP} == datalab.DATALAB_LABEL_MAP


def test_vendor_content_strips_block_html() -> None:
    from parse_bench.evaluation.layout_adapters.adapters import _build_vendor_content

    content = _build_vendor_content("Text", '<p block-type="Text"><a href="#">&lt; BACK</a> to <b>start</b></p>')
    assert content is not None and " ".join(content.text.split()) == "< BACK to start"
    sub = _build_vendor_content("Text", "<p>L<sub>1</sub> <b>Net</b>work<br>next</p>")
    assert sub is not None and sub.text == " L<sub>1</sub> Network next "
    for kept in ["<PL Number> and <Include a summary>", '<MARK Position="30 40"> <VIEW>', "the <IR> Framework"]:
        content = _build_vendor_content("Text", kept)
        assert content is not None and content.text == kept
    plain = _build_vendor_content("Text", "a < b and c > d")
    assert plain is not None and plain.text == "a < b and c > d"
