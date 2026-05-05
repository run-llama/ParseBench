from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
import pytest

from backend.indexer import IndexedDocumentInternal
from backend.loader import load_document
from backend.models import ArtifactFlags
from backend.gt_rules import _find_extract_field_metric_result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_image(path: Path) -> None:
    image = Image.new("RGB", (640, 480), color=(255, 255, 255))
    image.save(path)


def _make_doc(tmp_path: Path) -> IndexedDocumentInternal:
    source = tmp_path / "doc.png"
    _make_image(source)
    return IndexedDocumentInternal(
        doc_id="doc1",
        base_name="doc",
        relative_dir=".",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=None,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=False,
            has_v2_items_payload=True,
        ),
    )


def _make_parse_result_payload(
    *,
    pipeline_name: str,
    raw_output: dict,
    layout_items: list[dict],
    width: float = 640,
    height: float = 480,
) -> dict:
    return {
        "request": {
            "example_id": "doc1",
            "source_file_path": "/tmp/doc.png",
            "product_type": "parse",
            "schema_override": None,
            "config_override": None,
        },
        "pipeline_name": pipeline_name,
        "product_type": "parse",
        "raw_output": raw_output,
        "output": {
            "task_type": "parse",
            "example_id": "doc1",
            "pipeline_name": pipeline_name,
            "pages": [],
            "layout_pages": [
                {
                    "page_number": 1,
                    "width": width,
                    "height": height,
                    "items": layout_items,
                }
            ],
            "markdown": "",
        },
        "latency_in_ms": 1,
    }


def _make_layout_detection_result_payload(
    *,
    pipeline_name: str,
    raw_output: dict,
    width: float = 640,
    height: float = 480,
) -> dict:
    return {
        "request": {
            "example_id": "doc1",
            "source_file_path": "/tmp/doc.png",
            "product_type": "layout_detection",
            "schema_override": None,
            "config_override": None,
        },
        "pipeline_name": pipeline_name,
        "product_type": "layout_detection",
        "raw_output": raw_output,
        "output": {
            "task_type": "layout_detection",
            "example_id": "doc1",
            "pipeline_name": pipeline_name,
            "model": "llamaparse",
            "image_width": width,
            "image_height": height,
            "predictions": [],
            "markdown": "",
        },
        "latency_in_ms": 1,
    }


def _layer_map(loaded) -> dict[str, object]:
    return {layer.granularity: layer for layer in loaded.pages[0].granular_layers}


def test_loader_prefers_v2_items_file(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    v2_path = tmp_path / "doc.v2.items.json"
    raw_path = tmp_path / "doc.raw.json"

    _write_json(
        v2_path,
        {
            "pages": [
                {
                    "page_number": 1,
                    "page_width": 640,
                    "page_height": 480,
                    "items": [{"type": "text", "md": "from_v2", "bbox": []}],
                }
            ]
        },
    )
    _write_json(
        raw_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_raw", "bbox": []}],
                        }
                    ]
                }
            }
        },
    )

    doc.v2_items_path = v2_path
    doc.raw_path = raw_path

    loaded = load_document(doc)

    assert loaded.selected_grounding_source == "v2_items"
    assert loaded.pages[0].items[0].md == "from_v2"


def test_loader_falls_back_to_raw_then_result(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    raw_path = tmp_path / "doc.raw.json"
    result_path = tmp_path / "doc.result.json"

    _write_json(
        raw_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_raw", "bbox": []}],
                        }
                    ]
                }
            }
        },
    )
    _write_json(
        result_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_result", "bbox": []}],
                        }
                    ]
                }
            }
        },
    )

    doc.raw_path = raw_path
    doc.result_path = result_path

    loaded = load_document(doc)
    assert loaded.selected_grounding_source == "raw"
    assert loaded.pages[0].items[0].md == "from_raw"

    doc.raw_path = None
    loaded_result = load_document(doc)
    assert loaded_result.selected_grounding_source == "result"
    assert loaded_result.pages[0].items[0].md == "from_result"


def test_loader_prefers_result_layout_pages_over_legacy_sources(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    v2_path = tmp_path / "doc.v2.items.json"
    raw_path = tmp_path / "doc.raw.json"
    result_path = tmp_path / "doc.result.json"

    _write_json(
        v2_path,
        {
            "pages": [
                {
                    "page_number": 1,
                    "page_width": 640,
                    "page_height": 480,
                    "items": [{"type": "text", "md": "from_v2", "bbox": []}],
                }
            ]
        },
    )
    _write_json(
        raw_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_raw", "bbox": []}],
                        }
                    ]
                }
            }
        },
    )
    _write_json(
        result_path,
        {
            "output": {
                "markdown": "# From normalized document",
                "layout_pages": [
                    {
                        "page_number": 1,
                        "width": 640,
                        "height": 480,
                        "md": "# From normalized page",
                        "items": [
                            {
                                "type": "heading",
                                "value": "from_result_layout",
                                "bbox": {"x": 0.1, "y": 0.2, "w": 0.25, "h": 0.1},
                            }
                        ],
                    }
                ],
            }
        },
    )

    doc.v2_items_path = v2_path
    doc.raw_path = raw_path
    doc.result_path = result_path

    loaded = load_document(doc)

    assert loaded.selected_grounding_source == "result"
    assert loaded.pages[0].items[0].md == "from_result_layout"
    assert loaded.pages[0].items[0].type == "heading"
    assert loaded.pages[0].items[0].bboxes[0].x == 64.0
    assert loaded.pages[0].items[0].bboxes[0].y == 96.0
    assert loaded.pages[0].items[0].bboxes[0].w == 160.0
    assert loaded.pages[0].items[0].bboxes[0].h == 48.0
    assert loaded.selected_markdown_source == "result"
    assert loaded.pages[0].markdown == "# From normalized page"
    assert loaded.document_markdown == "# From normalized document"


def test_loader_falls_back_from_empty_normalized_tables_to_raw_items(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    raw_path = tmp_path / "doc.raw.json"
    result_path = tmp_path / "doc.result.json"

    _write_json(
        raw_path,
        {
            "raw_output": {
                "items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [
                                {
                                    "type": "table",
                                    "html": "<table><tr><td>from_raw_table</td></tr></table>",
                                    "bbox": [],
                                }
                            ],
                        }
                    ]
                }
            }
        },
    )
    _write_json(
        result_path,
        {
            "output": {
                "layout_pages": [
                    {
                        "page_number": 1,
                        "width": 640,
                        "height": 480,
                        "items": [
                            {
                                "type": "table",
                                "value": "",
                                "bbox": {"x": 0.1, "y": 0.2, "w": 0.25, "h": 0.1},
                            }
                        ],
                    }
                ],
            }
        },
    )

    doc.raw_path = raw_path
    doc.result_path = result_path

    loaded = load_document(doc)

    assert loaded.selected_grounding_source == "raw"
    assert loaded.pages[0].items[0].type == "table"
    assert loaded.pages[0].items[0].md == "<table><tr><td>from_raw_table</td></tr></table>"


def test_loader_prefers_raw_layout_pages_over_v2_items_sidecar(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    v2_path = tmp_path / "doc.v2.items.json"
    raw_path = tmp_path / "doc.raw.json"

    _write_json(
        v2_path,
        {
            "pages": [
                {
                    "page_number": 1,
                    "page_width": 640,
                    "page_height": 480,
                    "items": [{"type": "text", "md": "from_v2", "bbox": []}],
                }
            ]
        },
    )
    _write_json(
        raw_path,
        {
            "output": {
                "layout_pages": [
                    {
                        "page_number": 1,
                        "width": 640,
                        "height": 480,
                        "items": [
                            {
                                "type": "text",
                                "value": "from_raw_layout",
                                "layout_segments": [
                                    {"x": 0.5, "y": 0.25, "w": 0.125, "h": 0.2, "startIndex": 1, "endIndex": 4}
                                ],
                            }
                        ],
                    }
                ]
            }
        },
    )

    doc.v2_items_path = v2_path
    doc.raw_path = raw_path

    loaded = load_document(doc)

    assert loaded.selected_grounding_source == "raw"
    assert loaded.pages[0].items[0].md == "from_raw_layout"
    assert loaded.pages[0].items[0].bboxes[0].x == 320.0
    assert loaded.pages[0].items[0].bboxes[0].y == 120.0
    assert loaded.pages[0].items[0].bboxes[0].w == 80.0
    assert loaded.pages[0].items[0].bboxes[0].h == 96.0
    assert loaded.pages[0].items[0].bboxes[0].start_index == 1
    assert loaded.pages[0].items[0].bboxes[0].end_index == 4


def test_loader_accepts_raw_items_pages_payload(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    raw_path = tmp_path / "doc.raw.json"
    _write_json(
        raw_path,
        {
            "raw_output": {
                "items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_items", "bbox": []}],
                        }
                    ]
                }
            }
        },
    )

    doc.raw_path = raw_path
    loaded = load_document(doc)

    assert loaded.selected_grounding_source == "raw"
    assert loaded.pages[0].items[0].md == "from_items"


def test_loader_uses_item_html_when_markdown_is_missing(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    raw_path = tmp_path / "doc.raw.json"
    _write_json(
        raw_path,
        {
            "raw_output": {
                "items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [
                                {
                                    "type": "table",
                                    "html": "<table><tr><td>from_html</td></tr></table>",
                                    "bbox": [],
                                }
                            ],
                        }
                    ]
                }
            }
        },
    )

    doc.raw_path = raw_path
    loaded = load_document(doc)

    assert loaded.selected_grounding_source == "raw"
    assert loaded.pages[0].items[0].type == "table"
    assert loaded.pages[0].items[0].md == "<table><tr><td>from_html</td></tr></table>"


def test_loader_prefers_sidecar_markdown_when_available(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    raw_path = tmp_path / "doc.raw.json"
    markdown_path = tmp_path / "doc.md"
    _write_json(
        raw_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_raw", "bbox": []}],
                        }
                    ]
                },
                "v2_md": {
                    "pages": [
                        {
                            "page_number": 1,
                            "markdown": "# From raw markdown",
                        }
                    ]
                },
            }
        },
    )
    markdown_path.write_text("# From sidecar markdown", encoding="utf-8")

    doc.raw_path = raw_path
    doc.markdown_path = markdown_path

    loaded = load_document(doc)

    assert loaded.selected_markdown_source == "sidecar_md"
    assert loaded.document_markdown == "# From sidecar markdown"
    assert loaded.pages[0].markdown == "# From sidecar markdown"


def test_loader_extracts_page_markdown_from_raw_then_result(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    raw_path = tmp_path / "doc.raw.json"
    result_path = tmp_path / "doc.result.json"
    _write_json(
        raw_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_raw", "bbox": []}],
                        }
                    ]
                },
                "v2_md": {
                    "pages": [
                        {
                            "page_number": 1,
                            "markdown": "# Raw markdown",
                        }
                    ]
                },
            }
        },
    )
    _write_json(
        result_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [{"type": "text", "md": "from_result", "bbox": []}],
                        }
                    ]
                },
                "v2_md": {
                    "pages": [
                        {
                            "page_number": 1,
                            "markdown": "# Result markdown",
                        }
                    ]
                },
            }
        },
    )

    doc.raw_path = raw_path
    doc.result_path = result_path

    loaded = load_document(doc)
    assert loaded.selected_markdown_source == "raw"
    assert loaded.pages[0].markdown == "# Raw markdown"

    doc.raw_path = None
    loaded_result = load_document(doc)
    assert loaded_result.selected_markdown_source == "result"
    assert loaded_result.pages[0].markdown == "# Result markdown"


def test_loader_reads_v2_md_sidecar_payload(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)

    v2_items_path = tmp_path / "doc.v2.items.json"
    v2_md_path = tmp_path / "doc.v2.md.json"
    _write_json(
        v2_items_path,
        {
            "pages": [
                {
                    "page_number": 1,
                    "items": [{"type": "text", "md": "from_v2_items", "bbox": []}],
                }
            ]
        },
    )
    _write_json(
        v2_md_path,
        {
            "pages": [
                {
                    "page_number": 1,
                    "markdown": "# From v2 md sidecar",
                }
            ]
        },
    )

    doc.v2_items_path = v2_items_path
    doc.markdown_json_path = v2_md_path
    loaded = load_document(doc)

    assert loaded.selected_markdown_source == "sidecar_md"
    assert loaded.pages[0].markdown == "# From v2 md sidecar"
    assert loaded.document_markdown == "# From v2 md sidecar"


def test_loader_exposes_source_file_url_when_source_is_under_shared_root(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-experiments"
    shared_root.mkdir(parents=True)
    doc = _make_doc(shared_root)
    raw_path = shared_root / "doc.raw.json"

    _write_json(
        raw_path,
        {
            "raw_output": {
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "items": [],
                        }
                    ]
                }
            }
        },
    )
    doc.raw_path = raw_path

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))
    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_BASE_URL", "http://files.example.test")

    loaded = load_document(doc)

    assert loaded.source_file_url == "http://files.example.test/files/doc.png"


def test_loader_exposes_textract_granular_layers(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        {
                            "Id": "line-1",
                            "BlockType": "LINE",
                            "Text": "Record REC-0000",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.05}},
                        },
                        {
                            "Id": "word-1",
                            "BlockType": "WORD",
                            "Text": "Record",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.12, "Height": 0.05}},
                        },
                        {
                            "Id": "word-2",
                            "BlockType": "WORD",
                            "Text": "REC-0000",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.24, "Top": 0.2, "Width": 0.16, "Height": 0.05}},
                        },
                        {
                            "Id": "cell-1",
                            "BlockType": "CELL",
                            "RowIndex": 1,
                            "ColumnIndex": 1,
                            "RowSpan": 1,
                            "ColumnSpan": 1,
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.08, "Top": 0.18, "Width": 0.34, "Height": 0.08}},
                            "Relationships": [{"Type": "CHILD", "Ids": ["word-1", "word-2"]}],
                        },
                    ]
                }
            },
            layout_items=[
                {
                    "type": "text",
                    "value": "Record REC-0000",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05},
                }
            ],
        ),
    )

    doc.result_path = result_path
    loaded = load_document(doc)
    layers = _layer_map(loaded)

    line_layer = layers["line"]
    word_layer = layers["word"]
    cell_layer = layers["cell"]

    assert line_layer.availability == "available"
    assert [unit.text for unit in line_layer.units] == ["Record REC-0000"]
    assert word_layer.availability == "available"
    assert [unit.text for unit in word_layer.units] == ["Record", "REC-0000"]
    assert cell_layer.availability == "available"
    assert len(cell_layer.units) == 1
    assert cell_layer.units[0].text == "Record REC-0000"
    assert cell_layer.units[0].row_index == 0
    assert cell_layer.units[0].column_index == 0
    assert cell_layer.units[0].bbox.x == 51.2
    assert len(cell_layer.units[0].bboxes) == 1


def test_loader_exposes_llamaparse_cells_from_grounded_rows(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="llamaparse_local_cli2",
            raw_output={
                "v2_grounded_items": [
                    {
                        "page_number": 1,
                        "page_width": 640,
                        "page_height": 480,
                        "items": [
                            {
                                "type": "table",
                                "rows": [["Alpha", "42"]],
                                "grounding": {
                                    "rows": [
                                        [
                                            {
                                                "bbox": [
                                                    {"x": 100, "y": 120, "w": 34, "h": 20},
                                                    {"x": 146, "y": 120, "w": 34, "h": 20},
                                                ],
                                                "lines": [
                                                    {
                                                        "span": [0, 5],
                                                        "bbox": {"x": 100, "y": 120, "w": 80, "h": 20},
                                                        "words": [
                                                            {
                                                                "span": [0, 5],
                                                                "bbox": {"x": 100, "y": 120, "w": 80, "h": 20},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {
                                                "bbox": [{"x": 220, "y": 120, "w": 40, "h": 20}],
                                                "lines": [
                                                    {
                                                        "span": [0, 2],
                                                        "bbox": {"x": 220, "y": 120, "w": 40, "h": 20},
                                                        "words": [
                                                            {
                                                                "span": [0, 2],
                                                                "bbox": {"x": 220, "y": 120, "w": 40, "h": 20},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                        ]
                                    ]
                                },
                            }
                        ],
                    }
                ]
            },
            layout_items=[
                {
                    "type": "table",
                    "md": "| Alpha | 42 |",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1},
                }
            ],
        ),
    )

    doc.result_path = result_path
    loaded = load_document(doc)
    layers = _layer_map(loaded)

    cell_layer = layers["cell"]
    assert cell_layer.availability == "available"
    assert [unit.text for unit in cell_layer.units] == ["Alpha", "42"]
    assert cell_layer.units[0].bbox.x == 100
    assert cell_layer.units[0].bbox.w == 80
    assert len(cell_layer.units[0].bboxes) == 2
    assert cell_layer.units[0].bboxes[0].w == 34
    assert cell_layer.units[0].bboxes[1].x == 146
    assert cell_layer.units[1].bbox.w == 40
    assert len(cell_layer.units[1].bboxes) == 1


def test_loader_exposes_llamaparse_granular_layers_from_layout_detection_results(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"

    _write_json(
        result_path,
        _make_layout_detection_result_payload(
            pipeline_name="candidate_granular_bboxes",
            raw_output={
                "v2_items": {
                    "pages": [
                        {
                            "page_number": 1,
                            "page_width": 640,
                            "page_height": 480,
                            "items": [
                                {
                                    "type": "text",
                                    "md": "Alpha 42",
                                    "bbox": [{"x": 80, "y": 120, "w": 200, "h": 30}],
                                }
                            ],
                        }
                    ]
                },
                "v2_grounded_items": [
                    {
                        "page_number": 1,
                        "page_width": 640,
                        "page_height": 480,
                        "items": [
                            {
                                "type": "text",
                                "md": "Alpha 42",
                                "bbox": [{"x": 80, "y": 120, "w": 200, "h": 30}],
                                "grounding": {
                                    "source": "md",
                                    "lines": [
                                        {
                                            "span": [0, 8],
                                            "bbox": {"x": 80, "y": 120, "w": 200, "h": 30},
                                            "words": [
                                                {"span": [0, 5], "bbox": {"x": 80, "y": 120, "w": 90, "h": 30}},
                                                {"span": [6, 8], "bbox": {"x": 190, "y": 120, "w": 30, "h": 30}},
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            },
        ),
    )

    doc.result_path = result_path
    loaded = load_document(doc)
    layers = _layer_map(loaded)

    assert layers["line"].availability == "available"
    assert [unit.text for unit in layers["line"].units] == ["Alpha 42"]
    assert layers["word"].availability == "available"
    assert [unit.text for unit in layers["word"].units] == ["Alpha", "42"]


def test_loader_exposes_extract_result_grounded_items_without_layout_pages(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"

    _write_json(
        result_path,
        {
            "request": {
                "example_id": "doc1",
                "source_file_path": "/tmp/doc.png",
                "product_type": "extract",
            },
            "pipeline_name": "extract_pipeline_agentic_granular_bboxes_local",
            "product_type": "extract",
            "raw_output": {
                "data": {"vendor": "Acme Corp"},
                "v2_grounded_items": [
                    {
                        "page_number": 1,
                        "page_width": 640,
                        "page_height": 480,
                        "success": True,
                        "items": [
                            {
                                "type": "text",
                                "md": "Acme Corp",
                                "bbox": [{"x": 64, "y": 48, "w": 120, "h": 20}],
                                "grounding": {
                                    "source": "md",
                                    "lines": [
                                        {
                                            "span": [0, 9],
                                            "bbox": {"x": 64, "y": 48, "w": 120, "h": 20},
                                            "words": [
                                                {"span": [0, 4], "bbox": {"x": 64, "y": 48, "w": 52, "h": 20}},
                                                {"span": [5, 9], "bbox": {"x": 124, "y": 48, "w": 60, "h": 20}},
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            },
            "output": {"vendor": "Acme Corp"},
        },
    )

    doc.result_path = result_path
    loaded = load_document(doc)
    layers = _layer_map(loaded)

    assert loaded.selected_grounding_source == "result"
    assert loaded.pages[0].items[0].md == "Acme Corp"
    assert layers["line"].availability == "available"
    assert [unit.text for unit in layers["line"].units] == ["Acme Corp"]
    assert layers["word"].availability == "available"
    assert [unit.text for unit in layers["word"].units] == ["Acme", "Corp"]


def test_loader_extract_field_gt_rules_use_extract_citation_fallback(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"

    _write_json(
        result_path,
        {
            "request": {
                "example_id": "doc1",
                "source_file_path": "/tmp/doc.png",
                "product_type": "extract",
            },
            "pipeline_name": "extract_pipeline_agentic_granular_bboxes_local",
            "product_type": "extract",
            "output": {
                "task_type": "extract",
                "extracted_data": {"stock_list": [{"catalog_number": "CAT-001"}]},
                "field_citations": [
                    {
                        "field_path": "stock_list[0].catalog_number",
                        "page": 1,
                        "bbox": [0.60, 0.10, 0.08, 0.05],
                        "reference_text": "| Example Supply | Sample Item | CAT-001 | ITEM-0001 |",
                    }
                ],
            },
        },
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {
                "type": "object",
                "properties": {
                    "stock_list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"catalog_number": {"type": "string"}},
                        },
                    }
                },
            },
            "expected_output": {"stock_list": [{"catalog_number": "CAT-001"}]},
            "test_rules": [
                {
                    "id": "rule-catalog",
                    "type": "extract_field",
                    "field_path": "stock_list[0].catalog_number",
                    "expected_value": "CAT-001",
                    "bboxes": [{"page": 1, "bbox": [0.60, 0.10, 0.08, 0.05], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )

    doc.result_path = result_path
    loaded = load_document(doc)

    [item] = loaded.pages[0].items
    [rule] = loaded.pages[0].gt_rules
    assert item.value == "| Example Supply | Sample Item | CAT-001 | ITEM-0001 |"
    assert rule.predicted_granularity == "extract_field"
    assert rule.predicted_text == "CAT-001"
    assert rule.matched_unit_ids == [item.item_id]
    assert rule.iou == pytest.approx(1.0)
    # The citation fallback is display evidence only. Verdicts are a single
    # source of truth from evaluator rule_results, so without an evaluation
    # report these must remain ungraded.
    assert rule.localization_pass is None
    assert rule.classification_pass is None
    assert rule.attribution_pass is None
    assert rule.overall_pass is None
    assert len(rule.predicted_bboxes) == 1
    assert rule.predicted_bboxes[0].x == pytest.approx(384.0)


def test_loader_exposes_extract_field_gt_rules_from_adjacent_test_case(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        {
                            "Id": "line-1",
                            "BlockType": "LINE",
                            "Text": "Record REC-0000",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.05}},
                        },
                        {
                            "Id": "word-1",
                            "BlockType": "WORD",
                            "Text": "Record",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.12, "Height": 0.05}},
                        },
                        {
                            "Id": "word-2",
                            "BlockType": "WORD",
                            "Text": "REC-0000",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.24, "Top": 0.2, "Width": 0.16, "Height": 0.05}},
                        },
                    ]
                }
            },
            layout_items=[
                {
                    "type": "text",
                    "value": "Record REC-0000",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05},
                }
            ],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {"type": "object", "properties": {"record_id": {"type": "string"}}},
            "expected_output": {"record_id": "REC-0000"},
            "test_rules": [
                {
                    "id": "rule-account-number",
                    "type": "extract_field",
                    "field_path": "record_id",
                    "expected_value": "REC-0000",
                    "bboxes": [{"page": 1, "bbox": [0.24, 0.2, 0.16, 0.05], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )

    doc.result_path = result_path

    loaded = load_document(doc)
    rules = loaded.pages[0].gt_rules

    assert len(rules) == 1
    rule = rules[0]
    assert rule.rule_id == "rule-account-number"
    assert rule.field_path == "record_id"
    assert rule.gt_bbox.x == 153.6
    assert rule.predicted_granularity == "word"
    assert rule.predicted_text == "REC-0000"
    assert rule.predicted_bbox is not None
    assert rule.predicted_bbox.x == 153.6
    assert rule.matched_unit_ids == ["word-2"]


def test_loader_uses_explicit_test_case_path_for_gt_rules(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    external_dir = tmp_path / "dataset"
    external_dir.mkdir()
    test_case_path = external_dir / "doc.test.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        {
                            "Id": "word-1",
                            "BlockType": "WORD",
                            "Text": "42",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.5, "Top": 0.4, "Width": 0.08, "Height": 0.04}},
                        }
                    ]
                }
            },
            layout_items=[
                {
                    "type": "text",
                    "value": "42",
                    "bbox": {"x": 0.5, "y": 0.4, "w": 0.08, "h": 0.04},
                }
            ],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "expected_output": {"answer": "42"},
            "test_rules": [
                {
                    "id": "rule-answer",
                    "type": "extract_field",
                    "field_path": "answer",
                    "expected_value": "42",
                    "bboxes": [{"page": 1, "bbox": [0.5, 0.4, 0.08, 0.04], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )

    doc.result_path = result_path
    doc.test_case_path = test_case_path

    loaded = load_document(doc)

    assert len(loaded.pages[0].gt_rules) == 1
    assert loaded.pages[0].gt_rules[0].rule_id == "rule-answer"


def test_loader_extract_field_matching_uses_customer_numeric_value_rules(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        {
                            "Id": "line-1",
                            "BlockType": "LINE",
                            "Text": "Total $3,676.69",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.4, "Height": 0.05}},
                        },
                        {
                            "Id": "word-1",
                            "BlockType": "WORD",
                            "Text": "$3,676.69",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.24, "Top": 0.2, "Width": 0.16, "Height": 0.05}},
                        },
                    ]
                }
            },
            layout_items=[
                {
                    "type": "text",
                    "value": "Total $3,676.69",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.05},
                }
            ],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {"type": "object", "properties": {"amount": {"type": "number"}}},
            "expected_output": {"amount": 3676.69},
            "test_rules": [
                {
                    "id": "rule-amount",
                    "type": "extract_field",
                    "field_path": "amount",
                    "expected_value": 3676.69,
                    "bboxes": [{"page": 1, "bbox": [0.24, 0.2, 0.16, 0.05], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )

    doc.result_path = result_path
    loaded = load_document(doc)

    rule = loaded.pages[0].gt_rules[0]
    assert rule.predicted_granularity == "word"
    assert rule.predicted_text == "$3,676.69"
    assert rule.text_score == 1.0


def test_loader_extract_field_matching_uses_customer_date_value_rules(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        {
                            "Id": "line-1",
                            "BlockType": "LINE",
                            "Text": "January 2, 2024",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.5, "Top": 0.3, "Width": 0.2, "Height": 0.05}},
                        }
                    ]
                }
            },
            layout_items=[
                {
                    "type": "text",
                    "value": "January 2, 2024",
                    "bbox": {"x": 0.5, "y": 0.3, "w": 0.2, "h": 0.05},
                }
            ],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "format": "date"},
                    "candidate_name": {"type": "string"},
                },
            },
            "expected_output": {"start_date": "2024-01-02", "candidate_name": "Ada"},
            "test_rules": [
                {
                    "id": "rule-start-date",
                    "type": "extract_field",
                    "field_path": "start_date",
                    "expected_value": "2024-01-02",
                    "bboxes": [{"page": 1, "bbox": [0.5, 0.3, 0.2, 0.05], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )

    doc.result_path = result_path
    loaded = load_document(doc)

    rule = loaded.pages[0].gt_rules[0]
    assert rule.predicted_granularity == "line"
    assert rule.predicted_text == "January 2, 2024"
    assert rule.text_score == 1.0


def test_loader_exposes_layout_gt_rules_from_evaluation_report(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    source = suite_dir / "doc.png"
    _make_image(source)
    result_path = suite_dir / "doc.result.json"
    test_case_path = suite_dir / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    doc = IndexedDocumentInternal(
        doc_id="doc-layout",
        base_name="doc",
        relative_dir="suite",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=result_path,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        test_case_path=test_case_path,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=True,
            has_v2_items_payload=True,
        ),
    )

    payload = _make_layout_detection_result_payload(
        pipeline_name="candidate_granular_bboxes",
        raw_output={"v2_items": {"pages": [{"page_number": 1, "page_width": 640, "page_height": 480, "items": []}]}},
        width=640,
        height=480,
    )
    payload["request"]["example_id"] = "suite/doc"
    _write_json(result_path, payload)
    _write_json(
        test_case_path,
        {
            "test_rules": [
                {
                    "id": "layout-1",
                    "type": "layout",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.1],
                    "canonical_class": "Text",
                    "ro_index": 7,
                    "content": "alpha beta",
                }
            ]
        },
    )
    _write_json(
        report_path,
        {
            "per_example_results": [
                {
                    "example_id": "suite/doc",
                    "test_id": "suite/doc",
                    "metrics": [
                        {
                            "metric_name": "layout_element_rule_pass_rate",
                            "metadata": {
                                "rule_results": [
                                    {
                                        "element_id": "layout-1",
                                        "element_index": 0,
                                        "page": 1,
                                        "best_pred_class": "Text",
                                        "best_pred_class_norm": "Text",
                                        "best_pred_index": 4,
                                        "best_pred_ioa_gt": 0.93,
                                        "best_pred_iou": 0.81,
                                        "best_pred_bbox": [0.11, 0.21, 0.39, 0.29],
                                        "gt_text_norm": "alpha beta",
                                        "pred_text_norm": "alpha",
                                        "localization_pass": True,
                                        "localization_reason": "pass",
                                        "classification_pass": True,
                                        "classification_reason": "pass",
                                        "attribution_applicable": True,
                                        "attribution_pass": False,
                                        "attribution_reason": "f1_below_threshold",
                                        "attribution_method": "f1",
                                        "attribution_threshold": 0.8,
                                        "token_precision": 1.0,
                                        "token_recall": 0.5,
                                        "token_f1": 2 / 3,
                                        "missing_tokens": ["beta"],
                                        "extra_tokens": [],
                                        "normalized_attributes": {"text_role": "paragraph"},
                                    }
                                ]
                            },
                        }
                    ],
                }
            ]
        },
    )

    loaded = load_document(doc)

    assert len(loaded.pages[0].gt_rules) == 1
    rule = loaded.pages[0].gt_rules[0]
    assert rule.rule_type == "layout"
    assert rule.rule_id == "layout-1"
    assert rule.canonical_class == "Text"
    assert rule.gt_ro_index == 7
    assert rule.predicted_class == "Text"
    assert rule.predicted_text == "alpha"
    assert rule.predicted_bbox is not None
    assert rule.predicted_bbox.x == pytest.approx(70.4)
    assert rule.predicted_bbox.y == pytest.approx(100.8)
    assert rule.predicted_bbox.w == pytest.approx(179.2)
    assert rule.predicted_bbox.h == pytest.approx(38.4)
    assert rule.localization_pass is True
    assert rule.classification_pass is True
    assert rule.attribution_pass is False
    assert rule.overall_pass is False
    assert rule.iou == 0.81
    assert rule.token_f1 == 2 / 3
    assert rule.missing_tokens == ["beta"]


def test_loader_layout_gt_rules_fall_back_to_filtered_element_index(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    source = suite_dir / "doc.png"
    _make_image(source)
    result_path = suite_dir / "doc.result.json"
    test_case_path = suite_dir / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    doc = IndexedDocumentInternal(
        doc_id="doc-layout-index",
        base_name="doc",
        relative_dir="suite",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=result_path,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        test_case_path=test_case_path,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=True,
            has_v2_items_payload=True,
        ),
    )

    payload = _make_layout_detection_result_payload(
        pipeline_name="candidate_granular_bboxes",
        raw_output={"v2_items": {"pages": [{"page_number": 1, "page_width": 640, "page_height": 480, "items": []}]}},
        width=640,
        height=480,
    )
    payload["request"]["example_id"] = "suite/doc"
    _write_json(result_path, payload)
    _write_json(
        test_case_path,
        {
            "test_rules": [
                {
                    "id": "layout-ignored",
                    "type": "layout",
                    "page": 1,
                    "bbox": [0.05, 0.1, 0.1, 0.08],
                    "canonical_class": "Section",
                    "attributes": {"ignore": True},
                    "ro_index": 0,
                },
                {
                    "id": "layout-visible",
                    "type": "layout",
                    "page": 1,
                    "bbox": [0.2, 0.25, 0.2, 0.12],
                    "canonical_class": "Table",
                    "ro_index": 1,
                },
            ]
        },
    )
    _write_json(
        report_path,
        {
            "per_example_results": [
                {
                    "example_id": "suite/doc",
                    "metrics": [
                        {
                            "metric_name": "layout_element_rule_pass_rate",
                            "metadata": {
                                "rule_results": [
                                    {
                                        "element_index": 0,
                                        "page": 1,
                                        "best_pred_class": "Table",
                                        "best_pred_bbox": [0.2, 0.25, 0.4, 0.37],
                                        "localization_pass": True,
                                        "classification_pass": True,
                                        "attribution_applicable": False,
                                        "best_pred_iou": 1.0,
                                        "best_pred_ioa_gt": 1.0,
                                    }
                                ]
                            },
                        }
                    ],
                }
            ]
        },
    )

    loaded = load_document(doc)

    assert len(loaded.pages[0].gt_rules) == 1
    rule = loaded.pages[0].gt_rules[0]
    assert rule.rule_id == "layout-visible"
    assert rule.canonical_class == "Table"
    assert rule.predicted_class == "Table"
    assert rule.predicted_bbox is not None
    assert rule.predicted_bbox.x == pytest.approx(128.0)
    assert rule.predicted_bbox.y == pytest.approx(120.0)


def test_loader_refreshes_layout_gt_rules_when_evaluation_report_changes(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    source = suite_dir / "doc.png"
    _make_image(source)
    result_path = suite_dir / "doc.result.json"
    test_case_path = suite_dir / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    doc = IndexedDocumentInternal(
        doc_id="doc-layout-refresh",
        base_name="doc",
        relative_dir="suite",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=result_path,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        test_case_path=test_case_path,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=True,
            has_v2_items_payload=True,
        ),
    )

    payload = _make_layout_detection_result_payload(
        pipeline_name="candidate_granular_bboxes",
        raw_output={"v2_items": {"pages": [{"page_number": 1, "page_width": 640, "page_height": 480, "items": []}]}},
        width=640,
        height=480,
    )
    payload["request"]["example_id"] = "suite/doc"
    _write_json(result_path, payload)
    _write_json(
        test_case_path,
        {
            "test_rules": [
                {
                    "id": "layout-1",
                    "type": "layout",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.1],
                    "canonical_class": "Text",
                    "ro_index": 0,
                }
            ]
        },
    )

    def _write_report(predicted_class: str) -> None:
        _write_json(
            report_path,
            {
                "per_example_results": [
                    {
                        "example_id": "suite/doc",
                        "metrics": [
                            {
                                "metric_name": "layout_element_rule_pass_rate",
                                "metadata": {
                                    "rule_results": [
                                        {
                                            "element_id": "layout-1",
                                            "element_index": 0,
                                            "page": 1,
                                            "best_pred_class": predicted_class,
                                            "best_pred_class_norm": predicted_class,
                                            "best_pred_bbox": [0.1, 0.2, 0.4, 0.3],
                                            "localization_pass": True,
                                            "classification_pass": True,
                                            "attribution_applicable": False,
                                            "best_pred_iou": 0.9,
                                            "best_pred_ioa_gt": 0.95,
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                ]
            },
        )

    _write_report("Text")
    first_loaded = load_document(doc)
    assert first_loaded.pages[0].gt_rules[0].predicted_class == "Text"

    _write_report("Table")
    report_stat = report_path.stat()
    os.utime(report_path, ns=(report_stat.st_atime_ns, report_stat.st_mtime_ns + 1_000_000))

    second_loaded = load_document(doc)
    assert second_loaded.pages[0].gt_rules[0].predicted_class == "Table"


def test_loader_marks_azure_cell_layer_unavailable(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="azure_di_layout",
            raw_output={
                "pages": [
                    {
                        "page_number": 1,
                        "width": 2.0,
                        "height": 4.0,
                        "lines": [
                            {
                                "content": "Record number",
                                "polygon": [0.2, 0.4, 1.0, 0.4, 1.0, 0.8, 0.2, 0.8],
                            }
                        ],
                        "words": [
                            {
                                "content": "REC-0000",
                                "polygon": [1.1, 0.4, 1.6, 0.4, 1.6, 0.8, 1.1, 0.8],
                            }
                        ],
                    }
                ],
                "tables": [
                    {
                        "row_count": 1,
                        "column_count": 1,
                        "cells": [
                            {
                                "row_index": 0,
                                "column_index": 0,
                                "content": "Header",
                                "row_span": None,
                                "column_span": None,
                            }
                        ],
                        "bounding_regions": [{"page_number": 1, "polygon": [0.2, 1.0, 1.4, 1.0, 1.4, 2.0, 0.2, 2.0]}],
                    }
                ],
            },
            layout_items=[
                {
                    "type": "text",
                    "value": "Record number",
                    "bbox": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.1},
                }
            ],
        ),
    )

    doc.result_path = result_path
    loaded = load_document(doc)
    layers = _layer_map(loaded)

    assert layers["line"].availability == "available"
    assert layers["word"].availability == "available"
    assert layers["cell"].availability == "unavailable"
    assert "does not preserve exact cell polygons" in (layers["cell"].reason or "")


def test_loader_exposes_extract_field_gt_rules_with_multi_bbox_stray_and_verified(
    tmp_path: Path,
) -> None:
    """extract_field rules with evidence bboxes expand into one GT
    rule per evidence bbox, propagate tags + verified flag, skip empty-bbox
    rules, and carry null expected_value through unchanged.
    """
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="candidate_granular_bboxes",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        # Address line 1 words
                        {
                            "Id": "line-addr-1",
                            "BlockType": "LINE",
                            "Text": "123 Example Ave,",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.06, "Top": 0.25, "Width": 0.13, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-1a",
                            "BlockType": "WORD",
                            "Text": "123",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.06, "Top": 0.25, "Width": 0.03, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-1b",
                            "BlockType": "WORD",
                            "Text": "Example",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.10, "Top": 0.25, "Width": 0.015, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-1c",
                            "BlockType": "WORD",
                            "Text": "Ave",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.12, "Top": 0.25, "Width": 0.04, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-1d",
                            "BlockType": "WORD",
                            "Text": ",",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.165, "Top": 0.25, "Width": 0.025, "Height": 0.02}},
                        },
                        # Address line 2
                        {
                            "Id": "line-addr-2",
                            "BlockType": "LINE",
                            "Text": "Example City, CA 00000",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.06, "Top": 0.27, "Width": 0.18, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-2a",
                            "BlockType": "WORD",
                            "Text": "Example",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.06, "Top": 0.27, "Width": 0.05, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-2b",
                            "BlockType": "WORD",
                            "Text": "City,",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.115, "Top": 0.27, "Width": 0.035, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-2c",
                            "BlockType": "WORD",
                            "Text": "CA",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.155, "Top": 0.27, "Width": 0.02, "Height": 0.02}},
                        },
                        {
                            "Id": "word-addr-2d",
                            "BlockType": "WORD",
                            "Text": "00000",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.18, "Top": 0.27, "Width": 0.04, "Height": 0.02}},
                        },
                        # client_id
                        {
                            "Id": "line-cid",
                            "BlockType": "LINE",
                            "Text": "CLIENT-0001",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.4, "Top": 0.1, "Width": 0.1, "Height": 0.02}},
                        },
                        {
                            "Id": "word-cid",
                            "BlockType": "WORD",
                            "Text": "CLIENT-0001",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.4, "Top": 0.1, "Width": 0.1, "Height": 0.02}},
                        },
                        # stray token (evidence heuristic miss)
                        {
                            "Id": "line-stray",
                            "BlockType": "LINE",
                            "Text": "stray",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.7, "Top": 0.6, "Width": 0.1, "Height": 0.02}},
                        },
                        {
                            "Id": "word-stray",
                            "BlockType": "WORD",
                            "Text": "stray",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.7, "Top": 0.6, "Width": 0.1, "Height": 0.02}},
                        },
                    ]
                }
            },
            layout_items=[],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"},
                    "address": {"type": "string"},
                    "nickname": {"type": "string"},
                },
            },
            "expected_output": {
                "client_id": "CLIENT-0001",
                "address": "123 Example Ave,\nExample City, CA 00000",
                "nickname": None,
            },
            "test_rules": [
                # Simple single-bbox rule (verified=True implicitly via default)
                {
                    "type": "extract_field",
                    "id": "rule-client-id",
                    "field_path": "client_id",
                    "expected_value": "CLIENT-0001",
                    "bboxes": [{"page": 1, "bbox": [0.4, 0.1, 0.1, 0.02], "source_bbox_index": 0}],
                    "verified": True,
                    "tags": ["benchmark_fixture"],
                },
                # Multi-bbox rule: should expand into 2 GT rules (one per evidence bbox)
                {
                    "type": "extract_field",
                    "id": "rule-address",
                    "field_path": "address",
                    "expected_value": "123 Example Ave,\nExample City, CA 00000",
                    "bboxes": [
                        {"page": 1, "bbox": [0.06, 0.25, 0.13, 0.02], "source_bbox_index": 0},
                        {"page": 1, "bbox": [0.06, 0.27, 0.18, 0.02], "source_bbox_index": 1},
                    ],
                    "verified": True,
                    "tags": ["benchmark_fixture"],
                },
                # Stray rule: null expected_value, verified=False, stray tag
                {
                    "type": "extract_field",
                    "id": "rule-stray",
                    "field_path": "nickname",
                    "expected_value": None,
                    "bboxes": [{"page": 1, "bbox": [0.7, 0.6, 0.1, 0.02], "source_bbox_index": 406}],
                    "verified": False,
                    "tags": ["benchmark_fixture", "stray_evidence"],
                },
                # Empty-bbox rule: should be skipped (nothing to render)
                {
                    "type": "extract_field",
                    "id": "rule-empty",
                    "field_path": "client_id",
                    "expected_value": "CLIENT-0001",
                    "bboxes": [],
                    "verified": True,
                    "tags": ["benchmark_fixture"],
                },
            ],
        },
    )

    doc.result_path = result_path
    loaded = load_document(doc)

    rules = loaded.pages[0].gt_rules
    assert all(rule.rule_type == "extract_field" for rule in rules), [rule.rule_type for rule in rules]

    rules_by_id = {rule.rule_id: rule for rule in rules}
    # Single-bbox rule keeps its original id.
    assert "rule-client-id" in rules_by_id
    # Multi-bbox rule fans out into `id#<bbox_index>` entries.
    assert "rule-address#0" in rules_by_id
    assert "rule-address#1" in rules_by_id
    # Stray rule keeps its original id.
    assert "rule-stray" in rules_by_id
    # Empty-bbox rule is skipped entirely (no ghost entry).
    assert not any(rule_id.startswith("rule-empty") for rule_id in rules_by_id)
    # Total: 1 + 2 + 1 = 4 extract_field rules.
    assert len(rules) == 4

    # client_id rule: expected_value + tags preserved, verified=True, stray tag absent.
    client_rule = rules_by_id["rule-client-id"]
    assert client_rule.field_path == "client_id"
    assert client_rule.expected_value == "CLIENT-0001"
    assert client_rule.evidence_index == 0
    assert client_rule.verified is True
    assert "stray_evidence" not in client_rule.tags
    assert client_rule.tags == ["benchmark_fixture"]
    assert client_rule.source_bbox_index == 0
    # Best-match should pick up the word-level client_id prediction.
    assert client_rule.predicted_text == "CLIENT-0001"
    assert client_rule.predicted_granularity == "word"

    # Multi-bbox rule: evidence_index reflects the bbox position; source_bbox_index
    # mirrors the original payload positions (lossless round-trip).
    address_line_1 = rules_by_id["rule-address#0"]
    address_line_2 = rules_by_id["rule-address#1"]
    assert address_line_1.field_path == "address"
    assert address_line_1.evidence_index == 0
    assert address_line_1.source_bbox_index == 0
    assert address_line_2.evidence_index == 1
    assert address_line_2.source_bbox_index == 1
    # Each expanded rule carries the same rule-level expected_value + tags.
    assert address_line_1.expected_value == "123 Example Ave,\nExample City, CA 00000"
    assert address_line_2.expected_value == "123 Example Ave,\nExample City, CA 00000"
    assert address_line_1.verified is True and address_line_2.verified is True
    # GT bboxes differ per evidence bbox — not collapsed.
    assert address_line_1.gt_bbox.y != address_line_2.gt_bbox.y

    # Stray rule: verified=False, stray tag surfaces, null expected_value.
    stray_rule = rules_by_id["rule-stray"]
    assert stray_rule.expected_value is None
    assert stray_rule.verified is False
    assert "stray_evidence" in stray_rule.tags
    assert stray_rule.source_bbox_index == 406


@pytest.mark.parametrize("metric_name", ["parse_field_element_pass_rate", "extract_element_pass_rate"])
def test_loader_extract_field_gt_rules_pick_up_metric_rule_results(tmp_path: Path, metric_name: str) -> None:
    """When ``_evaluation_report.json`` carries field grounding metric metadata with per-rule
    ``rule_results``, the viz's ``GroundTruthRuleMatch`` should inherit
    loc_pass / cls_pass / attr_pass / overall_pass, the predicted_bboxes
    rendered in page-pixel coords, and the textual metadata used by the
    LCS text diff and the PDF overlay.

    The metric emits one entry per rule (not per GT bbox). Multi-bbox rules
    therefore share the same metric verdict — this is covered below.
    """
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    source = suite_dir / "doc.png"
    _make_image(source)
    result_path = suite_dir / "doc.result.json"
    test_case_path = suite_dir / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    doc = IndexedDocumentInternal(
        doc_id="doc-extract-metric",
        base_name="doc",
        relative_dir="suite",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=result_path,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        test_case_path=test_case_path,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=True,
            has_v2_items_payload=True,
        ),
    )

    payload = _make_parse_result_payload(
        pipeline_name="candidate_granular_bboxes",
        raw_output={},
        layout_items=[],
        width=640,
        height=480,
    )
    payload["request"]["example_id"] = "suite/doc"
    _write_json(result_path, payload)
    _write_json(
        test_case_path,
        {
            "data_schema": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "invoice_number": {"type": "string"},
                },
            },
            "expected_output": {"vendor": "Acme Corp", "invoice_number": "INV-001"},
            "test_rules": [
                {
                    "id": "rule-vendor",
                    "type": "extract_field",
                    "field_path": "vendor",
                    "expected_value": "Acme Corp",
                    "bboxes": [{"page": 1, "bbox": [0.10, 0.10, 0.20, 0.02], "source_bbox_index": 0}],
                    "verified": True,
                },
                {
                    "id": "rule-invoice",
                    "type": "extract_field",
                    "field_path": "invoice_number",
                    "expected_value": "INV-001",
                    "bboxes": [{"page": 1, "bbox": [0.50, 0.50, 0.10, 0.02], "source_bbox_index": 0}],
                    "verified": True,
                },
            ],
        },
    )
    _write_json(
        report_path,
        {
            "per_example_results": [
                {
                    "example_id": "suite/doc",
                    "test_id": "suite/doc",
                    "metrics": [
                        {
                            "metric_name": metric_name,
                            "metadata": {
                                "gt_count": 2,
                                "rule_results": [
                                    {
                                        "field_path": "vendor",
                                        "loc_pass": True,
                                        "cls_pass": True,
                                        "attr_pass": True,
                                        "element_pass": True,
                                        "granularity": "line",
                                        "iou": 0.92,
                                        "score": 1.0,
                                        "mode": "substring",
                                        "reason": "pass",
                                        "localization_reason": "pass",
                                        "matched_pred_bboxes": [[0.10, 0.10, 0.20, 0.02]],
                                        "matched_pred_text": "Acme Corp",
                                    },
                                    {
                                        "field_path": "invoice_number",
                                        "loc_pass": False,
                                        "cls_pass": True,
                                        "attr_pass": False,
                                        "element_pass": False,
                                        "granularity": "none",
                                        "iou": 0.0,
                                        "score": 0.0,
                                        "mode": "missing",
                                        "reason": "no_support_match",
                                        "localization_reason": "no_support_match",
                                        "matched_pred_bboxes": [],
                                        "matched_pred_text": "",
                                    },
                                ],
                            },
                        }
                    ],
                }
            ]
        },
    )

    loaded = load_document(doc)
    rules = {rule.rule_id: rule for rule in loaded.pages[0].gt_rules}
    assert "rule-vendor" in rules
    assert "rule-invoice" in rules

    vendor = rules["rule-vendor"]
    assert vendor.rule_type == "extract_field"
    assert vendor.localization_pass is True
    assert vendor.classification_pass is True
    assert vendor.attribution_pass is True
    assert vendor.overall_pass is True
    assert vendor.localization_reason == "pass"
    assert vendor.attribution_reason == "pass"
    assert vendor.attribution_method == "substring"
    assert vendor.text_score == pytest.approx(1.0)
    assert vendor.iou == pytest.approx(0.92)
    assert vendor.predicted_text == "Acme Corp"
    assert vendor.predicted_granularity == "line"
    # matched_pred_bboxes are scaled to page-pixel (page_width=640, page_height=480).
    assert len(vendor.predicted_bboxes) == 1
    pred_bbox = vendor.predicted_bboxes[0]
    assert pred_bbox.x == pytest.approx(64.0)  # 0.10 * 640
    assert pred_bbox.y == pytest.approx(48.0)  # 0.10 * 480
    assert pred_bbox.w == pytest.approx(128.0)  # 0.20 * 640
    assert pred_bbox.h == pytest.approx(9.6)  # 0.02 * 480

    invoice = rules["rule-invoice"]
    assert invoice.localization_pass is False
    assert invoice.classification_pass is True
    assert invoice.attribution_pass is False
    assert invoice.overall_pass is False
    assert invoice.localization_reason == "no_support_match"
    assert invoice.attribution_reason == "no_support_match"
    assert invoice.iou == pytest.approx(0.0)
    # Empty matched_pred_bboxes → viz loader should leave predicted_bboxes untouched
    # (viz's own heuristic may have populated an empty list already; either way,
    # the metric doesn't overwrite it with a bogus page-pixel bbox).
    assert invoice.predicted_bboxes == [] or all(bbox.w == 0 for bbox in invoice.predicted_bboxes)


@pytest.mark.parametrize(
    ("product_type", "metrics", "expected_metric_name"),
    [
        (
            "extract",
            ["parse_field_element_pass_rate", "extract_element_pass_rate"],
            "extract_element_pass_rate",
        ),
        (
            "parse",
            ["parse_field_element_pass_rate", "extract_element_pass_rate"],
            "parse_field_element_pass_rate",
        ),
        ("", ["parse_field_element_pass_rate"], "parse_field_element_pass_rate"),
        ("", ["extract_element_pass_rate"], "extract_element_pass_rate"),
    ],
)
def test_loader_extract_field_metric_prefers_product_specific_carrier(
    product_type: str,
    metrics: list[str],
    expected_metric_name: str,
) -> None:
    metric = _find_extract_field_metric_result(
        {
            "product_type": product_type,
            "metrics": [
                {
                    "metric_name": metric_name,
                    "metadata": {"carrier": metric_name, "rule_results": [{"field_path": "vendor"}]},
                }
                for metric_name in metrics
            ],
        }
    )

    assert metric is not None
    assert metric["metric_name"] == expected_metric_name


def test_loader_extract_field_metric_skips_non_rule_result_carriers() -> None:
    metric = _find_extract_field_metric_result(
        {
            "metrics": [
                {"metric_name": "parse_field_element_pass_rate", "metadata": {"score": 1.0}},
                {
                    "metric_name": "extract_element_pass_rate",
                    "metadata": {"rule_results": [{"field_path": "vendor"}]},
                },
            ],
        }
    )

    assert metric is not None
    assert metric["metric_name"] == "extract_element_pass_rate"


def test_loader_extract_field_metric_preserves_local_granular_evidence(tmp_path: Path) -> None:
    """Metric reports can carry a broad source snippet/bbox even when the
    page-local granular match identifies the exact word used for attribution.
    The visualizer should keep the local evidence for display and overlays
    while still inheriting the metric pass/fail fields.
    """
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    source = suite_dir / "doc.png"
    _make_image(source)
    result_path = suite_dir / "doc.result.json"
    test_case_path = suite_dir / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    doc = IndexedDocumentInternal(
        doc_id="doc-extract-metric-local-evidence",
        base_name="doc",
        relative_dir="suite",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=result_path,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        test_case_path=test_case_path,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=True,
            has_v2_items_payload=True,
        ),
    )

    payload = _make_parse_result_payload(
        pipeline_name="textract",
        raw_output={
            "textract_response": {
                "Blocks": [
                    {
                        "Id": "line-1",
                        "BlockType": "LINE",
                        "Text": "Supplier | Item Name | Catalog # | Item #",
                        "Page": 1,
                        "Geometry": {"BoundingBox": {"Left": 0.05, "Top": 0.10, "Width": 0.80, "Height": 0.05}},
                    },
                    {
                        "Id": "word-catalog",
                        "BlockType": "WORD",
                        "Text": "CAT-001",
                        "Page": 1,
                        "Geometry": {"BoundingBox": {"Left": 0.60, "Top": 0.10, "Width": 0.08, "Height": 0.05}},
                    },
                ]
            }
        },
        layout_items=[],
        width=640,
        height=480,
    )
    payload["request"]["example_id"] = "suite/doc"
    _write_json(result_path, payload)
    _write_json(
        test_case_path,
        {
            "data_schema": {
                "type": "object",
                "properties": {"stock_list": {"type": "array", "items": {"type": "object"}}},
            },
            "expected_output": {"stock_list": [{"catalog_number": "CAT-001"}]},
            "test_rules": [
                {
                    "id": "rule-catalog",
                    "type": "extract_field",
                    "field_path": "stock_list[0].catalog_number",
                    "expected_value": "CAT-001",
                    "bboxes": [{"page": 1, "bbox": [0.60, 0.10, 0.08, 0.05], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )
    _write_json(
        report_path,
        {
            "per_example_results": [
                {
                    "example_id": "suite/doc",
                    "test_id": "suite/doc",
                    "metrics": [
                        {
                            "metric_name": "parse_field_element_pass_rate",
                            "metadata": {
                                "gt_count": 1,
                                "rule_results": [
                                    {
                                        "field_path": "stock_list[0].catalog_number",
                                        "loc_pass": True,
                                        "cls_pass": True,
                                        "attr_pass": True,
                                        "element_pass": True,
                                        "granularity": "word",
                                        "iou": 1.0,
                                        "score": 1.0,
                                        "mode": "substring",
                                        "reason": "pass",
                                        "localization_reason": "pass",
                                        "matched_pred_bboxes": [[0.05, 0.10, 0.80, 0.05]],
                                        "matched_pred_text": "| Supplier | Item Name | Catalog # | Item # |",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        },
    )

    loaded = load_document(doc)
    [rule] = loaded.pages[0].gt_rules

    assert rule.overall_pass is True
    assert rule.localization_pass is True
    assert rule.attribution_method == "substring"
    assert rule.iou == pytest.approx(1.0)
    assert rule.predicted_text == "CAT-001"
    assert rule.predicted_granularity == "word"
    assert rule.matched_unit_ids == ["word-catalog"]
    assert len(rule.predicted_bboxes) == 1
    pred_bbox = rule.predicted_bboxes[0]
    assert pred_bbox.x == pytest.approx(384.0)  # 0.60 * 640
    assert pred_bbox.y == pytest.approx(48.0)  # 0.10 * 480
    assert pred_bbox.w == pytest.approx(51.2)  # 0.08 * 640
    assert pred_bbox.h == pytest.approx(24.0)  # 0.05 * 480


def test_loader_extract_field_metric_derives_array_cell_text_from_table_markdown(tmp_path: Path) -> None:
    """When the evaluator falls back to a table layout item, its matched text is
    the full markdown table. For array field paths, derive the row/cell value so
    the UI shows the prediction actually compared for that field.
    """
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    source = suite_dir / "doc.png"
    _make_image(source)
    result_path = suite_dir / "doc.result.json"
    test_case_path = suite_dir / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    doc = IndexedDocumentInternal(
        doc_id="doc-extract-metric-table-cell",
        base_name="doc",
        relative_dir="suite",
        source_kind="image",
        source_ext=".png",
        last_modified_ms=source.stat().st_mtime_ns // 1_000_000,
        source_path=source,
        raw_path=None,
        result_path=result_path,
        v2_items_path=None,
        markdown_path=None,
        markdown_json_path=None,
        test_case_path=test_case_path,
        artifact_flags=ArtifactFlags(
            has_v2_items_file=False,
            has_raw_file=False,
            has_result_file=True,
            has_v2_items_payload=True,
        ),
    )

    payload = _make_parse_result_payload(
        pipeline_name="candidate_granular_bboxes",
        raw_output={},
        layout_items=[],
        width=640,
        height=480,
    )
    payload["request"]["example_id"] = "suite/doc"
    _write_json(result_path, payload)
    _write_json(
        test_case_path,
        {
            "data_schema": {
                "type": "object",
                "properties": {
                    "employees_in_a_payroll": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"employee_name": {"type": "string"}, "post": {"type": "string"}},
                        },
                    }
                },
            },
            "expected_output": {
                "employees_in_a_payroll": [
                    {"employee_name": "Person Alpha", "post": "Role A"},
                    {"employee_name": "Person Beta", "post": "Role B"},
                ]
            },
            "test_rules": [
                {
                    "id": "rule-employee-name",
                    "type": "extract_field",
                    "field_path": "employees_in_a_payroll[1].employee_name",
                    "expected_value": "Person Beta",
                    "bboxes": [{"page": 1, "bbox": [0.30, 0.30, 0.10, 0.02], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )
    table_markdown = "\n".join(
        [
            "| Row # | Record Information<br/>Name | Record Information<br/>Role |",
            "| ----- | --------------------------- | --------------------------- |",
            "| 1 | Person Alpha | Role A |",
            "| 2 | Person Beto | Role B |",
        ]
    )
    _write_json(
        report_path,
        {
            "per_example_results": [
                {
                    "example_id": "suite/doc",
                    "test_id": "suite/doc",
                    "metrics": [
                        {
                            "metric_name": "parse_field_element_pass_rate",
                            "metadata": {
                                "gt_count": 1,
                                "rule_results": [
                                    {
                                        "field_path": "employees_in_a_payroll[1].employee_name",
                                        "loc_pass": True,
                                        "cls_pass": True,
                                        "attr_pass": False,
                                        "element_pass": False,
                                        "granularity": "layout_item",
                                        "iou": 1.0,
                                        "score": 0.52,
                                        "mode": "jaro_winkler",
                                        "reason": "jaro_winkler_below_threshold",
                                        "localization_reason": "pass",
                                        "matched_pred_bboxes": [[0.10, 0.10, 0.80, 0.80]],
                                        "matched_pred_text": table_markdown,
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        },
    )

    loaded = load_document(doc)
    [rule] = loaded.pages[0].gt_rules

    assert rule.overall_pass is False
    assert rule.localization_pass is True
    assert rule.attribution_method == "jaro_winkler"
    assert rule.predicted_text == "Person Beto"


def test_loader_extract_field_gt_rules_no_metric_keeps_defaults(tmp_path: Path) -> None:
    """Eval reports without final field grounding metrics leave attribution slots empty.

    The viewer should stay compatible with reports produced before the
    visualizable field grounding metric metadata was added.
    """
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={
                "textract_response": {
                    "Blocks": [
                        {
                            "Id": "line-1",
                            "BlockType": "LINE",
                            "Text": "Acme Corp",
                            "Page": 1,
                            "Geometry": {"BoundingBox": {"Left": 0.10, "Top": 0.10, "Width": 0.20, "Height": 0.02}},
                        },
                    ]
                }
            },
            layout_items=[{"type": "text", "value": "Acme Corp", "bbox": {"x": 0.10, "y": 0.10, "w": 0.20, "h": 0.02}}],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {"type": "object", "properties": {"vendor": {"type": "string"}}},
            "expected_output": {"vendor": "Acme Corp"},
            "test_rules": [
                {
                    "id": "rule-vendor",
                    "type": "extract_field",
                    "field_path": "vendor",
                    "expected_value": "Acme Corp",
                    "bboxes": [{"page": 1, "bbox": [0.10, 0.10, 0.20, 0.02], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )
    # Intentionally: no _evaluation_report.json

    doc.result_path = result_path
    loaded = load_document(doc)
    rules = loaded.pages[0].gt_rules
    assert len(rules) == 1
    vendor = rules[0]
    # Metric fields stay None when no report is present.
    assert vendor.localization_pass is None
    assert vendor.classification_pass is None
    assert vendor.attribution_pass is None
    assert vendor.overall_pass is None
    assert vendor.localization_reason is None
    assert vendor.attribution_method is None
    # Viz-computed fields remain populated by the best-match heuristic.
    assert vendor.predicted_text == "Acme Corp"


def test_loader_extract_field_gt_rules_ignore_temporary_metric_namespace(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    result_path = tmp_path / "doc.result.json"
    test_case_path = tmp_path / "doc.test.json"
    report_path = tmp_path / "_evaluation_report.json"

    _write_json(
        result_path,
        _make_parse_result_payload(
            pipeline_name="textract",
            raw_output={},
            layout_items=[{"type": "text", "value": "Acme Corp", "bbox": {"x": 0.10, "y": 0.10, "w": 0.20, "h": 0.02}}],
        ),
    )
    _write_json(
        test_case_path,
        {
            "data_schema": {"type": "object", "properties": {"vendor": {"type": "string"}}},
            "expected_output": {"vendor": "Acme Corp"},
            "test_rules": [
                {
                    "id": "rule-vendor",
                    "type": "extract_field",
                    "field_path": "vendor",
                    "expected_value": "Acme Corp",
                    "bboxes": [{"page": 1, "bbox": [0.10, 0.10, 0.20, 0.02], "source_bbox_index": 0}],
                    "verified": True,
                }
            ],
        },
    )

    temporary_metric_name = "extract_field_" + "element_pass_rate"
    _write_json(
        report_path,
        {
            "per_example_results": [
                {
                    "example_id": "doc1",
                    "test_id": "doc1",
                    "metrics": [
                        {
                            "metric_name": temporary_metric_name,
                            "metadata": {
                                "rule_results": [
                                    {
                                        "field_path": "vendor",
                                        "loc_pass": True,
                                        "cls_pass": True,
                                        "attr_pass": True,
                                        "element_pass": True,
                                    }
                                ]
                            },
                        }
                    ],
                }
            ]
        },
    )

    doc.result_path = result_path
    doc.test_case_path = test_case_path
    loaded = load_document(doc)

    [vendor] = loaded.pages[0].gt_rules
    assert vendor.rule_type == "extract_field"
    assert vendor.localization_pass is None
    assert vendor.classification_pass is None
    assert vendor.attribution_pass is None
    assert vendor.overall_pass is None
