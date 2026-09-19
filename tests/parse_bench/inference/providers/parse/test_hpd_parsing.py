"""Focused coverage for the HPD-Parsing provider and layout path."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from PIL import Image

from parse_bench.evaluation.layout_adapters.adapters import HpdParsingLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers.projection import project_layout_predictions
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.hpd_parsing import (
    HpdParsingProvider,
    _parse_blocks,
)
from parse_bench.schemas.layout_detection_output import (
    LAYOUT_MODEL_INFO,
    LayoutDetectionModel,
    LayoutOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

_OFFICIAL_NO_CONTENT_SENTINELS = (
    "[Non-Text]",
    "The image is too blurry to recognize any text content.",
    (
        "The image contains no text or characters. It is a graphical element "
        "(a horizontal line with a vertical line) and does not contain any chart, "
        "graph, or data points that can be extracted. Therefore, the correct OCR "
        "output is an empty string."
    ),
)


def _normalize_raw_response(
    monkeypatch: pytest.MonkeyPatch,
    raw_response: str | list[str],
    *,
    width: int = 1000,
    height: int = 1000,
    prompt_mode: str = "fork",
):
    monkeypatch.setenv("HPD_PARSING_SERVER_URL", "https://example.invalid/v1")
    pipeline = get_pipeline("hpd_parsing_vllm_parse")
    provider = HpdParsingProvider(pipeline.provider_name, pipeline.config)
    request = InferenceRequest(
        example_id="layout/raw-response",
        source_file_path="example.png",
        product_type="parse",
    )
    now = datetime.now()
    raw_responses = [raw_response] if isinstance(raw_response, str) else raw_response
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="parse",
        raw_output={
            "pages": [
                {
                    "page_index": page_index,
                    "width": width,
                    "height": height,
                    "raw_response": page_response,
                }
                for page_index, page_response in enumerate(raw_responses)
            ],
            "num_pages": len(raw_responses),
            "model": "PaddlePaddle/HPD-Parsing",
            "prompt_mode": prompt_mode,
        },
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )
    return provider.normalize(raw_result)


def test_plain_mode_preserves_single_page_response_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_response = "  # Plain output\n\nBody <custom attr=value>\n"
    result = _normalize_raw_response(monkeypatch, raw_response, prompt_mode="plain")

    assert result.output.pages[0].markdown == raw_response
    assert result.output.markdown == raw_response
    assert result.output.layout_pages == []


def test_plain_mode_preserves_multiple_pages_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_responses = ["First page\n", "\nSecond page", "Third page"]
    result = _normalize_raw_response(monkeypatch, raw_responses, prompt_mode="plain")

    assert [page.page_index for page in result.output.pages] == [0, 1, 2]
    assert [page.markdown for page in result.output.pages] == raw_responses
    assert result.output.markdown == "\n\n".join(raw_responses)
    assert result.output.layout_pages == []


def test_fork_mode_keeps_non_block_response_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _normalize_raw_response(monkeypatch, "ordinary full-page markdown", prompt_mode="fork")

    assert result.output.pages[0].markdown == ""
    assert result.output.markdown == ""
    assert result.output.layout_pages == []


def test_parse_blocks_maps_labels_and_keeps_text_with_invalid_geometry() -> None:
    items = _parse_blocks(
        "<BLOCK>title [10, 20, 900, 80]<CHILD>Heading"
        "<BLOCK>formula [20, 100, 600, 180]<CHILD>\\[x + y\\]"
        "<BLOCK>equation [20, 200, 600, 280]<CHILD>\\[a = b\\]"
        "<BLOCK>page_footnote [20, 800, 900, 850]<CHILD>Source note"
        "<BLOCK>page_number [450, 930, 550, 980]<CHILD>7"
        "<BLOCK>abandon [0, 0, 1000, 1000]<CHILD>drop me"
        "<BLOCK>text [bad, 1, 2, 3]<CHILD>keep text only"
    )

    assert [item["label"] for item in items] == [
        "Section-header",
        "Formula",
        "Formula",
        "Footnote",
        "Page-footer",
        "Text",
    ]
    assert items[1]["text"] == "x + y"
    assert items[2]["text"] == "a = b"
    assert items[-1] == {"label": "Text", "bbox": [], "text": "keep text only"}


@pytest.mark.parametrize(
    ("raw_label", "canonical_label"),
    [
        ("text", "Text"),
        ("title", "Section-header"),
        ("doc_title", "Title"),
        ("paragraph_title", "Section-header"),
        ("ref_text", "Text"),
        ("phonetic", "Text"),
        ("header", "Page-header"),
        ("footer", "Page-footer"),
        ("page_number", "Page-header"),
        ("aside_text", "Text"),
        ("page_footnote", "Footnote"),
        ("list", "List-item"),
        ("index", "Text"),
        ("image_caption", "Caption"),
        ("table_caption", "Caption"),
        ("code_caption", "Caption"),
        ("image_footnote", "Footnote"),
        ("table_footnote", "Footnote"),
        ("image", "Picture"),
        ("image_block", "Picture"),
        ("table", "Table"),
        ("chart", "Picture"),
        ("code", "Code"),
        ("algorithm", "Code"),
        ("equation", "Formula"),
    ],
)
def test_parse_blocks_supports_public_hpd_labels(raw_label: str, canonical_label: str) -> None:
    items = _parse_blocks(f"<BLOCK>{raw_label} [10, 20, 300, 200]<CHILD>block content")

    assert items == [
        {
            "label": canonical_label,
            "bbox": [10.0, 20.0, 300.0, 200.0],
            "text": "block content",
        }
    ]


def test_parse_blocks_keeps_geometry_only_blocks_separated_by_whitespace() -> None:
    items = _parse_blocks(
        "<BLOCK>table [10, 20, 300, 200] \n"
        "<BLOCK>list [10, 220, 300, 400]\n\n"
        "<BLOCK>chart [320, 20, 900, 400]  \n"
        "<BLOCK>image [320, 420, 900, 800]\n"
        "<BLOCK>text [10, 820, 900, 900]<CHILD>After the containers"
    )

    assert [(item["label"], item["text"]) for item in items] == [
        ("Table", ""),
        ("Picture", ""),
        ("Picture", ""),
        ("Text", "After the containers"),
    ]


def test_invalid_bboxes_keep_markdown_and_later_valid_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        "<BLOCK>text [10, broken, 300]<CHILD>Malformed geometry text"
        "<BLOCK>text [10, 120, 300]<CHILD>Short geometry text"
        "<BLOCK>text [100, 220, 900, 320]<CHILD>Valid geometry text",
    )

    assert result.output.markdown == ("Malformed geometry text\n\nShort geometry text\n\nValid geometry text")
    assert [item.value for item in result.output.layout_pages[0].items] == ["Valid geometry text"]

    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert [prediction.label for prediction in layout.predictions] == ["Text"]
    assert layout.predictions[0].bbox == pytest.approx([100.0, 220.0, 900.0, 320.0])


@pytest.mark.parametrize("marker", ["<FORK>", "<CHILD>", "<BLOCK>"])
def test_protocol_markers_truncate_later_branch_text(
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        f"<BLOCK>text [10, 20, 900, 200]<CHILD>Before{marker}discarded branch text",
    )

    assert result.output.markdown == "Before"
    assert result.output.layout_pages[0].items[0].value == "Before"


@pytest.mark.parametrize(
    "formula",
    [
        r"\[x + y\]",
        r"\(x + y\)",
        r"\[x + y",
        r"\(x + y",
    ],
)
def test_formula_delimiters_produce_valid_markdown(
    monkeypatch: pytest.MonkeyPatch,
    formula: str,
) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        f"<BLOCK>equation [10, 20, 900, 200]<CHILD>{formula}",
    )

    assert result.output.markdown == "$$\nx + y\n$$"
    assert result.output.layout_pages[0].items[0].value == "x + y"


@pytest.mark.parametrize("formula", [r"\[", r"\("])
def test_empty_truncated_formula_does_not_emit_unmatched_delimiter(
    monkeypatch: pytest.MonkeyPatch,
    formula: str,
) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        f"<BLOCK>equation [10, 20, 900, 200]<CHILD>{formula}",
    )

    assert result.output.markdown == ""
    assert result.output.layout_pages[0].items[0].value == ""


def test_normalize_preserves_new_public_label_content_and_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        "<BLOCK>ref_text [10, 20, 300, 100]<CHILD>Referenced paragraph"
        "<BLOCK>code [20, 120, 700, 300]<CHILD>print('hello')"
        "<BLOCK>code_caption [20, 310, 700, 350]<CHILD>Listing 1"
        "<BLOCK>image_block [100, 400, 900, 900]<CHILD>Architecture diagram",
    )

    assert result.output.markdown == ("Referenced paragraph\n\nprint('hello')\n\nListing 1\n\nArchitecture diagram")
    items = result.output.layout_pages[0].items
    assert [item.layout_segments[0].label for item in items] == [
        "Text",
        "Code",
        "Caption",
        "Picture",
    ]
    assert [item.value for item in items] == [
        "Referenced paragraph",
        "print('hello')",
        "Listing 1",
        "Architecture diagram",
    ]
    assert items[1].layout_segments[0].model_dump(include={"x", "y", "w", "h"}) == {
        "x": 0.02,
        "y": 0.12,
        "w": 0.68,
        "h": 0.18,
    }

    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert [prediction.label for prediction in layout.predictions] == [
        "Text",
        "Code",
        "Caption",
        "Picture",
    ]
    assert layout.predictions[1].bbox == pytest.approx([20.0, 120.0, 700.0, 300.0])


def test_truncated_table_is_closed_before_later_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        "<BLOCK>table [10, 20, 900, 300]<CHILD><table><tr><td colspan=2>First table"
        "<BLOCK>table_caption [10, 310, 900, 350]<CHILD>Table 1 caption"
        "<BLOCK>text [10, 360, 900, 450]<CHILD>Following paragraph"
        "<BLOCK>page_number [450, 920, 550, 970]<CHILD>7"
        "<BLOCK>table [10, 500, 900, 800]<CHILD><table><tr><td>Second table</td></tr></table>",
    )

    assert result.output.markdown.index("</table>") < result.output.markdown.index("Table 1 caption")
    assert result.output.layout_pages[0].items[0].value.endswith("</table>")
    parsed = BeautifulSoup(result.output.markdown, "html.parser")
    tables = parsed.find_all("table")
    assert len(tables) == 2
    assert tables[0].get_text(" ", strip=True) == "First table"
    assert tables[1].get_text(" ", strip=True) == "Second table"
    assert "Table 1 caption" not in tables[0].get_text(" ", strip=True)
    assert "Following paragraph" not in tables[0].get_text(" ", strip=True)
    assert parsed.get_text(" ", strip=True).split() == [
        "First",
        "table",
        "Table",
        "1",
        "caption",
        "Following",
        "paragraph",
        "7",
        "Second",
        "table",
    ]


def test_table_attribute_quoting_does_not_modify_non_table_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = '<table><tr><td colspan="2">Value</td></tr></table>'
    autolink = "<https://example.com?q=foo>"
    comparison = "a < x=1 > b"
    code = "`a < x=1 > b`"
    arbitrary = "literal <custom attr=value> text"
    formula = "x < y=1 > z"
    result = _normalize_raw_response(
        monkeypatch,
        "<BLOCK>table [10, 20, 900, 200]<CHILD><table><tr><td colspan=2>Value</td></tr></table>"
        f"<BLOCK>text [10, 220, 900, 260]<CHILD>{autolink}"
        f"<BLOCK>text [10, 280, 900, 320]<CHILD>{comparison}"
        f"<BLOCK>code [10, 340, 900, 380]<CHILD>{code}"
        f"<BLOCK>text [10, 400, 900, 440]<CHILD>{arbitrary}"
        f"<BLOCK>equation [10, 460, 900, 500]<CHILD>{formula}",
    )

    assert result.output.markdown == (
        f"{table}\n\n{autolink}\n\n{comparison}\n\n{code}\n\n{arbitrary}\n\n$$\n{formula}\n$$"
    )
    assert [item.value for item in result.output.layout_pages[0].items] == [
        table,
        autolink,
        comparison,
        code,
        arbitrary,
        formula,
    ]


@pytest.mark.parametrize("sentinel", _OFFICIAL_NO_CONTENT_SENTINELS)
def test_official_no_content_sentinels_are_layout_only(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: str,
) -> None:
    result = _normalize_raw_response(
        monkeypatch,
        f"<BLOCK>text [10, 20, 300, 100]<CHILD>{sentinel}<BLOCK>text [10, 120, 300, 200]<CHILD>Visible content",
    )

    assert result.output.markdown == "Visible content"
    items = result.output.layout_pages[0].items
    assert [item.value for item in items] == ["", "Visible content"]
    assert [item.layout_segments[0].label for item in items] == ["Text", "Text"]
    assert items[0].layout_segments[0].model_dump(include={"x", "y", "w", "h"}) == {
        "x": 0.01,
        "y": 0.02,
        "w": 0.29,
        "h": 0.08,
    }

    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert len(layout.predictions) == 2
    assert layout.predictions[0].bbox == [10.0, 20.0, 300.0, 100.0]


def test_provider_requires_public_server_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HPD_PARSING_SERVER_URL", raising=False)
    with pytest.raises(ProviderConfigError, match="HPD_PARSING_SERVER_URL"):
        HpdParsingProvider("hpd_parsing", {"server_url_env": "HPD_PARSING_SERVER_URL"})

    monkeypatch.setenv("HPD_PARSING_SERVER_URL", "https://example.invalid/v1")
    provider = HpdParsingProvider("hpd_parsing", {"server_url_env": "HPD_PARSING_SERVER_URL"})
    assert provider.provider_name == "hpd_parsing"


def test_normalize_builds_markdown_layout_and_adapter_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPD_PARSING_SERVER_URL", "https://example.invalid/v1")
    pipeline = get_pipeline("hpd_parsing_vllm_parse")
    provider = HpdParsingProvider(pipeline.provider_name, pipeline.config)
    request = InferenceRequest(
        example_id="layout/example",
        source_file_path="example.png",
        product_type="parse",
    )
    now = datetime.now()
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="parse",
        raw_output={
            "pages": [
                {
                    "page_index": 0,
                    "width": 1000,
                    "height": 2000,
                    "raw_response": (
                        "<BLOCK>title [100, 100, 900, 200]<CHILD>Overview"
                        "<BLOCK>text [100, 250, 900, 400]<CHILD>Body text"
                        "<BLOCK>equation [100, 410, 900, 440]<CHILD>\\[e = mc^2\\]"
                        "<BLOCK>page_footnote [100, 445, 900, 449]<CHILD>Source note"
                        "<BLOCK>table [100, 450, 900, 800]"
                        "<CHILD><table><tr><td colspan=2>Value</td></tr></table>"
                        "<BLOCK>chart [100, 850, 900, 980]"
                    ),
                }
            ],
            "num_pages": 1,
            "model": "PaddlePaddle/HPD-Parsing",
            "prompt_mode": "fork",
        },
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )

    result = provider.normalize(raw_result)

    assert result.output.markdown.startswith("## Overview\n\nBody text")
    assert "$$\ne = mc^2\n$$" in result.output.markdown
    assert "Source note" in result.output.markdown
    assert 'colspan="2"' in result.output.markdown
    assert [item.type for item in result.output.layout_pages[0].items] == [
        "text",
        "text",
        "text",
        "text",
        "table",
        "image",
    ]

    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, HpdParsingLayoutAdapter)
    layout = adapter.to_layout_output(result)
    assert layout.model is LayoutDetectionModel.HPD_PARSING_LAYOUT
    assert [prediction.label for prediction in layout.predictions] == [
        "Section-header",
        "Text",
        "Formula",
        "Footnote",
        "Table",
        "Picture",
    ]
    assert layout.predictions[0].bbox == [100.0, 200.0, 900.0, 400.0]
    serialized = layout.model_dump(mode="json")
    assert serialized["model"] == "hpd_parsing_layout"
    assert LayoutOutput.model_validate(serialized).model is LayoutDetectionModel.HPD_PARSING_LAYOUT
    assert LAYOUT_MODEL_INFO[LayoutDetectionModel.HPD_PARSING_LAYOUT]["name"] == "HPD-Parsing Layout"


def test_empty_list_container_does_not_hide_or_duplicate_child_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HPD_PARSING_SERVER_URL", "https://example.invalid/v1")
    pipeline = get_pipeline("hpd_parsing_vllm_parse")
    provider = HpdParsingProvider(pipeline.provider_name, pipeline.config)
    request = InferenceRequest(
        example_id="layout/list-container",
        source_file_path="example.png",
        product_type="parse",
    )
    now = datetime.now()
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="parse",
        raw_output={
            "pages": [
                {
                    "page_index": 0,
                    "width": 1000,
                    "height": 1000,
                    "raw_response": (
                        "<BLOCK>list [50, 100, 950, 900]\n"
                        "<BLOCK>text [100, 150, 900, 300]<CHILD>First child item"
                        "<BLOCK>page_footnote [100, 800, 900, 850]<CHILD>Child footnote"
                    ),
                }
            ],
            "num_pages": 1,
            "model": "PaddlePaddle/HPD-Parsing",
            "prompt_mode": "fork",
        },
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )

    result = provider.normalize(raw_result)
    assert result.output.markdown == "First child item\n\nChild footnote"
    assert [item.layout_segments[0].label for item in result.output.layout_pages[0].items] == [
        "Text",
        "Footnote",
    ]

    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert layout.model is LayoutDetectionModel.HPD_PARSING_LAYOUT
    assert [prediction.label for prediction in layout.predictions] == ["Text", "Footnote"]
    projected = project_layout_predictions(
        result,
        layout,
        evaluation_view="canonical",
        target_ontology="canonical",
    )
    assert [prediction["class_name"] for prediction in projected] == ["Text", "Footnote"]


def _provider_for_test(monkeypatch: pytest.MonkeyPatch) -> tuple[HpdParsingProvider, PipelineSpec]:
    monkeypatch.setenv("HPD_PARSING_SERVER_URL", "https://example.invalid/v1")
    pipeline = get_pipeline("hpd_parsing_vllm_parse")
    return HpdParsingProvider(pipeline.provider_name, pipeline.config), pipeline


def _request_for_file(path: Path) -> InferenceRequest:
    return InferenceRequest(example_id="errors/example", source_file_path=str(path), product_type="parse")


@pytest.mark.parametrize("extension", [".tif", ".tiff"])
def test_run_inference_preserves_all_tiff_frames_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extension: str,
) -> None:
    provider, pipeline = _provider_for_test(monkeypatch)
    source = tmp_path / f"pages{extension}"
    frames = [Image.new("RGB", (2, 2), (value, 0, 0)) for value in (10, 100, 200)]
    frames[0].save(source, format="TIFF", save_all=True, append_images=frames[1:])
    seen_values: list[int] = []

    def response_for_frame(image: Image.Image) -> str:
        value = image.convert("RGB").getpixel((0, 0))[0]
        seen_values.append(value)
        return f"<BLOCK>text [10, 20, 900, 200]<CHILD>Frame {value}"

    monkeypatch.setattr(provider, "_call_endpoint", response_for_frame)

    raw_result = provider.run_inference(pipeline, _request_for_file(source))
    assert seen_values == [10, 100, 200]
    assert raw_result.raw_output["num_pages"] == 3
    assert [page["page_index"] for page in raw_result.raw_output["pages"]] == [0, 1, 2]
    assert [page["raw_response"].rsplit("<CHILD>", 1)[1] for page in raw_result.raw_output["pages"]] == [
        "Frame 10",
        "Frame 100",
        "Frame 200",
    ]

    result = provider.normalize(raw_result)
    assert [page.markdown for page in result.output.pages] == ["Frame 10", "Frame 100", "Frame 200"]
    assert result.output.markdown == "Frame 10\n\nFrame 100\n\nFrame 200"


class _OpenAIError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, response_status: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=response_status) if response_status is not None else None


class _FailingCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def create(self, **_kwargs: object) -> None:
        self.calls += 1
        raise self.error


def _set_client_error(provider: HpdParsingProvider, error: Exception) -> _FailingCompletions:
    completions = _FailingCompletions(error)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return completions


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (_OpenAIError("slow down", status_code=429), ProviderRateLimitError),
        (_OpenAIError("request timed out", status_code=408), ProviderTransientError),
        (_OpenAIError("server failed", status_code=500), ProviderTransientError),
        (_OpenAIError("server failed", status_code=599), ProviderTransientError),
        (_OpenAIError("bad request says timeout 503", status_code=400), ProviderPermanentError),
        (_OpenAIError("missing", status_code=404), ProviderPermanentError),
        (_OpenAIError("gateway", response_status=502), ProviderTransientError),
    ],
)
def test_openai_http_statuses_map_to_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_error: type[Exception],
) -> None:
    provider, _pipeline = _provider_for_test(monkeypatch)
    _set_client_error(provider, error)

    with pytest.raises(expected_error):
        provider._call_endpoint(Image.new("RGB", (1, 1)))


def test_rate_limit_retries_then_raises_final_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider, pipeline = _provider_for_test(monkeypatch)
    source = tmp_path / "page.png"
    Image.new("RGB", (1, 1)).save(source)
    completions = _set_client_error(provider, _OpenAIError("slow down", status_code=429))
    monkeypatch.setattr("parse_bench.inference.providers.parse.hpd_parsing.time.sleep", lambda _delay: None)

    with pytest.raises(ProviderRateLimitError, match="HTTP 429"):
        provider.run_inference(pipeline, _request_for_file(source))
    assert completions.calls == 3


def test_run_inference_raises_final_transient_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider, pipeline = _provider_for_test(monkeypatch)
    source = tmp_path / "page.png"
    source.write_bytes(b"not read because inference is mocked")
    attempts = 0

    def fail_transient(_source_path: Path) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        raise ProviderTransientError("endpoint unavailable")

    monkeypatch.setattr(provider, "_run_inference_pages", fail_transient)
    monkeypatch.setattr("parse_bench.inference.providers.parse.hpd_parsing.time.sleep", lambda _delay: None)

    with pytest.raises(ProviderTransientError, match="endpoint unavailable"):
        provider.run_inference(pipeline, _request_for_file(source))
    assert attempts == 3


def test_run_inference_preserves_permanent_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider, pipeline = _provider_for_test(monkeypatch)
    source = tmp_path / "page.png"
    source.write_bytes(b"not read because inference is mocked")
    attempts = 0

    def fail_permanently(_source_path: Path) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        raise ProviderPermanentError("render failed")

    monkeypatch.setattr(provider, "_run_inference_pages", fail_permanently)

    with pytest.raises(ProviderPermanentError, match="render failed"):
        provider.run_inference(pipeline, _request_for_file(source))
    assert attempts == 1


def test_run_inference_classifies_unexpected_render_errors_as_permanent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider, pipeline = _provider_for_test(monkeypatch)
    source = tmp_path / "page.png"
    source.write_bytes(b"invalid image")

    def fail_render(_source_path: Path) -> dict[str, object]:
        raise OSError("bad image")

    monkeypatch.setattr(provider, "_run_inference_pages", fail_render)

    with pytest.raises(ProviderPermanentError, match="HPD-Parsing inference failed: bad image"):
        provider.run_inference(pipeline, _request_for_file(source))


def test_minimal_import_does_not_load_unrelated_provider_dependencies() -> None:
    script = """
import os
import sys

for name in (
    "aiohttp",
    "pypdf",
    "requests",
    "parse_bench.inference.providers.parse.mistral_ocr",
    "parse_bench.inference.providers.parse.paddleocr",
):
    sys.modules[name] = None

os.environ["HPD_PARSING_SERVER_URL"] = "https://example.invalid/v1"
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers import create_provider

provider = create_provider(get_pipeline("hpd_parsing_vllm_parse"))
print(type(provider).__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.stdout.strip() == "HpdParsingProvider"
