from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderConfigError, ProviderTransientError
from parse_bench.inference.providers.parse._layout_utils import (
    close_open_ended_bands,
    extract_layout_blocks_lenient,
    parse_layout_blocks,
)
from parse_bench.inference.providers.parse.amazon_nova import AmazonNovaProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


class _FakeBedrockClient:
    """Captures the Converse kwargs and replays a canned response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


def _provider(**attrs: Any) -> AmazonNovaProvider:
    provider = object.__new__(AmazonNovaProvider)
    defaults: dict[str, Any] = {
        "_model": "us.amazon.nova-2-lite-v1:0",
        "_region": "us-east-1",
        "_dpi": 150,
        "_max_tokens": 32768,
        "_timeout": 300,
        "_reasoning_effort": None,
        "_temperature": 0,
        "_top_p": None,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(provider, key, value)
    return provider


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="amazon_nova_test",
        provider_name="amazon_nova",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def test_geo_profile_is_priced_at_the_regional_rate() -> None:
    assert _provider(_model="us.amazon.nova-2-lite-v1:0")._get_pricing() == (0.33, 2.75)
    assert _provider(_model="amazon.nova-2-lite-v1:0")._get_pricing() == (0.33, 2.75)


def test_global_profile_is_priced_at_the_cross_region_global_rate() -> None:
    assert _provider(_model="global.amazon.nova-2-lite-v1:0")._get_pricing() == (0.30, 2.50)


def test_unknown_model_pricing_remains_unknown() -> None:
    assert _provider(_model="us.amazon.nova-9-mystery-v1:0")._get_pricing() is None


def test_unknown_pricing_does_not_abort_completed_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 1})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (10, 20), "white")],
    )
    provider = _provider(_model="us.amazon.nova-9-mystery-v1:0")
    provider._parse_image_with_layout = lambda image: (
        [],
        "[]",
        {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 2},
        "end_turn",
    )

    result = provider.run_inference(_pipeline(), _request(source))

    assert result.raw_output["num_api_calls"] == 1
    assert result.raw_output["total_tokens"] == 2
    assert "cost_usd" not in result.raw_output
    assert "cost_per_page_usd" not in result.raw_output


def test_constructor_disables_botocore_retries_and_uses_declared_dpi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3

    captured: dict[str, Any] = {}
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

    def client(service_name: str, **kwargs: Any) -> object:
        captured["service_name"] = service_name
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(boto3, "client", client)

    provider = AmazonNovaProvider("amazon_nova")

    assert provider._dpi == AmazonNovaProvider.PDF_RENDER_DPI == 150
    assert captured["service_name"] == "bedrock-runtime"
    assert captured["config"].retries == {"total_max_attempts": 1, "mode": "standard"}


def test_converse_sends_temperature_and_no_reasoning_by_default() -> None:
    provider = _provider()
    provider._client = _FakeBedrockClient(
        {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "stopReason": "end_turn",
        }
    )

    text, usage, stop_reason = provider._converse(Image.new("RGB", (64, 64), "white"), "system", "user")

    (call,) = provider._client.calls
    assert call["modelId"] == "us.amazon.nova-2-lite-v1:0"
    assert call["inferenceConfig"] == {"maxTokens": 32768, "temperature": 0}
    assert "additionalModelRequestFields" not in call
    assert call["messages"][0]["content"][0]["image"]["format"] == "jpeg"
    assert isinstance(call["messages"][0]["content"][0]["image"]["source"]["bytes"], bytes)
    assert text == "hello"
    assert usage == {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 15}
    assert stop_reason == "end_turn"


def test_converse_enables_reasoning_without_sampling_params() -> None:
    provider = _provider(_reasoning_effort="high", _temperature=None)
    provider._client = _FakeBedrockClient(
        {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
    )

    provider._converse(Image.new("RGB", (64, 64), "white"), "system", "user")

    (call,) = provider._client.calls
    assert call["inferenceConfig"] == {"maxTokens": 32768}
    assert call["additionalModelRequestFields"] == {
        "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
    }


def test_redacted_reasoning_blocks_are_not_part_of_the_parsed_text() -> None:
    response = {
        "output": {
            "message": {
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "[REDACTED]"}}},
                    {"text": "# Heading"},
                ]
            }
        },
        "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
    }

    assert AmazonNovaProvider._extract_text(response) == "# Heading"


def test_reasoning_effort_rejects_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")

    with pytest.raises(ProviderConfigError, match="rejects temperature/top_p"):
        AmazonNovaProvider("amazon_nova", {"reasoning_effort": "low", "temperature": 0.5})


def test_invalid_reasoning_effort_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")

    with pytest.raises(ProviderConfigError, match="Invalid reasoning_effort"):
        AmazonNovaProvider("amazon_nova", {"reasoning_effort": "maximum"})


def test_content_filtered_response_never_becomes_page_content() -> None:
    """Bedrock returns HTTP 200 with a canned filter notice; it is not document text."""
    provider = _provider()
    provider._client = _FakeBedrockClient(
        {
            "output": {
                "message": {"content": [{"text": " - The generated text has been blocked by our content filters."}]}
            },
            "usage": {"inputTokens": 673, "outputTokens": 0, "totalTokens": 673},
            "stopReason": "content_filtered",
        }
    )

    with pytest.raises(ProviderTransientError, match="content filter"):
        provider._converse(Image.new("RGB", (64, 64), "white"), "system", "user")


def test_empty_response_is_rejected_rather_than_parsed_as_a_blank_page() -> None:
    provider = _provider()
    provider._client = _FakeBedrockClient(
        {
            "output": {"message": {"content": [{"text": "   "}]}},
            "usage": {"inputTokens": 5, "outputTokens": 0, "totalTokens": 5},
            "stopReason": "end_turn",
        }
    )

    with pytest.raises(ProviderTransientError, match="no text"):
        provider._converse(Image.new("RGB", (64, 64), "white"), "system", "user")


@pytest.mark.parametrize("malformed", ["not layout markup", "[ ]", "[] trailing text"])
def test_nonempty_malformed_layout_is_not_accepted_as_blank(malformed: str) -> None:
    provider = _provider()
    provider._converse = lambda *args: (
        malformed,
        {"input_tokens": 3, "output_tokens": 2, "thinking_tokens": 0, "total_tokens": 5},
        "end_turn",
    )

    with pytest.raises(ProviderTransientError, match="malformed non-empty layout") as caught:
        provider._parse_image_with_layout(Image.new("RGB", (64, 64), "white"))

    assert caught.value.attempt_stats == {
        "input_tokens": 3,
        "output_tokens": 2,
        "thinking_tokens": 0,
        "total_tokens": 5,
    }


def test_exact_empty_array_is_the_only_blank_layout_representation() -> None:
    provider = _provider()
    provider._converse = lambda *args: (
        "  []\n",
        {"input_tokens": 3, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 4},
        "end_turn",
    )

    items, raw_text, usage, stop_reason = provider._parse_image_with_layout(Image.new("RGB", (64, 64), "white"))

    assert items == []
    assert raw_text == "  []\n"
    assert usage["total_tokens"] == 4
    assert stop_reason == "end_turn"


def test_pdf_run_inference_renders_and_calls_bedrock_one_page_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered_pages: list[Image.Image] = []
    render_calls: list[int] = []

    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert dpi == 150
        assert first_page == last_page
        if rendered_pages:
            with pytest.raises(ValueError, match="Operation on closed image"):
                rendered_pages[-1].getpixel((0, 0))
        render_calls.append(first_page)
        image = Image.new("RGB", (10 + first_page, 20), "white")
        rendered_pages.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    provider = _provider()
    provider._client = _FakeBedrockClient(
        {
            "output": {"message": {"content": [{"text": '<div data-bbox="[0,0,10,10]" data-label="Text">page</div>'}]}},
            "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
            "stopReason": "end_turn",
        }
    )

    result = provider.run_inference(_pipeline(), _request(source))

    assert render_calls == [1, 2]
    assert len(provider._client.calls) == 2
    assert result.raw_output["num_pages"] == 2
    assert [page["width"] for page in result.raw_output["pages"]] == [11, 12]
    with pytest.raises(ValueError, match="Operation on closed image"):
        rendered_pages[-1].getpixel((0, 0))


def test_failed_attempt_without_usage_omits_precise_document_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 1})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (10, 20), "white")],
    )
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    provider = _provider()
    calls = 0

    def parse_page(image: Image.Image) -> tuple[list[dict[str, object]], str, dict[str, int], str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderTransientError("transport failed before usage was reported")
        return (
            [{"bbox": [0, 0, 10, 10], "label": "Text", "text": "page"}],
            '<div data-bbox="[0,0,10,10]" data-label="Text">page</div>',
            {"input_tokens": 2, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 3},
            "end_turn",
        )

    provider._parse_image_with_layout = parse_page

    result = provider.run_inference(_pipeline(), _request(source))

    assert calls == 2
    assert result.raw_output["num_api_calls"] == 2
    assert "total_tokens" not in result.raw_output
    assert "cost_usd" not in result.raw_output
    assert "page_usages" not in result.raw_output


def test_single_image_run_inference_calls_bedrock_without_pdf_rasterization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (13, 17), "white").save(source)
    real_open = Image.open
    close_calls = 0

    class TrackedImageContext:
        def __init__(self, path: str | Path, *args: Any, **kwargs: Any) -> None:
            self.image = real_open(path, *args, **kwargs)

        def __enter__(self) -> Image.Image:
            return self.image

        def __exit__(self, *args: object) -> None:
            nonlocal close_calls
            close_calls += 1
            self.image.close()

    monkeypatch.setattr(Image, "open", TrackedImageContext)
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda *args, **kwargs: pytest.fail("single images must not be rasterized as PDFs"),
    )
    provider = _provider()
    provider._client = _FakeBedrockClient(
        {
            "output": {"message": {"content": [{"text": '<div data-bbox="[0,0,10,10]" data-label="Text">page</div>'}]}},
            "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
            "stopReason": "end_turn",
        }
    )

    result = provider.run_inference(_pipeline(), _request(source))

    assert len(provider._client.calls) == 1
    assert result.raw_output["num_pages"] == 1
    assert result.raw_output["pages"][0]["width"] == 13
    assert close_calls == 1


NESTED_LAYOUT = (
    '<div data-bbox="[0,0,1000,146]" data-label="Text">\n'
    '<div data-bbox="[10,20,300,60]" data-label="Title">Chapter 4</div>\n'
    '<div data-bbox="[10,80,900,140]" data-label="Text">SURVEY</div>\n'
    "</div>"
)


def test_nested_divs_lose_the_child_box_and_leak_markup_under_the_strict_parser() -> None:
    """Documents the failure mode the lenient reader exists to fix."""
    blocks = parse_layout_blocks(NESTED_LAYOUT)

    assert blocks[0]["label"] == "Text"
    assert "<div" in blocks[0]["text"]


def test_lenient_reader_recovers_leaf_boxes_from_nova_nested_divs() -> None:
    blocks = extract_layout_blocks_lenient(NESTED_LAYOUT)

    assert [(b["label"], b["bbox"], b["text"]) for b in blocks] == [
        ("Title", [10, 20, 300, 60], "Chapter 4"),
        ("Text", [10, 80, 900, 140], "SURVEY"),
    ]


def test_lenient_reader_matches_the_strict_parser_on_compliant_output() -> None:
    compliant = (
        '<div data-bbox="[1,2,3,4]" data-label="Text">hi</div>\n'
        '<div data-bbox="[5,6,7,8]" data-label="Title">there</div>'
    )

    assert extract_layout_blocks_lenient(compliant) == parse_layout_blocks(compliant)
    assert extract_layout_blocks_lenient("no wrappers here") == []


SEMANTIC_TAG_LAYOUT = (
    '<TABLE data-bbox="[0, 301, 1000, 456]" data-label="Table">\n'
    "<table><tr><td>7</td></tr></table>\n"
    "</TABLE>\n"
    '<p data-bbox="[0, 456, 1000, 480]" data-label="Text">Caption line</p>'
)


def test_semantic_tag_wrappers_are_scored_instead_of_dropped() -> None:
    """Nova wraps elements in <TABLE>/<p> rather than <div>; same bbox, same label."""
    blocks = extract_layout_blocks_lenient(SEMANTIC_TAG_LAYOUT)

    assert [(b["label"], b["bbox"]) for b in blocks] == [
        ("Table", [0, 301, 1000, 456]),
        ("Text", [0, 456, 1000, 480]),
    ]
    # The inner real <table> survives so GriTS sees a table, not a wrapper.
    assert blocks[0]["text"] == "<table><tr><td>7</td></tr></table>"


def test_plain_inner_tags_without_a_bbox_are_never_treated_as_wrappers() -> None:
    content = '<div data-bbox="[1,2,3,4]" data-label="Table"><table><tr><td>a</td></tr></table></div>'

    assert extract_layout_blocks_lenient(content) == [
        {"bbox": [1, 2, 3, 4], "label": "Table", "text": "<table><tr><td>a</td></tr></table>"}
    ]


UNCLOSED_WRAPPERS = (
    '<div data-bbox="[0, 0, 1000, 1000]" data-label="Text">\n'
    "lead paragraph\n"
    '<TABLE data-bbox="[0, 301, 1000, 456]" data-label="Table">\n'
    "<table><tr><td>7</td></tr></table>\n"
    '<p data-bbox="[0, 456, 1000, 480]" data-label="Text">caption</p>\n'
    "</div>"
)


def test_unclosed_wrappers_end_at_the_next_element_and_lose_nothing() -> None:
    """Nova opens <TABLE data-bbox=...> and never closes it."""
    blocks = extract_layout_blocks_lenient(UNCLOSED_WRAPPERS)

    assert [(b["label"], b["bbox"]) for b in blocks] == [
        ("Text", [0, 0, 1000, 1000]),
        ("Table", [0, 301, 1000, 456]),
        ("Text", [0, 456, 1000, 480]),
    ]
    assert blocks[0]["text"] == "lead paragraph"
    assert blocks[1]["text"] == "<table><tr><td>7</td></tr></table>"
    assert blocks[2]["text"] == "caption"


def test_whitespace_only_container_wrappers_are_skipped() -> None:
    content = (
        '<div data-bbox="[0,0,1000,1000]" data-label="Text">\n'
        '<div data-bbox="[1,2,3,4]" data-label="Title">real</div>\n'
        "</div>"
    )

    assert [b["label"] for b in extract_layout_blocks_lenient(content)] == ["Title"]


def test_open_ended_bands_end_where_the_next_element_starts() -> None:
    """Nova pins y2 to the page bottom; its own filled-in boxes are contiguous bands."""
    items = [
        {"bbox": [0, 0, 1000, 1000], "label": "Title", "text": "a"},
        {"bbox": [0, 87, 1000, 1000], "label": "Text", "text": "b"},
        {"bbox": [0, 208, 1000, 1000], "label": "Section-header", "text": "c"},
    ]

    assert [i["bbox"] for i in close_open_ended_bands(items)] == [
        [0, 0, 1000, 87],
        [0, 87, 1000, 208],
        [0, 208, 1000, 1000],
    ]


def test_boxes_with_a_real_bottom_edge_are_left_alone() -> None:
    items = [
        {"bbox": [0, 301, 1000, 456], "label": "Table", "text": "t"},
        {"bbox": [0, 456, 1000, 480], "label": "Text", "text": "u"},
    ]

    assert [i["bbox"] for i in close_open_ended_bands(items)] == [
        [0, 301, 1000, 456],
        [0, 456, 1000, 480],
    ]


def test_elements_sharing_one_top_edge_are_not_given_invented_extents() -> None:
    """Fully degenerate output stays degenerate rather than being fabricated into bands."""
    items = [{"bbox": [0, 0, 1000, 1000], "label": "Text", "text": str(n)} for n in range(4)]

    assert [i["bbox"] for i in close_open_ended_bands(items)] == [[0, 0, 1000, 1000]] * 4
