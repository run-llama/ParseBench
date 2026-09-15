"""Regression coverage for shared Qwen parse pipelines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.qwen import QwenProvider
from parse_bench.inference.providers.registry import create_provider
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="qwen",
        provider_name="qwen3_5",
        product_type=ProductType.PARSE,
        config={},
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="qwen-multipage",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


class _FakeResponse:
    status = 200

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "[]"}}]}

    async def text(self) -> str:
        return ""


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse()


@pytest.mark.parametrize("prompt_mode", ["parse", "layout"])
def test_qwen_preserves_all_pdf_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_mode: str,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    provider = QwenProvider(
        "qwen",
        {"server_url": "https://example.invalid", "prompt_mode": prompt_mode},
    )
    calls: list[bytes] = []
    monkeypatch.setattr(
        provider,
        "_pdf_to_images_with_size",
        lambda _path: [(b"first", 100, 200), (b"second", 300, 400)],
    )

    async def fake_run(image: bytes, width: int, height: int) -> dict[str, Any]:
        calls.append(image)
        number = len(calls)
        if prompt_mode == "layout":
            return {
                "prompt_mode": "layout",
                "_config": {"model": "qwen3.8-27b-fp8"},
                "pages": [
                    {
                        "page_index": 0,
                        "width": width,
                        "height": height,
                        "layout_items": [
                            {
                                "bbox": [0, 0, 100, 100],
                                "category": "Text",
                                "text": f"page {number}",
                            }
                        ],
                    }
                ],
            }
        return {
            "prompt_mode": "parse",
            "markdown": f"page {number}",
            "_config": {"model": "qwen3.8-27b-fp8"},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run)
    raw = provider.run_inference(_pipeline(), _request(source))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert [page["page_index"] for page in raw.raw_output["pages"]] == [0, 1]
    assert normalized.output.markdown == "page 1\n\npage 2"
    if prompt_mode == "layout":
        assert [page.page_number for page in normalized.output.layout_pages] == [1, 2]
        assert [(page.width, page.height) for page in normalized.output.layout_pages] == [
            (100.0, 200.0),
            (300.0, 400.0),
        ]
        layout_output = create_layout_adapter("qwen3_8").to_layout_output(normalized)
        assert layout_output.model == LayoutDetectionModel.QWEN3_8_LAYOUT
    else:
        assert "markdown" not in raw.raw_output


@pytest.mark.parametrize("prompt_mode", ["parse", "layout"])
def test_qwen_normalizes_unexpected_error_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_mode: str,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    provider = QwenProvider(
        "qwen",
        {"server_url": "https://example.invalid", "prompt_mode": prompt_mode},
    )
    monkeypatch.setattr(
        provider,
        "_pdf_to_images_with_size",
        lambda _path: [(b"page", 100, 200)],
    )

    async def fail_run(_pages: list[tuple[bytes, int, int]]) -> dict[str, Any]:
        raise RuntimeError("request failed")

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail_run)
    raw = provider.run_inference(_pipeline(), _request(source))

    assert raw.raw_output["prompt_mode"] == prompt_mode
    assert provider.normalize(raw).output.markdown == ""

    raw.raw_output.pop("prompt_mode")
    assert provider.normalize(raw).output.markdown == ""


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen_sends_explicit_reasoning_mode(enable_thinking: bool) -> None:
    provider = QwenProvider(
        "qwen",
        {
            "server_url": "https://example.invalid",
            "prompt_mode": "layout",
            "enable_thinking": enable_thinking,
        },
    )
    session = _FakeSession()

    assert asyncio.run(provider._call_api(session, "encoded-image")) == "[]"

    payload = session.calls[0]["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": enable_thinking}
    assert payload["messages"][0]["content"][1]["text"] == provider._prompt


def test_qwen38_layout_pipelines_differ_only_by_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN3_8_27B_SERVER_URL", "https://example.invalid")
    non_thinking = get_pipeline("qwen3_8_27b_parse_with_layout")
    thinking = get_pipeline("qwen3_8_27b_thinking_parse_with_layout")

    assert non_thinking.provider_name == thinking.provider_name == "qwen3_8"
    assert non_thinking.product_type == thinking.product_type == ProductType.PARSE
    assert non_thinking.config["enable_thinking"] is False
    assert thinking.config["enable_thinking"] is True
    assert {**non_thinking.config, "enable_thinking": True} == thinking.config
    assert thinking.config["prompt_mode"] == "layout"
    assert "table_format" not in thinking.config
    assert "mode" not in thinking.config
    assert type(create_provider(thinking)).__name__ == "QwenProvider"
    assert type(create_layout_adapter("qwen3_8")).__name__ == "QwenLayoutAdapter"


def test_qwen_reasoning_effort_and_sampling_overrides() -> None:
    provider = QwenProvider(
        "qwen",
        {
            "server_url": "https://example.invalid",
            "enable_thinking": True,
            "reasoning_effort": "medium",
            "top_p": 0.8,
            "top_k": 20,
        },
    )
    session = _FakeSession()

    asyncio.run(provider._call_api(session, "encoded-image"))

    payload = session.calls[0]["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20


def test_qwen_omits_sampling_overrides_when_unset() -> None:
    provider = QwenProvider("qwen", {"server_url": "https://example.invalid"})
    session = _FakeSession()

    asyncio.run(provider._call_api(session, "encoded-image"))

    payload = session.calls[0]["json"]
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    "config",
    [
        {"enable_thinking": True, "reasoning_effort": "ultra"},
        {"enable_thinking": False, "reasoning_effort": "low"},
    ],
)
def test_qwen_rejects_invalid_reasoning_effort(config: dict[str, object]) -> None:
    with pytest.raises(ProviderConfigError):
        QwenProvider("qwen", {"server_url": "https://example.invalid", **config})
