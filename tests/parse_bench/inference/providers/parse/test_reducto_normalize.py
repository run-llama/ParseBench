"""Regression tests for the Reducto normalizer.

Covers: joining ALL chunks (not just ``chunks[0]``), per-page markdown from
block ``bbox.page``, and unwrapping the URL-based large-result payload whether
it arrives as a bare chunk list or a response envelope.
"""

from datetime import datetime

import pytest

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.reducto import (
    ReductoProvider,
    _build_pages,
    _coerce_chunks,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _provider() -> ReductoProvider:
    return ReductoProvider("reducto", {"api_key": "test-key"})


def _raw_result(raw_output: dict) -> RawInferenceResult:
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(example_id="doc", source_file_path="/tmp/doc.pdf", product_type=ProductType.PARSE),
        pipeline=PipelineSpec(
            pipeline_name="reducto", provider_name="reducto", product_type=ProductType.PARSE, config={}
        ),
        pipeline_name="reducto",
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _block(content: str, page: int) -> dict:
    bbox = {"page": page, "left": 0.1, "top": 0.1, "width": 0.5, "height": 0.1}
    return {"type": "Text", "content": content, "bbox": bbox}


_CHUNKS = [
    {"content": "chunk one", "blocks": [_block("p1 a", 1), _block("p1 b", 1)]},
    {"content": "chunk two", "blocks": [_block("p2 a", 2), _block("p3 a", 3)]},
    {"content": "", "blocks": []},
]


def test_normalize_joins_all_chunks_and_builds_pages() -> None:
    result = _provider().normalize(_raw_result({"result": {"type": "full", "chunks": _CHUNKS}}))

    assert result.output.markdown == "chunk one\n\nchunk two"
    assert [p.page_index for p in result.output.pages] == [0, 1, 2]
    assert result.output.pages[0].markdown == "p1 a\n\np1 b"
    assert result.output.pages[2].markdown == "p3 a"
    assert [lp.page_number for lp in result.output.layout_pages] == [1, 2, 3]


def test_build_pages_drops_non_positive_pages() -> None:
    pages = _build_pages([{"blocks": [_block("bad", 0), _block("ok", 2), {"content": "x", "bbox": {"page": "n/a"}}]}])
    assert [(p.page_index, p.markdown) for p in pages] == [(0, "x"), (1, "ok")]


def test_coerce_chunks_accepts_list_and_envelopes() -> None:
    assert _coerce_chunks(_CHUNKS) == _CHUNKS
    assert _coerce_chunks({"chunks": _CHUNKS}) == _CHUNKS
    assert _coerce_chunks({"result": {"chunks": _CHUNKS}}) == _CHUNKS
    with pytest.raises(ProviderPermanentError):
        _coerce_chunks({"unexpected": 1})
    with pytest.raises(ProviderPermanentError):
        _coerce_chunks(["not-a-mapping"])


@pytest.mark.parametrize("payload", [_CHUNKS, {"chunks": _CHUNKS}, {"result": {"chunks": _CHUNKS}}])
def test_normalize_unwraps_url_result(monkeypatch, payload) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    monkeypatch.setattr("requests.get", lambda *_, **__: _Resp())
    result = _provider().normalize(_raw_result({"result": {"type": "url", "url": "https://example.invalid/r.json"}}))
    assert result.output.markdown == "chunk one\n\nchunk two"
    assert len(result.output.pages) == 3


def test_normalize_rejects_bad_result_shapes() -> None:
    with pytest.raises(ProviderPermanentError, match="must be an object"):
        _provider().normalize(_raw_result({"result": ["chunks"]}))
    with pytest.raises(ProviderPermanentError, match="usable 'url'"):
        _provider().normalize(_raw_result({"result": {"type": "url", "url": ""}}))
