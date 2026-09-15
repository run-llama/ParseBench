"""Tests for Extend parse per-page markdown reconstruction."""

from __future__ import annotations

import pytest

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.extend_parse import ExtendParseProvider, _build_pages


def _block(content: str, page: int | str | None) -> dict:
    block: dict = {"content": content}
    if page is not None:
        block["metadata"] = {"page": {"number": page}}
    return block


def test_build_pages_groups_blocks_by_page_number() -> None:
    chunks = [
        {"blocks": [_block("a", 1), _block("b", 2)]},
        {"blocks": [_block("c", 1), _block("", 2)]},
        "not-a-chunk",
    ]
    pages = _build_pages(chunks)
    assert [(p.page_index, p.markdown) for p in pages] == [(0, "a\n\nc"), (1, "b")]


def test_build_pages_falls_back_to_block_page_fields_and_drops_invalid() -> None:
    chunks = [
        {
            "blocks": [
                {"content": "legacy", "page": 3},
                {"content": "camel", "pageNumber": "2"},
                {"content": "unknown"},
                {"content": "zero", "metadata": {"page": {"number": 0}}},
                {"content": "junk", "metadata": {"page": {"number": "x"}}},
            ]
        }
    ]
    pages = _build_pages(chunks)
    assert [(p.page_index, p.markdown) for p in pages] == [(0, "unknown\n\njunk"), (1, "camel"), (2, "legacy")]


def test_credits_per_page_rejects_negative_override() -> None:
    provider = ExtendParseProvider.__new__(ExtendParseProvider)
    provider._base_config = {}
    with pytest.raises(ProviderConfigError):
        provider._credits_per_page({"credits_per_page": -1})
    assert provider._credits_per_page({"engine": "parse_light"}) == 0.5
