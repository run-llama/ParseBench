"""Tests for the KDL Frontier Parser nano provider wiring."""

from __future__ import annotations

import pytest

from parse_bench.inference.providers.parse.kdl_frontier_nano import KdlFrontierNanoProvider, _NanoEngine


def test_nano_engine_sets_bearer_header_only_when_api_key_given() -> None:
    engine = _NanoEngine("https://example.invalid/v1", "m", 1, 10.0, api_key="secret")
    assert engine._headers == {"Authorization": "Bearer secret"}
    assert _NanoEngine("https://example.invalid", "m", 1, 10.0)._headers == {}


def test_provider_reads_api_key_from_config_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    provider = KdlFrontierNanoProvider("kdl_frontier_nano", {"endpoint_url": "https://example.invalid/v1"})
    assert provider._api_key == "env-key"

    provider = KdlFrontierNanoProvider(
        "kdl_frontier_nano", {"endpoint_url": "https://example.invalid/v1", "api_key": "cfg-key"}
    )
    assert provider._api_key == "cfg-key"

    monkeypatch.delenv("VLLM_API_KEY")
    provider = KdlFrontierNanoProvider("kdl_frontier_nano", {"endpoint_url": "https://example.invalid/v1"})
    assert provider._api_key == ""
