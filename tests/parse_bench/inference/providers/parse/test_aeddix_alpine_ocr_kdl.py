"""Tests for the aeddix-alpine-ocr provider wiring (the kdl_frontier_nano pipeline with other weights)."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.layout_adapters.adapters import KdlFrontierNanoLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.aeddix_alpine_ocr_kdl import AeddixAlpineOcrKdlProvider
from parse_bench.inference.providers.parse.kdl_frontier_nano import KdlFrontierNanoProvider
from parse_bench.inference.providers.registry import create_provider


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL",
        "AEDDIX_ALPINE_OCR_KDL_MODEL",
        "KDL_NANO_ENDPOINT_URL",
        "KDL_NANO_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_endpoint_and_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL", "http://localhost:8000/v1/")
    provider = create_provider(get_pipeline("aeddix_alpine_ocr_kdl"))
    assert isinstance(provider, AeddixAlpineOcrKdlProvider)
    assert provider._endpoint_url == "http://localhost:8000/v1"
    assert provider._model == "aeddix-alpine-ocr"
    assert provider._dpi == 144

    monkeypatch.setenv("AEDDIX_ALPINE_OCR_KDL_MODEL", "served-name")
    assert create_provider(get_pipeline("aeddix_alpine_ocr_kdl"))._model == "served-name"


def test_config_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL", "http://env.invalid/v1")
    monkeypatch.setenv("AEDDIX_ALPINE_OCR_KDL_MODEL", "env-name")
    provider = AeddixAlpineOcrKdlProvider(
        "aeddix_alpine_ocr_kdl", {"endpoint_url": "http://cfg.invalid/v1", "model": "cfg-name"}
    )
    assert provider._endpoint_url == "http://cfg.invalid/v1"
    assert provider._model == "cfg-name"


def test_kdl_variables_do_not_configure_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KDL_NANO_ENDPOINT_URL", "http://kdl.invalid/v1")
    monkeypatch.setenv("KDL_NANO_MODEL", "kdl-frontier-parser-nano")
    with pytest.raises(ProviderConfigError, match="AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL"):
        create_provider(get_pipeline("aeddix_alpine_ocr_kdl"))

    monkeypatch.setenv("AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL", "http://localhost:8000/v1")
    assert create_provider(get_pipeline("aeddix_alpine_ocr_kdl"))._model == "aeddix-alpine-ocr"


def test_pipeline_is_kdl_frontier_nano_unchanged() -> None:
    ours, kdl = get_pipeline("aeddix_alpine_ocr_kdl"), get_pipeline("kdl_frontier_nano")
    assert ours.provider_name == "aeddix_alpine_ocr_kdl"
    assert ours.product_type == kdl.product_type
    assert ours.config == kdl.config
    assert issubclass(AeddixAlpineOcrKdlProvider, KdlFrontierNanoProvider)
    assert [name for name, value in vars(AeddixAlpineOcrKdlProvider).items() if callable(value)] == ["__init__"]
    for method in ("run_inference", "normalize", "_load_page_images"):
        assert getattr(AeddixAlpineOcrKdlProvider, method) is getattr(KdlFrontierNanoProvider, method)


def test_grounding_uses_the_kdl_layout_adapter() -> None:
    assert isinstance(create_layout_adapter("aeddix_alpine_ocr_kdl"), KdlFrontierNanoLayoutAdapter)
