"""ByteVerity provider: decision-table integrity/completeness and an offline end-to-end run (mocked API, no network)."""

import json
import math
import os
import sys
import types
from types import SimpleNamespace as NS

import pytest

pymupdf = pytest.importorskip("pymupdf")

from parse_bench.inference.providers.parse.byteverity import tables as bv_tables  # noqa: E402

TABLES = sorted(f[: -len(".table.json")] for f in os.listdir(bv_tables.TABLES_DIR) if f.endswith(".table.json"))


def test_seventeen_complete_pinned_tables():
    assert len(TABLES) == 17
    pins = json.load(open(os.path.join(bv_tables.TABLES_DIR, "PINS.json")))["tables"]
    assert sorted(pins) == TABLES
    for name in TABLES:
        t = bv_tables.table(name)  # raises on integrity or completeness failure
        raw = json.load(open(os.path.join(bv_tables.TABLES_DIR, f"{name}.table.json")))
        assert len(t.cells) == math.prod(len(i["values"]) for i in raw["inputs"])


def test_tampered_table_is_refused(tmp_path, monkeypatch):
    for f in os.listdir(bv_tables.TABLES_DIR):
        (tmp_path / f).write_bytes(open(os.path.join(bv_tables.TABLES_DIR, f), "rb").read())
    t = json.load(open(tmp_path / "page_route.table.json"))
    k = next(iter(t["cells"]))
    t["cells"][k][0] = "accept_snap" if t["cells"][k][0] != "accept_snap" else "escalate"
    (tmp_path / "page_route.table.json").write_text(json.dumps(t))
    monkeypatch.setattr(bv_tables, "TABLES_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="integrity"):
        bv_tables.DecisionTable("page_route")


def test_out_of_domain_facts_are_refused():
    r = bv_tables.table("page_route").evaluate(
        {"vlm_status": "ok", "text_layer": "bogus", "recall": "na", "escalated": False}
    )
    assert r["status"] == "INVALID_INPUT"


class _Completions:
    calls: list = []

    def create(self, **kw):
        _Completions.calls.append(kw)
        usage = NS(
            prompt_tokens=3000,
            completion_tokens=1200,
            prompt_tokens_details=NS(cached_tokens=1000),
            completion_tokens_details=NS(reasoning_tokens=200),
        )
        body = (
            '<div data-bbox="[100,100,900,200]" data-label="Title">Hello</div>\n'
            '<div data-bbox="[100,300,900,400]" data-label="Text">World</div>'
        )
        return NS(usage=usage, choices=[NS(message=NS(content=body))])


def test_offline_end_to_end_with_mocked_api(tmp_path, monkeypatch):
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda api_key, timeout, **kw: NS(chat=NS(completions=_Completions()))
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("SX_LAYOUT_LIVE", "0")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello")
    page.insert_text((72, 144), "World")
    pdf = tmp_path / "t.pdf"
    doc.save(str(pdf))

    from parse_bench.inference.providers.parse.byteverity.provider import ByteVerityProvider
    from parse_bench.schemas.pipeline import PipelineSpec
    from parse_bench.schemas.pipeline_io import InferenceRequest
    from parse_bench.schemas.product import ProductType

    cfg = {
        "model": "gpt-6-luna",
        "escalate_model": "gpt-6-luna",
        "effort": "low",
        "stage": "full",
        "transport": "openai",
    }
    prov = ByteVerityProvider("byteverity", cfg)
    spec = PipelineSpec(
        pipeline_name="byteverity_parse", provider_name="byteverity", product_type=ProductType.PARSE, config=cfg
    )
    raw = prov.run_inference(
        spec, InferenceRequest(example_id="text/t", source_file_path=str(pdf), product_type=ProductType.PARSE)
    )
    ro = raw.raw_output
    assert ro["transport"] == "openai" and ro["usage_calls"]
    assert all(c["model"] == "gpt-6-luna" for c in ro["usage_calls"])
    assert abs(ro["cost_usd"] - sum(c["cost_usd"] for c in ro["usage_calls"])) < 1e-12
    assert "Hello" in prov.normalize(raw).output.markdown


def test_api_transport_does_not_need_codex(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setenv("HOME", "/nonexistent")
    from parse_bench.inference.providers.parse.byteverity.provider import ByteVerityProvider

    prov = ByteVerityProvider("byteverity", {"model": "gpt-6-luna", "transport": "openai", "stage": "full"})
    assert prov._codex is None
