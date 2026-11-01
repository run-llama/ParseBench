"""End-to-end coverage of the public extension seams in parse_bench.extensions."""

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from parse_bench.evaluation.layout_adapters import registry as adapter_registry
from parse_bench.evaluation.layout_adapters.adapters import NormalizedLayoutOutputAdapter
from parse_bench.evaluation.layout_label_mappers import registry as mapper_registry
from parse_bench.evaluation.layout_label_mappers.base import MappingContext
from parse_bench.evaluation.layout_label_mappers.mappers import CanonicalPassthroughMapper
from parse_bench.evaluation.metrics.parse import rules_base
from parse_bench.evaluation.metrics.parse.rules_bag import MissingWordPercentRule
from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.extensions import (
    ExtensionProductType,
    ParseRuleBase,
    ParseTestRule,
    register_layout_adapter,
    register_layout_label_mapper,
    register_output_model,
    register_product_type,
    register_rule_type,
    registered_parse_rule_types,
    registered_product_types,
)
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel, LayoutOutput
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType, coerce_product_type
from parse_bench.test_cases import parse_rule_schemas
from parse_bench.test_cases.schema import ParseTestCase

# --- rule types -------------------------------------------------------------


class _ShoutRuleSchema(ParseRuleBase):
    type: Literal["shout"] = "shout"
    text: str


class _ShoutRule(ParseTestRule):
    """Passes when the rule text appears upper-cased in the markdown."""

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        text = self._rule_data.text.upper()
        return (text in md_content, f"looked for {text!r}")


def test_register_rule_type_validates_and_scores() -> None:
    register_rule_type("shout", _ShoutRuleSchema, _ShoutRule)
    assert "shout" in registered_parse_rule_types()

    case = ParseTestCase.model_validate(
        {
            "test_id": "ext/doc",
            "group": "ext",
            "file_path": "doc.pdf",
            "test_rules": [{"type": "shout", "text": "hello"}, {"type": "present", "text": "hello"}],
        }
    )
    assert case.test_rules is not None
    assert isinstance(case.test_rules[0], _ShoutRuleSchema)

    rule = create_test_rule(case.test_rules[0])
    assert isinstance(rule, _ShoutRule)
    assert rule.run("HELLO world")[0] is True
    assert rule.run("hello world")[0] is False


def test_register_rule_type_rejects_replacing_builtin() -> None:
    with pytest.raises(ValueError, match="built-in"):
        register_rule_type("present", _ShoutRuleSchema, _ShoutRule)


# --- product types ----------------------------------------------------------


def test_register_product_type_accepted_in_pipeline_spec() -> None:
    qa = register_product_type("qa_ext")
    assert isinstance(qa, ExtensionProductType)
    assert qa == "qa_ext" and qa.value == "qa_ext" and qa.name == "QA_EXT"
    assert register_product_type("qa_ext") is qa  # idempotent
    assert "qa_ext" in registered_product_types()

    spec = PipelineSpec(pipeline_name="p", provider_name="x", product_type="qa_ext")
    assert spec.product_type == "qa_ext"
    assert spec.product_type.value == "qa_ext"
    assert PipelineSpec.model_validate_json(spec.model_dump_json()).product_type == qa

    builtin = PipelineSpec(pipeline_name="p", provider_name="x", product_type="parse")
    assert builtin.product_type is ProductType.PARSE
    assert builtin.product_type.value == "parse"


def test_unknown_product_type_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown product type"):
        coerce_product_type("definitely_not_registered")
    with pytest.raises(ValueError, match="built-in"):
        register_product_type("parse")


# --- output models ----------------------------------------------------------


class _QAOutput(BaseModel):
    task_type: Literal["qa_ext"] = "qa_ext"
    example_id: str
    answers: list[str] = Field(default_factory=list)


def _result(output: BaseModel | dict[str, Any]) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(example_id="e", source_file_path="doc.pdf", product_type="parse"),
        pipeline_name="p",
        product_type="parse",
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_register_output_model_round_trips_through_json() -> None:
    register_output_model("qa_ext", _QAOutput)
    result = _result({"task_type": "qa_ext", "example_id": "e", "answers": ["42"]})
    assert isinstance(result.output, _QAOutput)

    reloaded = InferenceResult.model_validate_json(result.model_dump_json())
    assert isinstance(reloaded.output, _QAOutput)
    assert reloaded.output.answers == ["42"]


def test_builtin_outputs_still_dispatch() -> None:
    result = _result({"task_type": "parse", "example_id": "e", "pipeline_name": "p", "markdown": "# hi"})
    assert isinstance(result.output, ParseOutput)
    with pytest.raises(ValueError, match="Unknown output task_type"):
        _result({"task_type": "nope", "example_id": "e"})
    with pytest.raises(ValueError, match="built-in"):
        register_output_model("parse", _QAOutput)


def test_parse_test_case_round_trip_keeps_subclass_rule_fields() -> None:
    """Rules are typed on the base class so extensions validate; serialization must still emit subclass fields."""
    case = ParseTestCase(
        test_id="t/x",
        group="t",
        file_path="x.pdf",
        test_rules=[{"type": "missing_specific_word", "word": "hi"}, {"type": "is_latex", "formula": "x^2"}],
    )
    dumped = case.model_dump()
    assert dumped["test_rules"][0]["word"] == "hi"
    assert dumped["test_rules"][1]["formula"] == "x^2"
    reloaded = ParseTestCase.model_validate(dumped)
    assert reloaded.test_rules is not None and reloaded.test_rules[0].word == "hi"  # type: ignore[union-attr]


# --- import weight ----------------------------------------------------------

_PROVIDER_IMPORT_PROBE = textwrap.dedent(
    """
    import sys

    import parse_bench.evaluation.layout_adapters
    import parse_bench.evaluation.layout_label_mappers
    import parse_bench.extensions

    packages = tuple(f"parse_bench.inference.providers.{p}." for p in ("extract", "layoutdet", "parse"))
    helpers = ("parse_bench.inference.providers.layoutdet.adapters", "parse_bench.inference.providers.layoutdet.base")
    print(sorted(m for m in sys.modules if m.startswith(packages) and m not in helpers))

    from parse_bench.inference.pipelines import get_pipeline
    from parse_bench.inference.providers.registry import create_provider

    print(type(create_provider(get_pipeline("qwen3_8_27b_thinking_parse_with_layout"))).__name__)
    """
)


def test_extension_seam_imports_no_provider_until_create_provider() -> None:
    """Importing the extension seam or the evaluation registries must not import a provider module.

    Each provider pulls in its SDK, and an import of parse_bench.inference.providers.* used to load
    them all. create_provider must still find a built-in provider on first use. Measured in a fresh
    interpreter: this session's sys.modules already holds the providers.
    """
    done = subprocess.run(
        [sys.executable, "-c", _PROVIDER_IMPORT_PROBE],
        env={**os.environ, "QWEN3_8_27B_SERVER_URL": "https://example.invalid"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["[]", "QwenProvider"]


class _MatchedAdapter(NormalizedLayoutOutputAdapter):
    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        return True


class _ReplacementAdapter(_MatchedAdapter):
    pass


class _MatchedMapper(CanonicalPassthroughMapper):
    @classmethod
    def matches(cls, context: MappingContext) -> bool:
        return True


class _ReplacementMapper(_MatchedMapper):
    pass


def test_adapter_override_replaces_aliases_and_shape_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_registry, "_LAYOUT_ADAPTER_REGISTRY", [])
    monkeypatch.setattr(adapter_registry, "_PIPELINE_RESOLVERS", [])
    register_layout_adapter("parser", "alias", priority=100)(_MatchedAdapter)
    register_layout_adapter("unrelated")(NormalizedLayoutOutputAdapter)

    with pytest.raises(ValueError, match="already registered"):
        register_layout_adapter("parser", "alias")(_ReplacementAdapter)
    with pytest.raises(ValueError, match="all aliases"):
        register_layout_adapter("parser", _DANGEROUSLY_OVERRIDE=True)(_ReplacementAdapter)
    assert type(adapter_registry.create_layout_adapter("parser")) is _MatchedAdapter

    register_layout_adapter("parser", "alias", priority=0, _DANGEROUSLY_OVERRIDE=True)(_ReplacementAdapter)
    for key in ("parser", "alias"):
        assert type(adapter_registry.create_layout_adapter(key)) is _ReplacementAdapter
    assert type(adapter_registry.create_layout_adapter("unrelated")) is NormalizedLayoutOutputAdapter
    result = _result({"task_type": "parse", "example_id": "e", "pipeline_name": "p", "markdown": ""})
    assert type(adapter_registry.create_layout_adapter_for_result(result)) is _ReplacementAdapter


def test_mapper_override_replaces_provider_model_and_shape_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapper_registry, "_LAYOUT_LABEL_MAPPER_REGISTRY", [])
    model = LayoutDetectionModel.LLAMAPARSE
    keys = ("parser", f"model:{model.value}")
    register_layout_label_mapper(*keys, priority=100)(_MatchedMapper)
    register_layout_label_mapper("unrelated")(CanonicalPassthroughMapper)
    output = LayoutOutput(
        example_id="e", pipeline_name="p", model=model, image_width=100, image_height=100, predictions=[]
    )
    context = MappingContext(
        provider_name="parser", pipeline_name="p", model=model, raw_output={}, layout_output=output
    )

    with pytest.raises(ValueError, match="already registered"):
        register_layout_label_mapper(*keys)(_ReplacementMapper)
    with pytest.raises(ValueError, match="all aliases"):
        register_layout_label_mapper("parser", _DANGEROUSLY_OVERRIDE=True)(_ReplacementMapper)
    assert type(mapper_registry.resolve_layout_label_mapper(context)) is _MatchedMapper

    register_layout_label_mapper(*keys, priority=0, _DANGEROUSLY_OVERRIDE=True)(_ReplacementMapper)
    assert type(mapper_registry.resolve_layout_label_mapper(context)) is _ReplacementMapper
    assert type(mapper_registry.resolve_layout_label_mapper(replace(context, provider_name=None))) is _ReplacementMapper
    unmatched = replace(context, provider_name=None, model=LayoutDetectionModel.DOCLING_PARSE_LAYOUT)
    assert type(mapper_registry.resolve_layout_label_mapper(unmatched)) is _ReplacementMapper
    unrelated = replace(unmatched, provider_name="unrelated")
    assert type(mapper_registry.resolve_layout_label_mapper(unrelated)) is CanonicalPassthroughMapper


class _PresentOverrideSchema(ParseRuleBase):
    type: Literal["present"] = "present"
    text: str


@pytest.mark.parametrize("normalization", [None, "text-v2.1"])
def test_rule_override_replaces_validation_and_scoring(
    monkeypatch: pytest.MonkeyPatch, normalization: str | None
) -> None:
    monkeypatch.setattr(parse_rule_schemas, "_RULE_TYPE_TO_MODEL", dict(parse_rule_schemas._RULE_TYPE_TO_MODEL))
    monkeypatch.setattr(rules_base, "_RULE_CLASS_REGISTRY", dict(rules_base._RULE_CLASS_REGISTRY))
    register_rule_type("present", _PresentOverrideSchema, _ShoutRule, _DANGEROUSLY_OVERRIDE=True)
    case = ParseTestCase.model_validate(
        {
            "test_id": "ext/doc",
            "group": "ext",
            "file_path": "doc.pdf",
            "test_rules": [{"type": "present", "text": "hello", "text_normalization": normalization}],
        }
    )
    assert case.test_rules is not None
    assert isinstance(case.test_rules[0], _PresentOverrideSchema)
    rule = create_test_rule(case.test_rules[0])
    assert type(rule) is _ShoutRule
    assert rule.run("HELLO world")[0] is True
    assert rule.run("hello world")[0] is False


def test_invalid_rule_override_leaves_existing_schema_and_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parse_rule_schemas, "_RULE_TYPE_TO_MODEL", dict(parse_rule_schemas._RULE_TYPE_TO_MODEL))
    monkeypatch.setattr(rules_base, "_RULE_CLASS_REGISTRY", dict(rules_base._RULE_CLASS_REGISTRY))
    original = parse_rule_schemas.coerce_parse_rule({"type": "present", "text": "hello"})
    original_scorer = type(create_test_rule(original))
    with pytest.raises(TypeError, match="subclass ParseTestRule"):
        register_rule_type("present", _PresentOverrideSchema, object, _DANGEROUSLY_OVERRIDE=True)  # type: ignore[arg-type]
    unchanged = parse_rule_schemas.coerce_parse_rule({"type": "present", "text": "hello"})
    assert type(unchanged) is type(original)
    assert type(create_test_rule(unchanged)) is original_scorer


@pytest.mark.parametrize("combined_api", [False, True])
def test_normalization_precedence_changes_only_with_explicit_override(
    monkeypatch: pytest.MonkeyPatch, combined_api: bool
) -> None:
    monkeypatch.setattr(rules_base, "_RULE_CLASS_REGISTRY", dict(rules_base._RULE_CLASS_REGISTRY))
    monkeypatch.setattr(parse_rule_schemas, "_RULE_TYPE_TO_MODEL", dict(parse_rule_schemas._RULE_TYPE_TO_MODEL))
    payload = {"type": "missing_word_percent", "bag_of_word": {"café": 1}, "text_normalization": "text-v2.1"}
    for override, expected_score in ((False, 0.0), (True, 1.0), (False, 0.0)):
        if combined_api:
            register_rule_type(
                "missing_word_percent",
                parse_rule_schemas.ParseMissingWordPercentRule,
                MissingWordPercentRule,
                _DANGEROUSLY_OVERRIDE=override,
            )
        elif override:
            rules_base.register_rule_class("missing_word_percent", MissingWordPercentRule, _DANGEROUSLY_OVERRIDE=True)
        else:
            rules_base.register_rule_class("missing_word_percent", MissingWordPercentRule)
        rule = create_test_rule(payload)
        assert rule.run("cafe")[2] == expected_score
        assert rules_base.registered_rule_classes()["missing_word_percent"] is MissingWordPercentRule


@pytest.mark.parametrize("replace_keys", [("first",), ("first", "third")])
def test_adapter_override_preserves_equal_priority_shape_order(
    monkeypatch: pytest.MonkeyPatch, replace_keys: tuple[str, ...]
) -> None:
    monkeypatch.setattr(adapter_registry, "_LAYOUT_ADAPTER_REGISTRY", [])
    monkeypatch.setattr(adapter_registry, "_PIPELINE_RESOLVERS", [])
    register_layout_adapter("first", priority=90)(_MatchedAdapter)
    register_layout_adapter("second", priority=90)(_ReplacementAdapter)
    register_layout_adapter("third", priority=90)(_MatchedAdapter)
    result = _result({"task_type": "parse", "example_id": "e", "pipeline_name": "p", "markdown": ""})
    assert type(adapter_registry.create_layout_adapter_for_result(result)) is _MatchedAdapter
    register_layout_adapter(*replace_keys, priority=90, _DANGEROUSLY_OVERRIDE=True)(_MatchedAdapter)
    assert type(adapter_registry.create_layout_adapter_for_result(result)) is _MatchedAdapter


def test_mapper_override_preserves_equal_priority_provider_model_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapper_registry, "_LAYOUT_LABEL_MAPPER_REGISTRY", [])
    model = LayoutDetectionModel.LLAMAPARSE
    register_layout_label_mapper("parser", priority=90)(_MatchedMapper)
    register_layout_label_mapper(f"model:{model.value}", priority=90)(_ReplacementMapper)
    output = LayoutOutput(
        example_id="e", pipeline_name="p", model=model, image_width=100, image_height=100, predictions=[]
    )
    context = MappingContext(
        provider_name="parser", pipeline_name="p", model=model, raw_output={}, layout_output=output
    )
    assert type(mapper_registry.resolve_layout_label_mapper(context)) is _MatchedMapper
    register_layout_label_mapper("parser", priority=90, _DANGEROUSLY_OVERRIDE=True)(_MatchedMapper)
    assert type(mapper_registry.resolve_layout_label_mapper(context)) is _MatchedMapper
