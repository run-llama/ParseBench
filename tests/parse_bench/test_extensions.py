"""End-to-end coverage of the public extension seams in parse_bench.extensions."""

from datetime import datetime
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.extensions import (
    ExtensionProductType,
    ParseRuleBase,
    ParseTestRule,
    register_output_model,
    register_product_type,
    register_rule_type,
    registered_parse_rule_types,
    registered_product_types,
)
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType, coerce_product_type
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
