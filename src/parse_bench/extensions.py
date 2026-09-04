"""Public extension points for building a benchmark harness on top of parse-bench.

Everything a downstream package needs to add its own providers, pipelines,
products, rule types and output models without forking parse-bench:

- :func:`register_provider` / :func:`register_pipeline` — new parsers and configs.
- :func:`register_product_type` + :func:`register_output_model` — a new task
  (e.g. question answering) with its own normalized output schema.
- :func:`register_rule_type` — a new ``test_rules`` entry type: its pydantic
  schema and the class that scores it.
- :func:`register_layout_adapter` / :func:`register_layout_label_mapper` — how a
  provider's layout output maps onto the canonical ontology.
- :func:`register_pipeline_resolver` — let the package resolve a result's
  ``pipeline_name`` to a provider key through the harness's own pipeline registry.
- ``EvaluationRunner.register_evaluator(product_type, evaluator)`` — scoring for
  a registered product.

Registrations take effect when the extension module is imported, so an
extension package typically performs them in its top-level ``__init__``.
"""

from parse_bench.evaluation.layout_adapters.registry import register_layout_adapter, register_pipeline_resolver
from parse_bench.evaluation.layout_label_mappers.registry import register_layout_label_mapper
from parse_bench.evaluation.metrics.parse.rules_base import (
    ParseTestRule,
    register_rule_class,
    registered_rule_classes,
)
from parse_bench.inference.pipelines import register_pipeline
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline_io import register_output_model, registered_output_models
from parse_bench.schemas.product import (
    ExtensionProductType,
    register_product_type,
    registered_product_types,
)
from parse_bench.test_cases.parse_rule_schemas import (
    ParseRuleBase,
    register_parse_rule_model,
    registered_parse_rule_types,
)


def register_rule_type(rule_type: str, schema: type[ParseRuleBase], rule_class: type[ParseTestRule]) -> None:
    """Register a new ``test_rules`` type: its payload schema and its scoring class.

    ``schema`` must subclass ``ParseRuleBase`` and declare ``type`` as a
    ``Literal[rule_type]``; ``rule_class`` must subclass ``ParseTestRule`` and
    implement ``evaluate``. After registration the type validates inside
    ``ParseTestCase.test_rules`` and is scored by ``RuleBasedMetric`` like any
    built-in rule.
    """
    register_parse_rule_model(rule_type, schema)
    register_rule_class(rule_type, rule_class)


__all__ = [
    "ExtensionProductType",
    "ParseRuleBase",
    "ParseTestRule",
    "register_layout_adapter",
    "register_layout_label_mapper",
    "register_pipeline_resolver",
    "register_output_model",
    "register_pipeline",
    "register_product_type",
    "register_provider",
    "register_rule_class",
    "register_rule_type",
    "register_parse_rule_model",
    "registered_output_models",
    "registered_parse_rule_types",
    "registered_product_types",
    "registered_rule_classes",
]
