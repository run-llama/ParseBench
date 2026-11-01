# Extending parse-bench

`parse-bench` is designed to be the shared core of a larger benchmark harness.
A downstream package adds its own providers, pipelines, products, rule types
and output models through `parse_bench.extensions` without forking the scoring
code. Registrations take effect on import, so put them in your package's
top-level `__init__.py` and import that package before running the CLI.

```python
from parse_bench.extensions import (
    register_output_model, register_pipeline, register_product_type,
    register_provider, register_rule_type,
)
```

## Providers and pipelines

```python
from parse_bench.inference.providers.base import Provider
from parse_bench.schemas.pipeline import PipelineSpec

@register_provider("my_parser")
class MyParser(Provider):
    ...

register_pipeline(PipelineSpec(pipeline_name="my_parser_fast", provider_name="my_parser",
                               product_type="parse", base_config={"mode": "fast"}))
```

## A new product (task) with its own output

```python
from typing import Literal
from pydantic import BaseModel

QA = register_product_type("qa")          # accepted wherever a ProductType is expected

class QAOutput(BaseModel):
    task_type: Literal["qa"] = "qa"       # the discriminator InferenceResult.output dispatches on
    example_id: str
    answers: list[str]

register_output_model("qa", QAOutput)

# scoring: EvaluationRunner.register_evaluator("qa", QAEvaluator())
```

`QA == "qa"`, `QA.value` and `str(QA)` all work, so code written against the
built-in `ProductType` enum keeps working with registered products.

## A new rule type

```python
from typing import Literal
from parse_bench.extensions import ParseRuleBase, ParseTestRule, register_rule_type

class LinkRuleSchema(ParseRuleBase):      # validates the test.json payload
    type: Literal["link_attribution"] = "link_attribution"
    url: str

class LinkRule(ParseTestRule):            # scores it
    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        return (self._rule_data.url in md_content, "url present")

register_rule_type("link_attribution", LinkRuleSchema, LinkRule)
```

After registration the type loads inside `ParseTestCase.test_rules` and is
scored by `RuleBasedMetric` alongside the built-ins. Built-in rule types,
products and output models cannot be replaced.

## Layout adapters and label mappers

`register_layout_adapter` and `register_layout_label_mapper` attach a
provider's layout output to the canonical ontology used by the Visual
Grounding metrics. See `parse_bench/evaluation/layout_adapters/adapters.py`
for the built-in examples.

## CLI

The CLI is a Google Fire class; subclass `parse_bench.cli.BenchCLI` and add
attributes for extra command groups:

```python
from parse_bench.cli import BenchCLI

class MyBenchCLI(BenchCLI):
    def __init__(self) -> None:
        super().__init__()
        self.triage = TriageCLI()
```
