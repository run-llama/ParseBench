"""Evaluator for EXTRACT product type using annotation-based evaluation."""

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from parse_bench.evaluation.evaluators.base import BaseEvaluator
from parse_bench.evaluation.metrics.extract.json_subset_match_metric import (
    JsonSubsetMatchMetric,
)
from parse_bench.evaluation.metrics.extract.list_unwrap import normalize_list_prediction
from parse_bench.evaluation.metrics.extract.rule_based_metric import (
    ExtractRuleBasedMetric,
)
from parse_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_extract_field_grounding_metrics,
)
from parse_bench.evaluation.metrics.field_grounding.rule_filters import (
    filter_extract_field_rules,
)
from parse_bench.evaluation.metrics.field_grounding.value_compare import (
    compare_attributed_value,
    expected_type_for_field_path,
)
from parse_bench.evaluation.stats import build_operational_stats
from parse_bench.schemas.evaluation import EvaluationResult, MetricValue
from parse_bench.schemas.extract_output import ExtractOutput
from parse_bench.schemas.pipeline_io import InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.extract_field_paths import parse_field_path
from parse_bench.test_cases.schema import ExtractTestCase, TestCase

logger = logging.getLogger(__name__)
_LAYOUT_FAMILY_RULE_TYPES = frozenset({"layout"})
# Rule types owned by the extract evaluator (distinct from layout-family).
# Currently limited to extract_field; reserved for future extract-native rule types.
_EXTRACT_NATIVE_RULE_TYPES = frozenset({"extract_field"})


class ExtractEvaluator(BaseEvaluator):
    """
    Evaluator for EXTRACT product type.

    Supports two evaluation modes:
    1. Annotation-based: Compare extracted_data with expected_output using JsonSubsetMatchMetric
    2. Rule-based: Execute test rules against extracted_data using ExtractRuleBasedMetric
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        cosine_similarity: bool = False,
        normalize_dates: bool = True,
        weighted: bool = True,
        enable_rule_based: bool = True,
    ):
        """
        Initialize the extract evaluator.

        :param case_sensitive: Whether string comparison should be case-sensitive
        :param cosine_similarity: Use embedding similarity for strings (requires OpenAI API key)
        :param normalize_dates: Normalize date strings before comparison
        :param enable_rule_based: Enable rule-based metric evaluation (default: True)
        """
        self._accuracy_metric = JsonSubsetMatchMetric(
            case_sensitive=case_sensitive,
            cosine_similarity=cosine_similarity,
            normalize_dates=normalize_dates,
            weighted=weighted,
        )
        self._enable_rule_based = enable_rule_based
        self._rule_metric = ExtractRuleBasedMetric()

    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case to evaluate against
        :return: True if this evaluator can handle this case
        """
        # Must be EXTRACT product type
        if inference_result.product_type != ProductType.EXTRACT:
            return False

        # Must have ExtractOutput
        if not isinstance(inference_result.output, ExtractOutput):
            return False

        # Must be ExtractTestCase
        if not isinstance(test_case, ExtractTestCase):
            return False

        # Need either expected_output (for annotation-based) or test_rules (for rule-based)
        has_expected_output = test_case.expected_output is not None
        has_test_rules = test_case.test_rules is not None and len(test_case.test_rules) > 0

        return has_expected_output or has_test_rules

    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate an EXTRACT inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with expected output or test rules
        :return: Evaluation result with accuracy metrics
        :raises ValueError: If neither expected_output nor test_rules are provided
        """
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing expected_output or test_rules, or invalid product type")

        if not isinstance(inference_result.output, ExtractOutput):
            raise ValueError("Inference result output is not ExtractOutput")

        if not isinstance(test_case, ExtractTestCase):
            raise ValueError("Test case must be ExtractTestCase for EXTRACT evaluation")

        raw_extracted_data = inference_result.output.extracted_data
        metrics: list[MetricValue] = []

        # Normalize per_table_row list projections back into the per-doc shape
        # used by extract_field rules. The adapter is a pure shape transform:
        # state is recorded on existing metric metadata, not as standalone
        # dashboard metrics.
        field_rules_for_unwrap = (
            test_case.get_extract_field_rules() if hasattr(test_case, "get_extract_field_rules") else []
        )
        scoring_field_rules = filter_extract_field_rules(field_rules_for_unwrap)
        normalization = normalize_list_prediction(
            raw_extracted_data,
            field_rules_for_unwrap,
            data_schema=test_case.data_schema,
        )
        extracted_data = normalization.extracted_data
        unwrap_skipped = [
            *normalization.skipped_field_paths,
            *normalization.alias_skipped_field_paths,
        ]

        # Annotation-based evaluation.
        #
        # Note: the accuracy metric is computed against the *unwrapped*
        # extracted_data vs the full expected_output. On per_table_row runs
        # this honestly drops accuracy because scalar fields the prediction
        # doesn't emit (e.g. ``client_id``) still appear in expected_output.
        # That drop is a correct signal, not noise — if scalar coverage
        # matters, run a per_doc pipeline instead. See list_unwrap.py.
        if test_case.expected_output is not None:
            expected_output = test_case.expected_output

            # Calculate overall accuracy using the metric
            accuracy_metric = self._accuracy_metric.compute(
                expected=expected_output,
                actual=extracted_data,
                data_schema=test_case.data_schema,
            )
            metrics.append(accuracy_metric)

            # Calculate field-level accuracy if both are dicts
            if isinstance(expected_output, dict) and isinstance(extracted_data, dict):
                for key in expected_output.keys():
                    expected_value = expected_output.get(key)
                    actual_value = extracted_data.get(key)
                    field_result = self._accuracy_metric.compute(expected=expected_value, actual=actual_value)
                    metrics.append(
                        MetricValue(
                            metric_name=f"field_accuracy_{key}",
                            value=field_result.value,
                            metadata={"field": key, **field_result.metadata},
                        )
                    )

        # Per-rule extract_field metrics (separate name scheme: field_accuracy[path])
        self._emit_extract_field_metrics(
            test_case,
            extracted_data,
            metrics,
            field_rules=scoring_field_rules,
            skip_field_paths=unwrap_skipped,
        )
        grounding_metrics = compute_extract_field_grounding_metrics(
            extracted_data=extracted_data,
            field_rules=scoring_field_rules,
            field_citations=getattr(inference_result.output, "field_citations", []),
            data_schema=test_case.data_schema,
            skip_field_paths=unwrap_skipped,
            list_unwrap_applied=normalization.applied,
            list_unwrap_mode=normalization.mode,
            alias_skipped_field_paths=normalization.alias_skipped_field_paths,
            normalized_top_level_keys=normalization.normalized_top_level_keys,
            list_unwrap_warnings=normalization.warnings,
        )
        metrics.extend(grounding_metrics)

        # Rule-based evaluation
        if self._enable_rule_based:
            if not test_case.test_rules:
                logger.debug(
                    f"Skipping rule-based metric: test_rules not provided "
                    f"(test_id: {test_case.test_id}, "
                    f"example_id: {inference_result.request.example_id})"
                )
            else:
                extract_rules = [
                    rule
                    for rule in test_case.test_rules
                    if isinstance(rule, dict) and rule.get("type") not in _LAYOUT_FAMILY_RULE_TYPES
                ]
                if not extract_rules:
                    logger.debug(
                        f"Skipping extract rule metric: only layout-family rules present "
                        f"(test_id: {test_case.test_id}, example_id: {inference_result.request.example_id})"
                    )
                    return_metric = None
                else:
                    # Execute rules
                    rule_result = self._rule_metric.compute(
                        expected=extract_rules,
                        actual=extracted_data,
                    )
                    metrics.append(rule_result)
                    return_metric = rule_result

                # Add per-type pass rates when we actually executed extract rules
                if return_metric and return_metric.metadata and "rule_results" in return_metric.metadata:
                    rule_results = return_metric.metadata["rule_results"]
                    rule_types: dict[str, list[dict[str, Any]]] = {}
                    for result in rule_results:
                        rule_type = result.get("type", "unknown")
                        if rule_type not in rule_types:
                            rule_types[rule_type] = []
                        rule_types[rule_type].append(result)

                    for rule_type, type_results in rule_types.items():
                        passed = sum(1 for r in type_results if r.get("passed", False))
                        total = len(type_results)
                        pass_rate = passed / total if total > 0 else 0.0
                        metrics.append(
                            MetricValue(
                                metric_name=f"rule_{rule_type}_pass_rate",
                                value=pass_rate,
                                metadata={
                                    "passed": passed,
                                    "total": total,
                                    "rule_type": rule_type,
                                },
                            )
                        )

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=True,
            metrics=metrics,
            error=None,
            job_id=inference_result.raw_output.get("job_id"),
            parse_job_id=inference_result.raw_output.get("parse_job_id"),
            stats=stats,
        )

    def _emit_extract_field_metrics(
        self,
        test_case: ExtractTestCase,
        extracted_data: Any,
        metrics: list[MetricValue],
        *,
        field_rules: list[Any],
        skip_field_paths: Iterable[str] = (),
    ) -> None:
        """Emit per-rule and doc-level metrics for `extract_field` rules.

        Rules whose ``field_path`` is in ``skip_field_paths`` are dropped
        entirely — no per-rule metric is emitted and they don't count toward
        ``extract_value_pass_rate`` totals. This is used by the
        list-unwrap path on per_table_row predictions to avoid penalizing
        pipelines for scalar fields they structurally cannot emit.
        """
        if not field_rules:
            return

        skip_set = set(skip_field_paths)
        eligible_rules = [rule for rule in field_rules if rule.field_path not in skip_set]
        matched_rule_ids = _match_extract_field_rules_index_tolerant(
            eligible_rules,
            extracted_data,
            data_schema=test_case.data_schema,
        )
        total = 0
        passed = 0
        for rule in field_rules:
            if rule.field_path in skip_set:
                continue
            try:
                parse_field_path(rule.field_path)
            except ValueError:
                continue
            match = id(rule) in matched_rule_ids
            metrics.append(
                MetricValue(
                    metric_name=f"field_accuracy[{rule.field_path}]",
                    value=float(match),
                    metadata={
                        "verified": rule.verified,
                        "field_path": rule.field_path,
                    },
                )
            )
            total += 1
            passed += int(match)

        if total > 0:
            metrics.append(
                MetricValue(
                    metric_name="extract_value_pass_rate",
                    value=passed / total,
                    metadata={"total": total, "passed": passed},
                )
            )


def _field_value_match(expected: Any, actual: Any) -> bool:
    """Simple per-rule value match.

    * None ≡ None.
    * Booleans and numbers compare by equality (with bool/number cross-typing allowed).
    * Strings compare case-insensitively with whitespace collapsed.
    * Other mismatched types return False.
    """
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return _normalize_str(expected) == _normalize_str(actual)
    # Cross-type fallback: best-effort string compare.
    return _normalize_str(str(expected)) == _normalize_str(str(actual))


def _extract_field_value_match(
    *,
    field_path: str,
    expected: Any,
    actual: Any,
    data_schema: dict[str, Any] | None,
) -> bool:
    expected_type = expected_type_for_field_path(data_schema, field_path, expected)
    comparison = compare_attributed_value(
        expected,
        actual,
        expected_type=expected_type,
        source_kind="structured_value_no_citation_text",
    )
    return comparison.passed


def _normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _extract_field_pattern(field_path: str) -> tuple[str | None, ...] | None:
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return None
    return tuple(None if isinstance(token, int) else token for token in tokens)


def _iter_values_for_extract_field_pattern(source: Any, pattern: Iterable[str | None]) -> list[Any]:
    cursors = [source]
    for token in pattern:
        next_cursors: list[Any] = []
        if token is None:
            for cursor in cursors:
                if isinstance(cursor, list):
                    next_cursors.extend(item for item in cursor if item is not None)
        else:
            for cursor in cursors:
                if isinstance(cursor, dict) and token in cursor:
                    next_cursors.append(cursor[token])
        cursors = next_cursors
        if not cursors:
            return []
    return [cursor for cursor in cursors if not isinstance(cursor, (dict, list))]


def _match_extract_field_rules_index_tolerant(
    field_rules: list[Any],
    extracted_data: Any,
    *,
    data_schema: dict[str, Any] | None = None,
) -> set[int]:
    rules_by_pattern: dict[tuple[str | None, ...], list[Any]] = defaultdict(list)
    for rule in field_rules:
        pattern = _extract_field_pattern(rule.field_path)
        if pattern is not None:
            rules_by_pattern[pattern].append(rule)

    matched_rule_ids: set[int] = set()
    for pattern, rules in rules_by_pattern.items():
        predictions = _iter_values_for_extract_field_pattern(extracted_data, pattern)
        used_predictions: set[int] = set()
        for rule in rules:
            if rule.expected_value is None and not predictions:
                matched_rule_ids.add(id(rule))
                continue
            for pred_index, prediction in enumerate(predictions):
                if pred_index in used_predictions:
                    continue
                if not _extract_field_value_match(
                    field_path=rule.field_path,
                    expected=rule.expected_value,
                    actual=prediction,
                    data_schema=data_schema,
                ):
                    continue
                matched_rule_ids.add(id(rule))
                used_predictions.add(pred_index)
                break
    return matched_rule_ids
