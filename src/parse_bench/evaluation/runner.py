"""Evaluation runner for computing metrics on inference results."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unicodedata
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
from parse_bench.evaluation.evaluators.parse import ParseEvaluator
from parse_bench.evaluation.evaluators.qa import QAEvaluator
from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.stats import build_operational_stats
from parse_bench.schemas.evaluation import EvaluationResult, EvaluationSummary
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.pipeline_io import InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases import load_test_cases
from parse_bench.test_cases.parse_rule_schemas import get_rule_type
from parse_bench.test_cases.schema import (
    LayoutDetectionTestCase,
    ParseTestCase,
    TestCase,
)

if TYPE_CHECKING:
    from parse_bench.schemas.parse_output import ParseOutput


# Module-level worker function for ProcessPoolExecutor (must be picklable)
def _evaluate_single_worker(
    inference_result_dict: dict[str, Any],
    test_case_dict: dict[str, Any],
    test_case_type: str,
    eval_mode: str | bool,
    evaluator_type: str | None,
    default_layout_ontology: str = "basic",
    enable_teds: bool = False,
    skip_rules: bool = False,
) -> dict[str, Any]:
    """
    Worker function for parallel evaluation using ProcessPoolExecutor.

    This function runs in a separate process, so it must:
    1. Accept only picklable arguments (dicts, not Pydantic models)
    2. Create evaluators locally (they can't be pickled)
    3. Return a dict (not Pydantic model)

    :param inference_result_dict: Serialized InferenceResult
    :param test_case_dict: Serialized TestCase
    :param test_case_type: Type of test case ("parse", "layout_detection", etc.)
    :param eval_mode: "multi_task", True (cross_eval), or False (normal)
    :param evaluator_type: Type of evaluator to use (None for multi_task)
    :param default_layout_ontology: Default ontology to use when test case omits ontology
    :param enable_teds: Enable TEDS metric computation in parse evaluation
    :param skip_rules: Skip rule-based metric computation in parse evaluation
    :return: Serialized EvaluationResult dict
    """
    # Import here to avoid circular imports and ensure fresh state in worker
    from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.evaluation.layout_adapters import (
        create_layout_adapter_for_result,
    )
    from parse_bench.schemas.evaluation import EvaluationResult
    from parse_bench.schemas.pipeline_io import InferenceResult
    from parse_bench.schemas.product import ProductType
    from parse_bench.test_cases.schema import (
        LayoutDetectionTestCase,
        ParseTestCase,
    )

    try:
        # Deserialize inputs
        inference_result = InferenceResult.model_validate(inference_result_dict)

        # Deserialize test case based on type
        test_case: LayoutDetectionTestCase | ParseTestCase
        if test_case_type == "layout_detection":
            test_case = LayoutDetectionTestCase.model_validate(test_case_dict)
        elif test_case_type == "parse":
            test_case = ParseTestCase.model_validate(test_case_dict)
        else:
            raise ValueError(f"Unknown test_case_type: {test_case_type}")

        # Create evaluator based on type
        evaluators: dict[
            str,
            ParseEvaluator | LayoutDetectionEvaluator,
        ] = {
            "parse": ParseEvaluator(enable_teds=enable_teds, enable_rule_based=not skip_rules),
            "layout_detection": LayoutDetectionEvaluator(default_ontology=default_layout_ontology),
        }

        if eval_mode == "multi_task":
            # Multi-task evaluation needs special handling
            # For now, return error - multi_task is complex and rarely used
            # The main parallel path is for normal evaluations
            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=False,
                error="multi_task evaluation not supported in parallel mode",
            ).model_dump()

        elif eval_mode is True:  # is_cross_eval
            # Cross-evaluation: extract layout from PARSE result
            assert isinstance(test_case, LayoutDetectionTestCase)

            adapter = create_layout_adapter_for_result(inference_result)
            layout_output = adapter.to_layout_output(
                inference_result,
                page_filter=test_case.page_index + 1,
            )
            if not layout_output.predictions:
                return EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type="layout_detection",
                    success=False,
                    error=f"No layout data for page {test_case.page_index}",
                ).model_dump()

            # Build a synthetic LAYOUT_DETECTION result from adapted output.
            layout_inference_result = InferenceResult(
                request=inference_result.request,
                pipeline_name=inference_result.pipeline_name,
                product_type=ProductType.LAYOUT_DETECTION,
                raw_output=inference_result.raw_output,
                output=layout_output,
                started_at=inference_result.started_at,
                completed_at=inference_result.completed_at,
                latency_in_ms=inference_result.latency_in_ms,
            )
            layout_evaluator = evaluators["layout_detection"]
            result = layout_evaluator.evaluate(layout_inference_result, test_case)
            return result.model_dump()
        else:
            # Normal evaluation
            if evaluator_type is None or evaluator_type not in evaluators:
                return EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=inference_result.product_type.value,
                    success=False,
                    error=f"No evaluator for type: {evaluator_type}",
                ).model_dump()
            evaluator = evaluators[evaluator_type]
            result = evaluator.evaluate(inference_result, test_case)
            return result.model_dump()

    except Exception as e:
        # Return error result
        return {
            "test_id": test_case_dict.get("test_id", "unknown"),
            "example_id": inference_result_dict.get("request", {}).get("example_id", "unknown"),
            "pipeline_name": inference_result_dict.get("pipeline_name", "unknown"),
            "product_type": inference_result_dict.get("product_type", "unknown"),
            "success": False,
            "error": f"Worker error: {str(e)}",
            "metrics": [],
            "stats": [],
        }


def _scale_layout_output_coordinates(
    layout_output: LayoutOutput,
    target_width: int,
    target_height: int,
) -> LayoutOutput:
    """
    Scale layout output coordinates from source space to target space.

    :param layout_output: Layout output with predictions in source coordinate space
    :param target_width: Target image width (ground truth dimensions)
    :param target_height: Target image height (ground truth dimensions)
    :return: New LayoutOutput with scaled coordinates
    """
    if layout_output.image_width == 0 or layout_output.image_height == 0:
        return layout_output

    # Calculate scale factors
    x_scale = target_width / layout_output.image_width
    y_scale = target_height / layout_output.image_height

    # If no scaling needed, return as-is
    if abs(x_scale - 1.0) < 0.001 and abs(y_scale - 1.0) < 0.001:
        return layout_output

    def scale_bbox(bbox: list[float]) -> list[float]:
        """Scale bbox [x1, y1, x2, y2] to target space."""
        return [
            bbox[0] * x_scale,
            bbox[1] * y_scale,
            bbox[2] * x_scale,
            bbox[3] * y_scale,
        ]

    # Scale raw predictions
    scaled_predictions = []
    for pred in layout_output.predictions:
        scaled_pred = pred.model_copy(update={"bbox": scale_bbox(pred.bbox)})
        scaled_predictions.append(scaled_pred)

    return LayoutOutput(
        task_type=layout_output.task_type,
        example_id=layout_output.example_id,
        pipeline_name=layout_output.pipeline_name,
        model=layout_output.model,
        image_width=target_width,
        image_height=target_height,
        predictions=scaled_predictions,
    )


class EvaluationRunner:
    """
    Runs evaluation on saved inference results.

    Loads inference results from output directory, matches them with test cases,
    and computes metrics using product-specific evaluators.
    """

    def __init__(
        self,
        output_dir: Path,
        test_cases_dir: Path | None = None,
        multi_task: bool = True,
        enable_teds: bool = False,
        skip_rules: bool = False,
        layout_ontology: str = "basic",
    ):
        """
        Initialize the evaluation runner.

        :param output_dir: Directory containing inference results
        :param test_cases_dir: Optional directory containing test cases (if different from data)
        :param multi_task: Enable multi-task evaluation for mixed rule types
        :param enable_teds: Enable TEDS metric computation in parse evaluation
        :param skip_rules: Skip rule-based metric computation in parse evaluation
        :param layout_ontology: Default layout ontology when test case does not specify one
        """
        self.output_dir = Path(output_dir)
        self.test_cases_dir = Path(test_cases_dir) if test_cases_dir else None
        self.multi_task = multi_task
        self.enable_teds = enable_teds
        self.skip_rules = skip_rules
        self.layout_ontology = layout_ontology

        # Register default evaluators
        self._evaluators: dict[str, Any] = {}
        # Register ParseEvaluator for PARSE product type
        self.register_evaluator(
            "parse",
            ParseEvaluator(
                enable_teds=enable_teds,
                enable_rule_based=not skip_rules,
            ),
        )
        # Register QAEvaluator for PARSE product type with QA test cases
        self.register_evaluator("qa", QAEvaluator())
        # Register LayoutDetectionEvaluator for LAYOUT_DETECTION product type
        self.register_evaluator(
            "layout_detection",
            LayoutDetectionEvaluator(default_ontology=self.layout_ontology),
        )

    def register_evaluator(self, product_type: str, evaluator: Any) -> None:
        """
        Register a product-specific evaluator.

        :param product_type: Product type (e.g., 'extract', 'parse')
        :param evaluator: Evaluator instance implementing BaseEvaluator
        """
        self._evaluators[product_type] = evaluator

    def _load_inference_result(self, result_path: Path) -> InferenceResult | None:
        """
        Load an inference result from a JSON file.

        :param result_path: Path to the result JSON file
        :return: InferenceResult or None if loading fails
        """
        try:
            with open(result_path) as f:
                data = json.load(f)
            return InferenceResult.model_validate(data)
        except Exception:
            return None

    def _find_result_files(self, output_dir: Path) -> list[Path]:
        """
        Find all result JSON files in the output directory.

        :param output_dir: Directory to search
        :return: List of paths to result JSON files
        """
        result_files = []
        # Look for .result.json files (normalized results)
        for result_file in output_dir.rglob("*.result.json"):
            result_files.append(result_file)
        return sorted(result_files)

    def _evaluate_single(
        self,
        inference_result: InferenceResult,
        test_case: TestCase,
        evaluator: Any,
        eval_mode: str | bool,
    ) -> EvaluationResult:
        """
        Evaluate a single test case (thread-safe helper for parallel execution).

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with expected values
        :param evaluator: The evaluator to use (None for multi_task mode)
        :param eval_mode: "multi_task", True (cross_eval), or False (normal)
        :return: EvaluationResult
        """
        try:
            if eval_mode == "multi_task":
                # Multi-task evaluation: split rules and run both evaluators
                assert isinstance(test_case, (LayoutDetectionTestCase, ParseTestCase))
                return self._evaluate_multi_task(inference_result, test_case)
            elif eval_mode is True:  # is_cross_eval
                # Cross-evaluation: extract layout from PARSE result and evaluate
                assert isinstance(test_case, LayoutDetectionTestCase)

                adapter = create_layout_adapter_for_result(inference_result)
                layout_output = adapter.to_layout_output(
                    inference_result,
                    page_filter=test_case.page_index + 1,
                )
                if not layout_output.predictions:
                    return EvaluationResult(
                        test_id=test_case.test_id,
                        example_id=inference_result.request.example_id,
                        pipeline_name=inference_result.pipeline_name,
                        product_type="layout_detection",
                        success=False,
                        error=f"No layout data for page {test_case.page_index}",
                    )

                # Create a synthetic InferenceResult with layout output
                layout_inference_result = InferenceResult(
                    request=inference_result.request,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=ProductType.LAYOUT_DETECTION,
                    raw_output=inference_result.raw_output,
                    output=layout_output,
                    started_at=inference_result.started_at,
                    completed_at=inference_result.completed_at,
                    latency_in_ms=inference_result.latency_in_ms,
                )
                return evaluator.evaluate(layout_inference_result, test_case)  # type: ignore[no-any-return]
            else:
                return evaluator.evaluate(inference_result, test_case)  # type: ignore[no-any-return]
        except Exception as e:
            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=False,
                error=f"Evaluation error: {str(e)}",
            )

    def _match_result_with_test_case(
        self,
        inference_result: InferenceResult,
        test_cases: dict[str, TestCase],
    ) -> TestCase | None:
        """
        Match an inference result with a test case by example_id/test_id.

        :param inference_result: The inference result
        :param test_cases: Dictionary mapping test_id to TestCase
        :return: Matching TestCase or None
        """
        example_id = inference_result.request.example_id
        # Try direct match first
        if example_id in test_cases:
            return test_cases[example_id]
        # Fall back to NFC-normalized comparison so that filenames whose
        # accented characters were stored as NFD on one side and NFC on
        # the other (common after a macOS round-trip) still match.
        example_id_nfc = unicodedata.normalize("NFC", example_id)
        example_id_nfc_stem = example_id_nfc.rsplit(".", 1)[0]
        for test_id, test_case in test_cases.items():
            test_id_nfc = unicodedata.normalize("NFC", test_id)
            if test_id_nfc == example_id_nfc or test_id_nfc == example_id_nfc_stem:
                return test_case
        return None

    def _match_result_with_test_cases_multi(
        self,
        inference_result: InferenceResult,
        test_cases: dict[str, TestCase],
    ) -> list[TestCase]:
        """
        Match an inference result with multiple test cases by example_id prefix.

        Used for cross-evaluation where one PARSE result can match multiple
        LAYOUT_DETECTION test cases (one per page).

        :param inference_result: The inference result
        :param test_cases: Dictionary mapping test_id to TestCase
        :return: List of matching TestCases
        """
        example_id = inference_result.request.example_id
        matches = []

        # Try direct match first
        if example_id in test_cases:
            matches.append(test_cases[example_id])
            return matches

        # NFC-normalized fallback for accented filenames whose Unicode form
        # differs between sides (e.g. NFD result vs NFC test on disk).
        example_id_nfc = unicodedata.normalize("NFC", example_id)
        if example_id_nfc != example_id:
            for test_id, test_case in test_cases.items():
                if unicodedata.normalize("NFC", test_id) == example_id_nfc:
                    matches.append(test_case)
                    return matches

        # For multi-page: match test_ids that start with example_id + "/"
        # e.g., example_id="pdfs/uber" matches "pdfs/uber/page_0", "pdfs/uber/page_1"
        prefix_nfc = example_id_nfc + "/"
        for test_id, test_case in test_cases.items():
            if unicodedata.normalize("NFC", test_id).startswith(prefix_nfc):
                matches.append(test_case)

        return matches

    def run_evaluation(
        self,
        product_type: str | None = None,
        pipeline_name: str | None = None,
        group: str | None = None,
        verbose: bool = False,
        use_rich: bool | None = None,
        max_workers: int | None = None,
    ) -> EvaluationSummary:
        """
        Run evaluation on all inference results in the output directory.

        :param product_type: Optional filter by product type
        :param pipeline_name: Optional filter by pipeline name
        :param group: Optional filter by group name
        :param verbose: Show detailed information about skipped results
        :param use_rich: Whether to use Rich for progress indication (default: auto-detect)
        :param max_workers: Number of worker threads for parallel evaluation (default: CPU count)
        :return: Evaluation summary with aggregated metrics
        """
        # Auto-detect Rich usage if not specified
        if use_rich is None:
            use_rich = sys.stdout.isatty() and not verbose
        console = Console() if use_rich else None
        # Load test cases if test_cases_dir is provided
        test_cases_dict: dict[str, TestCase] = {}
        if self.test_cases_dir:
            test_cases = load_test_cases(
                root_dir=self.test_cases_dir,
                require_test_json=False,
                product_type=product_type,
            )
            # Filter by group if specified
            if group:
                original_count = len(test_cases)
                test_cases = [tc for tc in test_cases if tc.group == group]
                if verbose:
                    print(
                        f"📋 Filtered to {len(test_cases)} test cases in group '{group}' (from {original_count} total)"
                    )
            test_cases_dict = {tc.test_id: tc for tc in test_cases}
            if verbose:
                print(f"📋 Loaded {len(test_cases_dict)} test cases")
                if test_cases_dict:
                    sample_ids = list(test_cases_dict.keys())[:3]
                    print(f"   Sample test_ids: {sample_ids}")

        # Find all result files
        result_files = self._find_result_files(self.output_dir)
        if verbose:
            print(f"📁 Found {len(result_files)} result files")

        # Filter by group if specified
        # Result files are saved as: output_dir/group/test_id.result.json
        # So we can filter by checking the parent directory name
        # text_content and text_formatting share inference results in text/
        _INFERENCE_DIR = {"text_content": "text", "text_formatting": "text"}
        if group:
            original_file_count = len(result_files)
            match_dir = _INFERENCE_DIR.get(group, group)
            result_files = [f for f in result_files if f.parent.name == match_dir]
            if verbose:
                print(f"   Filtered to {len(result_files)} files in group '{group}' (from {original_file_count} total)")

        # Filter by pipeline if specified
        if pipeline_name:
            # Pipeline name is typically in the parent directory path
            result_files = [f for f in result_files if pipeline_name in str(f.parent)]
            if verbose:
                print(f"   Filtered to {len(result_files)} files for pipeline '{pipeline_name}'")

        # Load and evaluate each result
        evaluation_results: list[EvaluationResult] = []
        successful = 0
        failed = 0
        skipped = 0

        # Separate QA and non-QA evaluations
        qa_evaluation_tasks: list[tuple[InferenceResult, ParseTestCase, QAEvaluator]] = []
        # (inference_result, test_case, evaluator, eval_mode)
        # eval_mode: True = cross_eval, False = normal, "multi_task" = multi-task eval
        non_qa_evaluations: list[tuple[InferenceResult, TestCase, Any, bool | str]] = []

        # First pass: collect all evaluations and separate QA from non-QA
        for result_file in result_files:
            inference_result = self._load_inference_result(result_file)
            if not inference_result:
                skipped += 1
                if verbose:
                    print(f"⚠️  Skipped {result_file.name}: Failed to load inference result")
                continue

            # Filter by product type if specified
            # Allow cross-evaluation: PARSE results can be evaluated against LAYOUT_DETECTION tests
            is_cross_eval_allowed = (
                product_type == "layout_detection" and inference_result.product_type == ProductType.PARSE
            )
            if product_type and inference_result.product_type.value != product_type:
                if not is_cross_eval_allowed:
                    skipped += 1
                    if verbose:
                        print(
                            f"⚠️  Skipped {result_file.name}: Product type mismatch "
                            f"({inference_result.product_type.value} != {product_type})"
                        )
                    continue

            # Check for cross-evaluation: PARSE result against LAYOUT_DETECTION tests
            is_cross_eval_candidate = is_cross_eval_allowed and inference_result.product_type == ProductType.PARSE

            if is_cross_eval_candidate:
                # Cross-evaluation: match multiple layout test cases (one per page)
                matched_test_cases = self._match_result_with_test_cases_multi(inference_result, test_cases_dict)
                if not matched_test_cases:
                    skipped += 1
                    if verbose:
                        print(
                            f"⚠️  Skipped {result_file.name}: No matching layout test cases found "
                            f"(example_id: {inference_result.request.example_id})"
                        )
                    continue

                evaluator = self._evaluators.get("layout_detection")
                if not evaluator:
                    skipped += 1
                    if verbose:
                        print(f"⚠️  Skipped {result_file.name}: No layout_detection evaluator for cross-evaluation")
                    continue

                # Add one evaluation task per matched test case (per page)
                for test_case in matched_test_cases:
                    non_qa_evaluations.append((inference_result, test_case, evaluator, True))  # True = is_cross_eval
                continue

            # Regular matching: single test case
            test_case = self._match_result_with_test_case(inference_result, test_cases_dict)  # type: ignore[assignment]
            if not test_case:
                skipped += 1
                if verbose:
                    print(
                        f"⚠️  Skipped {result_file.name}: No matching test case found "
                        f"(example_id: {inference_result.request.example_id})"
                    )
                continue

            # Get appropriate evaluator
            # Expand qa_configs (plural) into per-question QA evaluation tasks
            has_qa_configs = isinstance(test_case, ParseTestCase) and test_case.qa_configs
            if has_qa_configs:
                evaluator = self._evaluators.get("qa")
                if evaluator:
                    assert isinstance(test_case, ParseTestCase)
                    for i, qc in enumerate(test_case.qa_configs, 1):  # type: ignore[arg-type]
                        per_q_tc = test_case.model_copy(
                            update={
                                "test_id": f"{test_case.test_id}#q{i}",
                                "qa_config": qc,
                                "qa_configs": None,
                            }
                        )
                        if evaluator.can_evaluate(inference_result, per_q_tc):
                            qa_evaluation_tasks.append((inference_result, per_q_tc, evaluator))
                continue

            is_qa_test = isinstance(test_case, ParseTestCase) and test_case.qa_config is not None

            if is_qa_test:
                evaluator = self._evaluators.get("qa")
                if not evaluator:
                    skipped += 1
                    if verbose:
                        print(f"⚠️  Skipped {result_file.name}: No QA evaluator registered")
                    continue
                if not evaluator.can_evaluate(inference_result, test_case):
                    skipped += 1
                    if verbose:
                        print(
                            f"⚠️  Skipped {result_file.name}: QA evaluator cannot handle this case "
                            f"(test_id: {test_case.test_id})"
                        )
                    continue
                qa_evaluation_tasks.append((inference_result, test_case, evaluator))  # type: ignore[arg-type]
            else:
                # Check for multi-task evaluation: test case has mixed rule types
                # Multi-task works with PARSE results, or LAYOUT_DETECTION results
                # that contain LlamaParse data (pages with markdown)
                is_llamaparse_output = self._is_llamaparse_output(inference_result)
                has_mixed = self._has_mixed_rules(test_case)
                is_multi_task_eval = (
                    self.multi_task
                    and (
                        inference_result.product_type == ProductType.PARSE
                        or (inference_result.product_type == ProductType.LAYOUT_DETECTION and is_llamaparse_output)
                    )
                    and has_mixed
                )

                if is_multi_task_eval:
                    # Multi-task evaluation: split rules and run both evaluators
                    # Use None evaluator as marker; actual evaluators called
                    # in _evaluate_multi_task
                    non_qa_evaluations.append((inference_result, test_case, None, "multi_task"))
                    continue

                # Check for cross-evaluation: PARSE result against LayoutDetectionTestCase
                is_cross_eval = (
                    isinstance(test_case, LayoutDetectionTestCase)
                    and inference_result.product_type == ProductType.PARSE
                )

                if is_cross_eval:
                    # Cross-evaluation: extract layout from PARSE result
                    evaluator = self._evaluators.get("layout_detection")
                    if not evaluator:
                        skipped += 1
                        if verbose:
                            print(f"⚠️  Skipped {result_file.name}: No layout_detection evaluator for cross-evaluation")
                        continue
                    # Mark this as cross-evaluation for special handling later
                    non_qa_evaluations.append((inference_result, test_case, evaluator, True))  # True = is_cross_eval
                else:
                    result_product_type = inference_result.product_type.value
                    evaluator = self._evaluators.get(result_product_type)
                    if not evaluator:
                        skipped += 1
                        if verbose:
                            print(
                                f"⚠️  Skipped {result_file.name}: No evaluator registered for "
                                f"product type: {result_product_type}"
                            )
                        continue
                    if not evaluator.can_evaluate(inference_result, test_case):
                        skipped += 1
                        if verbose:
                            reason = "Evaluator cannot evaluate this case"
                            print(
                                f"⚠️  Skipped {result_file.name}: {reason} "
                                f"(test_id: {test_case.test_id}, "
                                f"example_id: {inference_result.request.example_id})"
                            )
                        continue
                    non_qa_evaluations.append((inference_result, test_case, evaluator, False))  # False = not cross-eval

        # Count QA test cases for progress indication
        qa_test_cases = len(qa_evaluation_tasks)
        total_to_evaluate = len(non_qa_evaluations) + qa_test_cases

        # Plain-text progress logging for CI/non-TTY environments.
        # The log_progress closure is called unconditionally at each evaluation
        # site; it no-ops when Rich progress bars are active.
        eval_done = 0

        if not use_rich:
            print("=== Evaluation Plan ===")
            print(f"  Result files found: {len(result_files)} | Skipped: {skipped}")
            print(
                f"  Documents to evaluate: {total_to_evaluate} ({len(non_qa_evaluations)} standard, {qa_test_cases} QA)"
            )
            print("=======================")

        def log_progress(test_id: str, status: str = "") -> None:
            """Log evaluation progress as plain text (no-op when Rich is active)."""
            nonlocal eval_done
            eval_done += 1
            if use_rich:
                return
            status_suffix = f": {status}" if status else ""
            print(
                f"  [{eval_done}/{total_to_evaluate}] {test_id}{status_suffix}",
                flush=True,
            )

        # Create progress bars if using Rich
        progress: Progress | None = None
        qa_task_id: int | None = None
        total_task_id: int | None = None

        if use_rich and console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="bright_blue",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                expand=True,
            )
            if qa_test_cases > 0:
                qa_task_id = progress.add_task(
                    "[yellow]QA Evaluation (LLM calls)[/yellow]",
                    total=qa_test_cases,
                )
            total_task_id = progress.add_task(
                "[bold green]Total Evaluation[/bold green]",
                total=len(result_files),
            )
            progress.start()

        try:
            # Separate multi_task evaluations from parallelizable evaluations
            # Multi_task requires instance methods (_evaluate_multi_task, _evaluators)
            # that cannot be pickled and sent to worker processes
            multi_task_evaluations = [
                (inf, tc, ev, mode) for inf, tc, ev, mode in non_qa_evaluations if mode == "multi_task"
            ]
            parallelizable_evaluations = [
                (inf, tc, ev, mode) for inf, tc, ev, mode in non_qa_evaluations if mode != "multi_task"
            ]

            # Process multi_task evaluations in main process (cannot be parallelized)
            for inf_result, tc, _, _ in multi_task_evaluations:
                eval_result = self._evaluate_single(inf_result, tc, None, "multi_task")
                evaluation_results.append(eval_result)
                if eval_result.success:
                    successful += 1
                    log_progress(tc.test_id, "OK")
                elif eval_result.error and "No layout data" in eval_result.error:
                    skipped += 1
                    log_progress(tc.test_id, "skipped (no layout data)")
                else:
                    failed += 1
                    log_progress(tc.test_id, "FAILED")
                if progress and total_task_id is not None:
                    progress.update(total_task_id, advance=1)  # type: ignore[arg-type]

            # Process non-QA evaluations in parallel using ProcessPoolExecutor
            # Default to CPU count, but cap at 8 for CI environments
            num_workers = max_workers or min(os.cpu_count() or 4, 8)

            if parallelizable_evaluations:
                # Prepare tasks for ProcessPoolExecutor
                # We need to serialize data since processes don't share memory
                worker_tasks: list[
                    tuple[
                        dict,
                        dict,
                        str,
                        str | bool,
                        str | None,
                        str,
                        bool,
                        bool,
                    ]
                ] = []
                for inf_result, tc, _eval_obj, mode in parallelizable_evaluations:
                    # Serialize inference result and test case to dicts
                    inf_dict = inf_result.model_dump()
                    tc_dict = tc.model_dump()

                    # Determine test case type
                    if isinstance(tc, LayoutDetectionTestCase):
                        tc_type = "layout_detection"
                    elif isinstance(tc, ParseTestCase):
                        tc_type = "parse"
                    else:
                        raise ValueError(f"Unknown test case type: {type(tc).__name__}")

                    # Determine evaluator type
                    if mode is True:  # cross_eval
                        eval_type = "layout_detection"
                    else:
                        eval_type = inf_result.product_type.value

                    worker_tasks.append(
                        (
                            inf_dict,
                            tc_dict,
                            tc_type,
                            mode,
                            eval_type,
                            self.layout_ontology,
                            self.enable_teds,
                            self.skip_rules,
                        )
                    )

                # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    futures = [executor.submit(_evaluate_single_worker, *task) for task in worker_tasks]

                    # Per-worker timeout: 8 minutes per evaluation
                    worker_timeout = 8 * 60

                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            result_dict = future.result(timeout=worker_timeout)
                            eval_result = EvaluationResult.model_validate(result_dict)
                            evaluation_results.append(eval_result)

                            if eval_result.success:
                                successful += 1
                                log_progress(eval_result.test_id, "OK")
                            elif eval_result.error and "No layout data" in eval_result.error:
                                skipped += 1
                                log_progress(eval_result.test_id, "skipped (no layout data)")
                            else:
                                failed += 1
                                log_progress(eval_result.test_id, "FAILED")
                        except FuturesTimeoutError:
                            failed += 1
                            log_progress("unknown", f"FAILED (worker timed out after {worker_timeout}s)")
                        except Exception:
                            # Worker process error
                            failed += 1
                            log_progress("unknown", "FAILED (worker error)")

                        # Update progress (can't do this in worker due to separate processes)
                        completed += 1
                        if progress and total_task_id is not None:
                            progress.update(total_task_id, completed=completed)  # type: ignore[arg-type]

            # Process QA evaluations concurrently
            if qa_evaluation_tasks:
                qa_results, qa_success, qa_failed = asyncio.run(
                    self._run_qa_evaluations_async(
                        qa_evaluation_tasks,
                        progress,
                        qa_task_id,
                        total_task_id,
                        log_progress,
                    )
                )
                evaluation_results.extend(qa_results)
                successful += qa_success
                failed += qa_failed
        finally:
            # Stop progress display
            if progress:
                progress.stop()

        # Stamp tags from test cases onto evaluation results
        for result in evaluation_results:
            tc = test_cases_dict.get(result.test_id)  # type: ignore[assignment]
            if tc is not None:
                result.tags = list(tc.tags)

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(evaluation_results)

        # Aggregate operational stats
        aggregate_stats = self._aggregate_stats(evaluation_results)

        # Compute confusion matrix for layout detection evaluations
        confusion_matrix = None
        if product_type == "layout_detection" and test_cases_dict:
            layout_evaluator = self._evaluators.get("layout_detection")
            if isinstance(layout_evaluator, LayoutDetectionEvaluator):
                # Collect inference results into dict for confusion matrix computation
                inference_results_dict: dict[str, InferenceResult] = {}
                for result_file in result_files:
                    inference_result = self._load_inference_result(result_file)
                    if inference_result:
                        inference_results_dict[inference_result.request.example_id] = inference_result

                # Compute confusion matrix
                try:
                    confusion_matrix = layout_evaluator.compute_confusion_matrix(
                        inference_results=inference_results_dict,
                        test_cases=test_cases_dict,
                        iou_threshold=0.5,
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute confusion matrix: {e}", file=sys.stderr)

        # Aggregate per-tag metrics
        tag_metrics = self._aggregate_tag_metrics(evaluation_results)

        return EvaluationSummary(
            total_examples=len(evaluation_results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            aggregate_metrics=aggregate_metrics,
            per_example_results=evaluation_results,
            confusion_matrix=confusion_matrix,
            tag_metrics=tag_metrics,
            completed_at=None,  # Will be set by caller
            aggregate_stats=aggregate_stats,
        )

    def _aggregate_metrics(self, evaluation_results: list[EvaluationResult]) -> dict[str, float]:
        """
        Aggregate metrics across all evaluation results.

        :param evaluation_results: List of individual evaluation results
        :return: Dictionary of aggregated metric values
        """
        if not evaluation_results:
            return {}

        # Collect all metric values by metric name
        metric_values: dict[str, list[float]] = {}
        # Also collect counts from metadata (for rules, etc.)
        metric_counts: dict[str, list[tuple[int, int]]] = {}  # (passed, total) pairs
        # Track scores where tables were predicted (for _predicted aggregates)
        # Applies to any metric with "tables_predicted" metadata (TEDS, GriTS, etc.)
        predicted_values: dict[str, list[float]] = {}
        metric_count_sums: dict[str, list[int]] = {}  # count totals

        for result in evaluation_results:
            if not result.success:
                continue
            for metric in result.metrics:
                if metric.metric_name not in metric_values:
                    metric_values[metric.metric_name] = []
                metric_values[metric.metric_name].append(metric.value)

                # Track scores where tables were predicted and expected
                if metric.metadata and metric.metadata.get("tables_predicted", False):
                    key = f"{metric.metric_name}_predicted"
                    if key not in predicted_values:
                        predicted_values[key] = []
                    predicted_values[key].append(metric.value)

                # Extract counts from metadata if available
                if metric.metadata and "passed" in metric.metadata and "total" in metric.metadata:
                    if metric.metric_name not in metric_counts:
                        metric_counts[metric.metric_name] = []
                    passed = metric.metadata.get("passed", 0)
                    total = metric.metadata.get("total", 0)
                    if isinstance(passed, int) and isinstance(total, int):
                        metric_counts[metric.metric_name].append((passed, total))

                if metric.metadata and "count" in metric.metadata:
                    count = metric.metadata.get("count")
                    if isinstance(count, int):
                        if metric.metric_name not in metric_count_sums:
                            metric_count_sums[metric.metric_name] = []
                        metric_count_sums[metric.metric_name].append(count)

        # Compute averages
        aggregate: dict[str, float] = {}
        for metric_name, values in metric_values.items():
            if values:
                aggregate[f"avg_{metric_name}"] = sum(values) / len(values)
                aggregate[f"min_{metric_name}"] = min(values)
                aggregate[f"max_{metric_name}"] = max(values)

        # Aggregate counts for metrics that have them
        for metric_name, count_pairs in metric_counts.items():
            total_passed = sum(passed for passed, _ in count_pairs)
            total_rules = sum(total for _, total in count_pairs)
            if total_rules > 0:
                aggregate[f"total_{metric_name}_passed"] = float(total_passed)
                aggregate[f"total_{metric_name}_evaluated"] = float(total_rules)

        # Add _predicted aggregates (only docs where tables were predicted)
        for key, values in predicted_values.items():
            if values:
                aggregate[f"avg_{key}"] = sum(values) / len(values)
                aggregate[f"min_{key}"] = min(values)
                aggregate[f"max_{key}"] = max(values)

        # Aggregate explicit count totals (e.g., unmatched elements)
        for metric_name, counts in metric_count_sums.items():
            aggregate[f"total_{metric_name}"] = float(sum(counts))

        return aggregate

    def _aggregate_tag_metrics(self, evaluation_results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
        """
        Aggregate metrics grouped by tag.

        Groups results by tag, then calls _aggregate_metrics for each group.
        Adds example_count to each tag's metrics.

        :param evaluation_results: List of individual evaluation results
        :return: Dict keyed by tag name, each value containing aggregated metrics
        """
        from collections import defaultdict

        tag_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
        for result in evaluation_results:
            for tag in result.tags:
                tag_groups[tag].append(result)

        tag_metrics: dict[str, dict[str, float]] = {}
        for tag, results in sorted(tag_groups.items()):
            metrics = self._aggregate_metrics(results)
            metrics["example_count"] = float(len(results))
            tag_metrics[tag] = metrics

        return tag_metrics

    def _aggregate_stats(self, evaluation_results: list[EvaluationResult]) -> dict[str, dict[str, Any]]:
        """
        Aggregate operational stats across all evaluation results.

        Collects values by stat name from successful results and computes
        total, avg, min, max, p50, p95, p99, count for each.

        :param evaluation_results: List of individual evaluation results
        :return: Dict keyed by stat name, each value containing aggregates + unit
        """
        # Collect values and units by stat name
        stat_values: dict[str, list[float]] = {}
        stat_units: dict[str, str] = {}
        for r in evaluation_results:
            if not r.success:
                continue
            for s in r.stats:
                stat_values.setdefault(s.name, []).append(s.value)
                stat_units[s.name] = s.unit

        aggregate: dict[str, dict[str, Any]] = {}
        for name, values in stat_values.items():
            values_sorted = sorted(values)
            n = len(values_sorted)

            def percentile(p: int, n: int = n, values_sorted: list[float] = values_sorted) -> float:
                idx = int(n * p / 100)
                return values_sorted[min(idx, n - 1)]

            aggregate[name] = {
                "total": sum(values),
                "avg": sum(values) / n,
                "min": min(values),
                "max": max(values),
                "p50": percentile(50),
                "p90": percentile(90),
                "p95": percentile(95),
                "p99": percentile(99),
                "count": n,
                "unit": stat_units[name],
            }

        return aggregate

    def _has_mixed_rules(self, test_case: TestCase) -> bool:
        """
        Check if test case has both layout and non-layout rules.

        :param test_case: Test case to check
        :return: True if test case has mixed rule types
        """
        # Get test_rules from the test case
        rules = []
        if isinstance(test_case, (ParseTestCase, LayoutDetectionTestCase)):
            rules = list(test_case.test_rules or [])

        if not rules:
            return False

        has_layout = any(get_rule_type(r) == "layout" for r in rules)
        has_non_layout = any(get_rule_type(r) is not None and get_rule_type(r) != "layout" for r in rules)

        return has_layout and has_non_layout

    def _is_llamaparse_output(self, inference_result: InferenceResult) -> bool:
        """
        Check if inference result exposes parse-capable normalized output.

        This is used to determine if multi-task evaluation can be performed
        even when product_type is LAYOUT_DETECTION.

        :param inference_result: Inference result to check
        :return: True if output is LlamaParse format with pages and markdown
        """
        from parse_bench.schemas.layout_detection_output import LayoutOutput
        from parse_bench.schemas.parse_output import ParseOutput

        if isinstance(inference_result.output, ParseOutput):
            if inference_result.output.layout_pages:
                return True
            return len(inference_result.output.pages) > 0

        # Layout detection outputs can still carry full document markdown
        # (e.g., normalized from LlamaParse layout runs).
        if isinstance(inference_result.output, LayoutOutput):
            if inference_result.output.markdown.strip():
                return True
        return False

    def _create_parse_output_from_raw(self, inference_result: InferenceResult) -> ParseOutput | None:
        """
        Create a ParseOutput from normalized inference output.

        Used in multi-task evaluation to create a synthetic PARSE output when
        the original product_type was LAYOUT_DETECTION and markdown is present.

        :param inference_result: Inference result
        :return: ParseOutput or None if conversion fails
        """
        from parse_bench.schemas.layout_detection_output import LayoutOutput
        from parse_bench.schemas.parse_output import PageIR, ParseOutput

        if isinstance(inference_result.output, ParseOutput):
            return inference_result.output

        # For layout runs that still provide markdown, synthesize minimal
        # ParseOutput so parse/order rules can be evaluated in multi-task mode.
        if isinstance(inference_result.output, LayoutOutput):
            markdown = inference_result.output.markdown
            if isinstance(markdown, str) and markdown.strip():
                return ParseOutput(
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    pages=[PageIR(page_index=0, markdown=markdown)],
                    markdown=markdown,
                )
        return None

    def _evaluate_multi_task(
        self,
        inference_result: InferenceResult,
        test_case: LayoutDetectionTestCase | ParseTestCase,
    ) -> EvaluationResult:
        """
        Evaluate mixed rule types by splitting rules and running appropriate evaluators.

        For test cases with mixed rules (table, order, layout, etc.):
        1. Split rules into layout vs non-layout
        2. Evaluate non-layout rules with ParseEvaluator
        3. Evaluate layout rules with LayoutDetectionEvaluator (cross-eval from PARSE)
        4. Merge metrics into single result

        :param inference_result: The inference result to evaluate
        :param test_case: Test case with mixed rule types
        :return: Combined evaluation result with metrics from both evaluators
        """
        from parse_bench.schemas.evaluation import MetricValue

        # Get all rules from the test case
        all_rules = test_case.test_rules or []

        # Split rules by type
        layout_rules = [r for r in all_rules if get_rule_type(r) == "layout"]
        parse_rules = [r for r in all_rules if get_rule_type(r) != "layout"]

        all_metrics: list[MetricValue] = []
        errors: list[str] = []

        # Evaluate parse rules (table, order, present, absent, etc.)
        if parse_rules:
            temp_parse_test_case = ParseTestCase(
                test_id=test_case.test_id,
                group=test_case.group,
                file_path=test_case.file_path,
                test_rules=parse_rules,  # type: ignore[arg-type]
                expected_markdown=None,
            )

            # Create a synthetic PARSE inference result if needed
            # This allows ParseEvaluator to work even when the original
            # product_type was LAYOUT_DETECTION (auto-detected from test cases)
            parse_inference_result = inference_result
            if inference_result.product_type != ProductType.PARSE:
                parse_output = self._create_parse_output_from_raw(inference_result)
                if parse_output:
                    parse_inference_result = InferenceResult(
                        request=inference_result.request,
                        pipeline_name=inference_result.pipeline_name,
                        product_type=ProductType.PARSE,
                        raw_output=inference_result.raw_output,
                        output=parse_output,
                        started_at=inference_result.started_at,
                        completed_at=inference_result.completed_at,
                        latency_in_ms=inference_result.latency_in_ms,
                    )

            parse_evaluator = self._evaluators.get("parse")
            can_eval = (
                parse_evaluator.can_evaluate(parse_inference_result, temp_parse_test_case) if parse_evaluator else False
            )
            if parse_evaluator and can_eval:
                try:
                    parse_result = parse_evaluator.evaluate(parse_inference_result, temp_parse_test_case)
                    all_metrics.extend(parse_result.metrics)
                except Exception as e:
                    errors.append(f"Parse evaluation error: {e}")

        # Evaluate layout rules (cross-evaluation from PARSE output)
        if layout_rules:
            metadata = test_case.metadata if isinstance(test_case, LayoutDetectionTestCase) else None
            # For multi-page documents, layout rules may span multiple pages
            # Create test case with all layout rules (page_index=0 as default)
            temp_layout_test_case = LayoutDetectionTestCase(
                test_id=test_case.test_id,
                group=test_case.group,
                file_path=test_case.file_path,
                test_rules=layout_rules,
                source_dataset=metadata.get("source_dataset") if metadata else None,
                # Not used for multi-page; GT filtering is done by
                # get_layout_annotations.
                page_index=0,
                metadata=metadata,
            )

            adapter = create_layout_adapter_for_result(inference_result)
            layout_output = adapter.to_layout_output(inference_result)

            if layout_output.predictions:
                layout_evaluator = self._evaluators.get("layout_detection")
                if layout_evaluator:
                    try:
                        # Create synthetic inference result with layout output
                        layout_inference_result = InferenceResult(
                            request=inference_result.request,
                            pipeline_name=inference_result.pipeline_name,
                            product_type=ProductType.LAYOUT_DETECTION,
                            raw_output=inference_result.raw_output,
                            output=layout_output,
                            started_at=inference_result.started_at,
                            completed_at=inference_result.completed_at,
                            latency_in_ms=inference_result.latency_in_ms,
                        )
                        layout_result = layout_evaluator.evaluate(layout_inference_result, temp_layout_test_case)
                        all_metrics.extend(layout_result.metrics)
                    except Exception as e:
                        errors.append(f"Layout evaluation error: {e}")
            else:
                errors.append("Could not extract layout from PARSE output")

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=len(errors) == 0,
            metrics=all_metrics,
            error="; ".join(errors) if errors else None,
            stats=stats,
        )

    async def _evaluate_qa_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        evaluator: QAEvaluator,
        inference_result: InferenceResult,
        test_case: ParseTestCase,
        progress: Progress | None,
        qa_task_id: int | None,
        total_task_id: int | None,
        log_progress: Callable[[str, str], None] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a QA test case with semaphore-based concurrency control.

        :param semaphore: Semaphore for concurrency control
        :param evaluator: QA evaluator instance
        :param inference_result: The inference result to evaluate
        :param test_case: The test case with qa_config
        :param progress: Rich progress bar (optional)
        :param qa_task_id: QA progress task ID (optional)
        :param total_task_id: Total progress task ID (optional)
        :param log_progress: Plain-text progress callback (optional)
        :return: Evaluation result
        """
        async with semaphore:
            # Update progress description
            if progress and qa_task_id is not None:
                progress.update(
                    qa_task_id,  # type: ignore[arg-type]
                    description=f"[yellow]QA Evaluation: {test_case.test_id}[/yellow]",
                )

            # Run evaluation in thread (LLM calls are synchronous)
            try:
                eval_result = await asyncio.to_thread(evaluator.evaluate, inference_result, test_case)
            except Exception as e:
                # Handle evaluation errors
                eval_result = EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=inference_result.product_type.value,
                    success=False,
                    error=f"Evaluation error: {str(e)}",
                )

            # Update progress after evaluation
            if log_progress:
                status = "OK" if eval_result.success else "FAILED"
                log_progress(test_case.test_id, f"QA {status}")
            if progress:
                if qa_task_id is not None:
                    progress.update(qa_task_id, advance=1)  # type: ignore[arg-type]
                if total_task_id is not None:
                    progress.update(total_task_id, advance=1)  # type: ignore[arg-type]

            return eval_result

    async def _run_qa_evaluations_async(
        self,
        qa_evaluation_tasks: list[tuple[InferenceResult, ParseTestCase, QAEvaluator]],
        progress: Progress | None,
        qa_task_id: int | None,
        total_task_id: int | None,
        log_progress: Callable[[str, str], None] | None = None,
    ) -> tuple[list[EvaluationResult], int, int]:
        """
        Run QA evaluations concurrently with semaphore-based concurrency control.

        :param qa_evaluation_tasks: List of (inference_result, test_case, evaluator) tuples
        :param progress: Rich progress bar (optional)
        :param qa_task_id: QA progress task ID (optional)
        :param total_task_id: Total progress task ID (optional)
        :param log_progress: Plain-text progress callback (optional)
        :return: Tuple of (results list, success count, failed count)
        """
        # Create semaphore for QA concurrency control (fixed at 20)
        max_concurrent_qa = 20
        semaphore = asyncio.Semaphore(max_concurrent_qa)

        # Create async tasks for QA evaluations
        qa_tasks = [
            self._evaluate_qa_with_semaphore(
                semaphore,
                evaluator,
                inference_result,
                test_case,
                progress,
                qa_task_id,
                total_task_id,
                log_progress,
            )
            for inference_result, test_case, evaluator in qa_evaluation_tasks
        ]

        # Run QA evaluations concurrently
        qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)

        # Process results
        results: list[EvaluationResult] = []
        success_count = 0
        failed_count = 0

        for result in qa_results:
            if isinstance(result, Exception):
                failed_count += 1
                # Create error result - we don't have test_case info here
                # This shouldn't happen, but handle it gracefully
                results.append(
                    EvaluationResult(
                        test_id="unknown",
                        example_id="unknown",
                        pipeline_name="unknown",
                        product_type="parse",
                        success=False,
                        error=f"Task execution error: {str(result)}",
                    )
                )
            else:
                results.append(result)  # type: ignore[arg-type]
                if result.success:  # type: ignore[union-attr]
                    success_count += 1
                else:
                    failed_count += 1

        return results, success_count, failed_count
