"""Rule-based metric for executing parse test rules.

Besides the markdown, some rules need side inputs: the structured parse
payload, the raw provider response (rotation checks), the staged source file
and the test case's own path (reference renders live next to it), and chart
rules share one parsed-table cache per document so a 200-rule chart page does
not re-parse its tables 200 times. :meth:`RuleBasedMetric.compute` accepts
these as keyword arguments and :meth:`RuleBasedMetric._prepare_rule` injects
them into every rule that declares the matching attribute. A harness that
scores extra rule types subclasses the metric and extends ``_prepare_rule``.
"""

import os
import signal
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, cast

from parse_bench.evaluation.metrics.base import Metric
from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule, RuleNotApplicable
from parse_bench.evaluation.metrics.parse.rules_chart import parse_chart_tables
from parse_bench.evaluation.metrics.parse.table_parsing import TableData
from parse_bench.evaluation.metrics.parse.test_rules import (
    MissingSpecificWordRule,
    RotateCheckRule,
    WordBagRule,
    create_test_rule,
)
from parse_bench.evaluation.metrics.parse.utils import normalize_text
from parse_bench.schemas.evaluation import MetricValue
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.test_cases.parse_rule_schemas import (
    ParseRuleBase,
    ParseRuleInput,
    get_rule_id,
    get_rule_layout_bindings,
    get_rule_layout_id,
    get_rule_layout_ids,
    get_rule_page,
    get_rule_type,
)

# Per-rule timeout in seconds. Rules that exceed this are marked as failed.
RULE_TIMEOUT_SECONDS = 120

# Fallback for ``_doc_rule_budget_seconds`` when BENCH_DOC_RULE_BUDGET_SECONDS is
# unset, blank or unparseable.
DOC_RULE_BUDGET_DEFAULT_SECONDS = 600.0


def _doc_rule_budget_seconds() -> float:
    """Cumulative wall-clock budget for ALL rules of a single document.

    The per-rule ``RULE_TIMEOUT_SECONDS`` alarm bounds one rule, but a document
    with hundreds of rules that each time out can still consume ``n_rules * 120s``
    (hours) and block the whole evaluation pool until the CI job hits its 6h cap.
    This aggregate budget is the backstop: once a document has spent this long on
    rules, the remaining rules are skipped (scored 0, same as a blank output) so
    the eval moves on. One or two pathological documents lose their rule scores
    instead of stalling the entire run.

    Override with ``BENCH_DOC_RULE_BUDGET_SECONDS`` (0 or negative disables the
    cap). The ``DOC_RULE_BUDGET_DEFAULT_SECONDS`` default bounds a single document
    to roughly ``budget + RULE_TIMEOUT_SECONDS`` in the worst case.
    """
    raw = os.environ.get("BENCH_DOC_RULE_BUDGET_SECONDS")
    if raw is None or raw.strip() == "":
        return DOC_RULE_BUDGET_DEFAULT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DOC_RULE_BUDGET_DEFAULT_SECONDS


class _RuleTimeoutError(Exception):
    """Raised when a single rule exceeds its time budget."""


# Rule types that consume the shared per-document table parse.
CHART_RULE_TYPES = frozenset({"chart_data_point", "chart_data_array_labels", "chart_data_array_data"})


@dataclass
class ChartTableCache:
    """One ``compute`` call's parsed chart tables.

    Pass an instance as ``chart_table_cache=`` to share the parse with a
    caller (a judge stage that re-reads the same tables, for example);
    otherwise the metric creates a private one per call. ``populated``
    distinguishes a cached table-free document from a cache that has not been
    attempted yet.
    """

    tables: list[TableData] = field(default_factory=list)
    populated: bool = False

    def tables_for(self, content: str) -> list[TableData]:
        """Parse ``content`` once; later calls reuse the same table objects."""
        if not self.populated:
            self.tables = parse_chart_tables(content)
            self.populated = True
        return self.tables


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _RuleTimeoutError()


# Sentinel for ``getattr`` probes: distinguishes "attribute declared as None"
# (inject) from "attribute not declared" (this rule does not take the input).
_ABSENT = object()


class RuleBasedMetric(Metric):
    """Metric for executing test rules against markdown content."""

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "rule_pass_rate"

    def _prepare_rule(self, rule: ParseTestRule, actual: str, kwargs: dict[str, Any]) -> None:
        """Hand a freshly created rule the side inputs it declares.

        Runs inside the per-rule timeout and error boundary, before
        ``rule.run``. Injection is attribute-driven so extension rule types get
        it for free: a rule that sets ``self.raw_output = None`` (or
        ``source_file_path`` / ``test_case_path``) in its constructor receives
        the matching ``compute`` keyword argument; a value the rule already
        carries is left alone. Chart rules receive the shared per-call table
        parse. Subclasses extend this to inject harness-specific inputs.
        """
        parse_output = kwargs.get("parse_output")
        if isinstance(parse_output, ParseOutput) and hasattr(rule, "parse_output"):
            rule.parse_output = parse_output

        raw_output = kwargs.get("raw_output")
        if isinstance(raw_output, dict) and getattr(rule, "raw_output", _ABSENT) is None:
            rule.raw_output = raw_output  # type: ignore[attr-defined]

        source_file_path = kwargs.get("source_file_path")
        if source_file_path and getattr(rule, "source_file_path", _ABSENT) is None:
            rule.source_file_path = str(source_file_path)  # type: ignore[attr-defined]

        test_case_path = kwargs.get("test_case_file_path")
        if test_case_path and getattr(rule, "test_case_path", _ABSENT) is None:
            rule.test_case_path = str(test_case_path)  # type: ignore[attr-defined]

        chart_table_cache = kwargs.get("chart_table_cache")
        if (
            rule.type in CHART_RULE_TYPES
            and rule.parsed_tables is None
            and isinstance(chart_table_cache, ChartTableCache)
        ):
            rule.parsed_tables = chart_table_cache.tables_for(actual)

    def compute(
        self,
        expected: list[ParseRuleInput] | None,
        actual: str,
        page: int | None = None,
        **kwargs: Any,
    ) -> MetricValue:
        """
        Execute test rules against markdown content.

        :param expected: List of test rule definitions (from test_rules)
        :param actual: Actual markdown content to test
        :param page: Optional page number (1-indexed) to filter rules
        :param kwargs: Side inputs handed to rules that declare them (see
            ``_prepare_rule``): ``parse_output``, ``raw_output``,
            ``source_file_path``, ``test_case_file_path`` and
            ``chart_table_cache``.
        :return: MetricValue with pass rate and per-rule results
        """
        chart_table_cache = kwargs.get("chart_table_cache")
        if not isinstance(chart_table_cache, ChartTableCache):
            kwargs["chart_table_cache"] = ChartTableCache()
        if not expected:
            return MetricValue(
                metric_name=self.name,
                value=1.0,  # No rules means pass
                metadata={"note": "No test rules provided"},
            )

        # Filter rules by page if page is specified
        rules_to_run = expected
        if page is not None:
            # Filter rules that match this page or have no page specified
            rules_to_run = [rule for rule in expected if get_rule_page(rule) is None or get_rule_page(rule) == page]

        if not rules_to_run:
            return MetricValue(
                metric_name=self.name,
                value=1.0,  # No rules for this page means pass
                metadata={"note": f"No test rules for page {page}"},
            )

        if not actual:
            # Blank output fails every rule. Emit full per-rule metadata so the
            # judge metric and per-type pass rates include this doc (otherwise
            # blank-output docs silently drop out of the aggregate averages,
            # inflating scores for tools that fail to parse hard documents).
            rule_results = [
                {
                    "type": get_rule_type(rule_data),
                    "id": rule_data.id if isinstance(rule_data, ParseRuleBase) else get_rule_id(rule_data),
                    "page": get_rule_page(rule_data),
                    "tags": rule_data.tags if isinstance(rule_data, ParseRuleBase) else [],
                    "layout_id": get_rule_layout_id(rule_data),
                    "layout_ids": get_rule_layout_ids(rule_data),
                    "layout_bindings": get_rule_layout_bindings(rule_data),
                    "passed": False,
                    "score": 0.0,
                    "explanation": "No markdown content provided",
                    # rotate_check entries need expected_angle so the per-angle
                    # breakdown includes blank-output docs too.
                    **(
                        {"expected_angle": rule_data.get("value")} if get_rule_type(rule_data) == "rotate_check" else {}
                    ),
                }
                for rule_data in rules_to_run
            ]
            return MetricValue(
                metric_name=self.name,
                value=0.0,
                metadata={
                    "note": "No markdown content provided",
                    "passed": 0,
                    "total": len(rules_to_run),
                    "ambiguous_anchor_failures": 0,
                    "rule_results": rule_results,
                },
            )

        # Execute each rule
        passed = 0
        ambiguous_anchor_failures = 0
        total = len(rules_to_run)
        rule_results = []
        missing_specific_word_cache: tuple[Counter[str], str] | None = None

        # Timing accumulators
        slow_rules: list[tuple[int, str, float]] = []  # (index, type, seconds)
        timed_out_rules: list[tuple[int, str]] = []  # (index, type)

        # Aggregate per-document budget: bounds the whole rule loop so a single
        # pathological document cannot consume the entire evaluation.
        doc_budget = _doc_rule_budget_seconds()
        skipped_over_budget = 0
        # Rules that turned out to be undefined for this document: (type, why).
        # These are dropped from rule_results entirely, so they neither pass nor
        # fail and never reach the aggregation.
        skipped_inapplicable: list[tuple[str, str]] = []

        def _skipped_rule_entry(rule_data: Any, explanation: str) -> dict[str, Any]:
            """Score a rule 0 without running it (skipped, not evaluated)."""
            return {
                "type": get_rule_type(rule_data),
                "id": rule_data.id if isinstance(rule_data, ParseRuleBase) else get_rule_id(rule_data),
                "page": get_rule_page(rule_data),
                "tags": rule_data.tags if isinstance(rule_data, ParseRuleBase) else [],
                "layout_id": get_rule_layout_id(rule_data),
                "layout_ids": get_rule_layout_ids(rule_data),
                "layout_bindings": get_rule_layout_bindings(rule_data),
                "passed": False,
                "score": 0.0,
                "explanation": explanation,
            }

        # Use signal.alarm for per-rule timeout (Unix only, main thread of worker process)
        use_alarm = hasattr(signal, "SIGALRM")

        # Pre-normalize content ONCE for all rules (major performance optimization).
        # Guard it with the per-rule timeout too: a pathological document (e.g. an
        # O(n^2) regex triggered by degenerate OCR content) must not stall a worker
        # before the rule-loop budget can even take effect. If normalization blows
        # its budget, skip the whole document's rules (scored 0) and move on.
        t_normalize_start = time.monotonic()
        normalize_timed_out = False
        norm_prev_handler = None
        if use_alarm:
            norm_prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(RULE_TIMEOUT_SECONDS)
        try:
            normalized_actual = normalize_text(actual)
        except _RuleTimeoutError:
            normalize_timed_out = True
            normalized_actual = ""
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, norm_prev_handler)
        t_normalize_elapsed = time.monotonic() - t_normalize_start

        if normalize_timed_out:
            print(
                f"  NORMALIZE TIMEOUT after {t_normalize_elapsed:.1f}s: skipping all {total} rules for this document",
                flush=True,
            )
            explanation = f"Skipped: normalization exceeded {RULE_TIMEOUT_SECONDS}s"
            return MetricValue(
                metric_name=self.name,
                value=0.0,
                metadata={
                    "passed": 0,
                    "total": total,
                    "ambiguous_anchor_failures": 0,
                    "skipped_over_budget": total,
                    "note": explanation,
                    "rule_results": [_skipped_rule_entry(r, explanation) for r in rules_to_run],
                },
            )
        print(f"  Pre-normalized content: {len(actual)} -> {len(normalized_actual)} chars ({t_normalize_elapsed:.1f}s)")

        # Re-install the per-rule alarm handler for the loop below.
        prev_handler = None
        if use_alarm:
            prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)

        # Timing baseline for the rule loop (drives the aggregate budget below).
        t_rules_start = time.monotonic()

        # Log every ~100 rules, but at least first and last
        log_interval = max(total // 10, 100) if total > 10 else total
        try:
            for i, rule_data in enumerate(rules_to_run):
                # Aggregate budget check BEFORE arming the alarm: if this document
                # has already spent its budget, skip every remaining rule (score 0)
                # so one bad document can't stall the whole eval pool.
                if doc_budget > 0 and (time.monotonic() - t_rules_start) > doc_budget:
                    remaining = rules_to_run[i:]
                    budget_explanation = f"Skipped: document rule budget ({doc_budget:.0f}s) exceeded"
                    for remaining_rule in remaining:
                        rule_results.append(_skipped_rule_entry(remaining_rule, budget_explanation))
                    skipped_over_budget = len(remaining)
                    print(
                        f"    BUDGET EXCEEDED after {time.monotonic() - t_rules_start:.1f}s"
                        f" ({doc_budget:.0f}s cap): skipping {skipped_over_budget} of {total} remaining rules",
                        flush=True,
                    )
                    break
                if i == 0 or (i + 1) % log_interval == 0:
                    elapsed = time.monotonic() - t_rules_start
                    print(f"  Processing rule {i + 1}/{total} ({elapsed:.1f}s elapsed)", flush=True)
                rule_id = rule_data.id if isinstance(rule_data, ParseRuleBase) else get_rule_id(rule_data)
                rule_tags = rule_data.tags if isinstance(rule_data, ParseRuleBase) else []
                rule_layout_id = get_rule_layout_id(rule_data)
                rule_layout_ids = get_rule_layout_ids(rule_data)
                rule_layout_bindings = get_rule_layout_bindings(rule_data)
                try:
                    t_rule_start = time.monotonic()
                    rule_type_name = get_rule_type(rule_data) or "unknown"

                    # Arm the alarm before rule creation + execution
                    if use_alarm:
                        signal.alarm(RULE_TIMEOUT_SECONDS)

                    rule = create_test_rule(rule_data)
                    self._prepare_rule(rule, actual, kwargs)
                    if isinstance(rule, MissingSpecificWordRule):
                        if missing_specific_word_cache is None:
                            missing_specific_word_cache = (
                                WordBagRule._extract_normalized_words_static(
                                    actual,
                                    include_table_cells=True,
                                ),
                                MissingSpecificWordRule.strip_apostrophes(normalized_actual),
                            )
                        rule.actual_words = missing_specific_word_cache[0]
                        rule.apostrophe_stripped_content = missing_specific_word_cache[1]
                    # Pass pre-normalized content to avoid redundant normalization
                    result = rule.run(actual, normalized_content=normalized_actual)

                    # Disarm the alarm
                    if use_alarm:
                        signal.alarm(0)

                    t_rule_elapsed = time.monotonic() - t_rule_start
                    if t_rule_elapsed > 2.0:
                        slow_rules.append((i, rule_type_name, t_rule_elapsed))
                    rule_passed, explanation = result[0], result[1]
                    score = result[2] if len(result) == 3 else (1.0 if rule_passed else 0.0)
                    rule_result_entry: dict[str, Any] = {
                        "type": get_rule_type(rule_data),
                        "id": rule_id,
                        "page": get_rule_page(rule_data),
                        "tags": rule_tags,
                        "layout_id": rule_layout_id,
                        "layout_ids": rule_layout_ids,
                        "layout_bindings": rule_layout_bindings,
                        "passed": rule_passed,
                        "score": score,
                        "explanation": explanation,
                    }
                    if rule.result_details:
                        rule_result_entry["details"] = rule.result_details
                    if isinstance(rule, RotateCheckRule):
                        rule_result_entry["expected_angle"] = rule.expected_angle
                    rule_results.append(rule_result_entry)
                    if rule_passed:
                        passed += 1
                    elif explanation.startswith("[AMBIGUOUS ANCHORS]"):
                        ambiguous_anchor_failures += 1
                except RuleNotApplicable as e:
                    # The rule carries no evaluable constraint (degenerate rule
                    # definition), so the parser had nothing to get right or
                    # wrong. Emit NO rule result: that is what keeps it out of
                    # the per-type pass rates, the normalized category scores
                    # and semantic_formatting. Counting it 0.0 would zero a
                    # category on a document that parsed correctly.
                    if use_alarm:
                        signal.alarm(0)
                    skipped_inapplicable.append((rule_type_name, str(e)))
                    continue
                except _RuleTimeoutError:
                    t_rule_elapsed = time.monotonic() - t_rule_start
                    timed_out_rules.append((i, rule_type_name))
                    print(
                        f"    TIMEOUT rule #{i}: type={rule_type_name}"
                        f" exceeded {RULE_TIMEOUT_SECONDS}s ({t_rule_elapsed:.1f}s)",
                        flush=True,
                    )
                    rule_results.append(
                        {
                            "type": get_rule_type(rule_data),
                            "id": rule_id,
                            "page": get_rule_page(rule_data),
                            "tags": rule_tags,
                            "layout_id": rule_layout_id,
                            "layout_ids": rule_layout_ids,
                            "layout_bindings": rule_layout_bindings,
                            "passed": False,
                            "score": 0.0,
                            "explanation": f"Rule timed out after {RULE_TIMEOUT_SECONDS}s",
                        }
                    )
                except Exception as e:
                    # Disarm the alarm on error
                    if use_alarm:
                        signal.alarm(0)
                    # If rule execution fails, count as failed
                    rule_results.append(
                        {
                            "type": get_rule_type(rule_data),
                            "id": rule_id,
                            "page": get_rule_page(rule_data),
                            "tags": rule_tags,
                            "layout_id": rule_layout_id,
                            "layout_ids": rule_layout_ids,
                            "layout_bindings": rule_layout_bindings,
                            "passed": False,
                            "score": 0.0,
                            "explanation": f"Error executing rule: {e}",
                        }
                    )
        finally:
            # Always disarm alarm and restore previous handler
            if use_alarm:
                signal.alarm(0)
                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)

        # Inapplicable rules leave the denominator as well as the numerator:
        # they were never evaluated, so they must not dilute the pass rate.
        total -= len(skipped_inapplicable)

        total_score = 0.0
        for r in rule_results:
            total_score += float(cast(float, r["score"]))
        if total > 0:
            pass_rate = total_score / total
        elif skipped_inapplicable:
            # Every rule for this document was undefined. Same convention as a
            # document with no rules at all: nothing to fail.
            pass_rate = 1.0
        else:
            pass_rate = 0.0
        t_rules_total = time.monotonic() - t_rules_start
        print(
            f"  Rules: done, {passed}/{total} passed ({pass_rate:.1%}) in {t_rules_total:.1f}s",
            flush=True,
        )
        if skipped_over_budget:
            print(
                f"    SKIPPED {skipped_over_budget} rule(s): document budget ({doc_budget:.0f}s) exceeded",
                flush=True,
            )
        if skipped_inapplicable:
            for rtype, why in skipped_inapplicable:
                print(f"    NOT APPLICABLE, excluded from scoring: type={rtype} ({why})", flush=True)
        if timed_out_rules:
            for idx, rtype in timed_out_rules:
                print(f"    TIMED OUT rule #{idx}: type={rtype}", flush=True)
        if slow_rules:
            for idx, rtype, secs in slow_rules:
                print(f"    slow rule #{idx}: type={rtype} took {secs:.1f}s", flush=True)

        return MetricValue(
            metric_name=self.name,
            value=pass_rate,
            metadata={
                "passed": passed,
                "total": total,
                "ambiguous_anchor_failures": ambiguous_anchor_failures,
                "skipped_over_budget": skipped_over_budget,
                "skipped_inapplicable": len(skipped_inapplicable),
                "skipped_inapplicable_detail": skipped_inapplicable,
                "rule_results": rule_results,
            },
        )
