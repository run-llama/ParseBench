"""Opt-in ``exact_choice`` grading for single-choice QA items."""

from parse_bench.evaluation.metrics.qa.answer_comparison import AnswerComparisonMetric


class TestExactChoiceOptIn:
    """Strict response parsing is opt-in and fail-closed."""

    def setup_method(self) -> None:
        self.metric = AnswerComparisonMetric()

    def test_exact_answer_line_matches(self) -> None:
        result = self.metric.compare(
            "Answer: A",
            "A",
            "single_choice",
            {"grading_mode": "exact_choice"},
        )
        assert result.value == 1.0
        assert result.metadata["format_valid"] is True

    def test_exact_mode_rejects_bare_choice_prose_and_multiletter(self) -> None:
        for prediction in ("A", "Answer: A because…", "Answer: CD", "Answer: A\nextra"):
            result = self.metric.compare(
                prediction,
                "A",
                "single_choice",
                {"grading_mode": "exact_choice"},
            )
            assert result.value == 0.0, prediction
            assert result.metadata["format_valid"] is False

    def test_exact_mode_accepts_any_single_letter_choice(self) -> None:
        for choice in ("C", "D"):
            result = self.metric.compare(
                f"Answer: {choice}",
                choice,
                "single_choice",
                {"grading_mode": "exact_choice"},
            )
            assert result.value == 1.0, choice
            assert result.metadata["format_valid"] is True

    def test_explicit_option_labels_narrow_exact_choice_set(self) -> None:
        metadata = {
            "grading_mode": "exact_choice",
            "options": "A) Alpha  C) Charlie  D) Delta",
        }
        assert self.metric.compare("Answer: C", "C", "single_choice", metadata).value == 1.0
        invalid = self.metric.compare("Answer: B", "C", "single_choice", metadata)
        assert invalid.value == 0.0
        assert invalid.metadata["format_valid"] is False

    def test_legacy_single_choice_remains_permissive(self) -> None:
        result = self.metric.compare("A", "A", "single_choice")
        assert result.value == 1.0

    def test_unknown_explicit_mode_fails_closed(self) -> None:
        result = self.metric.compare(
            "A",
            "A",
            "single_choice",
            {"grading_mode": "future_mode"},
        )
        assert result.value == 0.0
        assert result.metadata["passed"] is False
