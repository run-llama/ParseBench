"""Answer comparison metric for QA evaluation."""

import re
from collections.abc import Mapping
from typing import Any

from parse_bench.schemas.evaluation import MetricValue


class AnswerComparisonMetric:
    """Metric for comparing predicted answers with expected answers."""

    def compare(
        self,
        predicted: str,
        expected: str,
        question_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> MetricValue:
        """
        Compare predicted answer with expected answer.

        :param predicted: Predicted answer from LLM
        :param expected: Expected answer from test case
        :param question_type: Type of question ("single_choice", "multiple_choice", "numerical")
        :param metadata: Optional metadata (tolerance, options, etc.)
        :return: MetricValue with pass/fail and metadata
        """
        # ``exact_choice`` is deliberately an opt-in grading contract.  Do
        # not route legacy single-choice items through it: older datasets rely
        # on the permissive letter extraction below (and, in particular, may
        # use answer strings rather than a serialized model response).
        if isinstance(metadata, dict) and "grading_mode" in metadata:
            grading_mode = metadata.get("grading_mode")
            if grading_mode == "exact_choice":
                return self._compare_exact_choice(predicted, expected, metadata)
            return MetricValue(
                metric_name="qa_answer_match",
                value=0.0,
                metadata={
                    "passed": False,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": question_type,
                    "error": f"Unsupported grading mode: {grading_mode!r}",
                },
            )

        if question_type == "single_choice":
            return self._compare_single_choice(predicted, expected, metadata)
        elif question_type == "multiple_choice":
            return self._compare_multiple_choice(predicted, expected, metadata)
        elif question_type == "numerical":
            return self._compare_numerical(predicted, expected, metadata)
        elif question_type == "free_text":
            return self._compare_free_text(predicted, expected, metadata)
        else:
            return MetricValue(
                metric_name="qa_answer_match",
                value=0.0,
                metadata={
                    "passed": False,
                    "predicted": predicted,
                    "expected": expected,
                    "error": f"Unknown question type: {question_type}",
                },
            )

    def _compare_exact_choice(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any],
    ) -> MetricValue:
        """Grade one strict ``Answer: <letter>`` response.

        The parser intentionally has no fallback to bare-letter or prose
        extraction. It is used only when the annotation explicitly sets
        ``metadata.grading_mode`` to ``exact_choice``; all unconfigured QA
        items retain the historical comparison behavior. A configured answer
        may be any single letter A-Z. When an option grid is supplied, its
        labelled choices narrow the accepted set.
        """

        valid_choices = self._exact_choice_set(expected, metadata)
        predicted_choice = self._parse_exact_choice_response(predicted)
        expected_choice = expected.strip().upper()
        valid_expected = bool(re.fullmatch(r"[A-Z]", expected_choice)) and expected_choice in valid_choices
        choice_valid = predicted_choice is not None and predicted_choice in valid_choices
        passed = valid_expected and choice_valid and predicted_choice == expected_choice

        return MetricValue(
            metric_name="qa_answer_match",
            value=1.0 if passed else 0.0,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "question_type": "single_choice",
                "grading_mode": "exact_choice",
                "predicted_choice": predicted_choice,
                "format_valid": choice_valid,
                "expected_valid": valid_expected,
                "valid_choices": sorted(valid_choices),
                "options": metadata.get("options"),
            },
        )

    @staticmethod
    def _exact_choice_set(expected: str, metadata: dict[str, Any]) -> set[str]:
        """Return the configured exact-choice labels.

        A single-letter expected answer is sufficient configuration and keeps
        this mode useful for datasets without an option grid. If ``options``
        contains explicit labels, those labels are authoritative instead.
        Invalid/unlabelled option text falls back to A-Z so it cannot make a
        valid annotation impossible to grade merely because display text was
        omitted from a sidecar.
        """

        options = metadata.get("options")
        labels: set[str] = set()
        if isinstance(options, str):
            labels.update(
                label.upper() for label in re.findall(r"(?<![A-Za-z0-9])([A-Z])\s*[).:]", options, re.IGNORECASE)
            )
        elif isinstance(options, Mapping):
            labels.update(str(key).strip().upper() for key in options if re.fullmatch(r"[A-Za-z]", str(key).strip()))
        elif isinstance(options, (list, tuple, set)):
            for option in options:
                if isinstance(option, Mapping):
                    label = option.get("label", option.get("id", ""))
                else:
                    label = option
                if isinstance(label, str):
                    match = re.match(r"^\s*([A-Za-z])\s*[).:]", label)
                    if match:
                        labels.add(match.group(1).upper())

        if labels:
            return labels

        expected_choice = expected.strip().upper()
        if re.fullmatch(r"[A-Z]", expected_choice):
            return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return set()

    @staticmethod
    def _parse_exact_choice_response(response: str) -> str | None:
        """Parse exactly one ``Answer: <letter>`` response."""

        if not isinstance(response, str):
            return None
        candidate = response.strip()
        # ``\s`` would accept a newline at either end.  Reject all line breaks
        # explicitly so prose or a second answer can never be accepted.
        if "\n" in candidate or "\r" in candidate:
            return None
        match = re.fullmatch(r"Answer:[ \t]+([A-Z])[ \t]*", candidate, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _compare_single_choice(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare single choice answers."""
        # Normalize both answers
        pred_normalized = self._normalize_answer(predicted)
        exp_normalized = self._normalize_answer(expected)

        # Try exact match first
        if pred_normalized == exp_normalized:
            return MetricValue(
                metric_name="qa_answer_match",
                value=1.0,
                metadata={
                    "passed": True,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": "single_choice",
                },
            )

        # Try extracting letter from predicted answer
        pred_letter = self._extract_letter(predicted)
        exp_letter = self._extract_letter(expected)

        if pred_letter and exp_letter and pred_letter == exp_letter:
            return MetricValue(
                metric_name="qa_answer_match",
                value=1.0,
                metadata={
                    "passed": True,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": "single_choice",
                    "matched_letter": pred_letter,
                },
            )

        # Case-insensitive comparison
        if pred_normalized.lower() == exp_normalized.lower():
            return MetricValue(
                metric_name="qa_answer_match",
                value=1.0,
                metadata={
                    "passed": True,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": "single_choice",
                },
            )

        return MetricValue(
            metric_name="qa_answer_match",
            value=0.0,
            metadata={
                "passed": False,
                "predicted": predicted,
                "expected": expected,
                "question_type": "single_choice",
            },
        )

    def _compare_multiple_choice(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare multiple choice answers."""
        # Parse answers into sets (order-independent)
        pred_set = self._parse_multiple_choice(predicted)
        exp_set = self._parse_multiple_choice(expected)

        # Compare sets
        passed = pred_set == exp_set
        value = 1.0 if passed else 0.0

        return MetricValue(
            metric_name="qa_answer_match",
            value=value,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "predicted_set": sorted(pred_set),
                "expected_set": sorted(exp_set),
                "question_type": "multiple_choice",
            },
        )

    def _compare_numerical(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare numerical answers with optional tolerance."""
        # Extract numbers from strings
        pred_num = self._extract_number(predicted)
        exp_num = self._extract_number(expected)

        if pred_num is None or exp_num is None:
            return MetricValue(
                metric_name="qa_answer_match",
                value=0.0,
                metadata={
                    "passed": False,
                    "predicted": predicted,
                    "expected": expected,
                    "error": "Could not extract numbers from answers",
                    "question_type": "numerical",
                },
            )

        # Get tolerance from metadata
        tolerance = 0.0
        if metadata:
            tolerance_val = metadata.get("tolerance")
            if tolerance_val is not None:
                try:
                    tolerance = float(tolerance_val)
                except (ValueError, TypeError):
                    pass

        # Compare with tolerance
        diff = abs(pred_num - exp_num)
        passed = diff <= tolerance
        value = 1.0 if passed else 0.0

        return MetricValue(
            metric_name="qa_answer_match",
            value=value,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "predicted_number": pred_num,
                "expected_number": exp_num,
                "difference": diff,
                "tolerance": tolerance,
                "question_type": "numerical",
            },
        )

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison, matching official FinMME format."""
        # Use the same normalization as official FinMME eval
        normalized = (
            answer.replace("**", "")
            .replace(":", "")
            .replace("$\\boxed{", "")
            .replace("}$", "")
            .replace("\\$", "")
            .replace("$", "")
            .replace("{", "")
            .replace("\\boxed", "")
        )
        return normalized.strip()

    def _extract_letter(self, answer: str) -> str | None:
        """Extract letter code (A, B, C, etc.) from answer."""
        # Look for single letter at start or in parentheses
        match = re.search(r"\b([A-Z])\b", answer.upper())
        if match:
            return match.group(1)
        return None

    def _parse_multiple_choice(self, answer: str) -> set[str]:
        """
        Parse multiple choice answer into set of letters.

        Matches the official FinMME eval logic: extract any character
        that's a valid choice letter (A-Z).
        """
        # Normalize answer
        normalized = self._normalize_answer(answer.upper())

        # Extract any character that's a valid choice letter (A-Z)
        # This matches the official FinMME eval script logic
        valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = {c for c in normalized if c in valid_letters}

        return letters

    def _compare_free_text(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare free-text answers with case-insensitive exact match."""
        pred_normalized = predicted.strip().lower()
        exp_normalized = expected.strip().lower()

        if "," in exp_normalized:
            pred_set = {s.strip() for s in pred_normalized.split(",")}
            exp_set = {s.strip() for s in exp_normalized.split(",")}
            passed = pred_set == exp_set
        else:
            passed = pred_normalized == exp_normalized

        return MetricValue(
            metric_name="qa_answer_match",
            value=1.0 if passed else 0.0,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "question_type": "free_text",
            },
        )

    def _extract_number(self, text: str) -> float | None:
        """Extract number from text string."""
        # Remove common prefixes
        text = re.sub(
            r"^(answer|answer:|the answer is|the answer:)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()

        # Try to find number (including decimals, negatives, scientific notation)
        # Match numbers with optional commas, decimals, negatives
        pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?"
        match = re.search(pattern, text)
        if match:
            # Remove commas before parsing
            num_str = match.group(0).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                pass

        return None
