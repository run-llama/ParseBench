"""Direct rule evaluation is usable from a thread pool without touching SIGALRM."""

import signal
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from parse_bench.evaluation.metrics.parse import rule_based_metric as module
from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric

RULES = [{"type": "is_bold", "text": "hello", "id": "bold-hello"}]


def in_worker(call):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(call).result(timeout=30)


@pytest.mark.parametrize("actual, expected", [("**hello**", 1.0), ("plain hello", 0.0)])
def test_threaded_evaluation_warns_and_keeps_scores(actual, expected):
    with pytest.warns(RuntimeWarning, match="cannot interrupt running work") as captured:
        result = in_worker(lambda: RuleBasedMetric().compute(RULES, actual))
    assert result.value == expected
    assert result.metadata["total"] == 1
    assert len(captured) == 1


def test_worker_never_installs_or_cancels_process_alarm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a worker must not change process-global signal state")

    monkeypatch.setattr(signal, "signal", forbidden)
    monkeypatch.setattr(signal, "alarm", forbidden, raising=False)
    with pytest.warns(RuntimeWarning):
        assert in_worker(lambda: RuleBasedMetric().compute(RULES, "**hello**")).value == 1.0


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="requires Unix alarm")
def test_main_thread_retains_alarm_lifecycle(monkeypatch):
    original_handler = signal.getsignal(signal.SIGALRM)
    original_alarm = signal.alarm
    calls = []

    def recording_alarm(seconds):
        calls.append(seconds)
        return original_alarm(seconds)

    monkeypatch.setattr(signal, "alarm", recording_alarm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = RuleBasedMetric().compute(RULES, "**hello**")
    assert result.value == 1.0
    assert module.RULE_TIMEOUT_SECONDS in calls
    assert calls[-1] == 0
    assert signal.getsignal(signal.SIGALRM) == original_handler
    assert not any(isinstance(item.message, RuntimeWarning) for item in caught)


def test_platform_without_sigalrm_warns_without_changing_score(monkeypatch):
    monkeypatch.delattr(signal, "SIGALRM", raising=False)
    with pytest.warns(RuntimeWarning, match="without SIGALRM"):
        assert RuleBasedMetric().compute(RULES, "**hello**").value == 1.0


@pytest.mark.parametrize(
    "expected, actual, kwargs, expected_score",
    [
        ([], "**hello**", {}, 1.0),
        (RULES, "", {}, 0.0),
        ([{"type": "is_bold", "text": "hello", "page": 1}], "**hello**", {"page": 2}, 1.0),
    ],
)
def test_short_circuit_paths_do_not_warn(expected, actual, kwargs, expected_score):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = in_worker(lambda: RuleBasedMetric().compute(expected, actual, **kwargs))
    assert result.value == expected_score
    assert not any(isinstance(item.message, RuntimeWarning) for item in caught)


def test_worker_normalization_error_propagates(monkeypatch):
    def fail(_actual):
        raise ValueError("normalization regression fixture")

    monkeypatch.setattr(module, "normalize_text", fail)
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="normalization regression fixture"):
        in_worker(lambda: RuleBasedMetric().compute(RULES, "**hello**"))


def test_worker_rule_error_remains_a_failed_rule(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("rule regression fixture")

    monkeypatch.setattr(RuleBasedMetric, "_prepare_rule", fail)
    with pytest.warns(RuntimeWarning):
        result = in_worker(lambda: RuleBasedMetric().compute(RULES, "**hello**"))
    assert result.value == 0.0
    assert "rule regression fixture" in result.metadata["rule_results"][0]["explanation"]


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="requires Unix alarm")
def test_normalization_timeout_still_restores_main_handler(monkeypatch):
    original_handler = signal.getsignal(signal.SIGALRM)

    def timeout(_actual):
        raise module._RuleTimeoutError()

    monkeypatch.setattr(module, "normalize_text", timeout)
    result = RuleBasedMetric().compute(RULES, "**hello**")
    assert result.value == 0.0
    assert result.metadata["skipped_over_budget"] == 1
    assert signal.getsignal(signal.SIGALRM) == original_handler
    assert signal.alarm(0) == 0


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="requires Unix alarm")
def test_rule_timeout_still_restores_main_handler(monkeypatch):
    original_handler = signal.getsignal(signal.SIGALRM)

    def timeout(*args, **kwargs):
        raise module._RuleTimeoutError()

    monkeypatch.setattr(RuleBasedMetric, "_prepare_rule", timeout)
    result = RuleBasedMetric().compute(RULES, "**hello**")
    assert result.value == 0.0
    assert "timed out" in result.metadata["rule_results"][0]["explanation"]
    assert signal.getsignal(signal.SIGALRM) == original_handler
    assert signal.alarm(0) == 0


def test_concurrent_worker_results_do_not_share_signal_state():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(RuleBasedMetric().compute, RULES, text) for text in ("**hello**", "plain hello")]
            assert [future.result(timeout=10).value for future in futures] == [1.0, 0.0]
    assert len([item for item in caught if isinstance(item.message, RuntimeWarning)]) == 2
