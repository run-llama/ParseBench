"""The eval pool's per-worker timeout, and the stalled-batch recovery it drives.

The cap used to be written as ``for f in as_completed(futures): f.result(timeout=N)``,
which never fires: ``as_completed`` blocks (it was called without a timeout) until
each future is *already done*, so the ``result()`` timeout has nothing left to wait
for and the "worker timed out" branch was dead code. One pathological document --
stuck in a network wait inside the LLM-judge calls, worker at ~0% CPU -- held a
411-document parse evaluation open for over 90 minutes.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation import runner as runner_mod
from parse_bench.evaluation.runner import EvaluationRunner, _drain_eval_futures
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType

# --- _drain_eval_futures ------------------------------------------------------


def test_drain_hands_every_completed_future_to_the_callback() -> None:
    """The happy path drains everything and reports no casualties."""
    seen: list[int] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lambda n=i: n) for i in range(6)]  # type: ignore[misc]
        timed_out, unstarted = _drain_eval_futures(futures, 30, lambda f: seen.append(f.result()))
    assert sorted(seen) == list(range(6))
    assert (timed_out, unstarted) == (set(), set())


def test_drain_cuts_off_a_worker_that_exceeds_the_cap() -> None:
    """A worker still running when the window elapses is abandoned, and the work
    queued behind it is cancelled for a fresh pool rather than lost."""
    release = threading.Event()
    seen: list[str] = []

    def _hang() -> str:
        assert release.wait(timeout=30), "test never released the blocked worker"
        return "hung"

    try:
        # One worker: the hanging task occupies it and the rest never start.
        pool = ThreadPoolExecutor(max_workers=1)
        hung = pool.submit(_hang)
        queued = [pool.submit(lambda n=i: f"queued-{n}") for i in range(3)]  # type: ignore[misc]

        timed_out, unstarted = _drain_eval_futures([hung, *queued], 0.2, lambda f: seen.append(f.result()))

        assert timed_out == {hung}, "the running-but-stuck worker must be reported as timed out"
        assert unstarted == set(queued), "queued work must be cancelled, not blamed for the stall"
        assert seen == [], "nothing completed, so nothing should have been handed to the callback"
        assert all(f.cancelled() for f in queued)
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_drain_keeps_collecting_while_the_pool_makes_progress() -> None:
    """The window restarts on every completion: fast documents are all collected
    even though a slow one is hogging a worker the whole time."""
    release = threading.Event()
    seen: list[str] = []

    def _hang() -> str:
        assert release.wait(timeout=30), "test never released the blocked worker"
        return "hung"

    try:
        pool = ThreadPoolExecutor(max_workers=4)
        hung = pool.submit(_hang)
        fast = [pool.submit(lambda n=i: f"fast-{n}") for i in range(3)]  # type: ignore[misc]

        timed_out, unstarted = _drain_eval_futures([hung, *fast], 0.5, lambda f: seen.append(f.result()))

        assert sorted(seen) == ["fast-0", "fast-1", "fast-2"]
        assert timed_out == {hung}
        assert unstarted == set()
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_drain_reports_a_future_that_finishes_during_the_timeout_sweep() -> None:
    """A future that lands in the gap between the wait expiring and the sweep is
    a completed evaluation, not a timeout -- never fail one we actually have."""
    late: Future = Future()
    late.set_running_or_notify_cancel()
    seen: list[str] = []

    def _on_done(future: Future) -> None:
        seen.append(future.result())

    # Resolve it "just after" the wait gave up, before the sweep inspects it.
    late.set_result("landed")
    timed_out, unstarted = _drain_eval_futures([late], 0.05, _on_done)
    assert seen == ["landed"]
    assert (timed_out, unstarted) == (set(), set())


# --- end-to-end through run_evaluation ---------------------------------------


def _write_case(gt_dir: Path, out_dir: Path, name: str) -> str:
    """Create a matching (test case, result) pair; return the loader's test_id."""
    test_id = f"g/{name}"
    doc_dir = gt_dir / "g"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"{name}.pdf").write_bytes(b"%PDF-1.4 dummy")
    (doc_dir / f"{name}.test.json").write_text(
        json.dumps(
            {
                "data_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "expected_output": {"name": "Alice"},
            }
        )
    )
    now = datetime(2026, 1, 1, 0, 0, 0)
    result = InferenceResult(
        request=InferenceRequest(
            example_id=test_id,
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="test-pipe",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output={
            "task_type": "extract",
            "example_id": test_id,
            "pipeline_name": "test-pipe",
            "extracted_data": {"name": "Alice"},
        },
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    res_dir = out_dir / "test-pipe" / "g"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{name}.result.json").write_text(result.model_dump_json())
    return test_id


def test_hung_worker_is_reported_and_never_blocks_the_rest_of_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A document whose worker exceeds the cap is failed with the timeout line, and
    the documents queued behind it are still evaluated (on a fresh pool)."""
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "out"
    hung_id = _write_case(gt_dir, out_dir, "aaa_hangs")
    ok_ids = [_write_case(gt_dir, out_dir, name) for name in ("bbb_ok", "ccc_ok")]

    release = threading.Event()
    real_worker = runner_mod._evaluate_single_worker

    def _worker(*task: Any) -> dict[str, Any]:
        if task[1].get("test_id") == hung_id:
            assert release.wait(timeout=30), "test never released the blocked worker"
        return real_worker(*task)  # type: ignore[no-any-return]

    class _SerialThreadPool(ThreadPoolExecutor):
        """Stand-in for the process pool: one worker, so ``aaa_hangs`` occupies it
        and the other two documents sit queued behind it."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(max_workers=1)

    monkeypatch.setattr(runner_mod, "_evaluate_single_worker", _worker)
    monkeypatch.setattr(runner_mod, "ProcessPoolExecutor", _SerialThreadPool)
    monkeypatch.setattr(runner_mod, "_EVAL_WORKER_STALL_TIMEOUT_SECONDS", 0.5)

    try:
        summary = EvaluationRunner(output_dir=out_dir, test_cases_dir=gt_dir).run_evaluation(
            product_type="extract", use_rich=False, max_workers=1
        )
    finally:
        release.set()

    out = capsys.readouterr().out
    assert f"{hung_id}: FAILED (worker timed out after 0.5s)" in out

    by_id = {r.test_id: r for r in summary.per_example_results}
    assert summary.successful == 2, "documents queued behind the hung one must still be evaluated"
    assert summary.failed == 1, "the timed-out document is counted as a failure, not silently dropped"
    for ok_id in ok_ids:
        assert by_id[ok_id].success


def test_documents_queued_behind_a_stall_are_retried_before_being_blamed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real ProcessPool keeps its call queue pre-fed, so a document that never
    ran can still show up as RUNNING and be swept up in the stall. Those get a
    second attempt on the fresh pool; only the document that stalls again is
    failed, and it is failed exactly once."""
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "out"
    hung_id = _write_case(gt_dir, out_dir, "aaa_hangs")
    swept_id = _write_case(gt_dir, out_dir, "bbb_swept_up")
    queued_id = _write_case(gt_dir, out_dir, "ccc_queued")

    release = threading.Event()
    real_worker = runner_mod._evaluate_single_worker
    attempts: dict[str, int] = {}
    lock = threading.Lock()

    def _worker(*task: Any) -> dict[str, Any]:
        test_id = str(task[1].get("test_id"))
        with lock:
            attempts[test_id] = attempts.get(test_id, 0) + 1
            nth = attempts[test_id]
        if test_id == hung_id:
            # Never finishes on its own -- the real pathological document.
            assert release.wait(timeout=30), "test never released the blocked worker"
        elif test_id == swept_id and nth == 1:
            # Stands in for a pre-fed queue entry: occupies a worker slot past the
            # stall window on the first batch, then evaluates normally on the retry.
            time.sleep(3)
        return real_worker(*task)  # type: ignore[no-any-return]

    class _TwoWorkerThreadPool(ThreadPoolExecutor):
        """Two slots, so aaa + bbb are RUNNING when the batch stalls and ccc is
        left genuinely unstarted -- the shape that makes a retry worthwhile."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(max_workers=2)

    monkeypatch.setattr(runner_mod, "_evaluate_single_worker", _worker)
    monkeypatch.setattr(runner_mod, "ProcessPoolExecutor", _TwoWorkerThreadPool)
    monkeypatch.setattr(runner_mod, "_EVAL_WORKER_STALL_TIMEOUT_SECONDS", 0.5)

    try:
        summary = EvaluationRunner(output_dir=out_dir, test_cases_dir=gt_dir).run_evaluation(
            product_type="extract", use_rich=False, max_workers=2
        )
    finally:
        release.set()

    out = capsys.readouterr().out
    by_id = {r.test_id: r for r in summary.per_example_results}

    assert attempts[swept_id] == 2, "the swept-up document must actually be re-submitted"
    assert by_id[swept_id].success, "a healthy document caught in the stall must be recovered, not blamed"
    assert by_id[queued_id].success
    assert summary.successful == 2

    # The genuinely hung document stalls again on the retry and is failed then --
    # once, not once per attempt.
    assert out.count("FAILED (worker timed out after 0.5s)") == 1
    assert f"{hung_id}: FAILED (worker timed out after 0.5s)" in out
