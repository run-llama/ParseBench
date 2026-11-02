"""Tests for the leaderboard accountability scripts (scripts/leaderboard/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "leaderboard"))

import apply as lb_apply  # noqa: E402
import check as lb_check  # noqa: E402
from common import DIMENSIONS, MapEntry, ResultsMap, load_csv, load_map, overall, save_map  # noqa: E402

SHA = "a" * 40


def _row(provider: str, verified: str = "yes", **scores: str) -> dict[str, str]:
    row = {"Provider": provider, "Category": "X", "Overall": "50.00", "Verified": verified}
    row.update(dict.fromkeys(DIMENSIONS, "50.00"))
    row.update(scores)
    return row


def _map(*entries: MapEntry) -> ResultsMap:
    return ResultsMap(
        results_repo="org/results", data_repo="org/data", data_revision=SHA, scored_with=None, rows=list(entries)
    )


def test_repo_leaderboard_is_consistent_with_results_map() -> None:
    _, rows = load_csv()
    rmap = load_map()
    assert lb_check.static_errors(rows, rmap) == []
    assert lb_check.verified_column_errors(rows, rmap) == []


def test_static_errors_catch_unmapped_duplicate_and_unsafe_entries() -> None:
    rows = [_row("A"), _row("B")]
    rmap = _map(
        MapEntry("A", "verified", "pipe", SHA),
        MapEntry("C", "verified", "pipe", SHA),
        MapEntry("D", "verified", "bad name; rm -rf /", "main"),
        MapEntry("E", "verified"),
    )
    errors = "\n".join(lb_check.static_errors(rows, rmap))
    assert "'B' has no entry" in errors
    assert "'C' matches no leaderboard row" in errors
    assert "'pipe' is mapped to more than one row" in errors
    assert "pipeline 'bad name; rm -rf /' must match" in errors
    assert "results_revision 'main' must match" in errors
    assert "'E': verified entries need" in errors


def test_results_revision_accepts_commit_and_hf_pr_refs() -> None:
    assert lb_check.SAFE_REVISION.fullmatch(SHA)
    assert lb_check.SAFE_REVISION.fullmatch("refs/pr/12")
    assert not lb_check.SAFE_REVISION.fullmatch("main")


def test_changed_rows_detects_new_and_edited_scores_only() -> None:
    base = [_row("A"), _row("B")]
    head = [_row("A", Category="renamed"), _row("B", Charts="60.00"), _row("C")]
    assert lb_check.changed_rows(head, base) == ["B", "C"]


def test_new_values_rounds_dimensions_before_overall() -> None:
    result = {"scores": {c: 10.0 + i + 0.004 for i, c in enumerate(DIMENSIONS)}}
    old = dict.fromkeys(["Overall", *DIMENSIONS], 50.0)
    new = lb_apply.new_values(result, old)
    assert all(new[c] == round(10.0 + i, 2) for i, c in enumerate(DIMENSIONS))
    assert new["Overall"] == round(overall({c: new[c] for c in DIMENSIONS}), 2)


def test_new_values_keeps_old_value_for_dimension_that_errored() -> None:
    result = {"scores": {**dict.fromkeys(DIMENSIONS, 60.0), "Visual_Grounding": None}}
    old = dict.fromkeys(["Overall", *DIMENSIONS], 50.0)
    new = lb_apply.new_values(result, old)
    assert new["Visual_Grounding"] == 50.0 and new["Charts"] == 60.0
    assert new["Overall"] == pytest.approx(58.0)  # recomputed from the values that are published


def test_diff_rows_reports_deltas_and_errors() -> None:
    rows = [_row("A"), _row("B"), _row("U", verified="no")]
    rescore = {
        "results": {
            "pa": {"scores": dict.fromkeys(DIMENSIONS, 53.0)},
            "pb": {"error": "boom"},
        }
    }
    records = lb_apply.diff_rows(rows, rescore, {"A": "pa", "B": "pb"})
    assert [r["provider"] for r in records] == ["A", "B"]
    assert records[0]["delta"]["Charts"] == pytest.approx(3.0)
    assert lb_apply.max_abs_delta(records[0]) == pytest.approx(3.0)
    assert records[1]["error"] == "boom"


def test_diff_rows_reports_dimension_errors_and_coverage() -> None:
    details = {c: {"published": 503, "total": 503, "error": None} for c in DIMENSIONS}
    details["Visual_Grounding"]["error"] = "497 page(s) failed in the harness"
    details["Content_Faithfulness"]["published"] = 85
    scores = {**dict.fromkeys(DIMENSIONS, 50.0), "Visual_Grounding": None}
    records = lb_apply.diff_rows([_row("A")], {"results": {"pa": {"scores": scores, "details": details}}}, {"A": "pa"})
    rec = records[0]
    assert rec["dim_errors"] == {"Visual_Grounding": "497 page(s) failed in the harness"}
    assert rec["coverage"] == {"Content_Faithfulness": (85, 503)}
    assert rec["new"]["Visual_Grounding"] == 50.0  # old value kept
    assert "❌" in lb_apply.render_report("t", records, 2.0, "intro")


def test_apply_writes_scores_and_records_harness(tmp_path: Path) -> None:
    csv_path, map_path = tmp_path / "lb.csv", tmp_path / "results.yaml"
    fieldnames = ["Provider", "Category", "Overall", *DIMENSIONS, "Verified"]
    from common import write_csv

    write_csv(fieldnames, [_row("A"), _row("U", verified="no")], csv_path)
    save_map(_map(MapEntry("A", "verified", "pa", SHA), MapEntry("U", "unverified")), map_path)
    rescore = tmp_path / "rescore.json"
    rescore.write_text(
        json.dumps(
            {
                "scored_with": "b" * 40,
                "data_repo": "org/data",
                "data_revision": SHA,
                "results_repo": "org/results",
                "results": {"pa": {"scores": dict.fromkeys(DIMENSIONS, 60.0)}},
            }
        )
    )
    args = lb_apply.argparse.Namespace(rescore=rescore, csv=csv_path, map=map_path, report=None, threshold=2.0)
    assert lb_apply.cmd_apply(args) == 0
    _, rows = load_csv(csv_path)
    by = {r["Provider"]: r for r in rows}
    assert by["A"]["Charts"] == "60.00" and by["A"]["Overall"] == "60.00"
    assert by["U"]["Charts"] == "50.00"
    assert load_map(map_path).scored_with == "b" * 40


def test_rows_missing_more_than_one_percent_are_incomplete() -> None:
    assert lb_apply._incomplete({"coverage": {"Tables": (501, 503)}}) is False  # 2 pages: scored 0, still verified
    assert lb_apply._incomplete({"coverage": {"Content_Faithfulness": (85, 506)}}) is True
    assert lb_apply._incomplete({"coverage": {}}) is False


def test_merge_combines_shards_from_one_run(tmp_path: Path) -> None:
    import rescore

    header = {"scored_with": "c" * 40, "data_repo": "org/data", "data_revision": SHA, "results_repo": "org/results"}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({**header, "results": {"pb": {"scores": {}}}}))
    b.write_text(json.dumps({**header, "results": {"pa": {"error": "boom"}}}))
    out = tmp_path / "merged.json"
    assert rescore.cmd_merge(lb_apply.argparse.Namespace(shards=[a, b], out=out)) == 0
    merged = json.loads(out.read_text())
    assert list(merged["results"]) == ["pa", "pb"] and merged["scored_with"] == "c" * 40


def test_merge_rejects_shards_from_different_runs(tmp_path: Path) -> None:
    import rescore

    header = {"scored_with": "c" * 40, "data_repo": "org/data", "data_revision": SHA, "results_repo": "org/results"}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({**header, "results": {}}))
    b.write_text(json.dumps({**header, "scored_with": "d" * 40, "results": {}}))
    with pytest.raises(SystemExit):
        rescore.cmd_merge(lb_apply.argparse.Namespace(shards=[a, b], out=tmp_path / "m.json"))


def test_ground_truth_download_retries_dropped_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    import huggingface_hub
    import rescore
    from huggingface_hub.errors import HfHubHTTPError

    calls: list[dict] = []

    def flaky_snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            response = httpx.Response(499, request=httpx.Request("GET", "https://huggingface.co/x"))
            raise HfHubHTTPError("Client error '499 unknown'", response=response)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky_snapshot_download)
    monkeypatch.setattr(rescore.time, "sleep", lambda _seconds: None)
    path = rescore.ground_truth("org/data", SHA, tmp_path, None)
    assert len(calls) == 2 and calls[0]["max_workers"] == rescore.GT_DOWNLOAD_WORKERS
    assert (path / ".complete").read_text() == SHA
    assert rescore.ground_truth("org/data", SHA, tmp_path, None) == path and len(calls) == 2  # cached
