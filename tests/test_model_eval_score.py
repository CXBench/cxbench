from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from model_eval.run import REPO_ROOT
from model_eval.score import (
    ScoreError,
    _resolve_recorded_task_path,
    build_parser,
    build_score_plan,
    main,
    score_existing,
)


TASK_PATH = REPO_ROOT / "benchmarks" / "cxbench-120" / "tasks" / "BK-RAW-001.us.yaml"


def _write_task_run(run_dir: Path, *, recorded_task_path: str | None = None) -> None:
    run_dir.mkdir(parents=True)
    config = {
        "task_path": recorded_task_path or str(TASK_PATH),
        "task_id": "BK-RAW-001.us",
        "mode": "least_privilege",
        "grader": "hybrid",
        "judge_model": "openai:gpt-test",
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")
    (run_dir / "final_state.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "grade.json").write_text('{"sentinel": true}\n', encoding="utf-8")


def test_score_existing_trace_writes_separate_artifacts_without_mutating_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "task-run"
    _write_task_run(run_dir)
    source_files = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    plan = build_score_plan(run_dir / "trace.jsonl")
    rows, report = score_existing(plan, tmp_path / "scores", grader="rule")

    assert rows[0]["ok"] is True
    assert rows[0]["final_grader"] == "rule"
    assert rows[0]["trace_sha256"] == hashlib.sha256(b"").hexdigest()
    assert report["grader"] == "rule"
    assert (tmp_path / "scores" / "grades" / "BK-RAW-001.us.json").is_file()
    assert (tmp_path / "scores" / "report.md").is_file()
    assert (tmp_path / "scores" / "domain_metrics.csv").is_file()
    assert source_files == {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    }


def test_batch_plan_recovers_recorded_wsl_paths_and_validates_task_hash(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "batch"
    task_run_name = "BK-RAW-001.us-least_privilege-test"
    task_run_dir = batch_dir / "task_runs" / task_run_name
    recorded_task_path = (
        f"/mnt/{tmp_path.name}/cxbench-main/benchmarks/cxbench-120/tasks/BK-RAW-001.us.yaml"
    )
    _write_task_run(task_run_dir, recorded_task_path=recorded_task_path)
    manifest = {
        "run_id": "source-run",
        "benchmark": "cxbench-120",
        "model": "gpt-test",
        "mode": "least_privilege",
        "grader": "hybrid",
        "judge_model": "gpt-test",
        "judge_provider": "same",
        "agent_provider": "openai",
        "task_count": 1,
        "tasks": [
            {
                "path": recorded_task_path,
                "sha256": hashlib.sha256(TASK_PATH.read_bytes()).hexdigest(),
            }
        ],
    }
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = {
        "task_id": "BK-RAW-001.us",
        "run_dir": f"/mnt/d/old-checkout/task_runs/{task_run_name}",
        "ok": True,
    }
    (batch_dir / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    plan = build_score_plan(batch_dir)

    assert plan.source_kind == "batch"
    assert plan.inputs[0].task_path == TASK_PATH.resolve()
    assert plan.inputs[0].run_dir == task_run_dir.resolve()
    assert plan.inputs[0].judge_model == "openai:gpt-test"


def test_batch_plan_rejects_task_drift_before_scoring(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    (batch_dir / "task_runs").mkdir(parents=True)
    manifest = {
        "benchmark": "cxbench-120",
        "task_count": 1,
        "tasks": [{"path": str(TASK_PATH), "sha256": "0" * 64}],
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ScoreError, match="task changed since the original run"):
        build_score_plan(batch_dir)


def test_dry_run_validates_trace_without_writing_or_calling_a_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "task-run"
    output_dir = tmp_path / "scores"
    _write_task_run(run_dir)

    exit_code = main([str(run_dir / "trace.jsonl"), "--output", str(output_dir), "--dry-run"])

    assert exit_code == 0
    assert not output_dir.exists()
    output = capsys.readouterr().out
    assert "grader: hybrid" in output
    assert "scoreable_records: 1" in output
    assert "input_errors: 0" in output
    assert "no API request was sent and no file was written" in output


def test_dry_run_reports_a_missing_final_state_as_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "task-run"
    _write_task_run(run_dir)
    (run_dir / "final_state.json").unlink()

    exit_code = main([str(run_dir / "trace.jsonl"), "--dry-run"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "scoreable_records: 0" in output
    assert "input_errors: 1" in output
    assert "missing required artifact(s): final_state.json" in output


def test_existing_record_scorer_defaults_to_formal_hybrid_grader() -> None:
    args = build_parser().parse_args(["trace.jsonl"])

    assert args.grader == "hybrid"


@pytest.mark.parametrize(
    "selection", ["recorded_release", "manifest", "missing_release", "ambiguous"]
)
def test_recorded_task_resolution_preserves_release_after_checkout_rename(
    monkeypatch, tmp_path, selection
):
    checkout = tmp_path / "renamed-checkout"
    filename = TASK_PATH.name
    for release in ["original", "revised"]:
        path = checkout / "benchmarks" / release / "tasks" / filename
        path.parent.mkdir(parents=True)
        path.write_bytes(TASK_PATH.read_bytes())
    monkeypatch.setattr("model_eval.score.REPO_ROOT", checkout)
    old_root = tmp_path / "absent-original-checkout"
    expected = checkout / "benchmarks/original/tasks" / filename
    if selection == "recorded_release":
        recorded = old_root / "benchmarks/original/tasks" / filename
        assert _resolve_recorded_task_path(str(recorded)) == expected
    elif selection == "manifest":
        assert (
            _resolve_recorded_task_path(str(old_root / filename), benchmark="original") == expected
        )
    elif selection == "missing_release":
        recorded = old_root / "benchmarks/missing/tasks" / filename
        with pytest.raises(ScoreError, match="release cannot be found"):
            _resolve_recorded_task_path(str(recorded))
    else:
        with pytest.raises(ScoreError, match="ambiguous"):
            _resolve_recorded_task_path(str(old_root / filename))
