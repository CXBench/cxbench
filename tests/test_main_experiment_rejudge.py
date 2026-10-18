"""Offline regressions for rejudging; rule-only fixtures are not formal results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cxbench import grader, openai_compat
from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec
from main_experiment import rejudge, resume, score


@pytest.fixture
def recorded_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(rejudge.batch, "_load_dotenv", lambda _: {})
    monkeypatch.setattr(openai_compat, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://original.invalid/v1")

    def make(mode="least_privilege", old_grader="rule"):
        source = tmp_path / "original"
        task = TaskSpec.load(Path("tasks/CX-01.us.yaml"))
        runtime = Runtime(
            task,
            mode=mode,
            model="scripted:compliant" if mode == "least_privilege" else "scripted:direct_bad",
            runs_dir=source / "task_runs",
            grader="rule",
            user_sim="deterministic",
        )
        executed = runtime.run()
        directory = Path(executed["run_dir"])
        config = score._json(directory / "config.json")
        config.update(
            grader=old_grader,
            final_grader=old_grader,
            execution_grader="rule",
            judge_model="openai:old-judge",
        )
        resume._write_json(directory / "config.json", config)
        old_grade = executed["grade"]
        # Deliberately stale rule and outcome values must not drive the new verdict.
        old_grade.update(grader=old_grader, outcome_pass=False, rules=[])
        resume._write_json(directory / "grade.json", old_grade)
        row = {
            "task_id": task.id,
            "task_path": str(task.path.resolve()),
            "run_dir": str(directory),
            "domain": task.domain,
            "ok": True,
            "verdict": "fail",
            "final_grader": old_grader,
            "error": None,
            "elapsed_seconds": 12.5,
            "turns": 4,
        }
        resume._write_results(source / "results.jsonl", {task.id: row})
        resume._write_json(
            source / "manifest.json",
            {
                "run_id": source.name,
                "model": "original-agent",
                "agent_provider": "openai",
                "grader": old_grader,
                "final_grader": old_grader,
                "execution_grader": "rule",
                "judge_model": "old-judge",
                "judge_provider": "openai",
                "task_count": 1,
                "tasks": [
                    {"path": str(task.path.resolve()), "sha256": rejudge.batch._sha256(task.path)}
                ],
            },
        )
        abandoned = source / "task_runs" / "abandoned"
        abandoned.mkdir()
        (abandoned / "grade.json").write_text("not a completed attempt")
        monkeypatch.setattr(Runtime, "run", lambda *_: pytest.fail("must not rerun agent or user"))
        return source, task, directory

    return make


def _args(source, output, *extra):
    return [
        "--batch",
        str(source),
        "--output",
        str(output),
        "--judge-model",
        "new-judge",
        "--concurrency",
        "1",
        "--retry-delay",
        "0",
        *extra,
    ]


@pytest.mark.parametrize(
    "mode,raw_pass,findings,expected",
    [
        ("least_privilege", True, [True], True),
        ("direct", True, [True], False),
        ("least_privilege", True, [False], False),
        ("least_privilege", False, [True], True),
        ("least_privilege", False, [], False),
        ("least_privilege", True, [], True),
    ],
)
@pytest.mark.parametrize("old_grader", ["rule", "hybrid"])
def test_rejudge_matches_runtime_and_paper_scores(
    recorded_batch,
    tmp_path,
    monkeypatch,
    mode,
    raw_pass,
    findings,
    expected,
    old_grader,
):
    source, task, directory = recorded_batch(mode, old_grader)
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    requests = []
    raw = {
        "process_pass": raw_pass,
        "rationale": "test decision",
        "findings": [
            {
                "rule_id": "semantic-check",
                "passed": passed,
                "severity": "major",
                "seq": 1,
                "evidence": "recorded evidence",
            }
            for passed in findings
        ],
    }

    def judge(client, messages, **kwargs):
        requests.append(messages)
        assert client.config.model == "new-judge"
        assert client.config.base_url == "https://judge.invalid/v1"
        return {"_cxbench": {"model_returned": "new-judge"}}, dict(raw)

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", judge)
    output = tmp_path / "rejudged"
    assert rejudge.main(_args(source, output, "--base-url", "https://judge.invalid/v1")) == 0
    assert len(requests) == 1
    payload = json.loads(requests[0][1]["content"])
    assert payload["trace"] == score._jsonl(directory / "trace.jsonl")
    assert payload["final_state"] == score._json(directory / "final_state.json")
    (row,) = score._jsonl(output / "results.jsonl")
    grade = score._json(Path(row["run_dir"]) / "grade.json")
    assert grade["verdict"] == ("pass" if expected else "fail")
    assert grade["outcome_pass"] is True
    assert grade["grader"] == row["final_grader"] == "hybrid"
    assert row["elapsed_seconds"] == 12.5
    assert row["source_run_dir"] == str(directory)
    assert row["regrade_attempts"] == 1
    assert len(list((output / "task_runs").iterdir())) == 1
    manifest = score._json(output / "manifest.json")
    assert manifest["source_batch"] == str(source)
    assert manifest["source_judge_model"] == "old-judge"
    assert manifest["judge_model"] == "new-judge"
    config = score._json(Path(row["run_dir"]) / "config.json")
    assert config["judge_model"] == "openai:new-judge"
    summary = score._json(output / "paper_scores" / "summary.json")
    assert summary["per_model"]["original-agent"]["both_pass"] == int(expected)
    assert score._json(output / "metrics.json")["both_pass_rate"] == float(expected)
    assert len(score._jsonl(output / "failures.jsonl")) == int(not expected)
    assert original == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }

    monkeypatch.setenv("OPENAI_BASE_URL", "https://judge.invalid/v1")
    expected_grade = grader.grade_run(
        task,
        payload["trace"],
        payload["final_state"],
        mode,
        grader="hybrid",
        judge_model="openai:new-judge",
    )
    assert grade == expected_grade
    assert requests[0] == requests[1]


def test_dry_run_does_not_call_judge_or_write(recorded_batch, tmp_path, monkeypatch):
    source, _, _ = recorded_batch()
    monkeypatch.setattr(resume, "_stage", lambda *a, **k: pytest.fail("dry run must not judge"))
    output = tmp_path / "new"
    assert rejudge.main(_args(source, output, "--dry-run")) == 0
    assert not output.exists()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "hash", "trace", "state", "failed"])
def test_invalid_source_rejected_before_calls_or_writes(
    recorded_batch, tmp_path, monkeypatch, problem
):
    source, _, directory = recorded_batch()
    if problem in {"missing", "duplicate", "failed"}:
        rows = score._jsonl(source / "results.jsonl")
        if problem == "missing":
            rows = []
        elif problem == "duplicate":
            rows *= 2
        else:
            rows[0]["ok"] = False
        (source / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif problem == "hash":
        manifest = score._json(source / "manifest.json")
        manifest["tasks"][0]["sha256"] = "wrong"
        resume._write_json(source / "manifest.json", manifest)
    elif problem == "trace":
        (directory / "trace.jsonl").unlink()
    else:
        (directory / "final_state.json").write_text("invalid")
    monkeypatch.setattr(
        resume, "_stage", lambda *a, **k: pytest.fail("invalid source must not judge")
    )
    output = tmp_path / "new"
    with pytest.raises(SystemExit) as exc:
        rejudge.main(_args(source, output))
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--concurrency", "0"],
        ["--max-output-tokens", "0"],
        ["--retry-delay", "nan"],
        ["--judge-model", ""],
        ["--judge-model", "openai:"],
        ["--judge-provider", "deepseek", "--base-url", "https://example.invalid"],
    ],
)
def test_invalid_options_rejected(recorded_batch, tmp_path, extra):
    source, _, _ = recorded_batch()
    output = tmp_path / "new"
    with pytest.raises(SystemExit):
        rejudge.main(_args(source, output, *extra))
    assert not output.exists()


def test_existing_and_nested_outputs_are_rejected(recorded_batch, tmp_path):
    source, _, _ = recorded_batch()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("keep")
    for output in (source, existing, source / "nested", tmp_path):
        with pytest.raises(SystemExit):
            rejudge.main(_args(source, output))
    assert (existing / "sentinel").read_text() == "keep"
    assert not (source / "nested").exists()


def test_judge_retries_same_trace_and_accepts_valid_failure(recorded_batch, tmp_path, monkeypatch):
    source, _, _ = recorded_batch()
    requests = []

    def judge(client, messages, **kwargs):
        requests.append(messages)
        if len(requests) == 1:
            raise ValueError("transient invalid judge response")
        return {}, {"process_pass": False, "findings": [], "rationale": "failed"}

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", judge)
    output = tmp_path / "new"
    assert rejudge.main(_args(source, output)) == 0
    assert len(requests) == 2
    assert requests[0] == requests[1]
    (row,) = score._jsonl(output / "results.jsonl")
    assert row["regrade_attempts"] == 2
    assert row["verdict"] == "fail"


def test_transport_failure_does_not_publish_final_results(recorded_batch, tmp_path, monkeypatch):
    source, _, _ = recorded_batch()

    def unavailable(*args, **kwargs):
        raise rejudge.batch.ProviderTransportError("connection retries exhausted")

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", unavailable)
    output = tmp_path / "new"
    assert rejudge.main(_args(source, output)) == 2
    assert not (output / "results.jsonl").exists()
    assert not (output / "paper_scores").exists()


@pytest.fixture
def interrupted_rejudge(recorded_batch, tmp_path, monkeypatch):
    import shutil

    import yaml

    source, task, directory = recorded_batch(old_grader="hybrid")
    second_path = tmp_path / "second-task.yaml"
    data = yaml.safe_load(task.path.read_text())
    data["id"] = task.id + "-second"
    second_path.write_text(yaml.safe_dump(data))
    second = TaskSpec.load(second_path)
    second_run = directory.with_name(directory.name + "-second")
    shutil.copytree(directory, second_run)
    config = score._json(second_run / "config.json")
    config.update(task_id=second.id, task_path=str(second_path))
    resume._write_json(second_run / "config.json", config)
    old_grade = score._json(second_run / "grade.json")
    old_grade["task_id"] = second.id
    resume._write_json(second_run / "grade.json", old_grade)
    rows = score._jsonl(source / "results.jsonl")
    rows.append(
        {**rows[0], "task_id": second.id, "task_path": str(second_path), "run_dir": str(second_run)}
    )
    resume._write_results(source / "results.jsonl", {row["task_id"]: row for row in rows})
    manifest = score._json(source / "manifest.json")
    manifest["tasks"].append(
        {"path": str(second_path), "sha256": rejudge.batch._sha256(second_path)}
    )
    manifest["task_count"] = 2
    resume._write_json(source / "manifest.json", manifest)
    calls = []

    def judge(client, messages, **kwargs):
        task_id = json.loads(messages[1]["content"])["task"]["id"]
        calls.append(task_id)
        if task_id == second.id:
            raise rejudge.batch.ProviderTransportError("SSL EOF after 3 attempts")
        return {}, {"process_pass": False, "findings": [], "rationale": "valid compliance failure"}

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", judge)
    output = tmp_path / "rejudged"
    assert rejudge.main(_args(source, output)) == 2
    completed = output / "task_runs" / directory.name
    assert score._json(completed / "grade.json")["verdict"] == "fail"
    assert not (output / "results.jsonl").exists()
    calls.clear()
    return source, output, task, second, completed, calls


def _finish_judge(monkeypatch, calls):
    def judge(client, messages, **kwargs):
        calls.append(json.loads(messages[1]["content"])["task"]["id"])
        return {}, {"process_pass": True, "findings": [], "rationale": "passed"}

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", judge)


def test_resume_only_judges_unfinished_tasks_and_preserves_failures(
    interrupted_rejudge, monkeypatch
):
    source, output, first, second, completed, calls = interrupted_rejudge
    saved_grade = (completed / "grade.json").read_bytes()
    _finish_judge(monkeypatch, calls)
    assert rejudge.main(_args(source, output, "--resume")) == 0
    assert calls == [second.id]
    assert (completed / "grade.json").read_bytes() == saved_grade
    rows = {row["task_id"]: row for row in score._jsonl(output / "results.jsonl")}
    assert rows[first.id]["verdict"] == "fail"
    assert rows[second.id]["verdict"] == "pass"
    assert all(row["elapsed_seconds"] == 12.5 for row in rows.values())
    summary = score._json(output / "paper_scores" / "summary.json")
    assert summary["per_model"]["original-agent"]["completed"] == 2
    assert summary["per_model"]["original-agent"]["both_pass"] == 1
    calls.clear()
    assert rejudge.main(_args(source, output, "--resume")) == 0
    assert calls == []


@pytest.mark.parametrize("metadata", ["legacy", "truncated_config", "missing_config"])
def test_resume_recovers_grade_without_parent_snapshot_or_report(
    interrupted_rejudge, monkeypatch, metadata
):
    source, output, first, second, completed, calls = interrupted_rejudge
    (output / "results_partial.jsonl").write_text('{"task_id":')
    (completed / "report.md").unlink()
    config_path = completed / "config.json"
    if metadata == "legacy":
        original = score._json(source / "task_runs" / completed.name / "config.json")
        resume._write_json(config_path, original)
    elif metadata == "truncated_config":
        config_path.write_text('{"task_id":')
    else:
        config_path.unlink()
    _finish_judge(monkeypatch, calls)
    assert rejudge.main(_args(source, output, "--resume")) == 0
    assert calls == [second.id]
    assert score._json(config_path)["judge_model"] == "openai:new-judge"
    assert (completed / "report.md").is_file()
    assert len(score._jsonl(output / "results.jsonl")) == 2


def test_resume_retries_unfinished_grade_write(interrupted_rejudge, monkeypatch):
    source, output, first, second, completed, calls = interrupted_rejudge
    pending = next(path for path in (output / "task_runs").iterdir() if path != completed)
    (pending / "grade.json").write_text('{"task_id":')
    _finish_judge(monkeypatch, calls)
    assert rejudge.main(_args(source, output, "--resume")) == 0
    assert calls == [second.id]


def test_resume_dry_run_has_no_writes_or_requests(interrupted_rejudge, capsys):
    source, output, first, second, completed, calls = interrupted_rejudge
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert rejudge.main(_args(source, output, "--resume", "--dry-run")) == 0
    assert "reused: 1; to_grade: 1" in capsys.readouterr().out
    assert calls == []
    assert before == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--judge-model", "different-model"], "judge-model differs"),
        (["--judge-provider", "deepseek"], "judge-provider differs"),
        (["--max-output-tokens", "4096"], "max-output-tokens differs"),
    ],
)
def test_resume_rejects_different_judge_configuration(interrupted_rejudge, extra, message, capsys):
    source, output, first, second, completed, calls = interrupted_rejudge
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit) as exc:
        rejudge.main(_args(source, output, "--resume", *extra))
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert calls == []
    assert before == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "problem", ["source", "tasks", "trace", "state", "config", "judge", "grade"]
)
def test_resume_rejects_mismatched_or_invalid_evidence(interrupted_rejudge, problem):
    source, output, first, second, completed, calls = interrupted_rejudge
    manifest = score._json(output / "manifest.json")
    if problem == "source":
        manifest["source_batch"] = str(source.with_name("other"))
    elif problem == "tasks":
        manifest["tasks"][0]["sha256"] = "changed"
    elif problem == "trace":
        with (completed / "trace.jsonl").open("a") as handle:
            handle.write("\n")
    elif problem == "state":
        (completed / "final_state.json").write_text("{}")
    elif problem == "config":
        config = score._json(completed / "config.json")
        config["mode"] = "direct"
        resume._write_json(completed / "config.json", config)
    elif problem == "judge":
        grade = score._json(completed / "grade.json")
        grade["llm_judge"]["findings"] = "invalid"
        resume._write_json(completed / "grade.json", grade)
    else:
        (completed / "grade.json").write_text('{"task_id":')
    resume._write_json(output / "manifest.json", manifest)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit):
        rejudge.main(_args(source, output, "--resume"))
    assert calls == []
    assert before == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_resume_missing_output_is_rejected(recorded_batch, tmp_path):
    source, _, _ = recorded_batch()
    output = tmp_path / "missing"
    with pytest.raises(SystemExit):
        rejudge.main(_args(source, output, "--resume"))
    assert not output.exists()
