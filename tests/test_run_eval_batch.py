from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from scripts import run_eval_batch as batch


def _args(**overrides):
    return argparse.Namespace(
        **{
            "mode": "least_privilege",
            "model": "offline-test",
            "judge_model": "offline-judge",
            "user_sim": "llm",
            "user_model": "offline-user",
            "user_max_attempts": 3,
            "task_timeout": 1,
            **overrides,
        }
    )


def _hybrid_grade():
    return {
        "task_id": "task-1",
        "mode": "least_privilege",
        "grader": "hybrid",
        "verdict": "fail",
        "outcome_pass": True,
        "process_pass": False,
        "false_pass_demo": True,
        "rules": [],
        "llm_judge": {"process_pass": False, "rationale": "Semantic violation."},
    }


def _collect(worker, arguments, *, stage="agent", retry_delay=0):
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(
            batch._completed_results(pool, worker, arguments, retry_delay=retry_delay, stage=stage)
        )


def _process_retry_worker(state_path):
    attempt = int(state_path.read_text()) + 1 if state_path.exists() else 1
    state_path.write_text(str(attempt))
    return {
        "task_id": "process-task",
        "ok": attempt > 3,
        "verdict": "fail" if attempt > 3 else "error",
    }


def test_execution_errors_and_timeouts_retry_until_a_valid_fail(monkeypatch, tmp_path):
    task = SimpleNamespace(id="task-1", domain="banking", scenario="test")
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: task)
    attempts = []

    def run():
        attempts.append(1)
        if len(attempts) < 5:
            raise ConnectionError("temporary provider failure")
        if len(attempts) < 7:
            raise batch.RuntimeTimeoutError("temporary timeout")
        return {
            "run_dir": str(tmp_path),
            "grade": {"verdict": "fail", "outcome_pass": False, "process_pass": False},
        }

    monkeypatch.setattr(batch, "Runtime", lambda **_: SimpleNamespace(run=run))
    results = _collect(batch._run_one, [(tmp_path / "task.yaml", _args(), tmp_path, "rule")])

    assert len(attempts) == 7
    assert len(results) == 1
    assert results[0]["attempts"] == 7
    assert results[0]["ok"] is True
    assert results[0]["verdict"] == "fail"
    assert results[0]["outcome_pass"] is False
    assert results[0]["process_pass"] is False
    assert results[0]["error"] is None


def test_judge_retries_keep_the_original_trace(monkeypatch, tmp_path):
    trace = '{"event_type": "assistant_message"}\n'
    (tmp_path / "trace.jsonl").write_text(trace)
    (tmp_path / "config.json").write_text(
        json.dumps({"task_path": "task.yaml", "mode": "least_privilege", "grader": "rule"})
    )
    task = object()
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: task)
    seen = []

    def grade(run_dir, loaded_task, mode, **kwargs):
        seen.append(run_dir)
        assert loaded_task is task
        assert (run_dir / "trace.jsonl").read_text() == trace
        if len(seen) <= 5:
            raise ConnectionError("judge unavailable")
        final_grade = _hybrid_grade()
        (run_dir / "grade.json").write_text(json.dumps(final_grade))
        return final_grade

    monkeypatch.setattr(batch, "write_grade", grade)
    original = {
        "task_id": "task-1",
        "ok": True,
        "run_dir": str(tmp_path),
        "error": None,
        "attempts": 2,
    }
    results = _collect(batch._regrade_one, [(original, _args(grader="hybrid"))], stage="judge")

    assert seen == [tmp_path] * 6
    assert original["ok"] is True
    assert results[0]["ok"] is True
    assert results[0]["verdict"] == "fail"
    assert results[0]["attempts"] == 2
    assert results[0]["regrade_attempts"] == 6
    assert results[0]["final_grader"] == "hybrid"
    assert json.loads((tmp_path / "config.json").read_text())["grader"] == "hybrid"


def test_retries_only_failed_tasks_and_honors_delay(monkeypatch):
    attempts = {"ready": 0, "retry": 0}
    delays = []
    monkeypatch.setattr(batch.time, "sleep", delays.append)

    def worker(name):
        attempts[name] += 1
        if name == "retry" and attempts[name] < 4:
            raise ConnectionError("temporary failure")
        return {"task_id": name, "ok": True, "verdict": "fail"}

    results = _collect(worker, [("ready",), ("retry",)], retry_delay=0.25)

    assert attempts == {"ready": 1, "retry": 4}
    assert delays == [0.25] * 3
    assert len(results) == 2


def test_process_workers_retry_without_network(tmp_path):
    state_path = tmp_path / "attempts.txt"
    with ProcessPoolExecutor(max_workers=1) as pool:
        results = list(
            batch._completed_results(
                pool, _process_retry_worker, [(state_path,)], retry_delay=0, stage="agent"
            )
        )

    assert state_path.read_text() == "4"
    assert results == [{"task_id": "process-task", "ok": True, "verdict": "fail", "attempts": 4}]


def test_keyboard_interrupt_stops_retrying():
    attempts = []

    def worker(name):
        attempts.append(name)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _collect(worker, [("task",)])
    assert attempts == ["task"]


@pytest.mark.parametrize("delay", ["-1", "nan", "inf"])
def test_main_rejects_invalid_retry_delay(delay):
    with pytest.raises(SystemExit) as exc:
        batch.main(
            [
                "--task",
                "unused.yaml",
                "--model",
                "test",
                "--user-model",
                "test",
                f"--retry-delay={delay}",
            ]
        )
    assert exc.value.code == 2


def test_default_task_can_be_overridden_without_appending():
    parser = batch.build_parser(default_task="all/*.yaml", default_runs_dir="runs/main")
    args = parser.parse_args(["--model", "test", "--user-model", "test"])
    explicit = parser.parse_args(["--model", "test", "--user-model", "test", "--task", "one.yaml"])
    assert args.task is None
    assert args.default_task == "all/*.yaml"
    assert args.runs_dir == "runs/main"
    assert explicit.task == ["one.yaml"]


def test_main_writes_only_completed_results_and_restores_provider(monkeypatch, tmp_path):
    (tmp_path / "task.yaml").write_text("test fixture")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: None)
    restored = []
    monkeypatch.setattr(batch, "apply_provider_env", lambda *_: {"test": "previous"})
    monkeypatch.setattr(batch, "restore_provider_env", restored.append)
    attempts = []

    def worker(*_):
        attempts.append(1)
        if len(attempts) < 3:
            return {"ok": False, "task_id": "task-1", "verdict": "error", "error": "temporary"}
        return {
            "ok": True,
            "task_id": "task-1",
            "domain": "banking",
            "verdict": "fail",
            "outcome_pass": True,
            "process_pass": False,
        }

    monkeypatch.setattr(batch, "_run_one", worker)
    code = batch.main(
        [
            "--task",
            "task.yaml",
            "--model",
            "test",
            "--user-model",
            "test",
            "--grader",
            "rule",
            "--concurrency",
            "1",
            "--retry-delay",
            "0",
            "--run-id",
            "retry-test",
        ]
    )
    output = tmp_path / "runs/eval_batches/retry-test"
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    partial = [
        json.loads(line) for line in (output / "results_partial.jsonl").read_text().splitlines()
    ]
    metrics = json.loads((output / "metrics.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())

    assert code == 0
    assert rows == partial
    assert len(rows) == 1
    assert rows[0]["attempts"] == 3
    assert rows[0]["verdict"] == "fail"
    assert metrics["error_count"] == 0
    assert metrics["process_pass_rate"] == 0
    assert manifest["retry_delay"] == 0
    assert restored == [{"test": "previous"}]


def test_invalid_task_fails_before_creating_results(monkeypatch, tmp_path):
    (tmp_path / "invalid.yaml").write_text("not: a-valid-task\n")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        batch.main(
            [
                "--task",
                "invalid.yaml",
                "--model",
                "test",
                "--user-model",
                "test",
                "--run-id",
                "invalid",
            ]
        )
    assert not (tmp_path / "runs").exists()


def test_same_provider_hybrid_retries_judge_without_rerunning_agent(monkeypatch, tmp_path):
    (tmp_path / "task.yaml").write_text("test fixture")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: object())
    providers = []
    restored = []

    def apply_provider(provider, _):
        providers.append(provider)
        return {"provider": provider}

    monkeypatch.setattr(batch, "apply_provider_env", apply_provider)
    monkeypatch.setattr(batch, "restore_provider_env", restored.append)
    agent_calls = []
    judge_calls = []

    def worker(task_path, args, runs_dir, runtime_grader):
        agent_calls.append(task_path)
        assert runtime_grader == "rule"
        run_dir = runs_dir / "task-1"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps({"task_path": str(task_path), "mode": args.mode, "grader": runtime_grader})
        )
        (run_dir / "trace.jsonl").write_text("unchanged trace")
        (run_dir / "report.md").write_text("- Grader: `rule`\n- Verdict: `pass`\n")
        return {
            "ok": True,
            "task_id": "task-1",
            "domain": "banking",
            "run_dir": str(run_dir),
            "verdict": "pass",
            "outcome_pass": True,
            "process_pass": True,
            "error": None,
        }

    def grade(run_dir, *_args, **_kwargs):
        judge_calls.append(run_dir)
        assert (run_dir / "trace.jsonl").read_text() == "unchanged trace"
        if len(judge_calls) < 5:
            raise ConnectionError("judge temporarily unavailable")
        final_grade = _hybrid_grade()
        (run_dir / "grade.json").write_text(json.dumps(final_grade))
        return final_grade

    monkeypatch.setattr(batch, "_run_one", worker)
    monkeypatch.setattr(batch, "write_grade", grade)
    code = batch.main(
        [
            "--task",
            "task.yaml",
            "--model",
            "test",
            "--user-model",
            "test",
            "--judge-provider",
            "same",
            "--concurrency",
            "1",
            "--retry-delay",
            "0",
            "--run-id",
            "same-provider",
        ]
    )
    output = tmp_path / "runs/eval_batches/same-provider"
    manifest = json.loads((output / "manifest.json").read_text())
    result = json.loads((output / "results.jsonl").read_text())

    assert code == 0
    assert len(agent_calls) == 1
    assert len(judge_calls) == 5
    assert len(set(judge_calls)) == 1
    assert providers == ["openai", "openai"]
    assert restored == [{"provider": "openai"}, {"provider": "openai"}]
    assert manifest["execution_grader"] == result["execution_grader"] == "rule"
    assert manifest["final_grader"] == result["final_grader"] == "hybrid"
    assert result["ok"] is True
    assert result["verdict"] == "fail"
    assert result["attempts"] == 1
    assert result["regrade_attempts"] == 5

    run_dir = judge_calls[0]
    final_grade = json.loads((run_dir / "grade.json").read_text())
    report = (run_dir / "report.md").read_text()
    assert f"- Grader: `{final_grade['grader']}`" in report
    assert f"- Verdict: `{final_grade['verdict']}`" in report
    assert "- Process pass: `False`" in report
    assert "- Grader: `rule`" not in report
    assert "Semantic violation." in report


def test_report_write_errors_retry_before_publishing_completion(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"task_path": "task.yaml", "mode": "least_privilege", "grader": "rule"})
    )
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: object())
    monkeypatch.setattr(batch, "write_grade", lambda *_args, **_kwargs: _hybrid_grade())
    real_write_report = batch.write_report
    attempts = []

    def flaky_write_report(run_dir, grade):
        attempts.append(run_dir)
        if len(attempts) == 1:
            raise OSError("temporary report write failure")
        return real_write_report(run_dir, grade)

    monkeypatch.setattr(batch, "write_report", flaky_write_report)
    original = {"task_id": "task-1", "ok": True, "run_dir": str(tmp_path)}
    results = _collect(batch._regrade_one, [(original, _args(grader="hybrid"))], stage="judge")

    assert len(attempts) == 2
    assert results[0]["ok"] is True
    assert results[0]["regrade_attempts"] == 2
    assert results[0]["execution_grader"] == "rule"
    assert "- Grader: `hybrid`" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("transport_failure", [False, True])
def test_failed_attempt_is_archived_and_completed_runs_are_preserved(
    monkeypatch, tmp_path, transport_failure
):
    import tarfile

    task = SimpleNamespace(id="task-1", domain="banking", scenario="test")
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: task)
    runs_dir = tmp_path / "task_runs"
    existing = runs_dir / "already-completed"
    existing.mkdir(parents=True)
    (existing / "report.md").write_text("keep this completed result")
    attempts = []

    class ArtifactRuntime:
        def __init__(self, **kwargs):
            self.run_dir = kwargs["runs_dir"] / f"attempt-{len(attempts) + 1}"
            self.run_dir.mkdir()
            (self.run_dir / "drafts").mkdir()
            (self.run_dir / "config.json").write_text('{"task_id": "task-1"}')
            self.trace = [{"event_type": "user_message", "payload": {"text": "kept in archive"}}]
            attempts.append(self)

        def _write_trace(self):
            (self.run_dir / "trace.jsonl").write_text(json.dumps(self.trace[0]) + "\n")

        def run(self):
            if len(attempts) == 1:
                if transport_failure:
                    raise batch.ProviderTransportError("SSL request retries exhausted")
                raise ConnectionError("temporary task failure")
            self._write_trace()
            (self.run_dir / "report.md").write_text("valid failing result")
            return {
                "run_dir": str(self.run_dir),
                "grade": {
                    "verdict": "fail",
                    "outcome_pass": False,
                    "process_pass": False,
                },
            }

    monkeypatch.setattr(batch, "Runtime", ArtifactRuntime)
    arguments = [(tmp_path / "task.yaml", _args(), runs_dir, "rule")]
    if transport_failure:
        with pytest.raises(batch.ProviderTransportError):
            _collect(batch._run_one, arguments)
        assert len(attempts) == 1
    else:
        rows = _collect(batch._run_one, arguments)
        assert len(attempts) == 2
        assert rows[0]["verdict"] == "fail"
        assert not (attempts[1].run_dir / "drafts").exists()
    assert not attempts[0].run_dir.exists()
    assert (existing / "report.md").read_text() == "keep this completed result"
    archives = list((tmp_path / "failed_attempts").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        assert json.load(archive.extractfile("attempt-1/config.json"))["task_id"] == "task-1"
        assert b"kept in archive" in archive.extractfile("attempt-1/trace.jsonl").read()
        error = json.load(archive.extractfile("attempt-1/error.json"))
        assert error["type"] == (
            "ProviderTransportError" if transport_failure else "ConnectionError"
        )


def test_archive_failure_keeps_original_artifacts(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "task_runs" / "failed"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("original config")
    runtime = SimpleNamespace(run_dir=run_dir, trace=[])

    def archive_unavailable(*args, **kwargs):
        raise OSError("archive storage unavailable")

    monkeypatch.setattr(batch.tarfile, "open", archive_unavailable)
    batch._finish_attempt(runtime, run_dir.parent, batch.ProviderTransportError("SSL EOF"))
    assert (run_dir / "config.json").read_text() == "original config"
    assert "failed attempt retained" in capsys.readouterr().err


def test_judge_transport_failure_does_not_restart_or_remove_the_completed_trace(
    monkeypatch, tmp_path
):
    trace = '{"event_type":"assistant_message"}\n'
    (tmp_path / "trace.jsonl").write_text(trace)
    (tmp_path / "config.json").write_text(
        json.dumps({"task_path": "task.yaml", "mode": "least_privilege"})
    )
    monkeypatch.setattr(batch.TaskSpec, "load", lambda _: object())
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise batch.ProviderTransportError("SSL request retries exhausted")

    monkeypatch.setattr(batch, "write_grade", unavailable)
    original = {"task_id": "task-1", "ok": True, "run_dir": str(tmp_path)}
    with pytest.raises(batch.ProviderTransportError):
        _collect(batch._regrade_one, [(original, _args(grader="hybrid"))], stage="judge")
    assert calls == [1]
    assert (tmp_path / "trace.jsonl").read_text() == trace
    assert original["ok"] is True


def _process_transport_failure():
    raise batch.ProviderTransportError("SSL request retries exhausted")


def test_exhausted_transport_error_propagates_from_process_worker():
    with ProcessPoolExecutor(max_workers=1) as pool:
        with pytest.raises(batch.ProviderTransportError, match="SSL"):
            list(
                batch._completed_results(
                    pool,
                    _process_transport_failure,
                    [()],
                    retry_delay=0,
                    stage="agent",
                )
            )


def test_batch_cli_reports_exhausted_transport_without_retrying(monkeypatch, capsys):
    def unavailable(_):
        raise batch.ProviderTransportError("SSL request retries exhausted")

    monkeypatch.setattr(batch, "run_batch", unavailable)
    assert batch.main(["--task", "unused.yaml", "--model", "test", "--user-model", "test"]) == 2
    assert "batch paused" in capsys.readouterr().out
