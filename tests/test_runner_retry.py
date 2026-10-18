"""Offline retry regressions; rule grading here is test-only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from cxbench import cli, retry, runner
from cxbench.adapters import AgentAdapter


TASK = Path("tasks/CX-01.us.yaml")


@pytest.fixture
def delays(monkeypatch):
    recorded = []
    monkeypatch.setattr(retry.time, "sleep", recorded.append)
    return recorded


def _grade(verdict="fail"):
    return {
        "verdict": verdict,
        "outcome_pass": verdict == "pass",
        "process_pass": verdict == "pass",
        "false_pass_demo": False,
        "grader": "rule",
    }


def _raw_result(verdict="fail"):
    return {"run_dir": "completed", "report": "completed/report.md", "grade": _grade(verdict)}


def test_retries_beyond_old_attempt_limits_and_keeps_valid_failure(delays, capsys):
    attempts = 0
    expected = _raw_result()

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 9:
            raise OSError("temporary provider failure")
        return expected

    assert retry.retry_until_completed(operation, label="test") is expected
    assert attempts == 9
    assert delays == [5.0] * 8
    assert "attempt 8 failed" in capsys.readouterr().err


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_user_cancellation_is_not_retried(exception, delays):
    def operation():
        raise exception()

    with pytest.raises(exception):
        retry.retry_until_completed(operation, label="test")
    assert delays == []


def test_run_eval_restarts_with_fresh_runtime_and_preserves_fail(monkeypatch, delays):
    runtimes = []

    class Runtime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            runtimes.append(self)

        def run(self):
            if len(runtimes) < 3:
                raise TimeoutError("provider timed out")
            return _raw_result()

    monkeypatch.setattr(runner, "Runtime", Runtime)
    result = runner.run_eval(TASK, AgentAdapter(), grader="rule", user_sim="deterministic")

    assert result.verdict == "fail"
    assert result.error is None
    assert len({id(instance) for instance in runtimes}) == 3
    assert delays == [5.0, 5.0]


def test_run_eval_rejects_invalid_configuration_before_retry(delays):
    with pytest.raises(ValueError, match="requires judge_model"):
        runner.run_eval(TASK, AgentAdapter(), user_sim="deterministic")
    assert delays == []


@pytest.mark.parametrize("workers", [1, 2])
def test_suite_retries_execution_errors_without_error_results(
    tmp_path, monkeypatch, delays, workers
):
    for source in (TASK, Path("tasks/CX-02.us.yaml")):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    attempts = Counter()

    class Runtime:
        def __init__(self, task, **kwargs):
            self.task = task

        def run(self):
            attempts[self.task.id] += 1
            if attempts[self.task.id] == 1:
                raise OSError("temporary error")
            return _raw_result("pass" if self.task.id == "CX-01" else "fail")

    monkeypatch.setattr(runner, "Runtime", Runtime)
    result = runner.run_suite(
        tmp_path, AgentAdapter(), workers=workers, grader="rule", user_sim="deterministic"
    )

    assert sorted(attempts.values()) == [2, 2]
    assert result["total"] == 2
    assert result["errors"] == 0
    assert all(row["verdict"] in {"pass", "fail"} for row in result["results"])


def test_cli_run_retries_execution_errors(monkeypatch, delays, tmp_path):
    attempts = []

    class Runtime:
        def __init__(self, **kwargs):
            attempts.append(self)

        def run(self):
            if len(attempts) == 1:
                raise OSError("temporary error")
            return _raw_result()

    monkeypatch.setattr(cli, "Runtime", Runtime)
    args = SimpleNamespace(
        task=TASK,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        judge_model=None,
        user_sim="deterministic",
        user_model=None,
        user_max_attempts=3,
    )

    assert cli._run(args) == 0
    assert len(attempts) == 2
    assert delays == [5.0]


def test_cli_grade_retries_without_running_agent(monkeypatch, delays, tmp_path):
    config = {"task_path": str(TASK), "mode": "least_privilege", "grader": "rule"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "trace.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "final_state.json").write_text("{}", encoding="utf-8")
    calls = []

    def write_grade(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("temporary judge error")
        return _grade()

    monkeypatch.setattr(cli, "write_grade", write_grade)
    monkeypatch.setattr(cli, "write_report", lambda *args: tmp_path / "report.md")
    assert cli._grade(SimpleNamespace(run=tmp_path, grader=None, judge_model=None)) == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert delays == [5.0]


def test_cli_grade_rejects_missing_judge_before_retry(delays, tmp_path):
    config = {"task_path": str(TASK), "mode": "least_privilege", "grader": "hybrid"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert cli._grade(SimpleNamespace(run=tmp_path, grader=None, judge_model=None)) == 2
    assert delays == []


def test_retry_stops_when_suite_cancels_during_failure(delays):
    stop_event = Event()
    calls = []

    def operation():
        calls.append("attempt")
        stop_event.set()
        raise OSError("provider failure")

    with pytest.raises(InterruptedError, match="cancelled"):
        retry.retry_until_completed(operation, label="test", stop_event=stop_event)
    assert calls == ["attempt"]
    assert delays == []


def test_cancelled_retry_does_not_start_another_attempt(delays):
    stop_event = Event()
    stop_event.set()
    calls = []

    with pytest.raises(InterruptedError, match="cancelled"):
        retry.retry_until_completed(
            lambda: calls.append("attempt"), label="test", stop_event=stop_event
        )
    assert calls == []


def test_suite_signals_all_retries_when_interrupted(tmp_path, monkeypatch):
    (tmp_path / TASK.name).write_text(TASK.read_text(encoding="utf-8"), encoding="utf-8")
    events = []

    def interrupted_run_eval(*args, _stop_event, **kwargs):
        events.append(_stop_event)
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "run_eval", interrupted_run_eval)
    with pytest.raises(KeyboardInterrupt):
        runner.run_suite(tmp_path, AgentAdapter(), workers=2)

    assert len(events) == 1
    assert events[0].is_set()


def test_exhausted_transport_retries_are_not_restarted_by_the_task_loop(delays):
    from cxbench.openai_compat import ProviderTransportError

    calls = []

    def operation():
        calls.append(1)
        raise ProviderTransportError("SSL request retries exhausted")

    with pytest.raises(ProviderTransportError):
        retry.retry_until_completed(operation, label="test")
    assert calls == [1]
    assert delays == []
