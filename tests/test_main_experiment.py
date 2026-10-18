from __future__ import annotations

import json
import os

import pytest

from main_experiment import run


def _args(tmp_path, *extra):
    return [
        "--model",
        "test-agent",
        "--user-model",
        "test-user",
        "--runs-dir",
        str(tmp_path / "runs"),
        "--run-id",
        "test",
        *extra,
    ]


def test_dry_run_selects_published_corpus_without_writes_or_api(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("dry-run must not execute or score")

    monkeypatch.setattr(run.batch, "run_batch", unexpected)
    monkeypatch.setattr(run, "score_batches", unexpected)
    assert run.main(_args(tmp_path, "--dry-run")) == 0
    assert "tasks: 120" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()


def test_explicit_task_replaces_default_corpus(tmp_path, capsys):
    assert (
        run.main(
            _args(
                tmp_path,
                "--task",
                "benchmarks/cxbench-120-v20260708/tasks/BK-RAW-001.us.yaml",
                "--dry-run",
            )
        )
        == 0
    )
    assert "tasks: 1\n" in capsys.readouterr().out


def test_run_invokes_batch_then_offline_scoring_and_restores_environment(tmp_path, monkeypatch):
    order = []
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "999")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://original.invalid/v1")

    def execute(args):
        order.append("execute")
        assert args.grader == "hybrid"
        assert args.mode == "least_privilege"
        assert os.environ["OPENAI_MAX_OUTPUT_TOKENS"] == "8192"
        assert os.environ["OPENAI_BASE_URL"] == "https://example.invalid/v1"
        assert args.retry_delay == 0
        return 0

    def score(batches, output):
        order.append("score")
        assert batches == {"test-agent": tmp_path / "runs" / "test"}
        assert output == tmp_path / "runs" / "test" / "paper_scores"
        assert os.environ["OPENAI_MAX_OUTPUT_TOKENS"] == "999"
        return {}

    monkeypatch.setattr(run.batch, "run_batch", execute)
    monkeypatch.setattr(run, "score_batches", score)
    assert (
        run.main(
            _args(
                tmp_path,
                "--base-url",
                "https://example.invalid/v1",
                "--retry-delay",
                "0",
            )
        )
        == 0
    )
    assert order == ["execute", "score"]
    assert os.environ["OPENAI_BASE_URL"] == "https://original.invalid/v1"


def test_diagnostic_grader_does_not_publish_formal_scores(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run.batch, "run_batch", lambda args: 0)

    def unexpected(*args, **kwargs):
        pytest.fail("rule-only runs must not produce formal scores")

    monkeypatch.setattr(run, "score_batches", unexpected)
    assert run.main(_args(tmp_path, "--grader", "rule")) == 0
    assert "no formal paper scores" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option,value",
    [
        ("--retry-delay", "-1"),
        ("--concurrency", "0"),
        ("--max-output-tokens", "0"),
        ("--task-timeout", "0"),
    ],
)
def test_invalid_options_rejected_before_writes(tmp_path, option, value):
    with pytest.raises(SystemExit) as exc:
        run.main(_args(tmp_path, option, value, "--dry-run"))
    assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "runs" / "test"
    output.mkdir(parents=True)
    sentinel = output / "manifest.json"
    sentinel.write_text(json.dumps({"existing": True}))
    with pytest.raises(SystemExit) as exc:
        run.main(_args(tmp_path))
    assert exc.value.code == 2
    assert json.loads(sentinel.read_text()) == {"existing": True}


def test_score_output_cannot_replace_the_batch(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid output must be rejected before running the batch")

    monkeypatch.setattr(run.batch, "run_batch", unexpected)
    with pytest.raises(SystemExit) as exc:
        run.main(_args(tmp_path, "--score-output", str(tmp_path / "runs" / "test")))
    assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_resume_preserves_recovery_state_when_transport_retries_are_exhausted(
    monkeypatch, tmp_path, capsys
):
    from types import SimpleNamespace
    from main_experiment import resume

    out_dir = tmp_path / "runs" / "test"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "results_partial.jsonl"
    sentinel.write_text("original completed results")
    tasks = {
        "completed": SimpleNamespace(path=tmp_path / "completed.yaml"),
        "missing": SimpleNamespace(path=tmp_path / "missing.yaml"),
    }
    completed = {"completed": {"final_grader": "hybrid", "run_dir": "completed"}}
    monkeypatch.setattr(resume, "_validate", lambda *_: ({}, tasks, out_dir / "paper_scores"))
    monkeypatch.setattr(resume, "_recover", lambda *_: (completed, {}))
    calls = []

    def unavailable(args, directory, results, arguments, *, stage):
        calls.append(stage)
        assert directory == out_dir and results is completed
        assert [item[0] for item in arguments] == [tasks["missing"].path]
        raise run.batch.ProviderTransportError("SSL request retries exhausted")

    monkeypatch.setattr(resume, "_stage", unavailable)
    monkeypatch.setattr(
        resume, "_finish", lambda *_: pytest.fail("must not publish incomplete results")
    )
    assert resume.main(_args(tmp_path)) == 2
    assert calls == ["agent"]
    assert sentinel.read_text() == "original completed results"
    assert "tasks: 2; reused: 1; to_run: 1; to_grade: 1" in capsys.readouterr().out
