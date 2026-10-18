from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from model_eval.analysis import _write_summary, build_analysis
from model_eval.benchmark import (
    DEFAULT_CONCURRENCY,
    DEFAULT_EXECUTOR as BENCHMARK_DEFAULT_EXECUTOR,
    build_batch_command as build_benchmark_command,
    build_child_environment as build_benchmark_environment,
    build_parser as build_benchmark_parser,
    resolve_config as resolve_benchmark_config,
)
from model_eval.reporting import build_model_report, write_model_report
from model_eval.resume import _provider_environment
from model_eval.run import (
    DEFAULT_EXECUTOR,
    REPO_ROOT,
    Endpoint,
    LaunchConfig,
    build_batch_command,
    build_child_environment,
    build_parser,
    resolve_config,
)
from scripts import run_eval_batch


def test_primary_benchmark_is_wsl_first_process_eval(monkeypatch, tmp_path: Path) -> None:
    for name in (
        "CXBENCH_MODEL",
        "CXBENCH_BASE_URL",
        "CXBENCH_CONCURRENCY",
        "CXBENCH_EXECUTOR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CXBENCH_API_KEY", "test-key")
    monkeypatch.setenv("CXBENCH_WIRE_API", "responses")
    monkeypatch.setenv("CXBENCH_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    args = build_benchmark_parser().parse_args(
        ["--env-file", str(tmp_path / "missing.env"), "--limit", "1"]
    )

    config = resolve_benchmark_config(args)
    command = build_benchmark_command(config)
    environment = build_benchmark_environment(config)

    assert config.mode == "least_privilege"
    assert config.grader == "hybrid"
    assert config.executor == BENCHMARK_DEFAULT_EXECUTOR == "process"
    assert config.concurrency == DEFAULT_CONCURRENCY == 20
    assert command[command.index("--executor") + 1] == "process"
    assert environment["OPENAI_WIRE_API"] == "chat"
    assert "OPENAI_REASONING_EFFORT" not in environment
    assert "CXBENCH_REASONING_EFFORT" not in environment


def test_model_eval_defaults_to_process_workers() -> None:
    task = REPO_ROOT / "benchmarks/cxbench-120/tasks/BK-RAW-001.us.yaml"
    config = LaunchConfig(
        api_key="test-key",
        model="gpt-test",
        endpoint=Endpoint(base_url="https://api.openai.com/v1", wire_api="chat"),
        tasks=(task,),
        grader="hybrid",
        concurrency=32,
        task_timeout=600,
        run_id="test-run",
    )

    command = build_batch_command(config, "mediate")
    executor_index = command.index("--executor")
    concurrency_index = command.index("--concurrency")

    assert DEFAULT_EXECUTOR == "process"
    assert command[executor_index + 1] == "process"
    assert command[concurrency_index + 1] == "32"


def test_mode_ablation_uses_chat_without_reasoning_effort(monkeypatch, tmp_path: Path) -> None:
    for name in (
        "CXBENCH_MODEL",
        "CXBENCH_BASE_URL",
        "CXBENCH_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CXBENCH_API_KEY", "test-key")
    monkeypatch.setenv("CXBENCH_WIRE_API", "responses")
    monkeypatch.setenv("CXBENCH_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "xhigh")
    args = build_parser().parse_args(["--env-file", str(tmp_path / "missing.env"), "--limit", "1"])

    config = resolve_config(args)
    child_environment = build_child_environment(config)

    assert config.endpoint.wire_api == "chat"
    assert child_environment["OPENAI_WIRE_API"] == "chat"
    assert "OPENAI_REASONING_EFFORT" not in child_environment
    assert "CXBENCH_REASONING_EFFORT" not in child_environment


def test_chat_resume_clears_inherited_reasoning_effort(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    launcher = {
        "base_url": "https://api.openai.com/v1",
        "wire_api": "chat",
        "max_output_tokens": 8192,
    }

    with _provider_environment(
        launcher,
        env_file=tmp_path / "missing.env",
        api_key="test-key",
        transport_retries=2,
        retry_backoff_seconds=0.0,
    ):
        assert "OPENAI_REASONING_EFFORT" not in os.environ
        assert os.environ["OPENAI_WIRE_API"] == "chat"

    assert os.environ["OPENAI_REASONING_EFFORT"] == "high"


def test_domain_report_contains_all_three_pass_metrics(tmp_path: Path) -> None:
    tasks = [
        {"task_id": "bank-1", "domain": "banking", "task_path": "bank-1.yaml"},
        {"task_id": "bank-2", "domain": "banking", "task_path": "bank-2.yaml"},
    ]
    mediate = [
        {
            "task_id": "bank-1",
            "domain": "banking",
            "ok": True,
            "outcome_pass": True,
            "process_pass": True,
            "verdict": "pass",
        },
        {
            "task_id": "bank-2",
            "domain": "banking",
            "ok": True,
            "outcome_pass": True,
            "process_pass": False,
            "verdict": "fail",
        },
    ]
    direct = [dict(row) for row in mediate]

    analysis = build_analysis(
        experiment_dir=tmp_path,
        tasks=tasks,
        mode_results={"mediate": mediate, "direct": direct},
        model="gpt-test",
        grader="hybrid",
        benchmark="cxbench-test",
    )
    domain = next(row for row in analysis["domain_metrics"] if row["mode"] == "mediate")

    assert domain["both_pass_rate"] == 0.5
    assert domain["outcome_pass_rate"] == 1.0
    assert domain["process_pass_rate"] == 0.5

    summary_path = tmp_path / "summary.md"
    _write_summary(summary_path, analysis)
    summary = summary_path.read_text(encoding="utf-8")
    assert "| mode | domain | completed | both-pass | outcome-pass | process-pass |" in summary


def test_single_model_report_uses_all_scheduled_tasks_as_denominator() -> None:
    results = [
        {
            "task_id": "bank-1",
            "domain": "banking",
            "ok": True,
            "outcome_pass": True,
            "process_pass": True,
        },
        {
            "task_id": "bank-2",
            "domain": "banking",
            "ok": True,
            "outcome_pass": True,
            "process_pass": False,
        },
        {
            "task_id": "health-1",
            "domain": "healthcare",
            "ok": False,
            "outcome_pass": False,
            "process_pass": False,
            "verdict": "error",
        },
    ]
    report = build_model_report(
        {
            "benchmark": "cxbench-test",
            "run_id": "run-1",
            "model": "gpt-test",
            "wire_api": "chat",
            "mode": "least_privilege",
            "grader": "hybrid",
        },
        results,
    )

    banking = next(row for row in report["domains"] if row["domain"] == "banking")
    healthcare = next(row for row in report["domains"] if row["domain"] == "healthcare")

    assert banking["both_pass_rate"] == 0.5
    assert banking["outcome_pass_rate"] == 1.0
    assert banking["process_pass_rate"] == 0.5
    assert healthcare["n"] == 1
    assert healthcare["error_count"] == 1
    assert healthcare["both_pass_rate"] == 0.0


def test_single_model_report_writes_markdown_json_and_csv(tmp_path: Path) -> None:
    manifest = {
        "benchmark": "cxbench-test",
        "run_id": "run-1",
        "model": "gpt-test",
        "wire_api": "chat",
        "mode": "least_privilege",
        "grader": "hybrid",
        "executor": "process",
        "concurrency": 20,
    }
    result = {
        "task_id": "bank-1",
        "domain": "banking",
        "ok": True,
        "outcome_pass": True,
        "process_pass": True,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    write_model_report(tmp_path)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "- Wire API: `chat`" in markdown
    assert "- Reasoning effort: `not set`" in markdown
    assert "| domain | completed | both-pass | outcome-pass | process-pass | errors |" in markdown
    assert "| banking | 1/1 | 1/1 (100.00%) | 1/1 (100.00%) | 1/1 (100.00%) | 0 |" in markdown
    assert report["wire_api"] == "chat"
    assert report["reasoning_effort"] is None
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "domain_metrics.csv").is_file()


def test_batch_process_executor_rejects_invalid_task_before_running(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "invalid-task.yaml").write_text("not: a-valid-task\n", encoding="utf-8")
    monkeypatch.setattr(run_eval_batch, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_batch.py",
            "--task",
            "invalid-task.yaml",
            "--model",
            "gpt-test",
            "--user-model",
            "gpt-test",
            "--grader",
            "rule",
            "--executor",
            "process",
            "--concurrency",
            "2",
            "--runs-dir",
            "runs",
            "--run-id",
            "process-smoke",
        ],
    )

    with pytest.raises(ValueError):
        run_eval_batch.main()

    assert not (tmp_path / "runs" / "process-smoke").exists()
