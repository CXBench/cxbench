"""Exercise the public release paths without credentials or model calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from cxbench import benchmark_generation as generation
from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec
from main_experiment.run import DEFAULT_TASK
from model_eval.run import BENCHMARK_ID

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path", sorted((ROOT / "data/generation_examples").glob("*.yaml")), ids=lambda path: path.stem
)
def test_generation_domain_examples_execute(path, tmp_path):
    result = Runtime(
        TaskSpec.load(path),
        mode="least_privilege",
        model="scripted:compliant",
        user_sim="deterministic",
        grader="rule",
        runs_dir=tmp_path,
    ).run()
    assert result["grade"]["outcome_pass"]
    assert result["grade"]["process_pass"]


def test_release_defaults_select_revised_six_domain_corpus():
    assert BENCHMARK_ID in DEFAULT_TASK
    paths = sorted(ROOT.glob(DEFAULT_TASK))
    assert len(paths) == 120
    assert len({TaskSpec.load(path).domain for path in paths}) == 6
    assert all("judge-v20260917" in str(path) for path in paths)


def test_public_generation_fixture_accepts_and_executes(tmp_path):
    run = tmp_path / "generation"
    validation = generation.generate_benchmark(
        ROOT / "configs/benchmark_generation/smoke.yaml", run
    )
    assert validation["valid"]
    accepted = generation.accept_candidates(run, tmp_path / "accepted")
    assert len(accepted["accepted"]) == 1
    task = TaskSpec.load(accepted["accepted"][0])
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        user_sim="deterministic",
        grader="rule",
        runs_dir=tmp_path / "runs",
    ).run()
    assert result["grade"]["outcome_pass"]
    assert result["grade"]["process_pass"]


def test_local_generation_rejects_incomplete_scenarios_without_writing(tmp_path):
    output = tmp_path / "generation"
    with pytest.raises(ValueError, match="requires a complete 'task'"):
        generation.generate_benchmark(ROOT / "configs/benchmark_generation/banking.yaml", output)
    assert not output.exists()


def test_generation_refuses_overwrite_and_stale_acceptance(tmp_path):
    output = tmp_path / "generation"
    config = ROOT / "configs/benchmark_generation/smoke.yaml"
    generation.generate_benchmark(config, output)
    with pytest.raises(FileExistsError, match="not empty"):
        generation.generate_benchmark(config, output)
    candidate = next((output / "candidates").glob("*.yaml"))
    raw = yaml.safe_load(candidate.read_text())
    raw["tools"]["submit"] = ["invented_submit"]
    candidate.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="changed since validation"):
        generation.accept_candidates(output, tmp_path / "accepted")
    validation = generation.validate_candidate_run(output)
    assert not validation["valid"]
    assert any("invented_submit" in error for error in validation["candidates"][0]["errors"])


def test_codex_executor_uses_public_cli_and_prompt_stdin(tmp_path, monkeypatch):
    (tmp_path / "prompt.txt").write_text("Generate a candidate.")
    monkeypatch.setattr(generation.shutil, "which", lambda _: "/opt/bin/codex")
    calls = []
    monkeypatch.setattr(
        generation.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    generation._run_codex_worker(tmp_path)
    args, kwargs = calls[0]
    assert args[0][:2] == ["/opt/bin/codex", "exec"]
    assert args[0][-1] == "-"
    assert "--skip-git-repo-check" in args[0]
    assert kwargs["input"] == "Generate a candidate."
    assert kwargs["check"] is True


@pytest.mark.parametrize("resume", [False, True])
def test_crs_shell_preserves_smoke_selection_and_external_outputs(tmp_path, resume):
    executable = tmp_path / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'base': os.environ.get('CRS_OAI_BASE_URL')}))\n"
    )
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("CXBENCH_")}
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env.pop("CRS_OAI_BASE_URL", None)
    command = [
        "bash",
        str(ROOT / "run.sh"),
        "--smoke-test",
        "--dry-run",
        "--run-id",
        "smoke",
        "--runs-dir",
        str(tmp_path / "results"),
    ]
    if resume:
        command.append("--resume")
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    captured = json.loads(result.stdout.splitlines()[-1])
    argv = captured["args"]
    assert argv[argv.index("--task") + 1].endswith("BK-RAW-001.us.yaml")
    assert argv[argv.index("--runs-dir") + 1] == str(tmp_path / "results")
    assert argv[argv.index("-m") + 1] == f"main_experiment.{'resume' if resume else 'run'}"
    assert captured["base"] is None  # Leave dotenv endpoint selection to the provider loader.
