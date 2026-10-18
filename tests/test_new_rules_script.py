"""Compare original and revised rejudging through the same experiment path."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from cxbench import openai_compat
from cxbench.grader import LLMTraceJudge
from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec
from main_experiment import rejudge, resume, score

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("new_rules", ROOT / "scripts/rejudge.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(rejudge.batch, "_load_dotenv", lambda _: {})
    monkeypatch.setattr(openai_compat, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid/openai/v1")
    monkeypatch.setenv("OPENAI_WIRE_API", "responses")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    source = tmp_path / "source"
    original_task = tmp_path / "original-task.yaml"
    original_raw = yaml.safe_load((ROOT / "tasks/CX-01.us.yaml").read_text())
    original_raw["policy"]["llm_judge"] = {
        "rubric_version": "test-old",
        "checklist": [],
    }
    original_task.write_text(yaml.safe_dump(original_raw))
    task = TaskSpec.load(original_task)
    runtime = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=source / "task_runs",
        grader="rule",
        user_sim="deterministic",
    )
    executed = runtime.run()
    directory = Path(executed["run_dir"])
    config = score._json(directory / "config.json")
    config.update(judge_model="openai:original-judge", execution_grader="rule")
    resume._write_json(directory / "config.json", config)
    manifest = {
        "model": "original-agent",
        "agent_provider": "openai",
        "judge_model": "original-judge",
        "judge_provider": "openai",
        "wire_api": "chat",
        "max_output_tokens": "4096",
        "concurrency": 1,
        "executor": "thread",
        "retry_delay": 0,
        "reasoning_effort": "low",
        "base_url": "https://original.invalid/v1",
        "task_count": 1,
        "tasks": [{"path": str(task.path), "sha256": rejudge.batch._sha256(task.path)}],
    }
    resume._write_json(source / "manifest.json", manifest)
    resume._write_results(
        source / "results.jsonl",
        {
            task.id: {
                "task_id": task.id,
                "task_path": str(task.path),
                "run_dir": str(directory),
                "ok": True,
                "verdict": "pass",
                "error": None,
            }
        },
    )
    release = tmp_path / "release"
    (release / "tasks").mkdir(parents=True)
    raw = yaml.safe_load(task.path.read_text())
    raw["policy"]["llm_judge"] = {"rubric_version": "test-new", "checklist": []}
    replacement = release / "tasks" / task.path.name
    replacement.write_text(yaml.safe_dump(raw))
    output = tmp_path / "output"
    args = [
        "--batch",
        str(source),
        "--task-release",
        str(release),
        "--output",
        str(output),
        "--inherit-source-config",
    ]
    monkeypatch.setattr(Runtime, "run", lambda *_: pytest.fail("agent must never rerun"))
    return args, source, output, replacement, directory


def test_only_rules_change_request_configuration_and_evidence_are_identical(
    inputs, monkeypatch, tmp_path
):
    args, source, output, replacement, directory = inputs
    original_bytes = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    calls = []

    def fake_judge(client, messages, **kwargs):
        calls.append(
            (
                client.config,
                copy.deepcopy(messages),
                os.environ["OPENAI_MAX_OUTPUT_TOKENS"],
                os.environ["OPENAI_REASONING_EFFORT"],
            )
        )
        return {}, {"process_pass": True, "findings": [], "rationale": "offline"}

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", fake_judge)
    assert (
        rejudge.main(
            [
                "--batch",
                str(source),
                "--output",
                str(tmp_path / "baseline"),
                "--inherit-source-config",
            ]
        )
        == 0
    )
    assert script.main(args) == 0
    baseline, revised = calls
    assert baseline[0] == revised[0]
    assert revised[0].model == "original-judge"
    assert revised[0].base_url == "https://original.invalid/v1"
    assert revised[0].wire_api == "chat"
    assert baseline[2:] == revised[2:] == ("4096", "low")
    assert baseline[1][0] == revised[1][0]
    before, after = [json.loads(call[1][1]["content"]) for call in calls]
    assert before["task"].pop("llm_judge_contract") != after["task"].pop("llm_judge_contract")
    for payload in (before, after):
        payload["task"]["gold_policy"].pop("llm_judge", None)
    assert before == after
    assert original_bytes == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    new_run = output / "task_runs" / directory.name
    for name in ("trace.jsonl", "final_state.json"):
        assert (new_run / name).read_bytes() == (directory / name).read_bytes()
    assert score._json(new_run / "config.json")["task_path"] == str(replacement)
    assert (output / "paper_scores/report.md").exists()
    assert os.environ["OPENAI_WIRE_API"] == "responses"
    assert os.environ["OPENAI_REASONING_EFFORT"] == "high"
    # A completed resume must not pay for another judge call.
    assert script.main([*args, "--resume"]) == 0
    assert len(calls) == 2


@pytest.mark.parametrize(
    "mutation",
    ["business", "missing_task", "unknown_id", "missing_config", "model_mismatch"],
)
def test_rejects_non_rule_changes_and_ambiguous_configuration(inputs, mutation):
    args, source, output, replacement, directory = inputs
    if mutation == "business":
        raw = yaml.safe_load(replacement.read_text())
        raw["hidden_facts"]["new_fact"] = "not a rule"
        replacement.write_text(yaml.safe_dump(raw))
    elif mutation == "missing_task":
        raw = yaml.safe_load(replacement.read_text())
        raw["id"] = "OTHER"
        replacement.write_text(yaml.safe_dump(raw))
    elif mutation == "unknown_id":
        args += ["--task-id", "OTHER"]
    elif mutation == "missing_config":
        manifest = score._json(source / "manifest.json")
        manifest.pop("wire_api")
        resume._write_json(source / "manifest.json", manifest)
    else:
        config = score._json(directory / "config.json")
        config["judge_model"] = "openai:another-model"
        resume._write_json(directory / "config.json", config)
    with pytest.raises(SystemExit) as error:
        script.main([*args, "--dry-run"])
    assert error.value.code == 2
    assert not output.exists()


def test_dry_run_inherits_source_and_never_calls_provider(inputs, monkeypatch, capsys):
    args, _, output, _, _ = inputs
    monkeypatch.setattr(resume, "_stage", lambda *a, **k: pytest.fail("dry-run must not grade"))
    assert script.main([*args, "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "judge_model: original-judge" in text
    assert "wire_api: chat" in text
    assert "max_output_tokens: 4096" in text
    assert not output.exists()


def test_unrecorded_endpoint_uses_original_provider_environment(inputs, monkeypatch, capsys):
    args, source, _, _, _ = inputs
    manifest = score._json(source / "manifest.json")
    manifest.pop("base_url")
    resume._write_json(source / "manifest.json", manifest)
    clients = []

    def judge(client, messages, **kwargs):
        clients.append(client.config)
        return {}, {"process_pass": True, "findings": [], "rationale": "offline"}

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", judge)
    assert script.main(args) == 0
    assert clients[0].base_url == "https://wrong.invalid/openai/v1"
    assert "no recorded base_url" in capsys.readouterr().out


def test_resume_rejects_changed_rules_or_system_prompt(inputs, monkeypatch):
    args, _, output, replacement, _ = inputs
    monkeypatch.setattr(
        openai_compat.OpenAICompatibleClient,
        "create_validated",
        lambda *a, **k: (
            {},
            {"process_pass": True, "findings": [], "rationale": "offline"},
        ),
    )
    assert script.main(args) == 0
    original = replacement.read_text()
    replacement.write_text(original + "\n# changed rubric revision\n")
    with pytest.raises(SystemExit):
        script.main([*args, "--resume"])
    replacement.write_text(original)
    monkeypatch.setattr(LLMTraceJudge, "_system_prompt", lambda _: "changed prompt")
    with pytest.raises(SystemExit):
        script.main([*args, "--resume"])
    assert (output / "results.jsonl").exists()


@pytest.mark.parametrize("judge_model", [None, "replacement-judge"])
def test_responses_override_sends_structured_judge_request(inputs, monkeypatch, judge_model):
    args, source, output, _, _ = inputs
    if judge_model:
        args = [*args, "--judge-model", judge_model]
    requests = []
    original = (source / "manifest.json").read_bytes()

    def responses(client, payload):
        requests.append((client.config, copy.deepcopy(payload)))
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {"process_pass": True, "findings": [], "rationale": "offline"}
                            ),
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "_post_responses_stream", responses)
    monkeypatch.setattr(
        openai_compat.OpenAICompatibleClient,
        "_post_chat_stream",
        lambda *a: pytest.fail("Responses override must never call Chat Completions"),
    )
    assert script.main([*args, "--wire-api", "responses"]) == 0
    config, payload = requests[0]
    assert config.wire_api == "responses"
    assert config.base_url == "https://original.invalid/v1"
    assert payload["model"] == (judge_model or "original-judge")
    assert payload["max_output_tokens"] == 4096
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["text"]["format"] == {
        "type": "json_schema",
        **LLMTraceJudge._JUDGE_RESPONSE_FORMAT["json_schema"],
    }
    manifest = score._json(output / "manifest.json")
    assert manifest["wire_api"] == "responses"
    assert manifest["judge_model"] == (judge_model or "original-judge")
    assert manifest["concurrency"] == 1
    assert manifest["retry_delay"] == 0
    assert (source / "manifest.json").read_bytes() == original
    assert script.main([*args, "--wire-api", "responses", "--resume"]) == 0
    assert len(requests) == 1
    with pytest.raises(SystemExit):
        script.main([*args, "--wire-api", "chat", "--resume"])
    with pytest.raises(SystemExit):
        script.main(
            [*args, "--wire-api", "responses", "--judge-model", "another-judge", "--resume"]
        )


def test_responses_dry_run_displays_override(inputs, monkeypatch, capsys):
    args, _, output, _, _ = inputs
    monkeypatch.setattr(resume, "_stage", lambda *a, **k: pytest.fail("must not grade"))
    assert script.main([*args, "--wire-api", "responses", "--dry-run"]) == 0
    assert "wire_api: responses" in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize("override_options", [False, True])
def test_entrypoints_share_defaults_and_provider_overrides(
    inputs, monkeypatch, tmp_path, override_options
):
    _, source, output, replacement, directory = inputs
    calls = []

    def judge(client, messages, **kwargs):
        calls.append(
            (
                client.config,
                copy.deepcopy(messages),
                os.environ["OPENAI_MAX_OUTPUT_TOKENS"],
                os.environ["OPENAI_REASONING_EFFORT"],
            )
        )
        return {}, {"process_pass": True, "findings": [], "rationale": "offline"}

    monkeypatch.setattr(openai_compat.OpenAICompatibleClient, "create_validated", judge)
    args = ["--batch", str(source), "--judge-model", "selected-judge"]
    if override_options:
        args += [
            "--judge-provider",
            "same",
            "--base-url",
            "https://selected.invalid/v1",
            "--wire-api",
            "responses",
            "--max-output-tokens",
            "2048",
            "--reasoning-effort",
            "medium",
            "--concurrency",
            "2",
            "--executor",
            "thread",
            "--retry-delay",
            "0.25",
            "--task-release",
            str(replacement.parent.parent),
            "--task-id",
            "CX-01.us",
        ]
    baseline = tmp_path / "baseline"
    assert rejudge.main([*args, "--output", str(baseline)]) == 0
    assert script.main([*args, "--output", str(output)]) == 0
    assert calls[0] == calls[1]
    config, _, tokens, effort = calls[1]
    assert config.model == "selected-judge"
    assert config.base_url == (
        "https://selected.invalid/v1" if override_options else "https://wrong.invalid/openai/v1"
    )
    assert tokens == ("2048" if override_options else "8192")
    assert effort == ("medium" if override_options else "high")
    expected_manifest = score._json(baseline / "manifest.json")
    manifest = score._json(output / "manifest.json")
    for document in (expected_manifest, manifest):
        document.pop("run_id")
        document.pop("created_at")
    assert manifest == expected_manifest
    assert manifest["concurrency"] == (2 if override_options else 20)
    assert manifest["retry_delay"] == (0.25 if override_options else 5.0)
    original = score._json(directory / "config.json")["task_path"]
    assert manifest["tasks"][0]["path"] == (str(replacement) if override_options else original)


def test_judge_model_is_required_without_explicit_inheritance(inputs):
    args, _, output, _, _ = inputs
    args = [arg for arg in args if arg != "--inherit-source-config"]
    for entrypoint in (script.main, rejudge.main):
        with pytest.raises(SystemExit) as error:
            entrypoint([*args, "--dry-run"])
        assert error.value.code == 2
    assert not output.exists()
