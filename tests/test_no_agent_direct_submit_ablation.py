from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from cxbench.grader import grade_run
from cxbench.task_schema import TaskSpec
from model_eval.run import REPO_ROOT
from no_agent_direct_submit_ablation.run import (
    ABLATION_ID,
    DIRECT_MODE,
    RUNS_DIR,
    ablate_task_mapping,
    build_batch_command,
    build_parser,
    resolve_config,
    write_ablated_task_specs,
)


def test_ablation_removes_only_target_rule_without_mutating_source() -> None:
    raw = {
        "rules": ["confirmation_before_submit", ABLATION_ID, "submit_matches_confirmed_draft"],
        "policy": {
            "process_requirements": ["Keep every non-ablated requirement."],
            "grading_policy": {
                "deterministic_hard_rules": ["confirmation_before_submit", ABLATION_ID],
                "semantic_judge_rules": ["semantic-check"],
                "hybrid_hint_rules": ["hint-check"],
            },
        },
    }

    derived, removed_from = ablate_task_mapping(raw)

    assert raw["rules"][1] == ABLATION_ID
    assert derived["rules"] == ["confirmation_before_submit", "submit_matches_confirmed_draft"]
    assert derived["policy"]["process_requirements"] == raw["policy"]["process_requirements"]
    assert removed_from == ["rules", "policy.grading_policy.deterministic_hard_rules"]


def test_derived_task_is_ablated_and_source_is_unchanged(tmp_path: Path) -> None:
    source = REPO_ROOT / "benchmarks/cxbench-120-v20260708/tasks/BK-RAW-001.us.yaml"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    derived_paths, provenance = write_ablated_task_specs((source,), tmp_path / "task_specs")
    task = TaskSpec.load(derived_paths[0])

    assert ABLATION_ID not in task.rules
    assert ABLATION_ID not in task.policy["grading_policy"]["deterministic_hard_rules"]
    assert "confirmation_before_submit" in task.rules
    assert "submit_matches_confirmed_draft" in task.rules
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert provenance[0]["source_sha256"] == source_hash
    assert set(provenance[0]["removed_from"]) == {
        "rules",
        "policy.grading_policy.deterministic_hard_rules",
    }


def test_direct_submit_is_not_graded_by_removed_rule(tmp_path: Path) -> None:
    source = REPO_ROOT / "benchmarks/cxbench-120-v20260708/tasks/BK-RAW-001.us.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    derived, _ = ablate_task_mapping(raw)
    task = TaskSpec.from_mapping(derived, tmp_path / source.name)
    trace = [
        {
            "seq": 1,
            "actor": "agent",
            "event_type": "tool_call",
            "payload": {"tool_name": "submit_eft_dispute_case", "args": {}},
        }
    ]

    grade = grade_run(task, trace, {}, mode=DIRECT_MODE, grader="rule")

    assert ABLATION_ID not in {item["rule_id"] for item in grade["rules"]}


def test_launcher_is_direct_only_and_reuses_model_eval_batch(monkeypatch, tmp_path: Path) -> None:
    for name in ("CXBENCH_MODE", "CXBENCH_GRADER", "CXBENCH_MODEL"):
        monkeypatch.delenv(name, raising=False)
    args = build_parser().parse_args(
        ["--env-file", str(tmp_path / "missing.env"), "--limit", "1", "--run-id", "ablation-test"]
    )
    config = resolve_config(args, require_api_key=False)
    task_spec = REPO_ROOT / "benchmarks/cxbench-120-v20260708/tasks/BK-RAW-001.us.yaml"

    command = build_batch_command(config, (task_spec,))

    assert config.mode == DIRECT_MODE
    assert config.grader == "hybrid"
    assert command[1:3] == ["-m", "model_eval.batch_entrypoint"]
    assert command[command.index("--mode") + 1] == DIRECT_MODE
    assert command[command.index("--runs-dir") + 1] == RUNS_DIR.as_posix()


def test_launcher_rejects_non_direct_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "least_privilege"])
