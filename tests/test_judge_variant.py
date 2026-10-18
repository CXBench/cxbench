from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from cxbench.grader import LLMTraceJudge, validate_rule_ids
from cxbench.task_schema import TaskSpec


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "benchmarks/cxbench-banking-judge-v20260917"
SOURCE = ROOT / "benchmarks/cxbench-120-v20260708/tasks"
TASKS = sorted((RELEASE / "tasks").glob("*.yaml"))


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_revised_task_preserves_business_contract_and_provenance(path):
    original = yaml.safe_load((SOURCE / path.name).read_text())
    revised = yaml.safe_load(path.read_text())
    old_judge = original["policy"].pop("llm_judge")
    new_judge = revised["policy"].pop("llm_judge")

    assert revised == original
    assert new_judge["source_spec"] == old_judge["source_spec"]
    assert new_judge["source_rule_cards"] == old_judge["source_rule_cards"]
    old_ids = [item["id"] for item in old_judge["checklist"]]
    new_ids = [item["id"] for item in new_judge["checklist"]]
    if revised["domain"] != "banking":
        assert new_ids == old_ids
    else:
        assert set(old_ids) <= set(new_ids)

    task = TaskSpec.load(path)
    validate_rule_ids(task.rules, task_id=task.id)


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_revised_checklist_has_finite_evidence_and_complete_confirmation_display(path):
    task = TaskSpec.load(path)
    judge = task.policy["llm_judge"]
    ids = [item["id"] for item in judge["checklist"]]
    assert len(ids) == len(set(ids))
    for item in judge["checklist"]:
        assert item["severity"] in {"critical", "major", "minor"}
        for field in ("id", "applies_when", "pass_condition"):
            assert isinstance(item[field], str) and item[field].strip()
        for field in ("fail_if", "required_evidence"):
            assert isinstance(item[field], list) and item[field]
            assert all(isinstance(value, str) and value.strip() for value in item[field])
    display = judge["confirmation_display"]["field_representations"]
    assert set(display) == set(task.confirmation_contract)
    assert all(isinstance(value, str) and value.strip() for value in display.values())


def test_revised_release_manifest_covers_all_six_domains_and_hashes():
    manifest = json.loads((RELEASE / "release_manifest.json").read_text())
    assert manifest["task_count"] == len(TASKS) == 120
    assert manifest["domain_count"] == 6
    assert manifest["domains"] == {
        domain: 20
        for domain in (
            "banking",
            "telecom",
            "insurance",
            "healthcare",
            "utilities",
            "government_benefits",
        )
    }
    assert manifest["formal_grader"] == "hybrid"
    assert manifest["changed_fields"] == ["policy.llm_judge"]
    assert manifest["judge_system_prompt_version"] == LLMTraceJudge.SYSTEM_PROMPT_VERSION
    entries = manifest["task_files"]
    assert len(entries) == len({entry["id"] for entry in entries}) == 120
    assert Counter(entry["domain"] for entry in entries) == manifest["domains"]
    assert {RELEASE / entry["file"] for entry in entries} == set(TASKS)
    for entry in entries:
        path = RELEASE / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        source = RELEASE / entry["source_file"]
        assert source.resolve() == (SOURCE / path.name).resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["source_sha256"]
        task = TaskSpec.load(path)
        assert task.id == entry["id"]
        assert task.title == entry["title"]
        assert task.domain == entry["domain"]
        assert (
            task.policy["llm_judge"]["rubric_version"] == manifest["rubric_versions"][task.domain]
        )


def test_judge_fixed_prefix_is_task_independent_and_keeps_facts_in_payload():
    judge = object.__new__(LLMTraceJudge)
    judge.mode = "least_privilege"
    prompts = set()
    for domain in ("telecom", "insurance", "healthcare", "utilities", "government_benefits"):
        task = next(TaskSpec.load(path) for path in TASKS if TaskSpec.load(path).domain == domain)
        judge.task = task
        policy = copy.deepcopy(task.policy)
        prompts.add(judge._system_prompt())
        payload = judge._judge_payload([], {})
        assert payload["task"]["llm_judge_contract"] == policy["llm_judge"]
        assert payload["task"]["confirmation_contract"] == task.confirmation_contract
        assert task.id not in judge._system_prompt()
    assert len(prompts) == 1
    prompt = prompts.pop()
    for domain_fact in (
        "Reg E",
        "PHI",
        "check_step_up",
        "continued_benefits",
        "banking_reg_e_error_resolution",
    ):
        assert domain_fact not in prompt
