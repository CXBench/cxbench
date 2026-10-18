from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cxbench.benchmark_generation import (
    accept_candidates,
    generate_benchmark,
    validate_candidate_run,
)
from cxbench.task_schema import TaskSpec


def _write_generation_fixture(tmp_path: Path) -> Path:
    corpus = tmp_path / "reg_e.md"
    corpus.write_text(
        "# Reg E sample\n"
        "The bank must provide a clear disclosure before asking for confirmation.\n"
        "The agent shall create an intake draft before submission.\n",
        encoding="utf-8",
    )
    corpus_manifest = tmp_path / "corpus.yaml"
    corpus_manifest.write_text(
        yaml.safe_dump({"chunks": [{"id": "reg-e", "path": str(corpus)}]}), encoding="utf-8"
    )

    domain_rd = tmp_path / "banking-rd.md"
    domain_rd.write_text(
        "# Banking RD\nUse least privilege, draft, confirmation, and runtime gateway.\n",
        encoding="utf-8",
    )

    task = yaml.safe_load(Path("tasks/CX-01.us.yaml").read_text(encoding="utf-8"))
    task["id"] = "BK-GEN-01.us"
    task["title"] = "Generated Reg E dispute candidate"
    task["policy"] = {"mandatory_disclosure": task["policy"]["mandatory_disclosure"]}
    matrix = tmp_path / "scenario.yaml"
    matrix.write_text(
        yaml.safe_dump({"scenarios": [{"id": "banking-reg-e-001", "task": task}]}), encoding="utf-8"
    )

    config = tmp_path / "generation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "domain": "banking",
                "business_system": "banking_customer_ops_v0",
                "corpus_manifest": str(corpus_manifest),
                "domain_rd": str(domain_rd),
                "scenario_matrix": str(matrix),
                "target_count": 1,
                "user_sim_profiles": ["deterministic", "llm", "verify"],
            }
        ),
        encoding="utf-8",
    )
    return config


def _write_rule_seed_generation_fixture(tmp_path: Path) -> Path:
    registry = tmp_path / "rule_registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": "reg-e-test-v1",
                "rules": [
                    {
                        "rule_id": "reg-e-written-confirmation",
                        "corpus_id": "reg-e-1005-11",
                        "obligation_type": "disclosure",
                        "stable_summary": "Written confirmation must be explained without delaying investigation.",
                        "required_judge_checks": [
                            "written_confirmation_notice",
                            "investigation_not_delayed_for_written_confirmation",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    domain_rd = tmp_path / "banking-rd.md"
    domain_rd.write_text(
        "# Banking RD\nUse least privilege, draft, confirmation, and runtime gateway.\n",
        encoding="utf-8",
    )

    task = yaml.safe_load(Path("tasks/CX-01.us.yaml").read_text(encoding="utf-8"))
    task["id"] = "BK-SEED-01.us"
    task["title"] = "Rule-seeded Reg E dispute candidate"
    task["policy"] = {"mandatory_disclosure": ""}
    matrix = tmp_path / "scenario.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "id": "BK-SEED-01.us",
                        "rule_seeds": {"required": ["reg-e-written-confirmation"], "optional": []},
                        "task": task,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = tmp_path / "generation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "domain": "banking",
                "business_system": "banking_customer_ops_v0",
                "rule_registry": str(registry),
                "domain_rd": str(domain_rd),
                "scenario_matrix": str(matrix),
                "target_count": 1,
                "user_sim_profiles": ["llm"],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_generate_benchmark_writes_rule_selection_candidates_and_validation(tmp_path):
    config = _write_generation_fixture(tmp_path)
    out = tmp_path / "run"

    validation = generate_benchmark(config, out, workers=2)

    assert validation["valid"] is True
    assert validation["candidate_count"] == 1
    assert (out / "selected_rule_cards.jsonl").exists()
    assert (out / "selection_manifest.json").exists()
    assert (out / "worker_inputs" / "interface_manifest.json").exists()
    assert (out / "workers" / "worker_01" / "inputs" / "scenario_shard.json").exists()
    assert (out / "workers" / "worker_01" / "outputs" / "candidates").exists()
    assert (out / "prompts" / "worker_01.txt").exists()
    prompt = (out / "workers" / "worker_01" / "prompt.txt").read_text(encoding="utf-8")
    assert "Do not inspect source code" in prompt
    assert "Use only these inputs" in prompt
    assert "Domain RD excerpt" not in prompt
    candidate = out / "candidates" / "BK-GEN-01.us.yaml"
    task = TaskSpec.load(candidate)
    assert task.policy["source_refs"]
    assert task.policy["process_requirements"]
    assert task.policy["grading_notes"]
    assert task.policy["llm_judge"]["source_spec"] == "grading_specs/BK-GEN-01.us.json"
    assert (
        task.policy["llm_judge"]["checklist"][0]["id"] == "generated_process_requirements_followed"
    )
    assert validation["merged_grading_specs"][0]["merged"] is True


def test_generate_benchmark_uses_manual_rule_seeds(tmp_path):
    config = _write_rule_seed_generation_fixture(tmp_path)
    out = tmp_path / "run"

    validation = generate_benchmark(config, out)

    assert validation["valid"] is True
    seed_manifest = json.loads((out / "resolved_seed_manifest.json").read_text(encoding="utf-8"))
    assert seed_manifest["mode"] == "manual_rule_seed"
    assert seed_manifest["registry_version"] == "reg-e-test-v1"
    assert seed_manifest["rules"][0]["rule_id"] == "reg-e-written-confirmation"
    assert seed_manifest["rules"][0]["corpus_id"] == "reg-e-1005-11"
    assert seed_manifest["rules"][0]["canonical_text_sha256"]
    assert seed_manifest["rules"][0]["resolved_text_sha256"]
    assert seed_manifest["rules"][0]["registry_item_sha256"]
    assert seed_manifest["rules"][0]["required_judge_checks"] == [
        "written_confirmation_notice",
        "investigation_not_delayed_for_written_confirmation",
    ]
    assert seed_manifest["scenarios"][0]["required_rule_ids"] == ["reg-e-written-confirmation"]
    worker_seed_manifest = json.loads(
        (out / "worker_inputs" / "resolved_seed_manifest.json").read_text(encoding="utf-8")
    )
    assert worker_seed_manifest == seed_manifest
    task = TaskSpec.load(out / "candidates" / "BK-SEED-01.us.yaml")
    assert task.policy["source_refs"] == ["reg-e-written-confirmation"]
    assert task.policy["seed_manifest_version"] == "reg-e-test-v1"
    check_ids = [item["id"] for item in task.policy["llm_judge"]["checklist"]]
    assert "written_confirmation_notice" in check_ids
    assert "investigation_not_delayed_for_written_confirmation" in check_ids


def test_rule_seed_validation_rejects_unseeded_source_refs(tmp_path):
    config = _write_rule_seed_generation_fixture(tmp_path)
    out = tmp_path / "run"
    generate_benchmark(config, out)
    candidate = out / "candidates" / "BK-SEED-01.us.yaml"
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raw["policy"]["source_refs"].append("unseeded-rule")
    candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    validation = validate_candidate_run(out)

    assert validation["valid"] is False
    assert (
        "policy.source_refs includes unseeded rule ids: unseeded-rule"
        in validation["candidates"][0]["errors"]
    )


def test_rule_seed_generation_rejects_missing_manual_seed(tmp_path):
    config = _write_rule_seed_generation_fixture(tmp_path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    matrix = Path(raw["scenario_matrix"])
    matrix_raw = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    matrix_raw["scenarios"][0]["rule_seeds"]["required"] = ["missing-rule"]
    matrix.write_text(yaml.safe_dump(matrix_raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown rule seed: missing-rule"):
        generate_benchmark(config, tmp_path / "run")


def test_generate_benchmark_can_reuse_rule_selection(tmp_path):
    config = _write_generation_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_benchmark(config, first)

    validation = generate_benchmark(
        config, second, reuse_rule_selection=first / "selected_rule_cards.jsonl"
    )

    assert validation["valid"] is True
    task = TaskSpec.load(second / "candidates" / "BK-GEN-01.us.yaml")
    assert task.policy["llm_judge"]["source_rule_cards"]
    manifest = json.loads((second / "selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "reused"
    assert manifest["rule_card_ids"]


def test_accept_candidates_copies_only_valid_candidates(tmp_path):
    config = _write_generation_fixture(tmp_path)
    out = tmp_path / "run"
    dest = tmp_path / "accepted"
    generate_benchmark(config, out)

    manifest = accept_candidates(out, dest)

    assert len(manifest["accepted"]) == 1
    accepted_path = Path(manifest["accepted"][0])
    assert accepted_path.exists()
    assert TaskSpec.load(accepted_path).id == "BK-GEN-01.us"
    assert (out / "accept_manifest.json").exists()


class FakeQualityReviewClient:
    def __init__(self, config):
        self.config = config

    def create_validated(
        self, messages, *, validator, tools=None, response_format=None, validation_context
    ):
        obj = {
            "pass": False,
            "decision": "fix",
            "findings": [
                {
                    "rule_id": "tool_capability_alignment",
                    "severity": "critical",
                    "evidence": "candidate requires an unsupported workflow",
                    "recommendation": "repair the task before acceptance",
                }
            ],
            "rationale": "quality gate blocks this candidate",
        }
        return {"_cxbench": {"provider": "fake-review"}}, obj


def test_generate_benchmark_quality_gate_blocks_acceptance(monkeypatch, tmp_path):
    import cxbench.benchmark_generation as generation

    config = _write_generation_fixture(tmp_path)
    out = tmp_path / "run"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(generation, "OpenAICompatibleClient", FakeQualityReviewClient)

    validation = generate_benchmark(config, out, review_model="openai:fake-review")

    assert validation["valid"] is False
    assert validation["valid_count"] == 0
    assert validation["quality_gate"]["pass_count"] == 0
    candidate = validation["candidates"][0]
    assert candidate["structural_valid"] is True
    assert candidate["quality_review"]["pass"] is False
    assert "quality gate failed" in candidate["errors"][-1]
    assert (out / "quality_review.json").exists()
    assert (out / "quality_reviews" / "BK-GEN-01.us.json").exists()
    with pytest.raises(ValueError, match="no valid candidates"):
        accept_candidates(out, tmp_path / "accepted")


def test_quality_gate_static_findings_block_llm_accept():
    from cxbench.benchmark_generation import _normalize_quality_review

    review = _normalize_quality_review(
        {"pass": True, "decision": "accept", "findings": [], "rationale": "looks fine"},
        task_id="BK-BAD.us",
        candidate_path=Path("candidates/BK-BAD.us.yaml"),
        static_findings=[
            {
                "rule_id": "grader_rule_alignment",
                "severity": "critical",
                "evidence": "impossible deterministic rule",
                "recommendation": "remove or replace the rule",
            }
        ],
    )

    assert review["pass"] is False
    assert review["decision"] == "accept"
    assert review["findings"][0]["rule_id"] == "grader_rule_alignment"
