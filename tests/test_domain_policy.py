from __future__ import annotations

from pathlib import Path

from cxbench.adapters import OpenAICompatibleAgent
from cxbench.domain_policy import (
    domain_policy_context_for_task,
    load_domain_policy_sources,
    validate_domain_policy_index,
)
from cxbench.task_schema import TaskSpec


def _agent_system_prompt(task: TaskSpec) -> str:
    agent = OpenAICompatibleAgent.__new__(OpenAICompatibleAgent)
    agent.task = task
    agent.mode = "least_privilege"
    return agent._messages([], ["get_session_context"])[0]["content"]


def test_domain_policy_index_covers_all_markdown_sources() -> None:
    assert validate_domain_policy_index() == []


def test_task_domain_is_inferred_from_business_system() -> None:
    assert TaskSpec.load(Path("tasks/HC-01.us.yaml")).domain == "healthcare"
    assert TaskSpec.load(Path("tasks/GB-01.us.yaml")).domain == "government_benefits"
    assert TaskSpec.load(Path("tasks/CX-01.us.yaml")).domain == "banking"


def test_banking_system_prompt_includes_domain_wide_policy_context() -> None:
    task = TaskSpec.load(Path("tasks/CX-01.us.yaml"))
    prompt = _agent_system_prompt(task)

    assert "<CXBENCH_DOMAIN_POLICY_CONTEXT" in prompt
    assert "source_count=10" in prompt
    assert "12 CFR § 1005.11 - Procedures for resolving errors." in prompt
    # Included because it is banking-domain context, even though CX-01 is a Reg E dispute task.
    assert "31 CFR § 1020.220 - Customer identification program requirements for banks." in prompt
    assert "45 CFR § 147.136 - Internal claims and appeals" not in prompt


def test_domain_policy_context_precedes_task_specific_context_for_prefix_cache() -> None:
    task = TaskSpec.load(Path("tasks/CX-01.us.yaml"))
    prompt = _agent_system_prompt(task)

    domain_context_at = prompt.index("<CXBENCH_DOMAIN_POLICY_CONTEXT")
    first_source_at = prompt.index("<DOMAIN_POLICY_SOURCE")
    task_context_at = prompt.index("<CXBENCH_TASK_CONTEXT>")
    task_title_at = prompt.index("Task: Unauthorized debit-card transaction dispute")

    assert "Available tools:" not in prompt
    assert domain_context_at < first_source_at < task_context_at < task_title_at


def test_healthcare_system_prompt_includes_healthcare_policy_context() -> None:
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    prompt = _agent_system_prompt(task)

    assert 'domain="healthcare"' in prompt
    assert "45 CFR § 147.136 - Internal claims and appeals and external review processes." in prompt
    assert (
        "45 CFR § 164.514 - Other requirements relating to uses and disclosures of protected health information."
        in prompt
    )
    assert "12 CFR § 1005.11 - Procedures for resolving errors." not in prompt


def test_healthcare_system_prompt_includes_no_arg_denial_workflow() -> None:
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    prompt = _agent_system_prompt(task)

    assert '<CXBENCH_DOMAIN_WORKFLOW_BRIEF domain="healthcare">' in prompt
    assert "Call get_denial_notice_summary with no arguments" in prompt
    assert "claim_summary, denial_notice, plan_terms" in prompt
    assert prompt.index("<CXBENCH_DOMAIN_WORKFLOW_BRIEF") < prompt.index("<CXBENCH_TASK_CONTEXT>")


def test_insurance_context_includes_curated_flow_rules() -> None:
    task = TaskSpec.load(Path("tasks/IN-01.us.yaml"))
    context = domain_policy_context_for_task(task)

    assert "Insurance Claims Compliance Flow Rules" in context
    assert "ca-fcsp-no-misrep-r002" in context
    assert len(load_domain_policy_sources("insurance")) == 3
