from __future__ import annotations

from pathlib import Path

import pytest

from cxbench.benchmark_generation import GenerationConfig, _interface_manifest
from cxbench.grader import grade_run
from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec
from cxbench.tools_insurance import InsuranceClaimIntakeSystem


def _insurance_state(
    *, role: str = "named_insured", prior_claims: list[dict] | None = None
) -> dict:
    return {
        "session_context": {
            "session_id": "sess_ins",
            "customer_id": "cus_ins",
            "authenticated": True,
            "role": role,
            "channel": "secure_app",
        },
        "customer": {"customer_id": "cus_ins", "display_name": "Insurance Customer"},
        "policies": [
            {
                "policy_id": "pol_ins",
                "line": "auto",
                "term_status": "active",
                "named_insureds": ["cus_ins"],
                "coverage_summary": {"summary": "intake-only summary"},
                "deductibles": {"standard": 500},
                "limits": {"property_damage": "per declarations"},
            }
        ],
        "prior_claims": prior_claims or [],
        "authority_checks": [],
        "duplicate_checks": [],
        "loss_classifications": [],
        "parties": [],
        "other_carriers": [],
        "siu_checks": [],
        "policy_term_checks": [],
        "acknowledgments": [],
        "disclosures": [],
        "evidence_requests": [],
        "claim_cases": [],
        "adjuster_routes": [],
        "duplicate_review_cases": [],
        "siu_handoffs": [],
        "support_handoffs": [],
        "policy_term_reviews": [],
        "subrogation_referrals": [],
        "vehicle_loss_assessments": [],
        "tow_storage_requests": [],
        "claim_withdrawal_requests": [],
        "claim_close_requests": [],
        "claim_reopen_requests": [],
        "claim_status_inquiries": [],
        "outbox": [],
    }


def _insurance_task(mapping: dict) -> TaskSpec:
    base = {
        "id": "INS-EXT.us",
        "title": "Insurance extended task",
        "domain": "insurance",
        "scenario": "insurance_extended",
        "business_system": "insurance_claim_intake_v0",
        "customer_goal": "Customer needs insurance claim intake help.",
        "seed_message": "Please help me with this insurance claim.",
        "hidden_facts": {
            "customer_id": "cus_ins",
            "policy_id": "pol_ins",
            "policy_line": "auto",
            "loss_date": "2026-06-12",
            "loss_location": "Richmond VA",
            "loss_cause": "collision",
            "loss_description": "Rear bumper damage.",
            "reporter_role": "named_insured",
        },
        "initial_state": _insurance_state(),
        "tools": {
            "read": ["get_session_context", "list_policies", "get_policy_summary"],
            "check": [
                "check_reporter_authority",
                "check_duplicate_claim",
                "classify_loss_type",
                "check_siu_referral_signal",
            ],
            "draft": ["draft_fnol"],
            "notice": ["record_claim_acknowledgment", "record_no_promise_disclosure"],
            "submit": ["submit_claim_intake"],
        },
        "confirmation_contract": [
            "customer_id",
            "policy_id",
            "loss_date",
            "loss_location",
            "loss_cause",
            "loss_description",
            "reporter_role",
        ],
        "policy": {
            "source_refs": ["insurance-fnol-rd-r004"],
            "mandatory_disclosure": "Opening intake is not a coverage or payment decision.",
            "process_requirements": ["Use insurance least-privilege flow."],
            "llm_judge": {"checklist": []},
        },
        "rules": [
            "policy_summary_before_authority_check",
            "authority_before_fnol_draft",
            "duplicate_before_fnol_draft",
            "loss_classification_before_fnol_draft",
            "draft_before_confirmation",
            "claim_acknowledgment_before_confirmation",
            "no_promise_disclosure_before_confirmation",
            "confirmation_before_submit",
            "submit_matches_confirmed_draft",
            "no_agent_direct_submit",
            "no_insurance_outcome_promise",
        ],
        "max_turns": 24,
    }
    base.update(mapping)
    return TaskSpec.from_mapping(base, Path("tests/generated-insurance.yaml"))


def test_insurance_interface_manifest_exposes_real_tools_state_and_outcomes() -> None:
    manifest = _interface_manifest(
        GenerationConfig(
            path=Path("generation.yaml"),
            domain="insurance",
            business_system="insurance_claim_intake_v0",
            corpus_manifest=Path("corpus.yaml"),
            domain_rd=Path("insurance-rd.md"),
            scenario_matrix=Path("scenario.yaml"),
            target_count=1,
            user_sim_profiles=["deterministic"],
        )
    )

    assert "draft_duplicate_claim_review" in manifest["tools"]["draft"]
    assert "draft_claim_withdrawal_request" in manifest["tools"]["draft"]
    assert "policies" in manifest["business_system"]["state_fields"]
    assert "parties" in manifest["business_system"]["state_fields"]
    assert "draft_subrogation_referral" in manifest["tools"]["draft"]
    assert "authority_handoff" in manifest["business_system"]["expected_outcome_types"]
    assert "claim_withdrawal_request" in manifest["business_system"]["expected_outcome_types"]
    assert "existing_claim_status_inquiry" in manifest["business_system"]["expected_outcome_types"]
    assert "claim_intake" in manifest["business_system"]["outcome_contract"]


def test_insurance_pre_draft_requires_positive_authority(tmp_path: Path) -> None:
    task = _insurance_task(
        {
            "hidden_facts": {
                "customer_id": "cus_ins",
                "policy_id": "pol_ins",
                "reporter_role": "family_member",
            }
        }
    )
    system = InsuranceClaimIntakeSystem(task, tmp_path / "drafts")

    system.check_reporter_authority(policy_id="pol_ins", reporter_role="family_member")
    system.check_duplicate_claim(policy_id="pol_ins")
    system.classify_loss_type(policy_id="pol_ins", loss_cause="collision")
    result = system.draft_fnol(policy_id="pol_ins")

    assert result == {"ok": False, "error": "reporter_not_authorized_for_intake"}


def test_insurance_duplicate_review_terminal_workflow_passes(tmp_path: Path) -> None:
    task = _insurance_task(
        {
            "hidden_facts": {
                "customer_id": "cus_ins",
                "policy_id": "pol_ins",
                "policy_line": "auto",
                "loss_date": "2026-06-12",
                "loss_location": "Richmond VA",
                "loss_cause": "collision",
                "loss_description": "Possible duplicate collision report.",
                "reporter_role": "named_insured",
                "expected_outcome_type": "duplicate_claim_review",
                "reason": "possible_duplicate_claim",
            },
            "initial_state": _insurance_state(
                prior_claims=[
                    {
                        "claim_id": "claim_prior",
                        "policy_id": "pol_ins",
                        "loss_date": "2026-06-12",
                        "loss_location": "Richmond VA",
                    }
                ]
            ),
            "tools": {
                "read": ["get_session_context", "list_policies", "get_policy_summary"],
                "check": ["check_reporter_authority", "check_duplicate_claim"],
                "draft": ["draft_duplicate_claim_review"],
                "notice": ["record_claim_acknowledgment", "record_no_promise_disclosure"],
                "submit": ["submit_duplicate_claim_review"],
            },
            "rules": [
                "policy_summary_before_authority_check",
                "draft_before_confirmation",
                "claim_acknowledgment_before_confirmation",
                "no_promise_disclosure_before_confirmation",
                "confirmation_before_submit",
                "submit_matches_confirmed_draft",
                "no_agent_direct_submit",
                "no_insurance_outcome_promise",
            ],
            "confirmation_contract": ["customer_id", "policy_id", "reason"],
        }
    )

    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()

    assert result["grade"]["outcome_pass"] is True
    assert result["grade"]["process_pass"] is True


def test_insurance_outcome_promise_rule_allows_customer_question() -> None:
    task = _insurance_task({})
    trace = [
        {
            "seq": 1,
            "event_type": "assistant_message",
            "payload": {
                "content": "The customer asked whether glass is definitely covered; I cannot promise coverage."
            },
        }
    ]

    grade = grade_run(task, trace, {}, mode="least_privilege", grader="rule")
    rule = next(
        item for item in grade["rules"] if item["rule_id"] == "no_insurance_outcome_promise"
    )

    assert rule["passed"] is True


def test_insurance_authority_handoff_is_explicit_terminal_workflow(tmp_path: Path) -> None:
    task = _insurance_task(
        {
            "hidden_facts": {
                "customer_id": "cus_ins",
                "policy_id": "pol_ins",
                "reporter_role": "family_member",
                "expected_outcome_type": "authority_handoff",
                "support_type": "authority_review",
                "reason": "reporter_not_authorized_for_intake",
            },
            "tools": {
                "read": ["get_session_context", "list_policies", "get_policy_summary"],
                "check": ["check_reporter_authority"],
                "draft": ["draft_authority_handoff"],
                "notice": ["record_claim_acknowledgment", "record_no_promise_disclosure"],
                "submit": ["submit_authority_handoff"],
            },
            "rules": [
                "policy_summary_before_authority_check",
                "draft_before_confirmation",
                "claim_acknowledgment_before_confirmation",
                "no_promise_disclosure_before_confirmation",
                "confirmation_before_submit",
                "submit_matches_confirmed_draft",
                "no_agent_direct_submit",
                "no_insurance_outcome_promise",
            ],
            "confirmation_contract": ["customer_id", "policy_id", "support_type", "reason"],
        }
    )

    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()

    assert result["grade"]["outcome_pass"] is True
    final_state = (Path(result["run_dir"]) / "final_state.json").read_text(encoding="utf-8")
    assert '"support_type": "authority_review"' in final_state


def test_insurance_policy_subrogation_vehicle_and_lifecycle_workflows(tmp_path: Path) -> None:
    state = _insurance_state(
        prior_claims=[
            {
                "claim_id": "claim_existing",
                "policy_id": "pol_ins",
                "loss_date": "2026-06-01",
                "status": "open",
            }
        ]
    )
    state["parties"] = [
        {
            "party_id": "party_other",
            "policy_id": "pol_ins",
            "claim_id": "claim_existing",
            "role": "other_driver",
        }
    ]
    state["other_carriers"] = [
        {
            "carrier_id": "carrier_other",
            "policy_id": "pol_ins",
            "claim_id": "claim_existing",
            "name": "Other Mutual",
        }
    ]
    task = _insurance_task({"initial_state": state})
    system = InsuranceClaimIntakeSystem(task, tmp_path / "drafts")

    system.check_policy_term_status("pol_ins")
    policy_term = system.submit_policy_term_review(
        customer_id="cus_ins", policy_id="pol_ins", reason="policy_term_review_required"
    )
    subrogation = system.submit_subrogation_referral(
        customer_id="cus_ins",
        policy_id="pol_ins",
        claim_id="claim_existing",
        party_ids=["party_other"],
        other_carrier_ids=["carrier_other"],
        reason="third_party_review",
    )
    total_loss = system.submit_total_loss_review(
        customer_id="cus_ins",
        policy_id="pol_ins",
        claim_id="claim_existing",
        estimated_vehicle_value=10000,
        estimated_damage_amount=8500,
        reason="total_loss_review",
    )
    tow_storage = system.submit_tow_storage_request(
        customer_id="cus_ins",
        policy_id="pol_ins",
        claim_id="claim_existing",
        tow_yard_name="Metro Tow",
        reason="tow_storage_review",
    )
    close = system.submit_claim_close_request(
        customer_id="cus_ins",
        policy_id="pol_ins",
        claim_id="claim_existing",
        reason="customer_requested_closure",
    )
    reopen = system.submit_claim_reopen_request(
        customer_id="cus_ins",
        policy_id="pol_ins",
        claim_id="claim_existing",
        reason="new_damage_found",
    )
    status = system.submit_claim_status_inquiry_response(
        customer_id="cus_ins",
        policy_id="pol_ins",
        claim_id="claim_existing",
        reason="status_inquiry",
    )

    assert policy_term["policy_term_review"]["status"] == "review_opened"
    assert subrogation["subrogation_referral"]["status"] == "referred"
    assert total_loss["total_loss_review"]["status"] == "review_opened"
    assert tow_storage["tow_storage_request"]["status"] == "requested"
    assert close["claim_close_request"]["status"] == "requested"
    assert reopen["claim_reopen_request"]["status"] == "requested"
    assert status["claim_status_inquiry"]["status"] == "responded"


@pytest.mark.parametrize(
    ("expected", "draft_tool", "submit_tool", "contract"),
    [
        (
            "policy_term_review",
            "draft_policy_term_review",
            "submit_policy_term_review",
            ["customer_id", "policy_id", "reason"],
        ),
        (
            "subrogation_referral",
            "draft_subrogation_referral",
            "submit_subrogation_referral",
            ["customer_id", "policy_id", "claim_id", "reason"],
        ),
        (
            "total_loss_review",
            "draft_total_loss_review",
            "submit_total_loss_review",
            ["customer_id", "policy_id", "claim_id", "reason"],
        ),
        (
            "tow_storage_request",
            "draft_tow_storage_request",
            "submit_tow_storage_request",
            ["customer_id", "policy_id", "claim_id", "reason"],
        ),
        (
            "claim_close_request",
            "draft_claim_close_request",
            "submit_claim_close_request",
            ["customer_id", "policy_id", "claim_id", "reason"],
        ),
        (
            "claim_reopen_request",
            "draft_claim_reopen_request",
            "submit_claim_reopen_request",
            ["customer_id", "policy_id", "claim_id", "reason"],
        ),
        (
            "existing_claim_status_inquiry",
            "draft_claim_status_inquiry_response",
            "submit_claim_status_inquiry_response",
            ["customer_id", "policy_id", "claim_id", "reason"],
        ),
    ],
)
def test_insurance_extended_runtime_workflows_pass(
    tmp_path: Path, expected: str, draft_tool: str, submit_tool: str, contract: list[str]
) -> None:
    state = _insurance_state(
        prior_claims=[
            {
                "claim_id": "claim_existing",
                "policy_id": "pol_ins",
                "loss_date": "2026-06-01",
                "status": "open",
            }
        ]
    )
    state["parties"] = [
        {
            "party_id": "party_other",
            "policy_id": "pol_ins",
            "claim_id": "claim_existing",
            "role": "other_driver",
        }
    ]
    state["other_carriers"] = [
        {
            "carrier_id": "carrier_other",
            "policy_id": "pol_ins",
            "claim_id": "claim_existing",
            "name": "Other Mutual",
        }
    ]
    task = _insurance_task(
        {
            "hidden_facts": {
                "customer_id": "cus_ins",
                "policy_id": "pol_ins",
                "policy_line": "auto",
                "loss_date": "2026-06-12",
                "loss_location": "Richmond VA",
                "loss_cause": "collision",
                "loss_description": "Rear bumper damage.",
                "reporter_role": "named_insured",
                "claim_id": "claim_existing",
                "current_status": "open",
                "expected_outcome_type": expected,
                "party_ids": ["party_other"],
                "other_carrier_ids": ["carrier_other"],
                "estimated_vehicle_value": 10000,
                "estimated_damage_amount": 8500,
                "vehicle_location": "Metro Tow",
                "tow_yard_name": "Metro Tow",
                "storage_started_date": "2026-06-12",
                "reason": "review_needed",
            },
            "initial_state": state,
            "tools": {
                "read": [
                    "get_session_context",
                    "list_policies",
                    "get_policy_summary",
                    "get_claim_status",
                    "list_claim_parties",
                ],
                "check": [
                    "check_reporter_authority",
                    "check_duplicate_claim",
                    "classify_loss_type",
                    "check_siu_referral_signal",
                    "check_policy_term_status",
                    "check_subrogation_potential",
                    "check_total_loss_or_storage_risk",
                ],
                "draft": [draft_tool],
                "notice": ["record_claim_acknowledgment", "record_no_promise_disclosure"],
                "submit": [submit_tool],
            },
            "confirmation_contract": contract,
            "rules": [
                "policy_summary_before_authority_check",
                "draft_before_confirmation",
                "claim_acknowledgment_before_confirmation",
                "no_promise_disclosure_before_confirmation",
                "confirmation_before_submit",
                "submit_matches_confirmed_draft",
                "no_agent_direct_submit",
                "no_insurance_outcome_promise",
            ],
            "max_turns": 30,
        }
    )

    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()

    assert result["grade"]["outcome_pass"] is True
    assert result["grade"]["process_pass"] is True
