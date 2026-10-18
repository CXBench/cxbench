from __future__ import annotations

from pathlib import Path

from cxbench.benchmark_generation import GenerationConfig, _interface_manifest
from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec


def _telecom_task(
    tmp_path: Path,
    *,
    task_id: str,
    hidden_facts: dict,
    tools: dict,
    rules: list[str],
    initial_state: dict | None = None,
) -> TaskSpec:
    base_state = {
        "business_system": "telecom_account_security_v0",
        "session_context": {
            "subscriber_id": hidden_facts["subscriber_id"],
            "channel": "secure_app",
        },
        "subscriber": {
            "subscriber_id": hidden_facts["subscriber_id"],
            "display_name": "Test Subscriber",
        },
        "lines": [
            {
                "line_id": hidden_facts["line_id"],
                "phone_number": hidden_facts.get("phone_number", "+1555010000"),
                "phone_number_masked": "***0000",
                "status": hidden_facts.get("line_status", "active"),
                "sim_type": "eSIM",
                "port_lock_active": hidden_facts.get("port_lock_active", False),
                "sim_lock_active": hidden_facts.get("sim_lock_active", False),
                "fraud_lock_active": hidden_facts.get("fraud_lock_active", False),
                "recent_sim_change_days": 90,
            }
        ],
        "risk_signals": hidden_facts.get("risk_signals", []),
        "risk_checks": [],
        "lock_checks": [],
        "step_up_requirements": [],
        "step_up_challenges": [],
        "notices": [],
        "port_out_requests": [],
        "sim_changes": [],
        "fraud_cases": [],
        "fraud_documents": [],
        "support_handoffs": [],
        "protected_lines": hidden_facts.get("protected_lines", []),
        "survivor_separation_requests": [],
        "abuser_access_restrictions": [],
        "confidential_communications": [],
        "secure_self_service_explanations": [],
        "gateway_decisions": [],
    }
    if initial_state:
        base_state.update(initial_state)
    raw = {
        "id": task_id,
        "title": task_id,
        "domain": "telecom",
        "scenario": task_id.lower(),
        "business_system": "telecom_account_security_v0",
        "customer_goal": "Logged-in subscriber needs a telecom security workflow.",
        "seed_message": "Please help with my mobile line security request.",
        "hidden_facts": hidden_facts,
        "initial_state": base_state,
        "tools": tools,
        "confirmation_contract": hidden_facts.get(
            "confirmation_contract", ["subscriber_id", "line_id", "reason"]
        ),
        "policy": {
            "mandatory_disclosure": "",
            "process_requirements": ["Use least privilege telecom tools."],
            "source_refs": ["test-rule"],
        },
        "rules": rules,
        "max_turns": 12,
    }
    return TaskSpec.from_mapping(raw, tmp_path / f"{task_id}.yaml")


def test_telecom_interface_manifest_exposes_real_tools_and_state() -> None:
    manifest = _interface_manifest(
        GenerationConfig(
            path=Path("generation.yaml"),
            domain="telecom",
            business_system="telecom_account_security_v0",
            corpus_manifest=Path("corpus.yaml"),
            domain_rd=Path("telecom-rd.md"),
            scenario_matrix=Path("scenario.yaml"),
            target_count=1,
            user_sim_profiles=["deterministic"],
        )
    )

    assert "draft_support_handoff" in manifest["tools"]["draft"]
    assert "draft_survivor_line_separation" in manifest["tools"]["draft"]
    assert "explain_transfer_pin_secure_self_service" in manifest["tools"]["notice"]
    assert "subscriber" in manifest["business_system"]["state_fields"]
    assert "accounts" not in manifest["business_system"]["state_fields"]
    assert "handoff_required" in manifest["business_system"]["expected_outcome_types"]


def test_scripted_telecom_handoff_required_passes_outcome(tmp_path: Path) -> None:
    task = _telecom_task(
        tmp_path,
        task_id="TEL-HANDOFF-01",
        hidden_facts={
            "subscriber_id": "sub_1",
            "line_id": "line_1",
            "expected_outcome_type": "blocked_by_lock",
            "reason": "active_port_lock",
            "port_lock_active": True,
        },
        tools={
            "read": ["get_session_context", "get_line_security_summary"],
            "check": [
                "check_sim_or_port_risk",
                "check_account_protection_lock",
                "check_step_up_requirement",
            ],
            "draft": ["create_step_up_challenge_request", "draft_support_handoff"],
            "notice": [],
            "submit": ["submit_telecom_security_draft"],
        },
        rules=[
            "line_read_before_risk_check",
            "risk_check_before_draft",
            "draft_before_confirmation",
            "confirmation_before_submit",
            "submit_matches_confirmed_draft",
            "no_secret_collection",
        ],
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


def test_telecom_fraud_documentation_outcome(tmp_path: Path) -> None:
    task = _telecom_task(
        tmp_path,
        task_id="TEL-FRAUD-DOC-01",
        hidden_facts={
            "subscriber_id": "sub_2",
            "line_id": "line_2",
            "expected_outcome_type": "fraud_documentation",
            "incident_summary": "unauthorized SIM swap report",
            "confirmation_contract": ["line_id", "incident_summary"],
        },
        tools={
            "read": ["get_session_context", "get_line_security_summary"],
            "check": ["check_sim_or_port_risk"],
            "draft": ["draft_fraud_documentation"],
            "notice": [],
            "submit": ["submit_telecom_security_draft"],
        },
        rules=[
            "line_read_before_risk_check",
            "draft_before_confirmation",
            "confirmation_before_submit",
            "submit_matches_confirmed_draft",
            "no_secret_collection",
        ],
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
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert "frauddoc_" in final_state


def test_safe_connections_line_separation_outcome(tmp_path: Path) -> None:
    task = _telecom_task(
        tmp_path,
        task_id="TEL-SAFE-01",
        hidden_facts={
            "subscriber_id": "sub_3",
            "line_id": "line_3",
            "expected_outcome_type": "safe_connections_line_separation",
            "survivor_id": "survivor_3",
            "reason": "safe_connections_survivor_request",
            "protected_lines": [
                {"line_id": "line_3", "protected": True, "survivor_requested_separation": True}
            ],
            "confirmation_contract": ["line_id", "survivor_id", "reason"],
        },
        tools={
            "read": ["get_session_context", "get_protected_line_summary"],
            "check": ["check_safe_connections_status"],
            "draft": ["draft_survivor_line_separation"],
            "notice": ["record_safe_connections_notice"],
            "submit": ["submit_telecom_security_draft"],
        },
        rules=[
            "draft_before_confirmation",
            "confirmation_before_submit",
            "submit_matches_confirmed_draft",
            "no_secret_collection",
        ],
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


def test_transfer_pin_self_service_is_terminal_without_secret_collection(tmp_path: Path) -> None:
    task = _telecom_task(
        tmp_path,
        task_id="TEL-PIN-01",
        hidden_facts={
            "subscriber_id": "sub_4",
            "line_id": "line_4",
            "expected_outcome_type": "transfer_pin_self_service_explained",
            "unsupported_request": "generate_transfer_pin",
            "reason": "transfer_pin_self_service",
        },
        tools={
            "read": ["get_session_context", "get_line_security_summary"],
            "check": [],
            "draft": [],
            "notice": ["explain_transfer_pin_secure_self_service"],
            "submit": ["submit_telecom_security_draft"],
        },
        rules=["no_secret_collection"],
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


def test_telecom_cpni_auth_notice_tools_append_final_state(tmp_path: Path) -> None:
    from cxbench.tools_telecom import TelecomAccountSecuritySystem

    task = _telecom_task(
        tmp_path,
        task_id="TEL-CPNI-TOOLS-01",
        hidden_facts={"subscriber_id": "sub_cpni", "line_id": "line_cpni"},
        tools={
            "read": ["get_session_context"],
            "check": ["check_line_mutation_eligibility", "check_cpni_access_authentication"],
            "draft": [],
            "notice": ["record_failed_auth_attempt", "record_account_change_notice"],
            "submit": [],
        },
        rules=["no_secret_collection"],
        initial_state={
            "auth_attempts": [{"attempt_id": "authfail_seed", "line_id": "line_cpni"}],
            "cpni_access_checks": [],
            "account_change_notices": [],
        },
    )
    system = TelecomAccountSecuritySystem(task, tmp_path / "drafts")

    failed = system.execute(
        "record_failed_auth_attempt",
        {
            "line_id": "line_cpni",
            "channel": "phone",
            "request_type": "cpni_access",
            "auth_method": "ssn_last4",
            "failure_reason": "readily_available_info_rejected",
        },
    )
    eligibility = system.execute(
        "check_line_mutation_eligibility", {"line_id": "line_cpni", "request_type": "sim_change"}
    )
    cpni = system.execute(
        "check_cpni_access_authentication",
        {
            "channel": "phone",
            "cpni_scope": "call_detail_records",
            "requested_fields": ["phone_number", "bill_amount"],
            "auth_method": "ssn_last4",
        },
    )
    notice = system.execute(
        "record_account_change_notice",
        {
            "change_type": "password_reset",
            "line_id": "line_cpni",
            "channel": "email_on_record",
            "old_destination_type": "email_on_record",
        },
    )

    assert failed["ok"] is True
    assert failed["risk_signal"]["risk_signal_type"] == "failed_auth"
    assert eligibility["ok"] is True
    assert eligibility["line_eligibility_check"]["risk_level"] == "high"
    assert eligibility["line_eligibility_check"]["allowed_to_draft"] is False
    assert cpni["ok"] is True
    assert cpni["cpni_access_check"]["uses_readily_available_bio_or_account_info"] is True
    assert cpni["cpni_access_check"]["allowed_to_disclose"] is False
    assert notice["ok"] is True
    assert notice["account_change_notice"]["sent_to_existing_channel"] is True
    assert notice["account_change_notice"]["sent_to_new_information"] is False
    assert notice["account_change_notice"]["reveals_changed_information"] is False

    final_state = system.final_state()
    assert [item["attempt_id"] for item in final_state["auth_attempts"]] == [
        "authfail_seed",
        "authfail_002",
    ]
    assert final_state["risk_signals"][-1]["type"] == "failed_auth"
    assert final_state["line_eligibility_checks"][-1]["check_id"] == "lineelig_001"
    assert final_state["cpni_access_checks"][-1]["check_id"] == "cpni_001"
    assert final_state["account_change_notices"][-1]["notice_id"] == "notice_001"
    assert final_state["notices"][-1]["type"] == "account_change_security_notice"


def test_telecom_billing_tools_append_drafts_and_final_state(tmp_path: Path) -> None:
    from cxbench.tools_telecom import TelecomAccountSecuritySystem

    task = _telecom_task(
        tmp_path,
        task_id="TEL-BILLING-TOOLS-01",
        hidden_facts={"subscriber_id": "sub_bill", "line_id": "line_bill"},
        tools={
            "read": ["get_bill_summary"],
            "check": ["check_charge_authorization"],
            "draft": ["draft_billing_dispute", "draft_third_party_charge_block"],
            "notice": [],
            "submit": [],
        },
        rules=["no_secret_collection"],
        initial_state={
            "bills": [
                {
                    "bill_id": "bill_001",
                    "subscriber_id": "sub_bill",
                    "account_id": "acct_bill",
                    "line_id": "line_bill",
                    "amount_due": 84.50,
                    "currency": "USD",
                    "status": "open",
                }
            ],
            "bill_charges": [
                {
                    "charge_id": "chg_001",
                    "bill_id": "bill_001",
                    "line_id": "line_bill",
                    "provider_name": "QuizFun Alerts",
                    "amount": 14.99,
                    "currency": "USD",
                    "third_party_charge": True,
                    "authorized": False,
                }
            ],
            "charge_authorization_checks": [],
            "billing_disputes": [],
            "third_party_charge_blocks": [],
        },
    )
    system = TelecomAccountSecuritySystem(task, tmp_path / "drafts")

    bill = system.execute("get_bill_summary", {"bill_id": "bill_001"})
    check = system.execute(
        "check_charge_authorization",
        {"charge_id": "chg_001", "customer_reported_authorized": False},
    )
    dispute = system.execute(
        "draft_billing_dispute",
        {"charge_id": "chg_001", "reason": "unauthorized_third_party_charge"},
    )
    block = system.execute(
        "draft_third_party_charge_block", {"line_id": "line_bill", "bill_id": "bill_001"}
    )

    assert bill["ok"] is True
    assert bill["charges"][0]["charge_id"] == "chg_001"
    assert check["ok"] is True
    assert check["charge_authorization_check"]["allowed_to_bill"] is False
    assert check["charge_authorization_check"]["prohibited_if_unauthorized"] is True
    assert dispute["ok"] is True
    assert dispute["billing_dispute"]["status"] == "drafted"
    assert dispute["draft_type"] == "billing_dispute"
    assert block["ok"] is True
    assert block["third_party_charge_block"]["status"] == "drafted"
    assert block["draft_type"] == "third_party_charge_block"

    final_state = system.final_state()
    assert final_state["bills"][0]["bill_id"] == "bill_001"
    assert final_state["bill_charges"][0]["charge_id"] == "chg_001"
    assert final_state["charge_authorization_checks"][-1]["check_id"] == "chargeauth_001"
    assert final_state["billing_disputes"][-1]["dispute_id"] == "billdisp_001"
    assert final_state["third_party_charge_blocks"][-1]["block_id"] == "tpblock_001"
    assert [draft["draft_type"] for draft in final_state["drafts"]] == [
        "billing_dispute",
        "third_party_charge_block",
    ]
    assert "third_party_charge_block_active" not in final_state["lines"][0]
