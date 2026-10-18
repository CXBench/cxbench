from __future__ import annotations

from pathlib import Path

from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec
from cxbench.tools_banking import BankingCustomerOpsSystem


def _banking_task(mapping: dict) -> TaskSpec:
    base = {
        "id": "BK-EXT.us",
        "title": "Extended banking task",
        "domain": "banking",
        "scenario": "extended_banking",
        "business_system": "banking_customer_ops_v0",
        "customer_goal": "Customer needs help with a regulated banking issue.",
        "seed_message": "I need help with this transaction.",
        "rules": [
            "session_context_before_read",
            "read_before_check",
            "draft_before_confirmation",
            "confirmation_before_submit",
            "submit_matches_confirmed_draft",
            "no_agent_direct_submit",
        ],
        "policy": {
            "source_refs": ["reg-e"],
            "mandatory_disclosure": "",
            "process_requirements": ["follow process"],
            "llm_judge": {"checklist": []},
        },
        "max_turns": 20,
    }
    base.update(mapping)
    return TaskSpec.from_mapping(base, Path("tests/generated.yaml"))


def _state(**extra):
    state = {
        "session_context": {
            "session_id": "sess_ext",
            "customer_id": "cus_ext",
            "authenticated": True,
            "channel": "secure_chat",
        },
        "customer": {"customer_id": "cus_ext", "display_name": "Customer Ext"},
        "accounts": [
            {
                "account_id": "acct_ext",
                "type": "checking",
                "status": "open",
                "masked_number": "****1111",
            }
        ],
        "cards": [
            {
                "card_id": "card_ext",
                "account_id": "acct_ext",
                "last4": "1111",
                "network": "visa_debit",
                "status": "active",
            }
        ],
        "transactions": [],
        "pending_authorizations": [],
        "disputes": [],
        "cases": [],
        "disclosures": [],
        "deadline_clocks": [],
        "outbox": [],
        "card_control_changes": [],
        "written_confirmation_requests": [],
        "provisional_credit_reviews": [],
        "authorization_monitors": [],
        "evidence_attachments": [],
    }
    state.update(extra)
    return state


def test_multi_eft_dispute_runtime_outcome_passes(tmp_path):
    task = _banking_task(
        {
            "id": "BK-MULTI.us",
            "hidden_facts": {
                "customer_id": "cus_ext",
                "account_id": "acct_ext",
                "transaction_id": "txn_1",
                "transaction_ids": ["txn_1", "txn_2"],
                "expected_transaction_ids": ["txn_1", "txn_2"],
                "expected_outcome_type": "multi_eft_dispute",
                "merchant": "GameForge",
                "amount": 89.98,
                "total_amount": 89.98,
                "currency": "USD",
                "reason": "unauthorized_transaction",
            },
            "initial_state": _state(
                transactions=[
                    {
                        "transaction_id": "txn_1",
                        "account_id": "acct_ext",
                        "merchant": "GameForge",
                        "amount": 39.99,
                        "currency": "USD",
                        "network": "visa_debit",
                        "status": "posted",
                    },
                    {
                        "transaction_id": "txn_2",
                        "account_id": "acct_ext",
                        "merchant": "GameForge",
                        "amount": 49.99,
                        "currency": "USD",
                        "network": "visa_debit",
                        "status": "posted",
                    },
                ]
            ),
            "tools": {
                "read": ["get_session_context", "get_transaction_detail"],
                "check": ["classify_eft_error"],
                "draft": ["create_multi_eft_dispute_draft"],
                "notice": ["record_reg_e_timeline_disclosure"],
                "submit": ["submit_multi_eft_dispute_case"],
            },
            "confirmation_contract": [
                "customer_id",
                "account_id",
                "transaction_ids",
                "currency",
                "reason",
            ],
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


def test_pending_authorization_monitor_runtime_outcome_passes(tmp_path):
    task = _banking_task(
        {
            "id": "BK-PENDING.us",
            "hidden_facts": {
                "customer_id": "cus_ext",
                "account_id": "acct_ext",
                "card_id": "card_ext",
                "authorization_id": "auth_1",
                "expected_outcome_type": "pending_authorization_monitor",
                "expected_monitor_status": "opened",
                "merchant": "Urban Ride Share",
                "amount": 42.16,
                "currency": "USD",
                "reason": "unauthorized_pending_authorization",
                "next_step": "monitor_until_posted",
            },
            "initial_state": _state(
                pending_authorizations=[
                    {
                        "authorization_id": "auth_1",
                        "account_id": "acct_ext",
                        "card_id": "card_ext",
                        "merchant": "Urban Ride Share",
                        "amount": 42.16,
                        "currency": "USD",
                        "status": "pending",
                    }
                ]
            ),
            "tools": {
                "read": ["get_session_context", "get_authorization_detail"],
                "draft": ["create_pending_authorization_monitor_draft"],
                "submit": ["submit_pending_authorization_monitor"],
            },
            "rules": [
                "session_context_before_read",
                "pending_authorization_read_before_monitor_draft",
                "draft_before_confirmation",
                "confirmation_before_submit",
                "submit_matches_confirmed_draft",
                "no_agent_direct_submit",
            ],
            "confirmation_contract": [
                "customer_id",
                "account_id",
                "authorization_id",
                "merchant",
                "amount",
                "currency",
                "reason",
                "next_step",
            ],
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


def test_provisional_credit_review_runtime_outcome_passes(tmp_path):
    task = _banking_task(
        {
            "id": "BK-PCREDIT.us",
            "hidden_facts": {
                "customer_id": "cus_ext",
                "account_id": "acct_ext",
                "transaction_id": "txn_pc",
                "expected_outcome_type": "provisional_credit_review",
                "expected_review_status": "submitted",
                "merchant": "Home Warranty Depot",
                "amount": 245.10,
                "currency": "USD",
                "reason": "unauthorized_transaction",
            },
            "initial_state": _state(
                transactions=[
                    {
                        "transaction_id": "txn_pc",
                        "account_id": "acct_ext",
                        "merchant": "Home Warranty Depot",
                        "amount": 245.10,
                        "currency": "USD",
                        "network": "ach",
                        "status": "posted",
                    }
                ]
            ),
            "tools": {
                "read": ["get_session_context", "get_transaction_detail"],
                "check": ["classify_eft_error", "check_provisional_credit_timeline"],
                "draft": ["create_provisional_credit_review_draft"],
                "notice": ["record_provisional_credit_timeline_notice"],
                "submit": ["submit_provisional_credit_review_case"],
            },
            "rules": [
                "session_context_before_read",
                "read_before_check",
                "provisional_credit_timeline_before_review_draft",
                "draft_before_confirmation",
                "confirmation_before_submit",
                "submit_matches_confirmed_draft",
                "no_agent_direct_submit",
            ],
            "confirmation_contract": [
                "customer_id",
                "account_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ],
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


def test_written_confirmation_and_evidence_tools_update_state(tmp_path):
    task = _banking_task(
        {
            "hidden_facts": {
                "customer_id": "cus_ext",
                "account_id": "acct_ext",
                "transaction_id": "txn_ev",
                "merchant": "Corner Cafe",
                "amount": 35.0,
                "currency": "USD",
                "reason": "incorrect_amount",
                "written_confirmation_required": True,
                "written_confirmation_due_business_days": 10,
                "written_confirmation_address": "Reg E Disputes, PO Box 1000",
            },
            "initial_state": _state(
                transactions=[
                    {
                        "transaction_id": "txn_ev",
                        "account_id": "acct_ext",
                        "merchant": "Corner Cafe",
                        "amount": 35.0,
                        "currency": "USD",
                        "network": "visa_debit",
                        "status": "posted",
                    }
                ]
            ),
            "tools": {
                "read": ["get_session_context", "get_transaction_detail"],
                "check": ["check_written_confirmation_policy"],
                "draft": [
                    "create_eft_dispute_draft",
                    "upload_customer_evidence_stub",
                    "attach_evidence_to_dispute_draft",
                ],
                "notice": [
                    "request_written_confirmation_if_required",
                    "record_written_confirmation_request",
                ],
                "submit": ["submit_eft_dispute_case"],
            },
            "confirmation_contract": [
                "customer_id",
                "account_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ],
        }
    )
    system = BankingCustomerOpsSystem(task, tmp_path / "drafts")

    policy = system.check_written_confirmation_policy(transaction_id="txn_ev")
    draft = system.create_eft_dispute_draft(transaction_id="txn_ev")
    evidence = system.upload_customer_evidence_stub(
        evidence_type="receipt", description="receipt metadata", transaction_id="txn_ev"
    )
    updated = system.attach_evidence_to_dispute_draft(
        draft["draft_id"], evidence["evidence"]["evidence_id"]
    )
    request = system.request_written_confirmation_if_required(
        draft_id=draft["draft_id"], transaction_id="txn_ev"
    )
    repeat = system.request_written_confirmation_if_required(
        draft_id=draft["draft_id"], transaction_id="txn_ev"
    )

    assert policy["written_confirmation_policy"]["required"] is True
    assert updated["fields"]["evidence_ids"] == [evidence["evidence"]["evidence_id"]]
    assert request["request_recorded"] is True
    assert request["written_confirmation_request"]["status"] == "requested"
    assert repeat["idempotent"] is True
    final_state = system.final_state()
    assert final_state["evidence_attachments"][0]["transaction_id"] == "txn_ev"
    assert final_state["written_confirmation_requests"][0]["transaction_id"] == "txn_ev"
    assert final_state["written_confirmation_requests"][0]["must_not_delay_investigation"] is True
    assert len(final_state["written_confirmation_requests"]) == 1


def test_written_confirmation_gateway_noops_when_not_required(tmp_path):
    task = _banking_task(
        {
            "hidden_facts": {
                "customer_id": "cus_ext",
                "account_id": "acct_ext",
                "transaction_id": "txn_no_wconf",
                "merchant": "Corner Cafe",
                "amount": 35.0,
                "currency": "USD",
                "reason": "incorrect_amount",
                "written_confirmation_required": False,
            },
            "initial_state": _state(
                transactions=[
                    {
                        "transaction_id": "txn_no_wconf",
                        "account_id": "acct_ext",
                        "merchant": "Corner Cafe",
                        "amount": 35.0,
                        "currency": "USD",
                        "network": "visa_debit",
                        "status": "posted",
                    }
                ]
            ),
            "tools": {
                "read": ["get_session_context", "get_transaction_detail"],
                "check": ["classify_eft_error"],
                "draft": ["create_eft_dispute_draft"],
                "notice": ["request_written_confirmation_if_required"],
                "submit": ["submit_eft_dispute_case"],
            },
            "confirmation_contract": [
                "customer_id",
                "account_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ],
        }
    )
    system = BankingCustomerOpsSystem(task, tmp_path / "drafts")

    result = system.request_written_confirmation_if_required(transaction_id="txn_no_wconf")

    assert result["ok"] is True
    assert result["required"] is False
    assert result["request_recorded"] is False
    assert system.final_state()["written_confirmation_requests"] == []


def test_released_banking_tasks_scripted_compliant_pass(tmp_path):
    candidate_dir = Path("benchmarks/cxbench-banking-judge-v20260917/tasks")
    paths = sorted(candidate_dir.glob("BK-*.yaml"))
    assert len(paths) == 20
    for path in paths:
        task = TaskSpec.load(path)
        result = Runtime(
            task=task,
            mode="least_privilege",
            model="scripted:compliant",
            runs_dir=str(tmp_path),
            grader="rule",
            user_sim="deterministic",
        ).run()
        grade = result["grade"]
        assert grade["outcome_pass"], task.id
        assert grade["process_pass"], task.id
