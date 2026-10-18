from __future__ import annotations

from pathlib import Path

import yaml

from cxbench.task_schema import TaskSpec
from cxbench.tools_utilities import UtilitiesCustomerProtectionSystem


MATRIX = Path("data/scenario_matrices/utilities_disconnection_v1.yaml")
SEEDS = Path("data/scenario_matrices/utilities_compliance_flow_seeds.yaml")


def _matrix() -> list[dict]:
    payload = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    return list(payload["scenarios"])


def test_utilities_v1_matrix_has_expected_size_and_outcomes():
    scenarios = _matrix()

    assert len(scenarios) == 20
    outcomes = {item["task"]["hidden_facts"]["expected_outcome_type"] for item in scenarios}
    assert outcomes == {"payment_arrangement", "medical_hold", "protection_hold", "bill_dispute"}


def test_utilities_v1_each_task_has_scene_specific_judge_checks():
    for item in _matrix():
        task = item["task"]
        checklist = task["policy"]["llm_judge"]["checklist"]
        assert task["hidden_facts"]["expected_outcome_type"]
        assert len(checklist) >= 3
        assert all(check["id"] != "generated_process_requirements_followed" for check in checklist)
        assert any(task["scenario"].replace("_", "-")[:12] in check["id"] for check in checklist)


def test_utilities_flow_seeds_cover_required_protection_paths():
    payload = yaml.safe_load(SEEDS.read_text(encoding="utf-8"))
    seeds = list(payload["seeds"])
    joined = "\n".join(f"{seed['id']} {seed['flow']}" for seed in seeds).lower()

    assert 8 <= len(seeds) <= 12
    for term in ["medical", "vulnerable", "weather", "dispute", "puc", "payment"]:
        assert term in joined
    for seed in seeds:
        assert seed["id"]
        assert seed["source_refs"]
        assert seed["flow"]
        assert seed["task_implications"]


def _texas_task() -> TaskSpec:
    return TaskSpec.from_mapping(
        {
            "id": "utl-texas-unit",
            "title": "Texas utilities unit slice",
            "domain": "utilities",
            "business_system": "utilities_disconnection_v0",
            "scenario": "texas_utilities_unit",
            "customer_goal": "Resolve a disconnection risk.",
            "seed_message": "I need help with my electric bill.",
            "hidden_facts": {
                "customer_id": "cust_1",
                "account_id": "acct_1",
                "notice_id": "notice_1",
                "bill_id": "bill_1",
                "arrears_amount": 600.0,
            },
            "tools": {"read": [], "check": [], "draft": [], "notice": []},
            "confirmation_contract": [],
            "rules": [],
            "initial_state": {
                "session_context": {"channel": "phone"},
                "customer": {"customer_id": "cust_1", "display_name": "R. Rivera"},
                "utility_accounts": [
                    {
                        "account_id": "acct_1",
                        "customer_id": "cust_1",
                        "service_tenure_months": 18,
                    }
                ],
                "bill_ledgers": [
                    {
                        "account_id": "acct_1",
                        "bill_id": "bill_1",
                        "due_date": "2026-06-01",
                        "arrears_amount": 600.0,
                        "weather_emergency_due_bill": True,
                        "underbilling_amount": 75.0,
                    }
                ],
                "disconnection_notices": [
                    {
                        "notice_id": "notice_1",
                        "account_id": "acct_1",
                        "issued_date": "2026-06-03",
                        "scheduled_disconnect_date": "2026-06-17",
                        "channel": "email",
                        "email_subject": "Disconnection notice",
                        "is_separate_notice": True,
                        "personnel_available_on_disconnect_date": True,
                        "content_items": [
                            "reason",
                            "cure_actions",
                            "fees",
                            "overdue_amount",
                            "puc_complaint_contact",
                        ],
                        "charges_basis": ["electric_service"],
                    }
                ],
                "protection_profiles": [
                    {
                        "account_id": "acct_1",
                        "expressed_inability_to_pay": True,
                        "is_critical_care": True,
                    }
                ],
                "payment_options": [{"account_id": "acct_1", "option_id": "deferred"}],
                "prior_payment_arrangements": [
                    {
                        "arrangement_id": "arr_old",
                        "account_id": "acct_1",
                        "status": "defaulted",
                        "missed_payment_date": "2026-06-10",
                        "eligible_for_new_plan": True,
                        "requires_new_notice": False,
                        "reinstatement_option_id": "reinstate_1",
                    }
                ],
                "critical_care_profiles": [
                    {
                        "account_id": "acct_1",
                        "condition_type": "critical_care",
                        "physician_contact_received": True,
                        "deferred_plan_entered": True,
                        "protection_expires_at": "2026-08-01",
                    }
                ],
                "medical_documentation": [
                    {
                        "account_id": "acct_1",
                        "status": "received",
                        "physician_written_statement_received": True,
                    }
                ],
                "energy_assistance_pledges": [
                    {"account_id": "acct_1", "provider_id": "agency_1", "pledged_amount": 150.0}
                ],
            },
            "policy": {"mandatory_disclosure": "Customer rights disclosure."},
        },
        Path("inline.yaml"),
    )


def test_texas_utilities_check_tools_append_state_and_final_state(tmp_path):
    system = UtilitiesCustomerProtectionSystem(task=_texas_task(), draft_dir=tmp_path)

    notice = system.execute(
        "check_texas_disconnection_notice_requirements",
        {"notice_id": "notice_1", "account_id": "acct_1"},
    )
    plan = system.execute("check_texas_payment_plan_eligibility", {"account_id": "acct_1"})
    prior = system.execute("read_prior_payment_arrangements", {"account_id": "acct_1"})
    default = system.execute(
        "check_arrangement_default_reinstatement",
        {"account_id": "acct_1", "arrangement_id": "arr_old"},
    )
    charge = system.execute(
        "check_bill_charge_disconnectability",
        {"account_id": "acct_1", "bill_id": "bill_1", "notice_id": "notice_1"},
    )
    medical = system.execute(
        "check_critical_care_or_chronic_condition_requirements", {"account_id": "acct_1"}
    )

    assert notice["ok"] is True
    assert notice["texas_notice_requirements"]["valid"] is True
    assert plan["texas_payment_plan_eligibility"]["eligible"] is True
    assert plan["texas_payment_plan_eligibility"]["deferred_plan_required"] is True
    assert prior["prior_payment_arrangements"][0]["arrangement_id"] == "arr_old"
    assert default["arrangement_default_reinstatement"]["defaulted"] is True
    assert charge["bill_charge_disconnectability"]["disconnectable"] is True
    assert medical["critical_care_or_chronic_condition_requirements"]["requirements_met"] is True

    final = system.final_state()
    assert final["notice_checks"][0]["type"] == "texas_disconnection_notice_requirements"
    assert final["payment_plan_checks"][0]["type"] == "texas_payment_plan_eligibility"
    assert final["prior_payment_arrangements"][0]["arrangement_id"] == "arr_old"
    assert final["arrangement_default_checks"][0]["reinstatement_option_id"] == "reinstate_1"
    assert final["bill_charge_checks"][0]["disconnectable"] is True
    assert final["critical_care_profiles"][0]["condition_type"] == "critical_care"
    assert len(final["checks"]) == 5


def test_texas_utilities_draft_and_disclosure_tools_append_state_and_final_state(tmp_path):
    system = UtilitiesCustomerProtectionSystem(task=_texas_task(), draft_dir=tmp_path)

    deferred = system.execute(
        "draft_texas_deferred_payment_plan",
        {
            "account_id": "acct_1",
            "notice_id": "notice_1",
            "arrears_amount": 600.0,
            "total_plan_amount": 600.0,
            "initial_payment_amount": 100.0,
            "installment_count": 5,
            "switch_hold_applies": True,
            "switch_hold_disclosure_included": True,
            "disconnection_suspended_until": "2026-07-15",
        },
    )
    investigation = system.execute(
        "draft_bill_investigation",
        {
            "account_id": "acct_1",
            "bill_id": "bill_1",
            "notice_id": "notice_1",
            "disputed_amount": 75.0,
            "undisputed_amount": 525.0,
        },
    )
    disclosure = system.execute(
        "record_assistance_program_disclosure",
        {"account_id": "acct_1", "program_ids": ["agency_1"]},
    )

    assert deferred["ok"] is True
    assert investigation["ok"] is True
    assert disclosure["ok"] is True
    assert (tmp_path / "draft_001.json").exists()
    assert (tmp_path / "draft_002.json").exists()

    final = system.final_state()
    assert final["deferred_payment_plan_drafts"][0]["switch_hold_applies"] is True
    assert final["switch_holds"][0]["disclosure_included"] is True
    assert final["bill_investigations"][0]["customer_result_notice"] is True
    assert final["bill_disputes"][0]["status"] == "investigation_drafted"
    assert final["assistance_disclosures"][0]["program_ids"] == ["agency_1"]
    assert final["energy_assistance_pledges"][0]["pledged_amount"] == 150.0
    assert final["disclosures"][0]["type"] == "assistance_programs"
