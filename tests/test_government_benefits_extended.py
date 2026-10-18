from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from cxbench.task_schema import TaskSpec
from cxbench.tools_benefits import GovernmentBenefitsAppealSystem


MATRIX = Path("data/scenario_matrices/government_benefits_appeal_v1.yaml")
RELEASE = Path("benchmarks/cxbench-120")


def _tasks() -> list[dict]:
    payload = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    return [item["task"] for item in payload["scenarios"]]


def test_government_benefits_matrix_has_20_scene_specific_tasks():
    tasks = _tasks()

    assert len(tasks) == 20
    assert {task["domain"] for task in tasks} == {"government_benefits"}
    assert {task["hidden_facts"]["program"] for task in tasks} >= {"SNAP", "Medicaid"}
    assert all(
        task["hidden_facts"].get("expected_outcome_type") == "hearing_request" for task in tasks
    )

    for task in tasks:
        checklist = task["policy"]["llm_judge"]["checklist"]
        assert len(checklist) >= 4
        assert {item["id"] for item in checklist} != {"generated_process_requirements_followed"}
        joined = " ".join(item["pass_condition"] for item in checklist).lower()
        assert task["scenario"] in " ".join(item["id"] for item in checklist)
        assert "notice" in joined
        assert "case-file" in joined
        assert "continued" in joined or "aid-pending" in joined
        assert "promise" in joined or "guarantee" in joined


def test_government_benefits_access_and_aid_pending_coverage():
    tasks = _tasks()

    access_tasks = [task for task in tasks if task["hidden_facts"]["accommodation_requested"]]
    aid_pending_tasks = [
        task for task in tasks if task["hidden_facts"]["continued_benefits_elected"]
    ]
    non_aid_tasks = [
        task for task in tasks if not task["hidden_facts"]["continued_benefits_elected"]
    ]

    assert len(access_tasks) >= 5
    assert any(task["hidden_facts"]["language"] == "es" for task in access_tasks)
    assert any("large" in task["hidden_facts"]["accommodation"].lower() for task in access_tasks)
    assert aid_pending_tasks
    assert non_aid_tasks

    for task in access_tasks:
        assert "draft_language_or_disability_accommodation" in task["tools"]["draft"]
        assert "accommodation_before_hearing_draft" in task["rules"]


def test_government_benefits_release_matches_manifest():
    manifest = json.loads((RELEASE / "release_manifest.json").read_text(encoding="utf-8"))
    entries = [item for item in manifest["task_files"] if item["domain"] == "government_benefits"]

    assert len(entries) == 20
    assert len({item["id"] for item in entries}) == 20
    for entry in entries:
        task_path = RELEASE / entry["file"]
        task = TaskSpec.load(task_path)
        assert task.id == entry["id"]
        assert task.domain == "government_benefits"
        assert hashlib.sha256(task_path.read_bytes()).hexdigest() == entry["sha256"]


def _benefits_task(initial_state: dict) -> TaskSpec:
    return TaskSpec.from_mapping(
        {
            "id": "GB-TEST-001",
            "title": "Benefits tool slice test",
            "domain": "government_benefits",
            "scenario": "benefits_tool_slice",
            "business_system": "government_benefits_appeal_v0",
            "customer_goal": "Prepare a hearing request with access supports.",
            "seed_message": "I need help with this notice.",
            "hidden_facts": {
                "claimant_id": "claimant_1",
                "benefit_case_id": "case_1",
                "program": "Medicaid",
            },
            "tools": {"read": [], "check": [], "draft": [], "submit": []},
            "confirmation_contract": [],
            "rules": [],
            "initial_state": initial_state,
            "policy": {"mandatory_disclosure": "You may request a fair hearing."},
        },
        Path("inline-benefits-task.yaml"),
    )


def test_benefits_extended_tools_append_state_and_final_state(tmp_path):
    system = GovernmentBenefitsAppealSystem(
        task=_benefits_task(
            {
                "session_context": {
                    "claimant_id": "claimant_1",
                    "actor_id": "rep_1",
                    "language": "es",
                    "access_need": "large_print",
                },
                "benefit_case": {
                    "benefit_case_id": "case_1",
                    "program": "Medicaid",
                    "status": "active",
                },
                "adverse_action_notice": {
                    "notice_id": "snap_notice_1",
                    "benefit_case_id": "case_1",
                    "program": "SNAP",
                    "action": "reduce_allotment",
                    "reason": "budget_change",
                    "contents": {
                        "timely_10_day": True,
                        "has_hearing_right": True,
                        "has_office_contact": True,
                        "has_continued_benefits_text": True,
                        "has_overissuance_liability_text": True,
                        "has_legal_services_text": True,
                    },
                },
                "benefits_program_profiles": [
                    {
                        "profile_id": "us_federal_medicaid",
                        "program": "Medicaid",
                        "jurisdiction": "us_federal",
                        "source_profile": "us_federal_medicaid",
                        "supported_workflow": True,
                    }
                ],
                "benefit_service_notices": [
                    {
                        "notice_id": "med_notice_1",
                        "benefit_case_id": "case_1",
                        "program": "Medicaid",
                        "action": "reduce_service",
                        "reason": "medical_necessity_review",
                        "effective_date": "2026-07-01",
                        "notice_date": "2026-06-15",
                        "benefit_or_service_impact": "home_care_hours_reduced",
                        "service_type": "home_care",
                        "service_units": "20_hours_weekly",
                        "action_category": "covered_service",
                        "maintenance_of_services": {
                            "notice_type": "10_day",
                            "request_date": "2026-06-20",
                            "requested_before_action_date": True,
                            "services_maintained": True,
                        },
                        "renewal_id": "renewal_1",
                    }
                ],
                "medicaid_renewals": [
                    {
                        "renewal_id": "renewal_1",
                        "ex_parte_attempted": True,
                        "prepopulated_form_sent": True,
                        "response_deadline_days": 30,
                        "termination_date": "2026-07-01",
                        "submitted_within_90_days": True,
                        "other_affordability_program_screened": True,
                    }
                ],
                "case_file_indexes": [
                    {
                        "notice_id": "med_notice_1",
                        "benefit_case_id": "case_1",
                        "available_sections": ["notice", "service_plan", "internal_note"],
                        "hearing_relevant_sections": ["notice", "service_plan"],
                        "confidential_redactions": ["worker_phone"],
                        "excluded_uncontestable_materials": ["internal_note"],
                    }
                ],
                "language_access_needs": [
                    {
                        "language": "es",
                        "interpreter_required": True,
                        "translator_required": True,
                        "qualified_provider_required": True,
                        "critical_document_human_review": True,
                    }
                ],
                "disability_access_needs": [
                    {
                        "disability_access_need": "large_print",
                        "auxiliary_aid": "large_print_notice",
                        "format": "large_print",
                        "hearing_modality": "phone",
                        "reasonable_modification": "extra_response_time",
                    }
                ],
                "benefits_representative_authorities": [
                    {
                        "representative_id": "rep_1",
                        "claimant_id": "claimant_1",
                        "authorization_type": "written_designation",
                        "authorization_on_file": True,
                        "scope": ["request_hearing", "access_case_file", "receive_notice"],
                    }
                ],
            }
        ),
        draft_dir=tmp_path,
    )

    assert system.execute("check_benefits_program_profile", {"program": "Medicaid"})["ok"] is True
    assert (
        system.execute("lookup_benefit_service_notice", {"notice_id": "med_notice_1"})["notice"][
            "service_type"
        ]
        == "home_care"
    )
    assert (
        system.execute("check_snap_notice_contents", {"notice_id": "snap_notice_1"})[
            "snap_notice_contents"
        ]["adequate"]
        is True
    )
    assert (
        system.execute("check_medicaid_hearing_right", {"notice_id": "med_notice_1"})[
            "medicaid_hearing_right"
        ]["hearing_required"]
        is True
    )
    assert (
        system.execute("check_medicaid_maintenance_of_services", {"notice_id": "med_notice_1"})[
            "medicaid_maintenance_of_services"
        ]["services_maintained"]
        is True
    )
    assert (
        system.execute("check_medicaid_renewal_procedure", {"renewal_id": "renewal_1"})[
            "medicaid_renewal_procedure"
        ]["ex_parte_attempted"]
        is True
    )
    assert system.execute("lookup_case_file_index", {"notice_id": "med_notice_1"})[
        "case_file_index"
    ]["available_sections"]
    assert (
        system.execute(
            "check_case_file_release_scope",
            {"notice_id": "med_notice_1", "requested_sections": ["notice", "service_plan"]},
        )["case_file_release_scope"]["release_scope_ok"]
        is True
    )
    assert (
        system.execute("check_language_access_need", {"notice_id": "med_notice_1"})[
            "language_access_need"
        ]["language"]
        == "es"
    )
    language_draft = system.execute(
        "draft_language_assistance_request", {"notice_id": "med_notice_1"}
    )
    assert language_draft["ok"] is True
    assert (
        system.execute("check_disability_access_need", {"notice_id": "med_notice_1"})[
            "disability_access_need"
        ]["format"]
        == "large_print"
    )
    disability_draft = system.execute(
        "draft_disability_accommodation_request", {"notice_id": "med_notice_1"}
    )
    assert disability_draft["ok"] is True
    assert (
        system.execute("check_benefits_representative_authority", {"scope": "access_case_file"})[
            "representative_authority"
        ]["can_act"]
        is True
    )

    final_state = system.final_state()
    assert final_state["benefits_program_profile_checks"]
    assert final_state["snap_notice_content_checks"]
    assert final_state["notice_adequacy_checks"]
    assert final_state["medicaid_hearing_right_checks"]
    assert final_state["medicaid_maintenance_of_services_checks"]
    assert final_state["continued_benefits_checks"]
    assert final_state["medicaid_renewal_procedure_checks"]
    assert final_state["case_file_release_scope_checks"]
    assert final_state["language_access_checks"]
    assert final_state["disability_access_checks"]
    assert final_state["benefits_representative_authority_checks"]
    assert final_state["language_assistance_requests"][0]["draft_id"] == language_draft["draft_id"]
    assert (
        final_state["disability_accommodation_requests"][0]["draft_id"]
        == disability_draft["draft_id"]
    )
    assert final_state["benefit_service_notices"][0]["notice_id"] == "med_notice_1"
