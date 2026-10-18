import copy
from dataclasses import replace
from pathlib import Path

import yaml

from cxbench.task_schema import TaskSpec
from cxbench.tools_healthcare import HealthcareAppealSystem
from cxbench.adapters import _tool_schema


MATRIX = Path("data/scenario_matrices/healthcare_appeal_v1.yaml")
CANDIDATE_DIR = Path("benchmarks/cxbench-120-v20260708/tasks")


def _load_matrix_tasks():
    raw = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    return [item["task"] for item in raw["scenarios"]]


def test_healthcare_matrix_has_20_expected_outcomes_and_scene_checklists():
    tasks = _load_matrix_tasks()
    assert len(tasks) == 20
    for task in tasks:
        assert task["hidden_facts"].get("expected_outcome_type")
        checklist = task["policy"]["llm_judge"]["checklist"]
        assert len(checklist) >= 3
        assert {item["id"] for item in checklist} != {"generated_process_requirements_followed"}
        assert any(task["scenario"] in item["id"] for item in checklist)


def test_healthcare_generated_source_refs_cover_appeal_obligations():
    def has_ref(refs, ref):
        return ref in refs or f"{ref}-r001" in refs

    task_paths = sorted(CANDIDATE_DIR.glob("HC-*.yaml"))
    assert len(task_paths) == 20
    for task_path in task_paths:
        task = TaskSpec.load(task_path)
        refs = set(task.policy.get("source_refs", []))
        expected = task.hidden_facts.get("expected_outcome_type")
        if expected == "external_review_request_submitted":
            assert has_ref(refs, "cfr-45-147.136-external-review")
            assert has_ref(refs, "usc-42-18116-external-review")
        elif expected == "expedited_internal_appeal_submitted":
            assert has_ref(refs, "cfr-45-147.136-urgent-care-appeals")
        else:
            assert has_ref(refs, "cfr-45-147.136-internal-appeals-external-review")
        assert has_ref(refs, "cfr-45-164.514-d-minimum-necessary")


def test_healthcare_external_review_scenario_exposes_external_draft_tool(tmp_path):
    task_path = CANDIDATE_DIR / "HC-V1-005.us.yaml"
    task = TaskSpec.load(task_path)
    assert task.hidden_facts["expected_outcome_type"] == "external_review_request_submitted"
    assert "draft_external_review_request" in task.tools["draft"]

    system = HealthcareAppealSystem(task, tmp_path / "drafts")
    auth = system.check_authorized_representative(member_id=task.hidden_facts["member_id"])
    assert auth["ok"] is True
    system.log_phi_access_purpose(
        purpose="prepare_external_review",
        denial_id=task.hidden_facts["denial_id"],
        requested_categories=["claim_summary", "denial_notice", "plan_terms"],
    )
    claim = system.get_minimum_necessary_claim_summary(
        claim_id=task.hidden_facts["claim_id"],
        denial_id=task.hidden_facts["denial_id"],
    )
    assert claim["ok"] is True
    draft = system.draft_external_review_request(**task.hidden_facts)
    assert draft["ok"] is True
    assert draft["draft_type"] == "external_review"
    submitted = system.submit_draft(draft["draft_id"])
    assert submitted["ok"] is True
    assert submitted["appeal_case"]["requested_action"] == "external_review_request"


def test_healthcare_denial_summary_defaults_to_active_scoped_denial(tmp_path):
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    result = system.get_denial_notice_summary()

    assert result["ok"] is True
    assert result["denial_notice"]["denial_id"] == task.hidden_facts["denial_id"]
    assert result["denial_notice"]["claim_id"] == task.hidden_facts["claim_id"]


def test_healthcare_phi_access_categories_are_enforced(tmp_path):
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    result = system.log_phi_access_purpose(
        purpose="prepare_coverage_appeal",
        requested_categories=["claim_summary", "full_medical_record"],
    )

    assert result["ok"] is False
    assert result["error"] == "unsupported_phi_category"
    assert result["allowed_categories"] == ["claim_summary", "denial_notice", "plan_terms"]


def test_healthcare_tool_schema_exposes_phi_category_allowlist():
    schema = _tool_schema("log_phi_access_purpose")
    properties = schema["function"]["parameters"]["properties"]

    assert properties["requested_categories"]["items"]["enum"] == [
        "claim_summary",
        "denial_notice",
        "plan_terms",
    ]
    assert "Allowed PHI categories" in properties["requested_categories"]["description"]


def test_healthcare_tool_schema_documents_no_arg_denial_lookup():
    schema = _tool_schema("get_denial_notice_summary")
    properties = schema["function"]["parameters"]["properties"]

    assert schema["function"]["parameters"]["required"] == []
    assert "Omit to read the active/scoped denial" in properties["denial_id"]["description"]


def test_healthcare_urgent_scenario_records_clinical_handoff(tmp_path):
    task = TaskSpec.load(CANDIDATE_DIR / "HC-V1-003.us.yaml")
    system = HealthcareAppealSystem(task, tmp_path / "drafts")
    urgent = system.check_urgent_care_or_continuity_flags(task.hidden_facts["denial_id"])
    assert urgent["ok"] is True
    assert urgent["urgent_care_check"]["urgent_care"] is True
    assert urgent["urgent_care_check"]["requires_clinical_handoff"] is True
    assert system.final_state()["handoffs"] == [
        {"type": "clinical_review", "denial_id": task.hidden_facts["denial_id"]}
    ]


def test_healthcare_draft_tool_translates_natural_language_to_codes(tmp_path):
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    draft = system.draft_internal_appeal(
        member_id=task.hidden_facts["member_id"],
        representative_id=task.hidden_facts["representative_id"],
        plan_id=task.hidden_facts["plan_id"],
        denial_id=task.hidden_facts["denial_id"],
        claim_id=task.hidden_facts["claim_id"],
        requested_action="overturn the prior authorization denial through an internal appeal",
        appeal_reason="The MRI denial should be appealed with supporting medical necessity records.",
    )

    assert draft["ok"] is True
    fields = draft["fields"]
    assert fields["requested_action_code"] == "internal_appeal"
    assert fields["appeal_basis_code"] == "medical_necessity_supporting_records_available"
    assert fields["appeal_narrative"]
    assert fields["customer_statement"] == ""
    assert "appeal_summary" in fields
    # Legacy aliases stay present for old tasks and outcome graders.
    assert fields["requested_action"] == fields["requested_action_code"]
    assert fields["appeal_reason"] == fields["appeal_basis_code"]


def test_healthcare_draft_tool_separates_narrative_and_customer_statement(tmp_path):
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    draft = system.draft_internal_appeal(
        member_id=task.hidden_facts["member_id"],
        representative_id=task.hidden_facts["representative_id"],
        plan_id=task.hidden_facts["plan_id"],
        denial_id=task.hidden_facts["denial_id"],
        claim_id=task.hidden_facts["claim_id"],
        requested_action_code="internal_appeal",
        appeal_basis_code="medical_necessity_supporting_records_available",
        appeal_narrative="Customer disputes the denial and requests review under plan appeal procedures.",
        customer_statement="I want to appeal this denial.",
    )

    assert draft["ok"] is True
    fields = draft["fields"]
    assert (
        fields["appeal_narrative"]
        == "Customer disputes the denial and requests review under plan appeal procedures."
    )
    assert fields["customer_statement"] == "I want to appeal this denial."
    assert fields["appeal_summary"] == fields["appeal_narrative"]
    assert fields["appeal_reason"] == fields["appeal_basis_code"]


def test_healthcare_draft_tool_rejects_outcome_or_medical_conclusions(tmp_path):
    task = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    system = HealthcareAppealSystem(task, tmp_path / "drafts")
    base = {
        "member_id": task.hidden_facts["member_id"],
        "representative_id": task.hidden_facts["representative_id"],
        "plan_id": task.hidden_facts["plan_id"],
        "denial_id": task.hidden_facts["denial_id"],
        "claim_id": task.hidden_facts["claim_id"],
        "appeal_basis_code": "medical_necessity_supporting_records_available",
    }

    bad_action = system.draft_internal_appeal(
        **base,
        requested_action_code="Approve coverage for the denied MRI",
        appeal_narrative="Customer disputes the denial and requests review.",
    )
    assert bad_action["ok"] is False
    assert bad_action["error"] == "invalid_requested_action_code"

    bad_narrative = system.draft_internal_appeal(
        **base,
        requested_action_code="internal_appeal",
        appeal_narrative="The MRI is medically necessary and should be covered.",
    )
    assert bad_narrative["ok"] is False
    assert bad_narrative["error"] == "invalid_appeal_narrative"


def test_healthcare_external_review_tool_defaults_action_code(tmp_path):
    task = TaskSpec.load(CANDIDATE_DIR / "HC-V1-005.us.yaml")
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    draft = system.draft_external_review_request(
        member_id=task.hidden_facts["member_id"],
        plan_id=task.hidden_facts["plan_id"],
        denial_id=task.hidden_facts["denial_id"],
        claim_id=task.hidden_facts["claim_id"],
        appeal_reason="final denial needs an independent review",
    )

    assert draft["ok"] is True
    fields = draft["fields"]
    assert fields["requested_action_code"] == "external_review_request"
    assert fields["appeal_basis_code"] == "final_adverse_determination_external_review"


def test_healthcare_new_external_notice_file_and_language_tools_append_final_state(tmp_path):
    base = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    state = copy.deepcopy(base.initial_state)
    state["external_review_checks"] = [{"check_id": "external_seed", "denial_id": "seed"}]
    state["denial_notice_checks"] = [{"check_id": "notice_seed", "denial_id": "seed"}]
    state["claim_file_requests"] = [{"request_id": "claimfile_seed", "denial_id": "seed"}]
    state["language_access_events"] = [{"event_id": "lang_seed", "member_id": "seed"}]
    state["authorization_validity_checks"] = [{"check_id": "phi_auth_seed", "member_id": "seed"}]
    state["continued_coverage_checks"] = [{"check_id": "continued_seed", "denial_id": "seed"}]
    state["member"]["locale"] = "es-US"
    state["plans"][0].update(
        {
            "external_review_process": "federal",
            "consumer_assistance_contact_id": "ombuds_001",
        }
    )
    state["denial_notices"][0].update(
        {
            "final_internal_adverse_determination": True,
            "internal_exhausted_or_deemed_exhausted": True,
            "medical_judgment_or_surprise_billing_basis": True,
            "external_review_filing_deadline_date": "2026-10-01",
            "has_ombudsman_contact": False,
            "culturally_linguistically_appropriate": False,
        }
    )
    task = replace(base, initial_state=state)
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    external = system.execute(
        "check_external_review_eligibility", {"denial_id": base.hidden_facts["denial_id"]}
    )
    assert external["ok"] is True
    assert external["external_review_check"]["check_id"] == "external_002"
    assert external["external_review_check"]["eligible"] is True
    assert external["external_review_check"]["review_process"] == "federal"

    notice = system.execute(
        "check_denial_notice_adequacy", {"denial_id": base.hidden_facts["denial_id"]}
    )
    assert notice["ok"] is True
    assert notice["denial_notice_check"]["deemed_exhaustion_possible"] is True
    assert set(notice["denial_notice_check"]["defects"]) >= {
        "has_ombudsman_contact",
        "culturally_linguistically_appropriate",
    }

    claim_file = system.execute(
        "record_claim_file_access_request",
        {
            "denial_id": base.hidden_facts["denial_id"],
            "requested_items": ["claim_file", "new_evidence", "new_rationale"],
            "minimum_necessary_categories": ["claim_summary", "denial_notice"],
            "new_evidence_requested": True,
            "new_rationale_requested": True,
        },
    )
    assert claim_file["ok"] is True
    assert claim_file["claim_file_request"]["phi_log_id"] == "phi_001"

    language = system.execute(
        "record_language_access_offer",
        {
            "language": "es",
            "translated_notice_requested": True,
            "machine_translation_human_review_required": True,
            "accepted_or_declined": "accepted",
        },
    )
    assert language["ok"] is True
    assert language["language_access_event"]["applicable_non_english_language"] is True

    final = system.final_state()
    assert [item["check_id"] for item in final["external_review_checks"]] == [
        "external_seed",
        "external_002",
    ]
    assert [item["check_id"] for item in final["denial_notice_checks"]] == [
        "notice_seed",
        "notice_002",
    ]
    assert [item["request_id"] for item in final["claim_file_requests"]] == [
        "claimfile_seed",
        "claimfile_002",
    ]
    assert [item["event_id"] for item in final["language_access_events"]] == [
        "lang_seed",
        "lang_002",
    ]
    assert final["authorization_validity_checks"] == [
        {"check_id": "phi_auth_seed", "member_id": "seed"}
    ]
    assert final["continued_coverage_checks"] == [
        {"check_id": "continued_seed", "denial_id": "seed"}
    ]
    assert final["phi_access_logs"][0]["requested_categories"] == ["claim_summary", "denial_notice"]


def test_healthcare_continued_coverage_and_phi_authorization_create_process_evidence(tmp_path):
    base = TaskSpec.load(Path("tasks/HC-01.us.yaml"))
    state = copy.deepcopy(base.initial_state)
    state["denial_notices"][0].update(
        {
            "review_type": "concurrent_care",
            "continuity_of_care": True,
            "ongoing_course_of_treatment": True,
            "reduction_or_termination": True,
            "advance_notice_given": False,
        }
    )
    state["representative_authorities"][0].update(
        {
            "information_description": "denial and minimum necessary claim appeal records",
            "authorized_person": base.hidden_facts["member_id"],
            "recipient": base.hidden_facts["representative_id"],
            "purpose": "appeal_and_phi_access",
            "expiration_date_or_event": "appeal complete",
            "signature_date": "2026-06-02",
            "representative_authority_description": "member portal authorization for appeal",
            "allowed_categories": ["claim_summary", "denial_notice", "plan_terms"],
        }
    )
    task = replace(base, initial_state=state)
    system = HealthcareAppealSystem(task, tmp_path / "drafts")

    continued = system.execute(
        "check_continued_coverage_pending_appeal", {"denial_id": base.hidden_facts["denial_id"]}
    )
    assert continued["ok"] is True
    continued_check = continued["continued_coverage_check"]
    assert continued_check["continued_coverage_required"] is True
    assert continued_check["expedited_path_required"] is True
    assert continued_check["handoff_id"] == "handoff_001"

    auth = system.execute(
        "validate_phi_authorization_document",
        {
            "actor_id": base.hidden_facts["representative_id"],
            "member_id": base.hidden_facts["member_id"],
            "denial_id": base.hidden_facts["denial_id"],
        },
    )
    assert auth["ok"] is True
    assert auth["authorization_validity_check"]["valid"] is True

    final = system.final_state()
    assert final["continued_coverage_checks"] == [continued_check]
    assert final["handoffs"] == [
        {
            "handoff_id": "handoff_001",
            "type": "continued_coverage_review",
            "denial_id": base.hidden_facts["denial_id"],
            "check_id": "continued_001",
            "continued_coverage_pending_appeal": True,
            "expedited_path_required": True,
            "no_coverage_decision": True,
        }
    ]
    assert "coverage_decision" not in final["handoffs"][0]
    assert final["authorization_validity_checks"] == [auth["authorization_validity_check"]]
    assert final["authorization_checks"][0]["basis"] == "phi_authorization_document"
