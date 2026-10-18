from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .task_schema import TaskSpec


Draft = Dict[str, Any]


def _canonical_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _copy_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _copy_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _copy_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return [] if value in (None, "") else [str(value)]
    return [str(item) for item in value]


@dataclass
class GovernmentBenefitsAppealSystem:
    """In-memory SNAP fair-hearing system for trace-level compliance tasks."""

    task: TaskSpec
    draft_dir: Path
    session_context: Dict[str, Any] = field(default_factory=dict)
    benefit_case: Dict[str, Any] = field(default_factory=dict)
    adverse_action_notice: Dict[str, Any] = field(default_factory=dict)
    case_file_summary: Dict[str, Any] = field(default_factory=dict)
    benefits_program_profiles: List[Dict[str, Any]] = field(default_factory=list)
    benefit_service_notices: List[Dict[str, Any]] = field(default_factory=list)
    medicaid_renewals: List[Dict[str, Any]] = field(default_factory=list)
    case_file_indexes: List[Dict[str, Any]] = field(default_factory=list)
    language_access_needs: List[Dict[str, Any]] = field(default_factory=list)
    disability_access_needs: List[Dict[str, Any]] = field(default_factory=list)
    benefits_representative_authorities: List[Dict[str, Any]] = field(default_factory=list)
    notice_adequacy_checks: List[Dict[str, Any]] = field(default_factory=list)
    deadline_checks: List[Dict[str, Any]] = field(default_factory=list)
    continued_benefits_checks: List[Dict[str, Any]] = field(default_factory=list)
    benefits_program_profile_checks: List[Dict[str, Any]] = field(default_factory=list)
    snap_notice_content_checks: List[Dict[str, Any]] = field(default_factory=list)
    medicaid_hearing_right_checks: List[Dict[str, Any]] = field(default_factory=list)
    medicaid_maintenance_of_services_checks: List[Dict[str, Any]] = field(default_factory=list)
    medicaid_renewal_procedure_checks: List[Dict[str, Any]] = field(default_factory=list)
    case_file_release_scope_checks: List[Dict[str, Any]] = field(default_factory=list)
    language_access_checks: List[Dict[str, Any]] = field(default_factory=list)
    disability_access_checks: List[Dict[str, Any]] = field(default_factory=list)
    benefits_representative_authority_checks: List[Dict[str, Any]] = field(default_factory=list)
    rights_disclosures: List[Dict[str, Any]] = field(default_factory=list)
    hearing_cases: List[Dict[str, Any]] = field(default_factory=list)
    case_file_requests: List[Dict[str, Any]] = field(default_factory=list)
    accommodation_requests: List[Dict[str, Any]] = field(default_factory=list)
    language_assistance_requests: List[Dict[str, Any]] = field(default_factory=list)
    disability_accommodation_requests: List[Dict[str, Any]] = field(default_factory=list)
    drafts: Dict[str, Draft] = field(default_factory=dict)
    outbox: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        state = self.task.initial_state
        self.session_context = dict(state.get("session_context", {}))
        self.benefit_case = dict(state.get("benefit_case", {}))
        self.adverse_action_notice = dict(state.get("adverse_action_notice", {}))
        self.case_file_summary = dict(state.get("case_file_summary", {}))
        self.benefits_program_profiles = _copy_list(state.get("benefits_program_profiles", []))
        self.benefit_service_notices = _copy_list(state.get("benefit_service_notices", []))
        self.medicaid_renewals = _copy_list(state.get("medicaid_renewals", []))
        self.case_file_indexes = _copy_list(state.get("case_file_indexes", []))
        self.language_access_needs = _copy_list(state.get("language_access_needs", []))
        self.disability_access_needs = _copy_list(state.get("disability_access_needs", []))
        self.benefits_representative_authorities = _copy_list(
            state.get("benefits_representative_authorities", [])
        )
        self.notice_adequacy_checks = _copy_list(state.get("notice_adequacy_checks", []))
        self.deadline_checks = _copy_list(state.get("deadline_checks", []))
        self.continued_benefits_checks = _copy_list(state.get("continued_benefits_checks", []))
        self.benefits_program_profile_checks = _copy_list(
            state.get("benefits_program_profile_checks", [])
        )
        self.snap_notice_content_checks = _copy_list(state.get("snap_notice_content_checks", []))
        self.medicaid_hearing_right_checks = _copy_list(
            state.get("medicaid_hearing_right_checks", [])
        )
        self.medicaid_maintenance_of_services_checks = _copy_list(
            state.get("medicaid_maintenance_of_services_checks", [])
        )
        self.medicaid_renewal_procedure_checks = _copy_list(
            state.get("medicaid_renewal_procedure_checks", [])
        )
        self.case_file_release_scope_checks = _copy_list(
            state.get("case_file_release_scope_checks", [])
        )
        self.language_access_checks = _copy_list(state.get("language_access_checks", []))
        self.disability_access_checks = _copy_list(state.get("disability_access_checks", []))
        self.benefits_representative_authority_checks = _copy_list(
            state.get("benefits_representative_authority_checks", [])
        )
        self.rights_disclosures = _copy_list(state.get("rights_disclosures", []))
        self.hearing_cases = _copy_list(state.get("hearing_cases", []))
        self.case_file_requests = _copy_list(state.get("case_file_requests", []))
        self.accommodation_requests = _copy_list(state.get("accommodation_requests", []))
        self.language_assistance_requests = _copy_list(
            state.get("language_assistance_requests", [])
        )
        self.disability_accommodation_requests = _copy_list(
            state.get("disability_accommodation_requests", [])
        )
        self.outbox = _copy_list(state.get("outbox", []))

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_session_context": self.get_session_context,
            "lookup_benefit_case": self.lookup_benefit_case,
            "lookup_adverse_action_notice": self.lookup_adverse_action_notice,
            "check_benefits_program_profile": self.check_benefits_program_profile,
            "lookup_benefit_service_notice": self.lookup_benefit_service_notice,
            "check_snap_notice_contents": self.check_snap_notice_contents,
            "check_medicaid_hearing_right": self.check_medicaid_hearing_right,
            "check_medicaid_maintenance_of_services": self.check_medicaid_maintenance_of_services,
            "check_medicaid_renewal_procedure": self.check_medicaid_renewal_procedure,
            "lookup_case_file_index": self.lookup_case_file_index,
            "check_case_file_release_scope": self.check_case_file_release_scope,
            "check_language_access_need": self.check_language_access_need,
            "draft_language_assistance_request": self.draft_language_assistance_request,
            "check_disability_access_need": self.check_disability_access_need,
            "draft_disability_accommodation_request": self.draft_disability_accommodation_request,
            "check_benefits_representative_authority": self.check_benefits_representative_authority,
            "check_notice_adequacy": self.check_notice_adequacy,
            "check_fair_hearing_deadline": self.check_fair_hearing_deadline,
            "check_continued_benefits_eligibility": self.check_continued_benefits_eligibility,
            "draft_case_file_request": self.draft_case_file_request,
            "draft_language_or_disability_accommodation": self.draft_language_or_disability_accommodation,
            "draft_fair_hearing_request": self.draft_fair_hearing_request,
            "record_fair_hearing_rights_disclosure": self.record_fair_hearing_rights_disclosure,
            "submit_hearing_request": self.submit_hearing_request,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"ok": False, "error": "unknown_tool"}
        sig = inspect.signature(handler)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            return handler(**args)
        filtered_args = {k: v for k, v in args.items() if k in sig.parameters}
        return handler(**filtered_args)

    def get_session_context(self) -> Dict[str, Any]:
        return {"ok": True, "session_context": dict(self.session_context)}

    def lookup_benefit_case(self, benefit_case_id: Optional[str] = None) -> Dict[str, Any]:
        if benefit_case_id and benefit_case_id != self.benefit_case.get("benefit_case_id"):
            return {"ok": False, "error": "benefit_case_not_found"}
        public = {
            key: self.benefit_case.get(key)
            for key in (
                "benefit_case_id",
                "program",
                "household_size",
                "monthly_allotment",
                "certification_period",
                "status",
            )
            if key in self.benefit_case
        }
        return {"ok": True, "benefit_case": public}

    def lookup_adverse_action_notice(self, notice_id: Optional[str] = None) -> Dict[str, Any]:
        if notice_id and notice_id != self.adverse_action_notice.get("notice_id"):
            return {"ok": False, "error": "notice_not_found"}
        return {"ok": True, "notice": dict(self.adverse_action_notice)}

    def check_benefits_program_profile(
        self,
        program: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile = self._find_program_profile(
            program=program, jurisdiction=jurisdiction, profile_id=profile_id
        )
        effective_program = str(
            program or profile.get("program") or self.benefit_case.get("program") or ""
        )
        effective_jurisdiction = str(jurisdiction or profile.get("jurisdiction") or "us_federal")
        unsupported_reason = profile.get("unsupported_reason")
        supported_default = effective_program.upper() in {"SNAP", "MEDICAID"}
        check: Dict[str, Any] = {
            "check_id": f"profile_{len(self.benefits_program_profile_checks) + 1:03d}",
            "program": effective_program,
            "jurisdiction": effective_jurisdiction,
            "profile_id": profile_id
            or profile.get("profile_id")
            or self._default_profile_id(effective_program),
            "source_profile": profile.get("source_profile")
            or self._default_profile_id(effective_program),
            "supported_workflow": bool(
                profile.get("supported_workflow", supported_default) and not unsupported_reason
            ),
            "unsupported_reason": unsupported_reason,
        }
        self.benefits_program_profile_checks.append(check)
        return {"ok": True, "program_profile": check}

    def lookup_benefit_service_notice(
        self,
        notice_id: Optional[str] = None,
        benefit_case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        notice = self._find_notice(notice_id=notice_id, benefit_case_id=benefit_case_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        allowed = (
            "notice_id",
            "benefit_case_id",
            "program",
            "action",
            "reason",
            "effective_date",
            "notice_date",
            "benefit_or_service_impact",
            "service_type",
            "service_units",
            "prior_authorization_id",
            "mco_appeal_id",
            "renewal_id",
            "notice_channel",
        )
        return {"ok": True, "notice": {key: notice.get(key) for key in allowed if key in notice}}

    def check_snap_notice_contents(self, notice_id: str) -> Dict[str, Any]:
        notice = self._find_notice(notice_id=notice_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        contents = _copy_dict(notice.get("contents"))
        timely_10_day = bool(notice.get("timely_10_day", contents.get("timely_10_day", True)))
        check = {
            "check_id": f"snapnotice_{len(self.snap_notice_content_checks) + 1:03d}",
            "notice_id": notice_id,
            "timely_10_day": timely_10_day,
            "weekend_holiday_extension_applies": bool(
                notice.get(
                    "weekend_holiday_extension_applies",
                    contents.get("weekend_holiday_extension_applies", False),
                )
            ),
            "has_action": bool(notice.get("action") or contents.get("has_action")),
            "has_reason": bool(notice.get("reason") or contents.get("has_reason")),
            "has_hearing_right": bool(
                contents.get("has_hearing_right", notice.get("has_hearing_right", True))
            ),
            "has_office_contact": bool(
                contents.get("has_office_contact", notice.get("office_contact"))
            ),
            "has_continued_benefits_text": bool(
                contents.get("has_continued_benefits_text", notice.get("continued_benefits_text"))
            ),
            "has_overissuance_liability_text": bool(
                contents.get(
                    "has_overissuance_liability_text", notice.get("overissuance_liability_text")
                )
            ),
            "has_legal_services_text": bool(
                contents.get("has_legal_services_text", notice.get("legal_services_text"))
            ),
            "computer_match_type": notice.get("computer_match_type")
            or contents.get("computer_match_type"),
        }
        required_flags = (
            "timely_10_day",
            "has_action",
            "has_reason",
            "has_hearing_right",
            "has_office_contact",
            "has_continued_benefits_text",
            "has_overissuance_liability_text",
            "has_legal_services_text",
        )
        defects = _copy_string_list(
            notice.get("defects", notice.get("missing_required_fields", []))
        )
        defects.extend(flag for flag in required_flags if not check[flag])
        final_defects = sorted(set(str(item) for item in defects))
        adequate = not final_defects
        check["defects"] = final_defects
        check["adequate"] = adequate
        self.snap_notice_content_checks.append(check)
        self._append_notice_adequacy_summary(notice_id, adequate, final_defects)
        return {"ok": True, "snap_notice_contents": check}

    def check_medicaid_hearing_right(
        self,
        notice_id: str,
        action_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        notice = self._find_notice(notice_id=notice_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        category = str(
            action_category or notice.get("action_category") or notice.get("notice_category") or ""
        )
        automatic_change_exception = bool(notice.get("automatic_change_exception", False))
        covered_categories = {
            "eligibility",
            "covered_service",
            "liability",
            "promptness",
            "prior_authorization",
            "mco",
            "nursing_facility",
            "pasrr",
            "nemt_pahp",
        }
        hearing_required = bool(
            notice.get(
                "hearing_required",
                category in covered_categories and not automatic_change_exception,
            )
        )
        check = {
            "check_id": f"medhear_{len(self.medicaid_hearing_right_checks) + 1:03d}",
            "notice_id": notice_id,
            "action_category": category,
            "hearing_required": hearing_required,
            "automatic_change_exception": automatic_change_exception,
            "route": notice.get("hearing_route")
            or ("fair_hearing" if hearing_required else "no_hearing_exception"),
            "source_clause": notice.get("source_clause", "42 CFR 431.220"),
        }
        self.medicaid_hearing_right_checks.append(check)
        return {"ok": True, "medicaid_hearing_right": check}

    def check_medicaid_maintenance_of_services(
        self,
        notice_id: str,
        request_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        notice = self._find_notice(notice_id=notice_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        policy = _copy_dict(notice.get("maintenance_of_services")) or _copy_dict(
            notice.get("continued_benefits")
        )
        action_date = str(
            policy.get("action_date")
            or notice.get("action_date")
            or notice.get("effective_date")
            or ""
        )
        effective_request_date = str(
            request_date
            or policy.get("request_date")
            or self.task.hidden_facts.get("request_date", "")
        )
        requested_before_action_date = bool(
            policy.get(
                "requested_before_action_date",
                bool(
                    effective_request_date and action_date and effective_request_date <= action_date
                ),
            )
        )
        sole_law_policy_issue = bool(
            policy.get("sole_law_policy_issue", notice.get("sole_law_policy_issue", False))
        )
        services_maintained = bool(
            policy.get(
                "services_maintained", requested_before_action_date and not sole_law_policy_issue
            )
        )
        check = {
            "check_id": f"medmos_{len(self.medicaid_maintenance_of_services_checks) + 1:03d}",
            "notice_id": notice_id,
            "notice_type": policy.get("notice_type") or notice.get("notice_type"),
            "action_date": action_date,
            "request_date": effective_request_date,
            "requested_before_action_date": requested_before_action_date,
            "services_maintained": services_maintained,
            "sole_law_policy_issue": sole_law_policy_issue,
            "written_pending_reduction_notice_required": bool(
                policy.get("written_pending_reduction_notice_required", services_maintained)
            ),
            "recovery_risk_text": policy.get(
                "recovery_risk_text",
                "Continued Medicaid services may be subject to recovery if the agency action is upheld.",
            ),
        }
        self.medicaid_maintenance_of_services_checks.append(check)
        self.continued_benefits_checks.append(
            {
                "check_id": f"aidpending_{len(self.continued_benefits_checks) + 1:03d}",
                "notice_id": notice_id,
                "eligible": services_maintained,
                "elected": services_maintained,
                "risk_text": check["recovery_risk_text"],
                "program": "Medicaid",
            }
        )
        return {"ok": True, "medicaid_maintenance_of_services": check}

    def check_medicaid_renewal_procedure(self, renewal_id: Optional[str] = None) -> Dict[str, Any]:
        renewal = self._find_medicaid_renewal(renewal_id)
        if not renewal:
            return {"ok": False, "error": "renewal_not_found"}
        check = {
            "check_id": f"medrenew_{len(self.medicaid_renewal_procedure_checks) + 1:03d}",
            "renewal_id": renewal.get("renewal_id") or renewal_id,
            "ex_parte_attempted": bool(renewal.get("ex_parte_attempted", False)),
            "prepopulated_form_sent": bool(renewal.get("prepopulated_form_sent", False)),
            "response_deadline_days": renewal.get("response_deadline_days"),
            "termination_date": renewal.get("termination_date"),
            "submitted_within_90_days": bool(renewal.get("submitted_within_90_days", False)),
            "no_in_person_interview_required": bool(
                renewal.get("no_in_person_interview_required", True)
            ),
            "only_needed_information_requested": bool(
                renewal.get("only_needed_information_requested", True)
            ),
            "other_affordability_program_screened": bool(
                renewal.get("other_affordability_program_screened", False)
            ),
            "accessible_notice": bool(renewal.get("accessible_notice", True)),
        }
        self.medicaid_renewal_procedure_checks.append(check)
        return {"ok": True, "medicaid_renewal_procedure": check}

    def lookup_case_file_index(
        self,
        notice_id: Optional[str] = None,
        benefit_case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        index = self._find_case_file_index(notice_id=notice_id, benefit_case_id=benefit_case_id)
        if not index:
            return {"ok": False, "error": "case_file_index_not_found"}
        allowed = (
            "notice_id",
            "benefit_case_id",
            "available_sections",
            "hearing_relevant_sections",
            "default_sections",
            "confidential_redactions",
            "excluded_uncontestable_materials",
        )
        return {
            "ok": True,
            "case_file_index": {key: index.get(key) for key in allowed if key in index},
        }

    def check_case_file_release_scope(
        self,
        notice_id: Optional[str] = None,
        requested_sections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        index = self._find_case_file_index(notice_id=notice_id)
        if not index:
            return {"ok": False, "error": "case_file_index_not_found"}
        available_sections = _copy_string_list(
            index.get("available_sections", index.get("default_sections", []))
        )
        hearing_relevant_sections = _copy_string_list(
            index.get("hearing_relevant_sections", available_sections)
        )
        requested = _copy_string_list(
            requested_sections or index.get("requested_sections", hearing_relevant_sections)
        )
        excluded = _copy_string_list(index.get("excluded_uncontestable_materials", []))
        allowed_sections = set(hearing_relevant_sections) - set(excluded)
        check = {
            "check_id": f"filescope_{len(self.case_file_release_scope_checks) + 1:03d}",
            "notice_id": notice_id or index.get("notice_id"),
            "available_sections": available_sections,
            "hearing_relevant_sections": hearing_relevant_sections,
            "requested_sections": requested,
            "free_copy_required": bool(index.get("free_copy_required", True)),
            "confidential_redactions": _copy_string_list(index.get("confidential_redactions", [])),
            "excluded_uncontestable_materials": excluded,
            "release_scope_ok": set(requested).issubset(allowed_sections),
        }
        self.case_file_release_scope_checks.append(check)
        return {"ok": True, "case_file_release_scope": check}

    def check_language_access_need(
        self,
        notice_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        need = self._find_language_access_need(language=language)
        effective_language = str(
            language or need.get("language") or self.session_context.get("language", "en")
        )
        check = {
            "check_id": f"lang_{len(self.language_access_checks) + 1:03d}",
            "notice_id": notice_id,
            "language": effective_language,
            "interpreter_required": bool(
                need.get("interpreter_required", effective_language.lower() != "en")
            ),
            "translator_required": bool(need.get("translator_required", False)),
            "qualified_provider_required": bool(
                need.get("qualified_provider_required", effective_language.lower() != "en")
            ),
            "critical_document_human_review": bool(
                need.get("critical_document_human_review", False)
            ),
            "own_interpreter_not_required": bool(
                need.get("own_interpreter_not_required", effective_language.lower() != "en")
            ),
            "minor_interpreter_restricted": bool(
                need.get("minor_interpreter_restricted", effective_language.lower() != "en")
            ),
        }
        self.language_access_checks.append(check)
        return {"ok": True, "language_access_need": check}

    def draft_language_assistance_request(
        self,
        notice_id: str,
        language: Optional[str] = None,
        interpreter_required: Optional[bool] = None,
        translator_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        need = self._find_language_access_need(language=language)
        effective_language = str(
            language or need.get("language") or self.session_context.get("language", "en")
        )
        fields = {
            "claimant_id": self._claimant_id(),
            "benefit_case_id": self._benefit_case_id(),
            "notice_id": notice_id,
            "language": effective_language,
            "interpreter_required": bool(
                need.get("interpreter_required", effective_language.lower() != "en")
                if interpreter_required is None
                else interpreter_required
            ),
            "translator_required": bool(
                need.get("translator_required", False)
                if translator_required is None
                else translator_required
            ),
            "qualified_provider_required": bool(
                need.get("qualified_provider_required", effective_language.lower() != "en")
            ),
            "critical_document_human_review": bool(
                need.get("critical_document_human_review", False)
            ),
            "own_interpreter_not_required": bool(
                need.get("own_interpreter_not_required", effective_language.lower() != "en")
            ),
            "minor_interpreter_restricted": bool(
                need.get("minor_interpreter_restricted", effective_language.lower() != "en")
            ),
        }
        draft = self._create_draft("language_assistance_request", fields)
        request = {
            "request_id": f"langreq_{len(self.language_assistance_requests) + 1:03d}",
            "draft_id": draft["draft_id"],
            **fields,
            "status": "draft",
        }
        self.language_assistance_requests.append(request)
        self.accommodation_requests.append(
            {"request_id": request["request_id"], **fields, "status": "draft"}
        )
        return draft

    def check_disability_access_need(
        self,
        notice_id: Optional[str] = None,
        disability_access_need: Optional[str] = None,
    ) -> Dict[str, Any]:
        need = self._find_disability_access_need(disability_access_need=disability_access_need)
        check = {
            "check_id": f"disability_{len(self.disability_access_checks) + 1:03d}",
            "notice_id": notice_id,
            "disability_access_need": disability_access_need
            or need.get("disability_access_need")
            or self.session_context.get("access_need"),
            "auxiliary_aid": need.get("auxiliary_aid"),
            "format": need.get("format"),
            "hearing_modality": need.get("hearing_modality"),
            "reasonable_modification": need.get("reasonable_modification"),
            "no_surcharge": bool(need.get("no_surcharge", True)),
            "acceptance_optional": bool(need.get("acceptance_optional", True)),
        }
        self.disability_access_checks.append(check)
        return {"ok": True, "disability_access_need": check}

    def draft_disability_accommodation_request(
        self,
        notice_id: str,
        disability_access_need: Optional[str] = None,
        auxiliary_aid: Optional[str] = None,
        format: Optional[str] = None,
        hearing_modality: Optional[str] = None,
        reasonable_modification: Optional[str] = None,
    ) -> Dict[str, Any]:
        need = self._find_disability_access_need(disability_access_need=disability_access_need)
        fields = {
            "claimant_id": self._claimant_id(),
            "benefit_case_id": self._benefit_case_id(),
            "notice_id": notice_id,
            "disability_access_need": disability_access_need
            or need.get("disability_access_need")
            or self.session_context.get("access_need"),
            "auxiliary_aid": auxiliary_aid or need.get("auxiliary_aid"),
            "format": format or need.get("format"),
            "hearing_modality": hearing_modality or need.get("hearing_modality"),
            "reasonable_modification": reasonable_modification
            or need.get("reasonable_modification"),
            "no_surcharge": bool(need.get("no_surcharge", True)),
            "acceptance_optional": bool(need.get("acceptance_optional", True)),
        }
        draft = self._create_draft("disability_accommodation_request", fields)
        request = {
            "request_id": f"disreq_{len(self.disability_accommodation_requests) + 1:03d}",
            "draft_id": draft["draft_id"],
            **fields,
            "status": "draft",
        }
        self.disability_accommodation_requests.append(request)
        self.accommodation_requests.append(
            {"request_id": request["request_id"], **fields, "status": "draft"}
        )
        return draft

    def check_benefits_representative_authority(
        self,
        actor_id: Optional[str] = None,
        claimant_id: Optional[str] = None,
        representative_id: Optional[str] = None,
        scope: str = "request_hearing",
    ) -> Dict[str, Any]:
        claimant = str(claimant_id or self._claimant_id())
        actor = str(
            actor_id or representative_id or self.session_context.get("actor_id") or claimant
        )
        representative = str(representative_id or actor)
        if actor == claimant:
            authority: Dict[str, Any] = {
                "authorization_type": "self",
                "authorization_on_file": True,
                "scope": [scope],
                "limitations": [],
            }
        else:
            authority = self._find_representative_authority(actor, claimant, representative) or {}
        scopes_list = _copy_string_list(authority.get("scope", []))
        authorization_on_file = bool(
            authority.get("authorization_on_file", authority.get("status") == "active")
        )
        can_act = bool(
            authorization_on_file
            and (
                scope in scopes_list
                or "all" in scopes_list
                or "appeal_and_case_file" in scopes_list
            )
        )
        check = {
            "check_id": f"repauth_{len(self.benefits_representative_authority_checks) + 1:03d}",
            "actor_id": actor,
            "claimant_id": claimant,
            "representative_id": representative,
            "authorization_type": authority.get("authorization_type", "none"),
            "authorization_on_file": authorization_on_file,
            "scope": scope,
            "can_act": can_act,
            "limitations": _copy_string_list(authority.get("limitations", [])),
        }
        self.benefits_representative_authority_checks.append(check)
        return {"ok": True, "representative_authority": check}

    def check_notice_adequacy(self, notice_id: str) -> Dict[str, Any]:
        if notice_id != self.adverse_action_notice.get("notice_id"):
            return {"ok": False, "error": "notice_not_found"}
        missing = list(self.adverse_action_notice.get("missing_required_fields", []))
        check = {
            "check_id": f"noticechk_{len(self.notice_adequacy_checks) + 1:03d}",
            "notice_id": notice_id,
            "adequate": not missing,
            "missing_required_fields": missing,
            "defect_route": "appeal_or_handoff" if missing else None,
        }
        self.notice_adequacy_checks.append(check)
        return {"ok": True, "notice_adequacy": check}

    def check_fair_hearing_deadline(self, notice_id: str) -> Dict[str, Any]:
        if notice_id != self.adverse_action_notice.get("notice_id"):
            return {"ok": False, "error": "notice_not_found"}
        deadline = dict(self.adverse_action_notice.get("deadlines", {}))
        check = {
            "check_id": f"deadline_{len(self.deadline_checks) + 1:03d}",
            "notice_id": notice_id,
            "hearing_deadline": deadline.get("fair_hearing_request_by"),
            "aid_pending_deadline": deadline.get("continued_benefits_election_by"),
            "timely": bool(deadline.get("timely", True)),
            "good_cause_route": bool(deadline.get("good_cause_route", False)),
        }
        self.deadline_checks.append(check)
        return {"ok": True, "deadline_check": check}

    def check_continued_benefits_eligibility(
        self, notice_id: str, elect_continued_benefits: bool = False
    ) -> Dict[str, Any]:
        if notice_id != self.adverse_action_notice.get("notice_id"):
            return {"ok": False, "error": "notice_not_found"}
        policy = dict(self.adverse_action_notice.get("continued_benefits", {}))
        check = {
            "check_id": f"aidpending_{len(self.continued_benefits_checks) + 1:03d}",
            "notice_id": notice_id,
            "eligible": bool(policy.get("eligible", False)),
            "elected": bool(elect_continued_benefits and policy.get("eligible", False)),
            "risk_text": policy.get(
                "risk_text", "Benefits may need to be repaid if the agency action is upheld."
            ),
        }
        self.continued_benefits_checks.append(check)
        return {"ok": True, "continued_benefits": check}

    def draft_case_file_request(
        self, notice_id: str, requested_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        fields = {
            "claimant_id": self._claimant_id(),
            "benefit_case_id": self._benefit_case_id(),
            "notice_id": notice_id,
            "requested_sections": requested_sections
            or self.case_file_summary.get("default_sections", ["notice", "budget", "verification"]),
        }
        self.case_file_requests.append(
            {
                "request_id": f"file_{len(self.case_file_requests) + 1:03d}",
                **fields,
                "status": "draft",
            }
        )
        return self._create_draft("case_file_request", fields)

    def draft_language_or_disability_accommodation(
        self,
        notice_id: str,
        language: Optional[str] = None,
        accommodation: Optional[str] = None,
    ) -> Dict[str, Any]:
        fields = {
            "claimant_id": self._claimant_id(),
            "benefit_case_id": self._benefit_case_id(),
            "notice_id": notice_id,
            "language": language or self.session_context.get("language", "en"),
            "accommodation": accommodation or self.session_context.get("access_need", "none"),
        }
        self.accommodation_requests.append(
            {
                "request_id": f"access_{len(self.accommodation_requests) + 1:03d}",
                **fields,
                "status": "draft",
            }
        )
        return self._create_draft("accommodation_request", fields)

    def draft_fair_hearing_request(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_hearing_fields(fields)
        return self._create_draft("fair_hearing_request", normalized)

    def record_fair_hearing_rights_disclosure(
        self, draft_id: Optional[str] = None
    ) -> Dict[str, Any]:
        disclosure = {
            "disclosure_id": f"rights_{len(self.rights_disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "fair_hearing_rights",
            "text": self.task.mandatory_disclosure(),
            "language": self.session_context.get("language", "en"),
        }
        self.rights_disclosures.append(disclosure)
        return {"ok": True, "rights_disclosure": disclosure}

    def submit_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        if draft.get("draft_type") != "fair_hearing_request":
            return {
                "ok": False,
                "error": "unsupported_draft_type",
                "draft_type": draft.get("draft_type"),
            }
        return self.submit_hearing_request(**dict(draft.get("fields", {})))

    def submit_hearing_request(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_hearing_fields(fields)
        case = {
            "hearing_case_id": f"hearing_{len(self.hearing_cases) + 1:03d}",
            **normalized,
            "status": "submitted",
        }
        self.hearing_cases.append(case)
        self.outbox.append(
            {"type": "hearing_request_notice", "hearing_case_id": case["hearing_case_id"]}
        )
        return {"ok": True, "hearing_case": case}

    def final_state(self) -> Dict[str, Any]:
        return {
            "session_context": dict(self.session_context),
            "benefit_case": dict(self.benefit_case),
            "adverse_action_notice": dict(self.adverse_action_notice),
            "case_file_summary": dict(self.case_file_summary),
            "benefits_program_profiles": [dict(item) for item in self.benefits_program_profiles],
            "benefit_service_notices": [dict(item) for item in self.benefit_service_notices],
            "medicaid_renewals": [dict(item) for item in self.medicaid_renewals],
            "case_file_indexes": [dict(item) for item in self.case_file_indexes],
            "language_access_needs": [dict(item) for item in self.language_access_needs],
            "disability_access_needs": [dict(item) for item in self.disability_access_needs],
            "benefits_representative_authorities": [
                dict(item) for item in self.benefits_representative_authorities
            ],
            "notice_adequacy_checks": [dict(item) for item in self.notice_adequacy_checks],
            "deadline_checks": [dict(item) for item in self.deadline_checks],
            "continued_benefits_checks": [dict(item) for item in self.continued_benefits_checks],
            "benefits_program_profile_checks": [
                dict(item) for item in self.benefits_program_profile_checks
            ],
            "snap_notice_content_checks": [dict(item) for item in self.snap_notice_content_checks],
            "medicaid_hearing_right_checks": [
                dict(item) for item in self.medicaid_hearing_right_checks
            ],
            "medicaid_maintenance_of_services_checks": [
                dict(item) for item in self.medicaid_maintenance_of_services_checks
            ],
            "medicaid_renewal_procedure_checks": [
                dict(item) for item in self.medicaid_renewal_procedure_checks
            ],
            "case_file_release_scope_checks": [
                dict(item) for item in self.case_file_release_scope_checks
            ],
            "language_access_checks": [dict(item) for item in self.language_access_checks],
            "disability_access_checks": [dict(item) for item in self.disability_access_checks],
            "benefits_representative_authority_checks": [
                dict(item) for item in self.benefits_representative_authority_checks
            ],
            "rights_disclosures": [dict(item) for item in self.rights_disclosures],
            "case_file_requests": [dict(item) for item in self.case_file_requests],
            "accommodation_requests": [dict(item) for item in self.accommodation_requests],
            "language_assistance_requests": [
                dict(item) for item in self.language_assistance_requests
            ],
            "disability_accommodation_requests": [
                dict(item) for item in self.disability_accommodation_requests
            ],
            "drafts": [dict(item) for item in self.drafts.values()],
            "hearing_cases": [dict(item) for item in self.hearing_cases],
            "outbox": [dict(item) for item in self.outbox],
        }

    def _create_draft(self, draft_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"draft_type": draft_type, "fields": fields}
        draft_hash = _canonical_hash(payload)
        draft_id = f"draft_{len(self.drafts) + 1:03d}"
        draft = {"draft_id": draft_id, "draft_hash": draft_hash, **payload}
        self.drafts[draft_id] = draft
        path = self.draft_dir / f"{draft_id}.json"
        path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"ok": True, **draft}

    def _normalize_hearing_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        return {
            "claimant_id": str(
                fields.get("claimant_id") or facts.get("claimant_id") or self._claimant_id()
            ),
            "benefit_case_id": str(
                fields.get("benefit_case_id")
                or facts.get("benefit_case_id")
                or self._benefit_case_id()
            ),
            "notice_id": str(
                fields.get("notice_id")
                or facts.get("notice_id")
                or self.adverse_action_notice.get("notice_id", "")
            ),
            "program": str(
                fields.get("program")
                or self.benefit_case.get("program")
                or facts.get("program", "SNAP")
            ),
            "adverse_action": str(
                fields.get("adverse_action")
                or self.adverse_action_notice.get("action")
                or facts.get("adverse_action", "")
            ),
            "hearing_issue": str(
                fields.get("hearing_issue") or facts.get("hearing_issue", "contest_adverse_action")
            ),
            "continued_benefits_elected": bool(
                fields.get(
                    "continued_benefits_elected", facts.get("continued_benefits_elected", False)
                )
            ),
            "case_file_requested": bool(
                fields.get("case_file_requested", facts.get("case_file_requested", False))
            ),
            "accommodation_requested": bool(
                fields.get("accommodation_requested", facts.get("accommodation_requested", False))
            ),
            "language": str(
                fields.get("language")
                or facts.get("language")
                or self.session_context.get("language", "en")
            ),
        }

    def _claimant_id(self) -> str:
        return str(
            self.session_context.get("claimant_id") or self.task.hidden_facts.get("claimant_id", "")
        )

    def _benefit_case_id(self) -> str:
        return str(
            self.benefit_case.get("benefit_case_id")
            or self.task.hidden_facts.get("benefit_case_id", "")
        )

    def _default_profile_id(self, program: str) -> str:
        normalized = program.lower().replace(" ", "_")
        return f"us_federal_{normalized}" if normalized else "unknown"

    def _find_program_profile(
        self,
        program: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        for profile in self.benefits_program_profiles:
            if profile_id and profile.get("profile_id") != profile_id:
                continue
            if program and str(profile.get("program", "")).lower() != str(program).lower():
                continue
            if (
                jurisdiction
                and str(profile.get("jurisdiction", "")).lower() != str(jurisdiction).lower()
            ):
                continue
            return dict(profile)
        return {}

    def _find_notice(
        self,
        notice_id: Optional[str] = None,
        benefit_case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        notices = [*self.benefit_service_notices]
        if self.adverse_action_notice:
            notices.append(self.adverse_action_notice)
        for notice in notices:
            if notice_id and notice.get("notice_id") != notice_id:
                continue
            if benefit_case_id and notice.get("benefit_case_id") != benefit_case_id:
                continue
            return dict(notice)
        return {}

    def _append_notice_adequacy_summary(
        self, notice_id: str, adequate: bool, defects: List[str]
    ) -> None:
        self.notice_adequacy_checks.append(
            {
                "check_id": f"noticechk_{len(self.notice_adequacy_checks) + 1:03d}",
                "notice_id": notice_id,
                "adequate": adequate,
                "missing_required_fields": list(defects),
                "defect_route": "appeal_or_handoff" if defects else None,
            }
        )

    def _find_medicaid_renewal(self, renewal_id: Optional[str] = None) -> Dict[str, Any]:
        for renewal in self.medicaid_renewals:
            if renewal_id and renewal.get("renewal_id") != renewal_id:
                continue
            return dict(renewal)
        if renewal_id:
            for notice in self.benefit_service_notices:
                if notice.get("renewal_id") == renewal_id:
                    return dict(notice.get("renewal", {"renewal_id": renewal_id}))
        return {}

    def _find_case_file_index(
        self,
        notice_id: Optional[str] = None,
        benefit_case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        for index in self.case_file_indexes:
            if notice_id and index.get("notice_id") != notice_id:
                continue
            if benefit_case_id and index.get("benefit_case_id") != benefit_case_id:
                continue
            return dict(index)
        if self.case_file_summary:
            summary = dict(self.case_file_summary)
            if notice_id:
                summary.setdefault("notice_id", notice_id)
            if benefit_case_id:
                summary.setdefault("benefit_case_id", benefit_case_id)
            return summary
        return {}

    def _find_language_access_need(self, language: Optional[str] = None) -> Dict[str, Any]:
        effective = str(language or self.session_context.get("language", ""))
        for need in self.language_access_needs:
            if effective and str(need.get("language", "")).lower() != effective.lower():
                continue
            return dict(need)
        return {}

    def _find_disability_access_need(
        self, disability_access_need: Optional[str] = None
    ) -> Dict[str, Any]:
        effective = str(disability_access_need or self.session_context.get("access_need", ""))
        for need in self.disability_access_needs:
            if effective and str(need.get("disability_access_need", "")) != effective:
                continue
            return dict(need)
        return {}

    def _find_representative_authority(
        self,
        actor_id: str,
        claimant_id: str,
        representative_id: str,
    ) -> Optional[Dict[str, Any]]:
        for authority in self.benefits_representative_authorities:
            authority_actor = str(
                authority.get("actor_id") or authority.get("representative_id") or ""
            )
            if authority_actor and authority_actor not in {actor_id, representative_id}:
                continue
            if claimant_id and str(authority.get("claimant_id", claimant_id)) != claimant_id:
                continue
            return dict(authority)
        return None
