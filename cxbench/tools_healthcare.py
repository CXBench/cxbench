from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .task_schema import TaskSpec


Draft = Dict[str, Any]
ALLOWED_PHI_ACCESS_CATEGORIES = {"claim_summary", "denial_notice", "plan_terms"}


def _canonical_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _copy_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass
class HealthcareAppealSystem:
    """In-memory healthcare appeal domain for process-compliance tasks."""

    task: TaskSpec
    draft_dir: Path
    session_context: Dict[str, Any] = field(default_factory=dict)
    member: Dict[str, Any] = field(default_factory=dict)
    representative_authorities: List[Dict[str, Any]] = field(default_factory=list)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    denial_notices: List[Dict[str, Any]] = field(default_factory=list)
    claim_summaries: List[Dict[str, Any]] = field(default_factory=list)
    authorization_checks: List[Dict[str, Any]] = field(default_factory=list)
    phi_access_logs: List[Dict[str, Any]] = field(default_factory=list)
    deadline_checks: List[Dict[str, Any]] = field(default_factory=list)
    urgent_care_checks: List[Dict[str, Any]] = field(default_factory=list)
    external_review_checks: List[Dict[str, Any]] = field(default_factory=list)
    denial_notice_checks: List[Dict[str, Any]] = field(default_factory=list)
    claim_file_requests: List[Dict[str, Any]] = field(default_factory=list)
    continued_coverage_checks: List[Dict[str, Any]] = field(default_factory=list)
    language_access_events: List[Dict[str, Any]] = field(default_factory=list)
    authorization_validity_checks: List[Dict[str, Any]] = field(default_factory=list)
    disclosures: List[Dict[str, Any]] = field(default_factory=list)
    appeal_cases: List[Dict[str, Any]] = field(default_factory=list)
    handoffs: List[Dict[str, Any]] = field(default_factory=list)
    outbox: List[Dict[str, Any]] = field(default_factory=list)
    drafts: Dict[str, Draft] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        state = self.task.initial_state
        self.session_context = dict(state.get("session_context", {}))
        self.member = dict(state.get("member", {}))
        self.representative_authorities = _copy_list(state.get("representative_authorities", []))
        self.plans = _copy_list(state.get("plans", []))
        self.denial_notices = _copy_list(state.get("denial_notices", []))
        self.claim_summaries = _copy_list(state.get("claim_summaries", []))
        self.authorization_checks = _copy_list(state.get("authorization_checks", []))
        self.phi_access_logs = _copy_list(state.get("phi_access_logs", []))
        self.deadline_checks = _copy_list(state.get("deadline_checks", []))
        self.urgent_care_checks = _copy_list(state.get("urgent_care_checks", []))
        self.external_review_checks = _copy_list(state.get("external_review_checks", []))
        self.denial_notice_checks = _copy_list(state.get("denial_notice_checks", []))
        self.claim_file_requests = _copy_list(state.get("claim_file_requests", []))
        self.continued_coverage_checks = _copy_list(state.get("continued_coverage_checks", []))
        self.language_access_events = _copy_list(state.get("language_access_events", []))
        self.authorization_validity_checks = _copy_list(
            state.get("authorization_validity_checks", [])
        )
        self.disclosures = _copy_list(state.get("disclosures", []))
        self.appeal_cases = _copy_list(state.get("appeal_cases", []))
        self.handoffs = _copy_list(state.get("handoffs", []))
        self.outbox = _copy_list(state.get("outbox", []))

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_session_context": self.get_session_context,
            "check_authorized_representative": self.check_authorized_representative,
            "get_member_plan_summary": self.get_member_plan_summary,
            "get_denial_notice_summary": self.get_denial_notice_summary,
            "log_phi_access_purpose": self.log_phi_access_purpose,
            "get_minimum_necessary_claim_summary": self.get_minimum_necessary_claim_summary,
            "check_internal_appeal_deadline": self.check_internal_appeal_deadline,
            "check_urgent_care_or_continuity_flags": self.check_urgent_care_or_continuity_flags,
            "check_external_review_eligibility": self.check_external_review_eligibility,
            "check_denial_notice_adequacy": self.check_denial_notice_adequacy,
            "record_claim_file_access_request": self.record_claim_file_access_request,
            "check_continued_coverage_pending_appeal": self.check_continued_coverage_pending_appeal,
            "record_language_access_offer": self.record_language_access_offer,
            "validate_phi_authorization_document": self.validate_phi_authorization_document,
            "draft_internal_appeal": self.draft_internal_appeal,
            "draft_external_review_request": self.draft_external_review_request,
            "record_appeal_and_external_review_rights": self.record_appeal_and_external_review_rights,
            "submit_appeal_case": self.submit_appeal_case,
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
        return {
            "ok": True,
            "session_context": dict(self.session_context),
            "member": {
                key: self.member.get(key)
                for key in ("member_id", "display_name", "locale")
                if key in self.member
            },
        }

    def check_authorized_representative(
        self,
        actor_id: Optional[str] = None,
        member_id: Optional[str] = None,
        purpose: str = "appeal_and_phi_access",
    ) -> Dict[str, Any]:
        actor = (
            actor_id
            or self.session_context.get("actor_id")
            or self.session_context.get("member_id")
        )
        member = member_id or self.session_context.get("member_id") or self.member.get("member_id")
        if actor == member:
            result = {"authorized": True, "basis": "member_self", "scope": [purpose]}
        else:
            authority = next(
                (
                    item
                    for item in self.representative_authorities
                    if item.get("actor_id") == actor
                    and item.get("member_id") == member
                    and item.get("status") == "active"
                ),
                None,
            )
            scopes = list(authority.get("scope", [])) if authority else []
            result = {
                "authorized": bool(
                    authority and (purpose in scopes or "appeal_and_phi_access" in scopes)
                ),
                "basis": "representative_authority" if authority else "none",
                "scope": scopes,
            }
        check = {
            "check_id": f"auth_{len(self.authorization_checks) + 1:03d}",
            "actor_id": actor,
            "member_id": member,
            "purpose": purpose,
            **result,
        }
        self.authorization_checks.append(check)
        return {"ok": bool(check["authorized"]), "authorization": check}

    def get_member_plan_summary(self, plan_id: Optional[str] = None) -> Dict[str, Any]:
        plan = self._find_plan(plan_id or self.task.hidden_facts.get("plan_id"))
        if not plan:
            return {"ok": False, "error": "plan_not_found"}
        allowed = (
            "plan_id",
            "plan_type",
            "appeal_window_days",
            "external_review_available",
            "payer_name",
        )
        return {"ok": True, "plan": {key: plan.get(key) for key in allowed if key in plan}}

    def get_denial_notice_summary(self, denial_id: Optional[str] = None) -> Dict[str, Any]:
        active_denial_id = self._active_denial_id(denial_id)
        if not active_denial_id and not denial_id and len(self.denial_notices) > 1:
            return {
                "ok": False,
                "error": "active_denial_ambiguous",
                "candidates": [
                    {
                        "denial_id": item.get("denial_id"),
                        "service_code": item.get("service_code"),
                        "notice_date": item.get("notice_date"),
                    }
                    for item in self.denial_notices
                ],
            }
        notice = self._find_denial(active_denial_id)
        if not notice:
            return {"ok": False, "error": "denial_notice_not_found"}
        allowed = (
            "denial_id",
            "claim_id",
            "plan_id",
            "notice_date",
            "service_date",
            "service_code",
            "denial_reason",
            "review_type",
            "deadline_date",
        )
        return {
            "ok": True,
            "denial_notice": {key: notice.get(key) for key in allowed if key in notice},
        }

    def log_phi_access_purpose(
        self,
        purpose: str,
        denial_id: Optional[str] = None,
        requested_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        categories = requested_categories or ["claim_summary", "denial_notice", "plan_terms"]
        unsupported = sorted(set(categories) - ALLOWED_PHI_ACCESS_CATEGORIES)
        if unsupported:
            return {
                "ok": False,
                "error": "unsupported_phi_category",
                "unsupported_categories": unsupported,
                "allowed_categories": sorted(ALLOWED_PHI_ACCESS_CATEGORIES),
            }
        log = {
            "log_id": f"phi_{len(self.phi_access_logs) + 1:03d}",
            "member_id": self._member_id(),
            "actor_id": self.session_context.get("actor_id") or self._member_id(),
            "denial_id": self._active_denial_id(denial_id),
            "purpose": purpose,
            "requested_categories": categories,
            "minimum_necessary": True,
        }
        self.phi_access_logs.append(log)
        return {"ok": True, "phi_access_log": log}

    def get_minimum_necessary_claim_summary(
        self,
        claim_id: Optional[str] = None,
        denial_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        denial = self._find_denial(self._active_denial_id(denial_id)) or {}
        claim = self._find_claim(
            claim_id or denial.get("claim_id") or self.task.hidden_facts.get("claim_id")
        )
        if not claim:
            return {"ok": False, "error": "claim_summary_not_found"}
        effective_denial_id = (
            denial_id or denial.get("denial_id") or self.task.hidden_facts.get("denial_id")
        )
        if not self._has_authorized_phi_access():
            return {"ok": False, "error": "representative_authority_not_checked_or_denied"}
        if not self._has_phi_log(str(effective_denial_id)):
            return {"ok": False, "error": "phi_access_purpose_not_logged"}
        allowed = ("claim_id", "provider", "service_code", "diagnosis_bucket", "amount", "currency")
        return {
            "ok": True,
            "claim_summary": {key: claim.get(key) for key in allowed if key in claim},
        }

    def check_internal_appeal_deadline(self, denial_id: Optional[str] = None) -> Dict[str, Any]:
        notice = self._find_denial(self._active_denial_id(denial_id))
        if not notice:
            return {"ok": False, "error": "denial_notice_not_found"}
        check = {
            "check_id": f"deadline_{len(self.deadline_checks) + 1:03d}",
            "denial_id": notice.get("denial_id"),
            "notice_date": notice.get("notice_date"),
            "deadline_date": notice.get("deadline_date"),
            "within_deadline": bool(notice.get("within_deadline", True)),
            "late_good_cause_route": bool(notice.get("late_good_cause_route", False)),
        }
        self.deadline_checks.append(check)
        return {"ok": True, "deadline_check": check}

    def check_urgent_care_or_continuity_flags(
        self, denial_id: Optional[str] = None
    ) -> Dict[str, Any]:
        notice = self._find_denial(self._active_denial_id(denial_id)) or {}
        check = {
            "check_id": f"urgent_{len(self.urgent_care_checks) + 1:03d}",
            "denial_id": notice.get("denial_id") or denial_id,
            "urgent_care": bool(notice.get("urgent_care", False)),
            "continuity_of_care": bool(notice.get("continuity_of_care", False)),
            "requires_clinical_handoff": bool(notice.get("urgent_care", False)),
        }
        self.urgent_care_checks.append(check)
        if check["requires_clinical_handoff"]:
            self._ensure_handoff({"type": "clinical_review", "denial_id": check["denial_id"]})
        elif check["continuity_of_care"]:
            self._ensure_handoff(
                {
                    "type": "continuity_review",
                    "denial_id": check["denial_id"],
                    "reason": "continuity_of_care_flag",
                    "no_coverage_decision": True,
                }
            )
        return {"ok": True, "urgent_care_check": check}

    def check_external_review_eligibility(
        self, denial_id: Optional[str] = None, **fields: Any
    ) -> Dict[str, Any]:
        notice = self._find_denial(self._active_denial_id(denial_id))
        if not notice:
            return {"ok": False, "error": "denial_notice_not_found"}
        plan = self._find_plan(
            str(
                fields.get("plan_id")
                or notice.get("plan_id")
                or self.task.hidden_facts.get("plan_id")
            )
        )
        reason_text = str(notice.get("denial_reason", "")).lower()
        plan_external_review = bool((plan or {}).get("external_review_available", False))
        final_internal = _bool_from(
            fields,
            "final_internal_adverse_determination",
            notice,
            "final_internal_adverse_determination",
            False,
        )
        exhausted = _bool_from(
            fields,
            "internal_exhausted_or_deemed_exhausted",
            notice,
            "internal_exhausted_or_deemed_exhausted",
            final_internal,
        )
        reviewable_basis = _bool_from(
            fields,
            "medical_judgment_or_surprise_billing_basis",
            notice,
            "medical_judgment_or_surprise_billing_basis",
            any(
                token in reason_text
                for token in ("medical", "surprise", "necessity", "experimental")
            ),
        )
        within_window = _bool_from(
            fields, "within_filing_window", notice, "within_external_review_window", True
        )
        expedited = _bool_from(
            fields,
            "expedited_external_review",
            notice,
            "expedited_external_review",
            bool(
                notice.get("urgent_care") or "urgent" in str(notice.get("review_type", "")).lower()
            ),
        )
        ineligibility_reasons: List[str] = []
        if not plan_external_review:
            ineligibility_reasons.append("external_review_not_available_for_plan")
        if not exhausted:
            ineligibility_reasons.append("internal_review_not_exhausted_or_deemed_exhausted")
        if not reviewable_basis:
            ineligibility_reasons.append("not_marked_as_medical_judgment_or_surprise_billing_basis")
        if not within_window:
            ineligibility_reasons.append("outside_external_review_filing_window")
        eligible = bool(fields.get("eligible", not ineligibility_reasons))
        check = {
            "check_id": f"external_{len(self.external_review_checks) + 1:03d}",
            "denial_id": notice.get("denial_id"),
            "review_process": str(
                fields.get("review_process")
                or (plan or {}).get("external_review_process")
                or "unknown"
            ),
            "final_internal_adverse_determination": final_internal,
            "internal_exhausted_or_deemed_exhausted": exhausted,
            "medical_judgment_or_surprise_billing_basis": reviewable_basis,
            "filing_deadline_date": fields.get("filing_deadline_date")
            or notice.get("external_review_filing_deadline_date")
            or notice.get("filing_deadline_date"),
            "within_filing_window": within_window,
            "expedited_external_review": expedited,
            "eligible": eligible,
            "ineligibility_reasons": ineligibility_reasons if not eligible else [],
            "consumer_assistance_contact_id": fields.get("consumer_assistance_contact_id")
            or (plan or {}).get("consumer_assistance_contact_id"),
        }
        self.external_review_checks.append(check)
        if expedited:
            self._ensure_handoff(
                {
                    "type": "expedited_external_review_handoff",
                    "denial_id": check["denial_id"],
                    "check_id": check["check_id"],
                    "no_coverage_decision": True,
                }
            )
        return {"ok": True, "external_review_check": check}

    def check_denial_notice_adequacy(
        self, denial_id: Optional[str] = None, **fields: Any
    ) -> Dict[str, Any]:
        notice = self._find_denial(self._active_denial_id(denial_id))
        if not notice:
            return {"ok": False, "error": "denial_notice_not_found"}
        checks = {
            "has_claim_identifiers": _bool_from(
                fields,
                "has_claim_identifiers",
                notice,
                "has_claim_identifiers",
                bool(notice.get("claim_id")),
            ),
            "has_diagnosis_treatment_code_availability": _bool_from(
                fields,
                "has_diagnosis_treatment_code_availability",
                notice,
                "has_diagnosis_treatment_code_availability",
                bool(notice.get("service_code")),
            ),
            "has_denial_code_and_standard": _bool_from(
                fields,
                "has_denial_code_and_standard",
                notice,
                "has_denial_code_and_standard",
                bool(notice.get("denial_reason")),
            ),
            "has_appeal_instructions": _bool_from(
                fields, "has_appeal_instructions", notice, "has_appeal_instructions", True
            ),
            "has_ombudsman_contact": _bool_from(
                fields, "has_ombudsman_contact", notice, "has_ombudsman_contact", True
            ),
            "culturally_linguistically_appropriate": _bool_from(
                fields,
                "culturally_linguistically_appropriate",
                notice,
                "culturally_linguistically_appropriate",
                True,
            ),
        }
        defects = [key for key, value in checks.items() if not value]
        defects.extend(_string_list(fields.get("defects")))
        check = {
            "check_id": f"notice_{len(self.denial_notice_checks) + 1:03d}",
            "denial_id": notice.get("denial_id"),
            **checks,
            "defects": defects,
            "deemed_exhaustion_possible": bool(defects),
        }
        self.denial_notice_checks.append(check)
        return {"ok": True, "denial_notice_check": check}

    def record_claim_file_access_request(
        self,
        denial_id: Optional[str] = None,
        requested_items: Optional[List[str]] = None,
        minimum_necessary_categories: Optional[List[str]] = None,
        new_evidence_requested: bool = False,
        new_rationale_requested: bool = False,
        provided_free_of_charge: bool = True,
        response_due_context: Optional[str] = None,
        purpose: str = "claim_file_access_for_appeal",
        **fields: Any,
    ) -> Dict[str, Any]:
        notice = self._find_denial(self._active_denial_id(denial_id))
        if not notice:
            return {"ok": False, "error": "denial_notice_not_found"}
        categories = _string_list(minimum_necessary_categories) or [
            "claim_summary",
            "denial_notice",
            "plan_terms",
        ]
        unsupported = sorted(set(categories) - ALLOWED_PHI_ACCESS_CATEGORIES)
        if unsupported:
            return {
                "ok": False,
                "error": "unsupported_phi_category",
                "unsupported_categories": unsupported,
                "allowed_categories": sorted(ALLOWED_PHI_ACCESS_CATEGORIES),
            }
        phi_log = self.log_phi_access_purpose(
            purpose=purpose,
            denial_id=str(notice.get("denial_id")),
            requested_categories=categories,
        )
        if not phi_log.get("ok"):
            return phi_log
        items = _string_list(requested_items) or ["claim_file", "plan_guidelines"]
        request = {
            "request_id": f"claimfile_{len(self.claim_file_requests) + 1:03d}",
            "denial_id": notice.get("denial_id"),
            "requested_items": items,
            "minimum_necessary_categories": categories,
            "new_evidence_requested": bool(new_evidence_requested),
            "new_rationale_requested": bool(new_rationale_requested),
            "provided_free_of_charge": bool(provided_free_of_charge),
            "response_due_context": response_due_context or "appeal_full_and_fair_review",
            "phi_log_id": phi_log["phi_access_log"]["log_id"],
        }
        self.claim_file_requests.append(request)
        return {"ok": True, "claim_file_request": request}

    def check_continued_coverage_pending_appeal(
        self, denial_id: Optional[str] = None, **fields: Any
    ) -> Dict[str, Any]:
        notice = self._find_denial(self._active_denial_id(denial_id))
        if not notice:
            return {"ok": False, "error": "denial_notice_not_found"}
        review_type = str(notice.get("review_type", "")).lower()
        reason_text = str(notice.get("denial_reason", "")).lower()
        ongoing = _bool_from(
            fields,
            "ongoing_course_of_treatment",
            notice,
            "ongoing_course_of_treatment",
            bool(notice.get("continuity_of_care")),
        )
        reduction_or_termination = _bool_from(
            fields,
            "reduction_or_termination",
            notice,
            "reduction_or_termination",
            any(
                token in f"{review_type} {reason_text}"
                for token in ("concurrent", "continued", "termination", "reduction")
            ),
        )
        advance_notice = _bool_from(
            fields, "advance_notice_given", notice, "advance_notice_given", False
        )
        advance_review_available = _bool_from(
            fields,
            "advance_review_available",
            notice,
            "advance_review_available",
            bool(ongoing and reduction_or_termination and advance_notice),
        )
        continued_required = _bool_from(
            fields,
            "continued_coverage_required",
            notice,
            "continued_coverage_required",
            bool(ongoing and reduction_or_termination),
        )
        expedited_required = _bool_from(
            fields,
            "expedited_path_required",
            notice,
            "expedited_path_required",
            bool(
                notice.get("urgent_care")
                or "urgent" in review_type
                or (continued_required and not advance_review_available)
            ),
        )
        check_id = f"continued_{len(self.continued_coverage_checks) + 1:03d}"
        handoff_id = None
        if continued_required or expedited_required:
            handoff_id = f"handoff_{len(self.handoffs) + 1:03d}"
            self._ensure_handoff(
                {
                    "handoff_id": handoff_id,
                    "type": "continued_coverage_review",
                    "denial_id": notice.get("denial_id"),
                    "check_id": check_id,
                    "continued_coverage_pending_appeal": continued_required,
                    "expedited_path_required": expedited_required,
                    "no_coverage_decision": True,
                }
            )
        check = {
            "check_id": check_id,
            "denial_id": notice.get("denial_id"),
            "ongoing_course_of_treatment": ongoing,
            "reduction_or_termination": reduction_or_termination,
            "advance_notice_given": advance_notice,
            "advance_review_available": advance_review_available,
            "continued_coverage_required": continued_required,
            "expedited_path_required": expedited_required,
            "handoff_id": handoff_id,
        }
        self.continued_coverage_checks.append(check)
        return {"ok": True, "continued_coverage_check": check}

    def record_language_access_offer(
        self,
        member_id: Optional[str] = None,
        language: Optional[str] = None,
        applicable_non_english_language: Optional[bool] = None,
        qualified_interpreter_offered: bool = True,
        translated_notice_requested: bool = False,
        machine_translation_human_review_required: bool = False,
        privacy_independent_decision_protected: bool = True,
        accepted_or_declined: str = "offered",
        adult_or_minor_interpreter_restriction_acknowledged: bool = True,
        **fields: Any,
    ) -> Dict[str, Any]:
        effective_language = str(
            language or self.member.get("language") or self.member.get("locale") or "unknown"
        )
        if applicable_non_english_language is None:
            applicable_non_english_language = not effective_language.lower().startswith("en")
        event = {
            "event_id": f"lang_{len(self.language_access_events) + 1:03d}",
            "member_id": member_id or self._member_id(),
            "language": effective_language,
            "applicable_non_english_language": bool(applicable_non_english_language),
            "qualified_interpreter_offered": bool(qualified_interpreter_offered),
            "translated_notice_requested": bool(translated_notice_requested),
            "machine_translation_human_review_required": bool(
                machine_translation_human_review_required
            ),
            "privacy_independent_decision_protected": bool(privacy_independent_decision_protected),
            "accepted_or_declined": str(accepted_or_declined),
            "adult_or_minor_interpreter_restriction_acknowledged": bool(
                adult_or_minor_interpreter_restriction_acknowledged
            ),
        }
        self.language_access_events.append(event)
        return {"ok": True, "language_access_event": event}

    def validate_phi_authorization_document(
        self,
        actor_id: Optional[str] = None,
        member_id: Optional[str] = None,
        denial_id: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        actor = actor_id or self.session_context.get("actor_id") or self._member_id()
        member = member_id or self._member_id()
        authority = next(
            (
                item
                for item in self.representative_authorities
                if item.get("actor_id") == actor and item.get("member_id") == member
            ),
            {},
        )
        allowed_categories = _string_list(
            fields.get("allowed_categories") or authority.get("allowed_categories")
        ) or sorted(ALLOWED_PHI_ACCESS_CATEGORIES)
        values = {
            "information_description": fields.get("information_description")
            or authority.get("information_description"),
            "authorized_person": fields.get("authorized_person")
            or authority.get("authorized_person")
            or member,
            "recipient": fields.get("recipient") or authority.get("recipient") or actor,
            "purpose": fields.get("purpose") or authority.get("purpose") or "appeal_and_phi_access",
            "expiration_date_or_event": fields.get("expiration_date_or_event")
            or authority.get("expiration_date_or_event")
            or authority.get("expiration_date"),
            "signature_date": fields.get("signature_date") or authority.get("signature_date"),
            "representative_authority_description": fields.get(
                "representative_authority_description"
            )
            or authority.get("representative_authority_description")
            or authority.get("source"),
        }
        defective_reasons = [f"missing_{key}" for key, value in values.items() if not value]
        unsupported_categories = sorted(set(allowed_categories) - ALLOWED_PHI_ACCESS_CATEGORIES)
        if unsupported_categories:
            defective_reasons.append("unsupported_allowed_categories")
        revoked = bool(
            fields.get("revoked", authority.get("revoked", authority.get("status") == "revoked"))
        )
        if revoked:
            defective_reasons.append("authorization_revoked")
        if authority and authority.get("status") not in {None, "active"}:
            defective_reasons.append("representative_authority_not_active")
        defective_reasons.extend(_string_list(fields.get("defective_reasons")))
        check = {
            "check_id": f"phi_auth_{len(self.authorization_validity_checks) + 1:03d}",
            "actor_id": actor,
            "member_id": member,
            "denial_id": self._active_denial_id(denial_id),
            **values,
            "revoked": revoked,
            "defective_reasons": defective_reasons,
            "allowed_categories": allowed_categories,
            "valid": not defective_reasons,
        }
        auth_check = {
            "check_id": f"auth_{len(self.authorization_checks) + 1:03d}",
            "actor_id": actor,
            "member_id": member,
            "purpose": values["purpose"],
            "authorized": check["valid"],
            "basis": "phi_authorization_document",
            "scope": allowed_categories,
        }
        check["authorization_check_id"] = auth_check["check_id"]
        self.authorization_validity_checks.append(check)
        self.authorization_checks.append(auth_check)
        return {"ok": True, "authorization_validity_check": check}

    def draft_internal_appeal(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_appeal_fields(fields, default_action_code="internal_appeal")
        validation = _validate_appeal_draft_fields(normalized, fields)
        if validation:
            return validation
        return self._create_draft("internal_appeal", normalized)

    def draft_external_review_request(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_appeal_fields(
            fields, default_action_code="external_review_request"
        )
        validation = _validate_appeal_draft_fields(normalized, fields)
        if validation:
            return validation
        return self._create_draft("external_review", normalized)

    def record_appeal_and_external_review_rights(
        self, draft_id: Optional[str] = None
    ) -> Dict[str, Any]:
        disclosure = {
            "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "appeal_and_external_review_rights",
            "text": self.task.mandatory_disclosure(),
            "no_outcome_guarantee": True,
        }
        self.disclosures.append(disclosure)
        return {"ok": True, "disclosure": disclosure}

    def submit_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        draft_type = str(draft.get("draft_type"))
        if draft_type not in {"internal_appeal", "external_review"}:
            return {"ok": False, "error": "unsupported_draft_type", "draft_type": draft_type}
        return self.submit_appeal_case(**dict(draft.get("fields", {})))

    def submit_appeal_case(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_appeal_fields(fields)
        queue = (
            "external_reviews"
            if normalized.get("requested_action_code") == "external_review_request"
            else "internal_appeals"
        )
        case_id = f"appeal_{len(self.appeal_cases) + 1:03d}"
        case = {
            "case_id": case_id,
            **normalized,
            "status": "submitted",
            "queue": queue,
        }
        self.appeal_cases.append(case)
        self.outbox.append(
            {
                "type": "appeal_received_notice",
                "case_id": case_id,
                "denial_id": case.get("denial_id"),
            }
        )
        return {"ok": True, "appeal_case": case}

    def final_state(self) -> Dict[str, Any]:
        return {
            "session_context": dict(self.session_context),
            "member": dict(self.member),
            "representative_authorities": [dict(item) for item in self.representative_authorities],
            "plans": [dict(item) for item in self.plans],
            "denial_notices": [dict(item) for item in self.denial_notices],
            "claim_summaries": [dict(item) for item in self.claim_summaries],
            "authorization_checks": [dict(item) for item in self.authorization_checks],
            "phi_access_logs": [dict(item) for item in self.phi_access_logs],
            "deadline_checks": [dict(item) for item in self.deadline_checks],
            "urgent_care_checks": [dict(item) for item in self.urgent_care_checks],
            "external_review_checks": [dict(item) for item in self.external_review_checks],
            "denial_notice_checks": [dict(item) for item in self.denial_notice_checks],
            "claim_file_requests": [dict(item) for item in self.claim_file_requests],
            "continued_coverage_checks": [dict(item) for item in self.continued_coverage_checks],
            "language_access_events": [dict(item) for item in self.language_access_events],
            "authorization_validity_checks": [
                dict(item) for item in self.authorization_validity_checks
            ],
            "drafts": [dict(item) for item in self.drafts.values()],
            "disclosures": [dict(item) for item in self.disclosures],
            "appeal_cases": [dict(item) for item in self.appeal_cases],
            "handoffs": [dict(item) for item in self.handoffs],
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

    def _normalize_appeal_fields(
        self, fields: Dict[str, Any], default_action_code: str = "internal_appeal"
    ) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        denial_id = str(fields.get("denial_id") or facts.get("denial_id", ""))
        denial = self._find_denial(denial_id) or {}
        action_code = _canonical_requested_action_code(
            fields.get("requested_action_code")
            or fields.get("requested_action")
            or facts.get("requested_action_code")
            or facts.get("requested_action")
            or default_action_code,
            default=default_action_code,
        )
        basis_code = _canonical_appeal_basis_code(
            fields.get("appeal_basis_code")
            or fields.get("appeal_reason")
            or facts.get("appeal_basis_code")
            or facts.get("appeal_reason")
            or "appeal_denial"
        )
        narrative = _appeal_narrative(fields, facts, denial, action_code, basis_code)
        customer_statement = str(
            fields.get("customer_statement") or facts.get("customer_statement") or ""
        )
        return {
            "member_id": str(
                fields.get("member_id") or facts.get("member_id") or self._member_id()
            ),
            "representative_id": str(
                fields.get("representative_id") or facts.get("representative_id", "")
            ),
            "plan_id": str(
                fields.get("plan_id") or denial.get("plan_id") or facts.get("plan_id", "")
            ),
            "denial_id": denial_id,
            "claim_id": str(
                fields.get("claim_id") or denial.get("claim_id") or facts.get("claim_id", "")
            ),
            "requested_action_code": action_code,
            "appeal_basis_code": basis_code,
            "appeal_narrative": narrative,
            "customer_statement": customer_statement,
            # Legacy aliases keep existing generated tasks, graders, and reports compatible.
            "appeal_summary": narrative,
            "requested_action": action_code,
            "appeal_reason": basis_code,
            "urgency_requested": bool(
                fields.get("urgency_requested", facts.get("urgency_requested", False))
            ),
        }

    def _find_plan(self, plan_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next((plan for plan in self.plans if plan.get("plan_id") == plan_id), None)

    def _find_denial(self, denial_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next(
            (notice for notice in self.denial_notices if notice.get("denial_id") == denial_id), None
        )

    def _active_denial_id(self, denial_id: Optional[str] = None) -> Optional[str]:
        if denial_id:
            return denial_id
        scoped = self.task.hidden_facts.get("denial_id")
        if scoped:
            return str(scoped)
        active = [item for item in self.denial_notices if item.get("status", "active") == "active"]
        if len(active) == 1:
            return str(active[0].get("denial_id"))
        if len(self.denial_notices) == 1:
            return str(self.denial_notices[0].get("denial_id"))
        return None

    def _find_claim(self, claim_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next(
            (claim for claim in self.claim_summaries if claim.get("claim_id") == claim_id), None
        )

    def _has_phi_log(self, denial_id: str) -> bool:
        return any(
            log.get("denial_id") == denial_id and log.get("minimum_necessary")
            for log in self.phi_access_logs
        )

    def _has_authorized_phi_access(self) -> bool:
        member_id = self._member_id()
        return any(
            check.get("member_id") == member_id
            and check.get("authorized") is True
            and (
                check.get("purpose") in {"appeal_and_phi_access", "prepare_coverage_appeal"}
                or "appeal_and_phi_access" in check.get("scope", [])
            )
            for check in self.authorization_checks
        )

    def _member_id(self) -> str:
        return str(
            self.session_context.get("member_id")
            or self.member.get("member_id")
            or self.task.hidden_facts.get("member_id", "")
        )

    def _ensure_handoff(self, handoff: Dict[str, Any]) -> None:
        if handoff not in self.handoffs:
            self.handoffs.append(handoff)


def _canonical_requested_action_code(value: Any, *, default: str = "internal_appeal") -> str:
    text = _normalized_code_text(value)
    if text in {
        "internal_appeal",
        "expedited_internal_appeal",
        "external_review_request",
        "continuity_of_care_review",
        "representative_authority_handoff",
    }:
        return text
    words = text.replace("_", " ")
    if "external" in words or "independent" in words:
        return "external_review_request"
    if "expedited" in words or "urgent" in words:
        return "expedited_internal_appeal"
    if "continuity" in words or "continued care" in words:
        return "continuity_of_care_review"
    if "representative" in words and ("handoff" in words or "authority" in words):
        return "representative_authority_handoff"
    if (
        "appeal" in words
        or "overturn" in words
        or "denial" in words
        or "prior authorization" in words
    ):
        return "internal_appeal"
    return default


def _canonical_appeal_basis_code(value: Any) -> str:
    text = _normalized_code_text(value)
    if text in {
        "medical_necessity_supporting_records_available",
        "prior_authorization_denial",
        "service_not_medically_necessary_disputed",
        "coverage_policy_exception_requested",
        "urgent_care_or_continuity_risk",
        "final_adverse_determination_external_review",
        "missing_or_incomplete_denial_notice",
        "appeal_denial",
    }:
        return text
    words = text.replace("_", " ")
    if "external" in words or "final adverse" in words or "final denial" in words:
        return "final_adverse_determination_external_review"
    if "urgent" in words or "continuity" in words or "continued care" in words:
        return "urgent_care_or_continuity_risk"
    if "medical necessity" in words or ("medical" in words and "necess" in words):
        return "medical_necessity_supporting_records_available"
    if "supporting record" in words or "medical record" in words:
        return "medical_necessity_supporting_records_available"
    if "prior authorization" in words or "preauthorization" in words:
        return "prior_authorization_denial"
    if "policy exception" in words or "coverage exception" in words:
        return "coverage_policy_exception_requested"
    if "incomplete notice" in words or "missing notice" in words:
        return "missing_or_incomplete_denial_notice"
    return "appeal_denial"


def _appeal_narrative(
    fields: Dict[str, Any],
    facts: Dict[str, Any],
    denial: Dict[str, Any],
    action_code: str,
    basis_code: str,
) -> str:
    provided = (
        fields.get("appeal_narrative") or fields.get("appeal_summary") or fields.get("summary")
    )
    if provided:
        return str(provided)
    service = denial.get("service_code") or facts.get("service_code") or "the denied service"
    denial_id = (
        fields.get("denial_id") or facts.get("denial_id") or denial.get("denial_id") or "the denial"
    )
    action_text = action_code.replace("_", " ")
    basis_text = basis_code.replace("_", " ")
    return f"Request {action_text} for {denial_id} concerning {service} based on {basis_text}."


def _validate_appeal_draft_fields(
    normalized: Dict[str, Any], original: Dict[str, Any]
) -> Dict[str, Any] | None:
    action_text = _normalized_phrase(
        original.get("requested_action_code") or original.get("requested_action")
    )
    narrative_text = _normalized_phrase(normalized.get("appeal_narrative"))
    if action_text and _contains_outcome_request(action_text):
        return {
            "ok": False,
            "error": "invalid_requested_action_code",
            "reason": "requested action must be a workflow code, not a coverage/payment approval request",
            "requested_action_code": normalized.get("requested_action_code"),
        }
    if narrative_text and _contains_prohibited_narrative(narrative_text):
        return {
            "ok": False,
            "error": "invalid_appeal_narrative",
            "reason": "appeal narrative must be neutral and must not assert medical necessity or coverage approval",
        }
    return None


def _normalized_phrase(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _contains_outcome_request(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "approve coverage",
            "approve payment",
            "approve continued",
            "guarantee coverage",
            "guarantee payment",
            "will be covered",
            "will pay",
            "ensure coverage",
        )
    )


def _contains_prohibited_narrative(text: str) -> bool:
    if any(
        marker in text
        for marker in ("guarantee", "will be covered", "will pay", "approve coverage")
    ):
        return True
    return any(
        marker in text
        for marker in ("is medically necessary", "are medically necessary", "medically necessary")
    )


def _normalized_code_text(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", "_").split())


def _string_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, (tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _bool_from(
    fields: Dict[str, Any], field_key: str, source: Dict[str, Any], source_key: str, default: bool
) -> bool:
    if field_key in fields:
        return bool(fields.get(field_key))
    if source_key in source:
        return bool(source.get(source_key))
    return bool(default)
