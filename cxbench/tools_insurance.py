from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .task_schema import TaskSpec


def _canonical_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _copy_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass
class InsuranceClaimIntakeSystem:
    """In-memory FNOL system for process-compliance insurance intake tasks."""

    task: TaskSpec
    draft_dir: Path
    session_context: Dict[str, Any] = field(default_factory=dict)
    customer: Dict[str, Any] = field(default_factory=dict)
    policies: List[Dict[str, Any]] = field(default_factory=list)
    prior_claims: List[Dict[str, Any]] = field(default_factory=list)
    siu_indicators: List[Dict[str, Any]] = field(default_factory=list)
    parties: List[Dict[str, Any]] = field(default_factory=list)
    other_carriers: List[Dict[str, Any]] = field(default_factory=list)
    authority_checks: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_checks: List[Dict[str, Any]] = field(default_factory=list)
    loss_classifications: List[Dict[str, Any]] = field(default_factory=list)
    siu_checks: List[Dict[str, Any]] = field(default_factory=list)
    policy_term_checks: List[Dict[str, Any]] = field(default_factory=list)
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    disclosures: List[Dict[str, Any]] = field(default_factory=list)
    drafts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence_requests: List[Dict[str, Any]] = field(default_factory=list)
    claim_cases: List[Dict[str, Any]] = field(default_factory=list)
    adjuster_routes: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_review_cases: List[Dict[str, Any]] = field(default_factory=list)
    siu_handoffs: List[Dict[str, Any]] = field(default_factory=list)
    support_handoffs: List[Dict[str, Any]] = field(default_factory=list)
    policy_term_reviews: List[Dict[str, Any]] = field(default_factory=list)
    subrogation_referrals: List[Dict[str, Any]] = field(default_factory=list)
    vehicle_loss_assessments: List[Dict[str, Any]] = field(default_factory=list)
    tow_storage_requests: List[Dict[str, Any]] = field(default_factory=list)
    claim_withdrawal_requests: List[Dict[str, Any]] = field(default_factory=list)
    claim_close_requests: List[Dict[str, Any]] = field(default_factory=list)
    claim_reopen_requests: List[Dict[str, Any]] = field(default_factory=list)
    claim_status_inquiries: List[Dict[str, Any]] = field(default_factory=list)
    outbox: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        state = self.task.initial_state
        self.session_context = dict(state.get("session_context", {}))
        self.customer = dict(state.get("customer", {}))
        self.policies = _copy_list(state.get("policies", []))
        self.prior_claims = _copy_list(state.get("prior_claims", []))
        self.siu_indicators = _copy_list(state.get("siu_indicators", []))
        self.parties = _copy_list(state.get("parties", []))
        self.other_carriers = _copy_list(state.get("other_carriers", []))
        self.authority_checks = _copy_list(state.get("authority_checks", []))
        self.duplicate_checks = _copy_list(state.get("duplicate_checks", []))
        self.loss_classifications = _copy_list(state.get("loss_classifications", []))
        self.siu_checks = _copy_list(state.get("siu_checks", []))
        self.policy_term_checks = _copy_list(state.get("policy_term_checks", []))
        self.acknowledgments = _copy_list(state.get("acknowledgments", []))
        self.disclosures = _copy_list(state.get("disclosures", []))
        self.evidence_requests = _copy_list(state.get("evidence_requests", []))
        self.claim_cases = _copy_list(state.get("claim_cases", []))
        self.adjuster_routes = _copy_list(state.get("adjuster_routes", []))
        self.duplicate_review_cases = _copy_list(state.get("duplicate_review_cases", []))
        self.siu_handoffs = _copy_list(state.get("siu_handoffs", []))
        self.support_handoffs = _copy_list(state.get("support_handoffs", []))
        self.policy_term_reviews = _copy_list(state.get("policy_term_reviews", []))
        self.subrogation_referrals = _copy_list(state.get("subrogation_referrals", []))
        self.vehicle_loss_assessments = _copy_list(state.get("vehicle_loss_assessments", []))
        self.tow_storage_requests = _copy_list(state.get("tow_storage_requests", []))
        self.claim_withdrawal_requests = _copy_list(state.get("claim_withdrawal_requests", []))
        self.claim_close_requests = _copy_list(state.get("claim_close_requests", []))
        self.claim_reopen_requests = _copy_list(state.get("claim_reopen_requests", []))
        self.claim_status_inquiries = _copy_list(state.get("claim_status_inquiries", []))
        self.outbox = _copy_list(state.get("outbox", []))

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_session_context": self.get_session_context,
            "list_policies": self.list_policies,
            "get_policy_summary": self.get_policy_summary,
            "get_claim_status": self.get_claim_status,
            "list_claim_parties": self.list_claim_parties,
            "check_reporter_authority": self.check_reporter_authority,
            "check_duplicate_claim": self.check_duplicate_claim,
            "classify_loss_type": self.classify_loss_type,
            "check_siu_referral_signal": self.check_siu_referral_signal,
            "check_policy_term_status": self.check_policy_term_status,
            "check_subrogation_potential": self.check_subrogation_potential,
            "check_total_loss_or_storage_risk": self.check_total_loss_or_storage_risk,
            "draft_fnol": self.draft_fnol,
            "draft_evidence_request": self.draft_evidence_request,
            "create_evidence_request_draft": self.create_evidence_request_draft,
            "draft_authority_handoff": self.draft_authority_handoff,
            "draft_duplicate_claim_review": self.draft_duplicate_claim_review,
            "draft_siu_review_handoff": self.draft_siu_review_handoff,
            "draft_claim_support_handoff": self.draft_claim_support_handoff,
            "draft_policy_term_review": self.draft_policy_term_review,
            "draft_subrogation_referral": self.draft_subrogation_referral,
            "draft_total_loss_review": self.draft_total_loss_review,
            "draft_tow_storage_request": self.draft_tow_storage_request,
            "draft_claim_withdrawal_request": self.draft_claim_withdrawal_request,
            "draft_claim_close_request": self.draft_claim_close_request,
            "draft_claim_reopen_request": self.draft_claim_reopen_request,
            "draft_claim_status_inquiry_response": self.draft_claim_status_inquiry_response,
            "record_claim_acknowledgment": self.record_claim_acknowledgment,
            "record_no_promise_disclosure": self.record_no_promise_disclosure,
            "submit_claim_intake": self.submit_claim_intake,
            "submit_evidence_request": self.submit_evidence_request,
            "submit_authority_handoff": self.submit_authority_handoff,
            "submit_duplicate_claim_review": self.submit_duplicate_claim_review,
            "submit_siu_review_handoff": self.submit_siu_review_handoff,
            "submit_claim_support_handoff": self.submit_claim_support_handoff,
            "submit_policy_term_review": self.submit_policy_term_review,
            "submit_subrogation_referral": self.submit_subrogation_referral,
            "submit_total_loss_review": self.submit_total_loss_review,
            "submit_tow_storage_request": self.submit_tow_storage_request,
            "submit_claim_withdrawal_request": self.submit_claim_withdrawal_request,
            "submit_claim_close_request": self.submit_claim_close_request,
            "submit_claim_reopen_request": self.submit_claim_reopen_request,
            "submit_claim_status_inquiry_response": self.submit_claim_status_inquiry_response,
            "submit_adjuster_route": self.submit_adjuster_route,
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
            "customer": {
                key: self.customer.get(key)
                for key in ("customer_id", "display_name", "locale")
                if key in self.customer
            },
        }

    def list_policies(self) -> Dict[str, Any]:
        return {"ok": True, "policies": [self._public_policy(policy) for policy in self.policies]}

    def get_policy_summary(self, policy_id: Optional[str] = None) -> Dict[str, Any]:
        policy = self._find_policy(policy_id)
        if not policy:
            return {"ok": False, "error": "policy_not_found"}
        return {"ok": True, "policy_summary": self._public_policy(policy)}

    def get_claim_status(self, claim_id: Optional[str] = None) -> Dict[str, Any]:
        claim = self._find_claim(claim_id)
        if not claim:
            return {"ok": False, "error": "claim_not_found"}
        public = {
            key: claim.get(key)
            for key in (
                "claim_id",
                "policy_id",
                "loss_date",
                "loss_location",
                "loss_cause",
                "status",
                "assigned_adjuster",
                "last_update",
            )
            if key in claim
        }
        inquiry = {
            "inquiry_id": f"inq_{len(self.claim_status_inquiries) + 1:03d}",
            "claim_id": public.get("claim_id", ""),
            "policy_id": public.get("policy_id", ""),
            "status": public.get("status", "unknown"),
            "result": public,
        }
        self.claim_status_inquiries.append(inquiry)
        return {"ok": True, "claim_status": public}

    def list_claim_parties(
        self, claim_id: Optional[str] = None, policy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        parties = [
            dict(party)
            for party in self.parties
            if (not claim_id or party.get("claim_id") == claim_id)
            and (not policy_id or party.get("policy_id") == policy_id)
        ]
        carriers = [
            dict(carrier)
            for carrier in self.other_carriers
            if (not claim_id or carrier.get("claim_id") == claim_id)
            and (not policy_id or carrier.get("policy_id") == policy_id)
        ]
        return {"ok": True, "parties": parties, "other_carriers": carriers}

    def check_reporter_authority(
        self, policy_id: str, reporter_id: Optional[str] = None, reporter_role: Optional[str] = None
    ) -> Dict[str, Any]:
        policy = self._find_policy(policy_id)
        if not policy:
            return {"ok": False, "error": "policy_not_found"}
        role = str(
            reporter_role
            or self.task.hidden_facts.get(
                "reporter_role", self.session_context.get("role", "named_insured")
            )
        )
        allowed = role in {
            "named_insured",
            "listed_driver",
            "authorized_representative",
            "claimant",
        }
        check = {
            "check_id": f"auth_{len(self.authority_checks) + 1:03d}",
            "policy_id": policy_id,
            "reporter_id": reporter_id or self.session_context.get("customer_id"),
            "reporter_role": role,
            "can_open_intake": allowed,
        }
        self.authority_checks.append(check)
        return {"ok": True, "reporter_authority": check}

    def check_duplicate_claim(
        self, policy_id: str, loss_date: Optional[str] = None, loss_location: Optional[str] = None
    ) -> Dict[str, Any]:
        matches = [
            claim
            for claim in self.prior_claims
            if claim.get("policy_id") == policy_id
            and (not loss_date or claim.get("loss_date") == loss_date)
            and (not loss_location or claim.get("loss_location") == loss_location)
        ]
        check = {
            "check_id": f"dup_{len(self.duplicate_checks) + 1:03d}",
            "policy_id": policy_id,
            "possible_duplicate": bool(matches),
            "matched_claim_ids": [claim.get("claim_id") for claim in matches],
        }
        self.duplicate_checks.append(check)
        return {"ok": True, "duplicate_claim": check}

    def classify_loss_type(
        self, policy_id: str, loss_cause: str, loss_description: str = ""
    ) -> Dict[str, Any]:
        policy = self._find_policy(policy_id) or {}
        line = str(policy.get("line", self.task.hidden_facts.get("policy_line", "auto")))
        route = "auto_physical_damage" if line == "auto" else f"{line}_intake"
        classification = {
            "check_id": f"loss_{len(self.loss_classifications) + 1:03d}",
            "policy_id": policy_id,
            "loss_cause": loss_cause,
            "loss_description": loss_description,
            "loss_type": route,
            "coverage_decision_made": False,
        }
        self.loss_classifications.append(classification)
        return {"ok": True, "loss_classification": classification}

    def check_siu_referral_signal(self, policy_id: str, loss_cause: str = "") -> Dict[str, Any]:
        indicators = list(self.siu_indicators)
        check = {
            "check_id": f"siu_{len(self.siu_checks) + 1:03d}",
            "policy_id": policy_id,
            "neutral_review_required": bool(indicators),
            "indicator_count": len(indicators),
            "indicator_codes": [
                str(item.get("code", item.get("type", "indicator")))
                for item in indicators
                if isinstance(item, dict)
            ],
            "review_posture": "neutral_review",
            "accusation_made": False,
        }
        self.siu_checks.append(check)
        return {"ok": True, "siu_referral_signal": check}

    def check_policy_term_status(self, policy_id: str) -> Dict[str, Any]:
        policy = self._find_policy(policy_id)
        if not policy:
            return {"ok": False, "error": "policy_not_found"}
        term_status = str(policy.get("term_status", "unknown"))
        check = {
            "check_id": f"term_{len(self.policy_term_checks) + 1:03d}",
            "policy_id": policy_id,
            "term_status": term_status,
            "eligible_for_ordinary_intake": term_status == "active",
            "source": policy.get("term_source", "policy_admin_fixture"),
            "effective_date": policy.get("effective_date"),
            "expiration_date": policy.get("expiration_date"),
            "lapse_date": policy.get("lapse_date"),
        }
        self.policy_term_checks.append(check)
        return {"ok": True, "policy_term_status": check}

    def check_subrogation_potential(
        self, policy_id: str, claim_id: Optional[str] = None
    ) -> Dict[str, Any]:
        party_result = self.list_claim_parties(claim_id=claim_id, policy_id=policy_id)
        third_parties = [
            party
            for party in party_result["parties"]
            if party.get("role") in {"other_driver", "property_owner", "third_party"}
        ]
        carriers = party_result["other_carriers"]
        check = {
            "check_id": f"subro_{len(self.subrogation_referrals) + 1:03d}",
            "policy_id": policy_id,
            "claim_id": claim_id or self.task.hidden_facts.get("claim_id", ""),
            "subrogation_review_needed": bool(third_parties or carriers),
            "party_count": len(third_parties),
            "other_carrier_count": len(carriers),
        }
        return {"ok": True, "subrogation_check": check}

    def check_total_loss_or_storage_risk(
        self, policy_id: str, claim_id: Optional[str] = None, vehicle_location: Optional[str] = None
    ) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        estimated_value = float(facts.get("estimated_vehicle_value", 0) or 0)
        estimated_damage = float(facts.get("estimated_damage_amount", 0) or 0)
        storage_daily = float(facts.get("storage_daily_rate", 0) or 0)
        ratio = estimated_damage / estimated_value if estimated_value else 0.0
        assessment = {
            "assessment_id": f"veh_{len(self.vehicle_loss_assessments) + 1:03d}",
            "policy_id": policy_id,
            "claim_id": claim_id or facts.get("claim_id", ""),
            "vehicle_location": vehicle_location or facts.get("vehicle_location", ""),
            "estimated_vehicle_value": estimated_value,
            "estimated_damage_amount": estimated_damage,
            "damage_to_value_ratio": round(ratio, 4),
            "total_loss_review_needed": ratio
            >= float(facts.get("total_loss_review_threshold", 0.7) or 0.7),
            "tow_storage_review_needed": bool(
                storage_daily or facts.get("tow_yard_name") or facts.get("storage_started_date")
            ),
            "status": "review_recorded",
        }
        self.vehicle_loss_assessments.append(assessment)
        return {"ok": True, "vehicle_loss_assessment": assessment}

    def draft_fnol(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_fnol_fields(fields)
        gate = self._pre_draft_check(normalized["policy_id"])
        if not gate["ok"]:
            return gate
        return self._create_draft("fnol", normalized)

    def draft_evidence_request(
        self,
        draft_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        loss_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = {
            "evidence_request_id": f"evreq_{len(self.evidence_requests) + 1:03d}",
            "draft_id": draft_id,
            "policy_id": policy_id or self.task.hidden_facts.get("policy_id", ""),
            "loss_type": loss_type
            or self.task.hidden_facts.get("loss_type", "auto_physical_damage"),
            "requested_items": list(
                self.task.hidden_facts.get(
                    "requested_evidence",
                    ["photos", "repair_estimate", "police_report_if_available"],
                )
            ),
            "status": "drafted",
        }
        self.evidence_requests.append(request)
        path = self.draft_dir / f"{request['evidence_request_id']}.json"
        path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"ok": True, "evidence_request": request}

    def create_evidence_request_draft(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_evidence_request_fields(fields)
        return self._create_draft("evidence_request", payload)

    def draft_authority_handoff(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_support_handoff_fields(
            {**fields, "support_type": fields.get("support_type", "authority_review")}
        )
        return self._create_draft("authority_handoff", payload)

    def draft_duplicate_claim_review(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_duplicate_review_fields(fields)
        return self._create_draft("duplicate_claim_review", payload)

    def draft_siu_review_handoff(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_siu_handoff_fields(fields)
        return self._create_draft("siu_review_handoff", payload)

    def draft_claim_support_handoff(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_support_handoff_fields(fields)
        return self._create_draft("claim_support_handoff", payload)

    def draft_policy_term_review(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_policy_term_review_fields(fields)
        return self._create_draft("policy_term_review", payload)

    def draft_subrogation_referral(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_subrogation_fields(fields)
        return self._create_draft("subrogation_referral", payload)

    def draft_total_loss_review(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_total_loss_fields(fields)
        return self._create_draft("total_loss_review", payload)

    def draft_tow_storage_request(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_tow_storage_fields(fields)
        return self._create_draft("tow_storage_request", payload)

    def draft_claim_withdrawal_request(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_withdrawal_fields(fields)
        return self._create_draft("claim_withdrawal_request", payload)

    def draft_claim_close_request(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_claim_lifecycle_fields(
            fields, default_reason="customer_requested_closure"
        )
        return self._create_draft("claim_close_request", payload)

    def draft_claim_reopen_request(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_claim_lifecycle_fields(
            fields, default_reason="customer_requested_reopen"
        )
        return self._create_draft("claim_reopen_request", payload)

    def draft_claim_status_inquiry_response(self, **fields: Any) -> Dict[str, Any]:
        payload = self._normalize_claim_lifecycle_fields(fields, default_reason="status_inquiry")
        return self._create_draft("claim_status_inquiry_response", payload)

    def record_claim_acknowledgment(self, draft_id: Optional[str] = None) -> Dict[str, Any]:
        ack = {
            "acknowledgment_id": f"ack_{len(self.acknowledgments) + 1:03d}",
            "draft_id": draft_id,
            "type": "claim_intake_acknowledgment",
            "text": "We can open an intake and route it for review; this is not a coverage or payment decision.",
        }
        self.acknowledgments.append(ack)
        return {"ok": True, "acknowledgment": ack}

    def record_no_promise_disclosure(self, draft_id: Optional[str] = None) -> Dict[str, Any]:
        disclosure = {
            "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "no_coverage_or_payment_promise",
            "text": self.task.mandatory_disclosure(),
            "channel": self.session_context.get("channel", "in_app"),
        }
        self.disclosures.append(disclosure)
        return {"ok": True, "disclosure": disclosure}

    def submit_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        draft_type = str(draft.get("draft_type"))
        fields = dict(draft.get("fields", {}))
        if draft_type == "fnol":
            gate = self._gateway_preconditions(draft_id, str(fields.get("policy_id", "")))
            if not gate["ok"]:
                return gate
            return self.submit_claim_intake(**fields)
        if draft_type == "evidence_request":
            return self.submit_evidence_request(**fields)
        if draft_type == "authority_handoff":
            return self.submit_authority_handoff(**fields)
        if draft_type == "duplicate_claim_review":
            return self.submit_duplicate_claim_review(**fields)
        if draft_type == "siu_review_handoff":
            return self.submit_siu_review_handoff(**fields)
        if draft_type == "claim_support_handoff":
            return self.submit_claim_support_handoff(**fields)
        if draft_type == "policy_term_review":
            return self.submit_policy_term_review(**fields)
        if draft_type == "subrogation_referral":
            return self.submit_subrogation_referral(**fields)
        if draft_type == "total_loss_review":
            return self.submit_total_loss_review(**fields)
        if draft_type == "tow_storage_request":
            return self.submit_tow_storage_request(**fields)
        if draft_type == "claim_withdrawal_request":
            return self.submit_claim_withdrawal_request(**fields)
        if draft_type == "claim_close_request":
            return self.submit_claim_close_request(**fields)
        if draft_type == "claim_reopen_request":
            return self.submit_claim_reopen_request(**fields)
        if draft_type == "claim_status_inquiry_response":
            return self.submit_claim_status_inquiry_response(**fields)
        return {"ok": False, "error": "unsupported_draft_type", "draft_type": draft_type}

    def submit_claim_intake(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_fnol_fields(fields)
        claim = {
            "claim_id": f"claim_{len(self.claim_cases) + 1:03d}",
            **normalized,
            "status": "intake_opened",
            "coverage_decision": "not_made",
            "payment_decision": "not_made",
        }
        self.claim_cases.append(claim)
        route = {
            "route_id": f"route_{len(self.adjuster_routes) + 1:03d}",
            "claim_id": claim["claim_id"],
            "route": self._latest_loss_type(normalized["policy_id"]),
            "status": "queued_for_adjuster_review",
        }
        self.adjuster_routes.append(route)
        self.outbox.append({"type": "claim_intake_notice", "claim_id": claim["claim_id"]})
        return {"ok": True, "claim_case": claim, "adjuster_route": route}

    def submit_evidence_request(self, **fields: Any) -> Dict[str, Any]:
        request = {
            "evidence_request_id": fields.get(
                "evidence_request_id", f"evreq_{len(self.evidence_requests) + 1:03d}"
            ),
            **self._normalize_evidence_request_fields(fields),
            "status": "sent",
        }
        self.evidence_requests.append(request)
        self.outbox.append(
            {
                "type": "evidence_request_notice",
                "evidence_request_id": request["evidence_request_id"],
            }
        )
        return {"ok": True, "evidence_request": request}

    def submit_authority_handoff(self, **fields: Any) -> Dict[str, Any]:
        return self.submit_claim_support_handoff(
            **{**fields, "support_type": fields.get("support_type", "authority_review")}
        )

    def submit_duplicate_claim_review(self, **fields: Any) -> Dict[str, Any]:
        review = {
            "review_id": f"duprev_{len(self.duplicate_review_cases) + 1:03d}",
            **self._normalize_duplicate_review_fields(fields),
            "status": "review_opened",
        }
        self.duplicate_review_cases.append(review)
        self.outbox.append(
            {"type": "duplicate_claim_review_notice", "review_id": review["review_id"]}
        )
        return {"ok": True, "duplicate_claim_review": review}

    def submit_siu_review_handoff(self, **fields: Any) -> Dict[str, Any]:
        handoff = {
            "handoff_id": f"siu_{len(self.siu_handoffs) + 1:03d}",
            **self._normalize_siu_handoff_fields(fields),
            "status": "queued_for_neutral_review",
            "review_posture": "neutral_review",
        }
        self.siu_handoffs.append(handoff)
        self.outbox.append(
            {"type": "siu_review_handoff_notice", "handoff_id": handoff["handoff_id"]}
        )
        return {"ok": True, "siu_handoff": handoff}

    def submit_claim_support_handoff(self, **fields: Any) -> Dict[str, Any]:
        handoff = {
            "handoff_id": f"support_{len(self.support_handoffs) + 1:03d}",
            **self._normalize_support_handoff_fields(fields),
            "status": "queued",
        }
        self.support_handoffs.append(handoff)
        self.outbox.append(
            {"type": "claim_support_handoff_notice", "handoff_id": handoff["handoff_id"]}
        )
        return {"ok": True, "support_handoff": handoff}

    def submit_policy_term_review(self, **fields: Any) -> Dict[str, Any]:
        review = {
            "review_id": f"termrev_{len(self.policy_term_reviews) + 1:03d}",
            **self._normalize_policy_term_review_fields(fields),
            "status": "review_opened",
        }
        self.policy_term_reviews.append(review)
        self.outbox.append({"type": "policy_term_review_notice", "review_id": review["review_id"]})
        return {"ok": True, "policy_term_review": review}

    def submit_subrogation_referral(self, **fields: Any) -> Dict[str, Any]:
        referral = {
            "referral_id": f"subro_{len(self.subrogation_referrals) + 1:03d}",
            **self._normalize_subrogation_fields(fields),
            "status": "referred",
        }
        self.subrogation_referrals.append(referral)
        self.outbox.append(
            {"type": "subrogation_referral_notice", "referral_id": referral["referral_id"]}
        )
        return {"ok": True, "subrogation_referral": referral}

    def submit_total_loss_review(self, **fields: Any) -> Dict[str, Any]:
        review = {
            "assessment_id": f"tloss_{len(self.vehicle_loss_assessments) + 1:03d}",
            **self._normalize_total_loss_fields(fields),
            "status": "review_opened",
        }
        self.vehicle_loss_assessments.append(review)
        self.outbox.append(
            {"type": "total_loss_review_notice", "assessment_id": review["assessment_id"]}
        )
        return {"ok": True, "total_loss_review": review}

    def submit_tow_storage_request(self, **fields: Any) -> Dict[str, Any]:
        request = {
            "request_id": f"tow_{len(self.tow_storage_requests) + 1:03d}",
            **self._normalize_tow_storage_fields(fields),
            "status": "requested",
        }
        self.tow_storage_requests.append(request)
        self.outbox.append(
            {"type": "tow_storage_request_notice", "request_id": request["request_id"]}
        )
        return {"ok": True, "tow_storage_request": request}

    def submit_claim_withdrawal_request(self, **fields: Any) -> Dict[str, Any]:
        request = {
            "withdrawal_request_id": f"withdraw_{len(self.claim_withdrawal_requests) + 1:03d}",
            **self._normalize_withdrawal_fields(fields),
            "status": "requested",
        }
        self.claim_withdrawal_requests.append(request)
        self.outbox.append(
            {
                "type": "claim_withdrawal_request_notice",
                "withdrawal_request_id": request["withdrawal_request_id"],
            }
        )
        return {"ok": True, "claim_withdrawal_request": request}

    def submit_claim_close_request(self, **fields: Any) -> Dict[str, Any]:
        request = {
            "close_request_id": f"close_{len(self.claim_close_requests) + 1:03d}",
            **self._normalize_claim_lifecycle_fields(
                fields, default_reason="customer_requested_closure"
            ),
            "status": "requested",
        }
        self.claim_close_requests.append(request)
        self.outbox.append(
            {"type": "claim_close_request_notice", "close_request_id": request["close_request_id"]}
        )
        return {"ok": True, "claim_close_request": request}

    def submit_claim_reopen_request(self, **fields: Any) -> Dict[str, Any]:
        request = {
            "reopen_request_id": f"reopen_{len(self.claim_reopen_requests) + 1:03d}",
            **self._normalize_claim_lifecycle_fields(
                fields, default_reason="customer_requested_reopen"
            ),
            "status": "requested",
        }
        self.claim_reopen_requests.append(request)
        self.outbox.append(
            {
                "type": "claim_reopen_request_notice",
                "reopen_request_id": request["reopen_request_id"],
            }
        )
        return {"ok": True, "claim_reopen_request": request}

    def submit_claim_status_inquiry_response(self, **fields: Any) -> Dict[str, Any]:
        inquiry = {
            "inquiry_id": f"inq_{len(self.claim_status_inquiries) + 1:03d}",
            **self._normalize_claim_lifecycle_fields(fields, default_reason="status_inquiry"),
            "status": "responded",
        }
        self.claim_status_inquiries.append(inquiry)
        self.outbox.append(
            {"type": "claim_status_inquiry_notice", "inquiry_id": inquiry["inquiry_id"]}
        )
        return {"ok": True, "claim_status_inquiry": inquiry}

    def submit_adjuster_route(
        self, claim_id: str, route: str = "standard_adjuster_review"
    ) -> Dict[str, Any]:
        item = {
            "route_id": f"route_{len(self.adjuster_routes) + 1:03d}",
            "claim_id": claim_id,
            "route": route,
            "status": "queued",
        }
        self.adjuster_routes.append(item)
        return {"ok": True, "adjuster_route": item}

    def final_state(self) -> Dict[str, Any]:
        return {
            "business_system": "insurance_claim_intake_v0",
            "session_context": dict(self.session_context),
            "customer": dict(self.customer),
            "policies": [dict(item) for item in self.policies],
            "prior_claims": [dict(item) for item in self.prior_claims],
            "siu_indicators": [dict(item) for item in self.siu_indicators],
            "parties": [dict(item) for item in self.parties],
            "other_carriers": [dict(item) for item in self.other_carriers],
            "authority_checks": [dict(item) for item in self.authority_checks],
            "duplicate_checks": [dict(item) for item in self.duplicate_checks],
            "loss_classifications": [dict(item) for item in self.loss_classifications],
            "siu_checks": [dict(item) for item in self.siu_checks],
            "policy_term_checks": [dict(item) for item in self.policy_term_checks],
            "acknowledgments": [dict(item) for item in self.acknowledgments],
            "disclosures": [dict(item) for item in self.disclosures],
            "drafts": [dict(item) for item in self.drafts.values()],
            "evidence_requests": [dict(item) for item in self.evidence_requests],
            "claim_cases": [dict(item) for item in self.claim_cases],
            "adjuster_routes": [dict(item) for item in self.adjuster_routes],
            "duplicate_review_cases": [dict(item) for item in self.duplicate_review_cases],
            "siu_handoffs": [dict(item) for item in self.siu_handoffs],
            "support_handoffs": [dict(item) for item in self.support_handoffs],
            "policy_term_reviews": [dict(item) for item in self.policy_term_reviews],
            "subrogation_referrals": [dict(item) for item in self.subrogation_referrals],
            "vehicle_loss_assessments": [dict(item) for item in self.vehicle_loss_assessments],
            "tow_storage_requests": [dict(item) for item in self.tow_storage_requests],
            "claim_withdrawal_requests": [dict(item) for item in self.claim_withdrawal_requests],
            "claim_close_requests": [dict(item) for item in self.claim_close_requests],
            "claim_reopen_requests": [dict(item) for item in self.claim_reopen_requests],
            "claim_status_inquiries": [dict(item) for item in self.claim_status_inquiries],
            "outbox": [dict(item) for item in self.outbox],
        }

    def _pre_draft_check(self, policy_id: str) -> Dict[str, Any]:
        authority = self._latest_for_policy(self.authority_checks, policy_id)
        if not authority:
            return {"ok": False, "error": "reporter_authority_check_required_before_draft"}
        if authority.get("can_open_intake") is not True:
            return {"ok": False, "error": "reporter_not_authorized_for_intake"}
        duplicate = self._latest_for_policy(self.duplicate_checks, policy_id)
        if not duplicate:
            return {"ok": False, "error": "duplicate_claim_check_required_before_draft"}
        if duplicate.get("possible_duplicate") is True:
            return {"ok": False, "error": "duplicate_claim_review_required"}
        if not self._latest_for_policy(self.loss_classifications, policy_id):
            return {"ok": False, "error": "loss_classification_required_before_draft"}
        term = self._latest_for_policy(self.policy_term_checks, policy_id)
        if term and term.get("eligible_for_ordinary_intake") is not True:
            return {"ok": False, "error": "policy_term_review_required"}
        return {"ok": True}

    def _gateway_preconditions(self, draft_id: str, policy_id: str) -> Dict[str, Any]:
        gate = self._pre_draft_check(policy_id)
        if not gate["ok"]:
            return gate
        if not any(item.get("draft_id") == draft_id for item in self.acknowledgments):
            return {"ok": False, "error": "claim_acknowledgment_required_before_submit"}
        if not any(item.get("draft_id") == draft_id for item in self.disclosures):
            return {"ok": False, "error": "no_promise_disclosure_required_before_submit"}
        return {"ok": True}

    def _create_draft(self, draft_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"draft_type": draft_type, "fields": fields}
        draft_hash = _canonical_hash(payload)
        draft_id = f"draft_{len(self.drafts) + 1:03d}"
        draft = {"draft_id": draft_id, "draft_hash": draft_hash, **payload}
        self.drafts[draft_id] = draft
        path = self.draft_dir / f"{draft_id}.json"
        path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"ok": True, **draft}

    def _normalize_fnol_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(fields.get("policy_id") or facts.get("policy_id", "")),
            "policy_line": str(fields.get("policy_line") or facts.get("policy_line", "auto")),
            "loss_date": str(fields.get("loss_date") or facts.get("loss_date", "")),
            "loss_location": str(fields.get("loss_location") or facts.get("loss_location", "")),
            "loss_cause": str(fields.get("loss_cause") or facts.get("loss_cause", "")),
            "loss_description": str(
                fields.get("loss_description") or facts.get("loss_description", "")
            ),
            "reporter_role": str(
                fields.get("reporter_role") or facts.get("reporter_role", "named_insured")
            ),
        }

    def _normalize_evidence_request_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        requested_items = fields.get(
            "requested_items",
            facts.get(
                "requested_evidence", ["photos", "repair_estimate", "police_report_if_available"]
            ),
        )
        if isinstance(requested_items, str):
            requested_items = [item.strip() for item in requested_items.split(",") if item.strip()]
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(fields.get("policy_id") or facts.get("policy_id", "")),
            "draft_id": fields.get("draft_id"),
            "loss_type": str(
                fields.get("loss_type") or facts.get("loss_type", "auto_physical_damage")
            ),
            "requested_items": list(requested_items) if isinstance(requested_items, list) else [],
        }

    def _normalize_duplicate_review_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        policy_id = str(fields.get("policy_id") or facts.get("policy_id", ""))
        duplicate = self._latest_for_policy(self.duplicate_checks, policy_id) or {}
        matched = fields.get(
            "matched_claim_ids",
            duplicate.get("matched_claim_ids", facts.get("matched_claim_ids", [])),
        )
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": policy_id,
            "matched_claim_ids": list(matched)
            if isinstance(matched, list)
            else [str(matched)]
            if matched
            else [],
            "reason": str(fields.get("reason") or facts.get("reason", "possible_duplicate_claim")),
        }

    def _normalize_siu_handoff_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        policy_id = str(fields.get("policy_id") or facts.get("policy_id", ""))
        siu = self._latest_for_policy(self.siu_checks, policy_id) or {}
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": policy_id,
            "reason": str(fields.get("reason") or facts.get("reason", "neutral_siu_review")),
            "indicator_count": int(fields.get("indicator_count", siu.get("indicator_count", 0))),
            "indicator_codes": list(fields.get("indicator_codes", siu.get("indicator_codes", []))),
            "accusation_made": False,
        }

    def _normalize_support_handoff_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(fields.get("policy_id") or facts.get("policy_id", "")),
            "claim_id": str(fields.get("claim_id") or facts.get("claim_id", "")),
            "support_type": str(
                fields.get("support_type")
                or facts.get("support_type", facts.get("expected_outcome_type", "support_handoff"))
            ),
            "reason": str(
                fields.get("reason") or facts.get("reason", "unsupported_or_exception_workflow")
            ),
        }

    def _normalize_policy_term_review_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        policy_id = str(fields.get("policy_id") or facts.get("policy_id", ""))
        term = self._latest_for_policy(self.policy_term_checks, policy_id) or {}
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": policy_id,
            "term_status": str(
                fields.get("term_status")
                or term.get("term_status", facts.get("term_status", "unknown"))
            ),
            "review_source": str(
                fields.get("review_source") or term.get("source", "policy_admin_fixture")
            ),
            "reason": str(
                fields.get("reason") or facts.get("reason", "policy_term_review_required")
            ),
        }

    def _normalize_subrogation_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        party_ids = fields.get("party_ids", facts.get("party_ids", []))
        carrier_ids = fields.get("other_carrier_ids", facts.get("other_carrier_ids", []))
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(fields.get("policy_id") or facts.get("policy_id", "")),
            "claim_id": str(fields.get("claim_id") or facts.get("claim_id", "")),
            "party_ids": list(party_ids)
            if isinstance(party_ids, list)
            else [str(party_ids)]
            if party_ids
            else [],
            "other_carrier_ids": list(carrier_ids)
            if isinstance(carrier_ids, list)
            else [str(carrier_ids)]
            if carrier_ids
            else [],
            "reason": str(
                fields.get("reason") or facts.get("reason", "third_party_or_other_carrier_review")
            ),
        }

    def _normalize_total_loss_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(fields.get("policy_id") or facts.get("policy_id", "")),
            "claim_id": str(fields.get("claim_id") or facts.get("claim_id", "")),
            "vehicle_location": str(
                fields.get("vehicle_location") or facts.get("vehicle_location", "")
            ),
            "estimated_vehicle_value": float(
                fields.get("estimated_vehicle_value", facts.get("estimated_vehicle_value", 0)) or 0
            ),
            "estimated_damage_amount": float(
                fields.get("estimated_damage_amount", facts.get("estimated_damage_amount", 0)) or 0
            ),
            "reason": str(fields.get("reason") or facts.get("reason", "total_loss_review")),
        }

    def _normalize_tow_storage_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(fields.get("policy_id") or facts.get("policy_id", "")),
            "claim_id": str(fields.get("claim_id") or facts.get("claim_id", "")),
            "tow_yard_name": str(fields.get("tow_yard_name") or facts.get("tow_yard_name", "")),
            "vehicle_location": str(
                fields.get("vehicle_location") or facts.get("vehicle_location", "")
            ),
            "storage_started_date": str(
                fields.get("storage_started_date") or facts.get("storage_started_date", "")
            ),
            "reason": str(fields.get("reason") or facts.get("reason", "tow_storage_review")),
        }

    def _normalize_withdrawal_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        prior_claim = self.prior_claims[0] if self.prior_claims else {}
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(
                fields.get("policy_id") or facts.get("policy_id", prior_claim.get("policy_id", ""))
            ),
            "claim_id": str(
                fields.get("claim_id") or facts.get("claim_id", prior_claim.get("claim_id", ""))
            ),
            "reason": str(
                fields.get("reason") or facts.get("reason", "customer_withdrawal_request")
            ),
        }

    def _normalize_claim_lifecycle_fields(
        self, fields: Dict[str, Any], *, default_reason: str
    ) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        prior_claim = self._find_claim(
            str(fields.get("claim_id") or facts.get("claim_id", ""))
        ) or (self.prior_claims[0] if self.prior_claims else {})
        return {
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "policy_id": str(
                fields.get("policy_id") or facts.get("policy_id", prior_claim.get("policy_id", ""))
            ),
            "claim_id": str(
                fields.get("claim_id") or facts.get("claim_id", prior_claim.get("claim_id", ""))
            ),
            "current_status": str(
                fields.get("current_status") or prior_claim.get("status", "unknown")
            ),
            "reason": str(fields.get("reason") or facts.get("reason", default_reason)),
        }

    def _latest_loss_type(self, policy_id: str) -> str:
        item = self._latest_for_policy(self.loss_classifications, policy_id) or {}
        return str(
            item.get(
                "loss_type", self.task.hidden_facts.get("loss_type", "standard_adjuster_review")
            )
        )

    def _find_policy(self, policy_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not policy_id:
            policy_id = str(self.task.hidden_facts.get("policy_id", ""))
        return next(
            (policy for policy in self.policies if policy.get("policy_id") == policy_id), None
        )

    def _find_claim(self, claim_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not claim_id:
            claim_id = str(self.task.hidden_facts.get("claim_id", ""))
        for collection in (self.claim_cases, self.prior_claims):
            match = next((claim for claim in collection if claim.get("claim_id") == claim_id), None)
            if match:
                return match
        return None

    @staticmethod
    def _latest_for_policy(items: List[Dict[str, Any]], policy_id: str) -> Optional[Dict[str, Any]]:
        for item in reversed(items):
            if item.get("policy_id") == policy_id:
                return item
        return None

    @staticmethod
    def _public_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: policy.get(key)
            for key in (
                "policy_id",
                "line",
                "term_status",
                "named_insureds",
                "coverage_summary",
                "deductibles",
                "limits",
            )
            if key in policy
        }
