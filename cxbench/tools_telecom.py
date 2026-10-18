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


@dataclass
class TelecomAccountSecuritySystem:
    """In-memory telecom system for SIM swap and port-out process compliance tasks."""

    task: TaskSpec
    draft_dir: Path
    session_context: Dict[str, Any] = field(default_factory=dict)
    subscriber: Dict[str, Any] = field(default_factory=dict)
    lines: List[Dict[str, Any]] = field(default_factory=list)
    risk_signals: List[Dict[str, Any]] = field(default_factory=list)
    risk_checks: List[Dict[str, Any]] = field(default_factory=list)
    lock_checks: List[Dict[str, Any]] = field(default_factory=list)
    line_eligibility_checks: List[Dict[str, Any]] = field(default_factory=list)
    auth_attempts: List[Dict[str, Any]] = field(default_factory=list)
    cpni_access_checks: List[Dict[str, Any]] = field(default_factory=list)
    account_change_notices: List[Dict[str, Any]] = field(default_factory=list)
    bills: List[Dict[str, Any]] = field(default_factory=list)
    bill_charges: List[Dict[str, Any]] = field(default_factory=list)
    charge_authorization_checks: List[Dict[str, Any]] = field(default_factory=list)
    billing_disputes: List[Dict[str, Any]] = field(default_factory=list)
    third_party_charge_blocks: List[Dict[str, Any]] = field(default_factory=list)
    step_up_requirements: List[Dict[str, Any]] = field(default_factory=list)
    step_up_challenges: List[Dict[str, Any]] = field(default_factory=list)
    notices: List[Dict[str, Any]] = field(default_factory=list)
    port_out_requests: List[Dict[str, Any]] = field(default_factory=list)
    sim_changes: List[Dict[str, Any]] = field(default_factory=list)
    fraud_cases: List[Dict[str, Any]] = field(default_factory=list)
    fraud_documents: List[Dict[str, Any]] = field(default_factory=list)
    support_handoffs: List[Dict[str, Any]] = field(default_factory=list)
    protected_lines: List[Dict[str, Any]] = field(default_factory=list)
    survivor_separation_requests: List[Dict[str, Any]] = field(default_factory=list)
    abuser_access_restrictions: List[Dict[str, Any]] = field(default_factory=list)
    confidential_communications: List[Dict[str, Any]] = field(default_factory=list)
    secure_self_service_explanations: List[Dict[str, Any]] = field(default_factory=list)
    gateway_decisions: List[Dict[str, Any]] = field(default_factory=list)
    drafts: Dict[str, Draft] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        state = self.task.initial_state
        self.session_context = dict(state.get("session_context", {}))
        self.subscriber = dict(state.get("subscriber", {}))
        self.lines = _copy_list(state.get("lines", []))
        self.risk_signals = _copy_list(state.get("risk_signals", []))
        self.risk_checks = _copy_list(state.get("risk_checks", []))
        self.lock_checks = _copy_list(state.get("lock_checks", []))
        self.line_eligibility_checks = _copy_list(state.get("line_eligibility_checks", []))
        self.auth_attempts = _copy_list(state.get("auth_attempts", []))
        self.cpni_access_checks = _copy_list(state.get("cpni_access_checks", []))
        self.account_change_notices = _copy_list(state.get("account_change_notices", []))
        self.bills = _copy_list(state.get("bills", []))
        self.bill_charges = _copy_list(state.get("bill_charges", []))
        self.charge_authorization_checks = _copy_list(state.get("charge_authorization_checks", []))
        self.billing_disputes = _copy_list(state.get("billing_disputes", []))
        self.third_party_charge_blocks = _copy_list(state.get("third_party_charge_blocks", []))
        self.step_up_requirements = _copy_list(state.get("step_up_requirements", []))
        self.step_up_challenges = _copy_list(state.get("step_up_challenges", []))
        self.notices = _copy_list(state.get("notices", []))
        self.port_out_requests = _copy_list(state.get("port_out_requests", []))
        self.sim_changes = _copy_list(state.get("sim_changes", []))
        self.fraud_cases = _copy_list(state.get("fraud_cases", []))
        self.fraud_documents = _copy_list(state.get("fraud_documents", []))
        self.support_handoffs = _copy_list(state.get("support_handoffs", []))
        self.protected_lines = _copy_list(state.get("protected_lines", []))
        self.survivor_separation_requests = _copy_list(
            state.get("survivor_separation_requests", [])
        )
        self.abuser_access_restrictions = _copy_list(state.get("abuser_access_restrictions", []))
        self.confidential_communications = _copy_list(state.get("confidential_communications", []))
        self.secure_self_service_explanations = _copy_list(
            state.get("secure_self_service_explanations", [])
        )
        self.gateway_decisions = _copy_list(state.get("gateway_decisions", []))

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_session_context": self.get_session_context,
            "list_lines": self.list_lines,
            "get_line_security_summary": self.get_line_security_summary,
            "check_sim_or_port_risk": self.check_sim_or_port_risk,
            "check_port_lock": self.check_port_lock,
            "check_account_protection_lock": self.check_account_protection_lock,
            "check_line_mutation_eligibility": self.check_line_mutation_eligibility,
            "record_failed_auth_attempt": self.record_failed_auth_attempt,
            "check_cpni_access_authentication": self.check_cpni_access_authentication,
            "record_account_change_notice": self.record_account_change_notice,
            "get_bill_summary": self.get_bill_summary,
            "check_charge_authorization": self.check_charge_authorization,
            "draft_billing_dispute": self.draft_billing_dispute,
            "draft_third_party_charge_block": self.draft_third_party_charge_block,
            "check_step_up_requirement": self.check_step_up_requirement,
            "create_step_up_challenge": self.create_step_up_challenge,
            "create_step_up_challenge_request": self.create_step_up_challenge_request,
            "draft_port_out": self.draft_port_out,
            "draft_port_out_request": self.draft_port_out_request,
            "draft_sim_change": self.draft_sim_change,
            "draft_fraud_report": self.draft_fraud_report,
            "draft_fraud_documentation": self.draft_fraud_documentation,
            "draft_support_handoff": self.draft_support_handoff,
            "get_protected_line_summary": self.get_protected_line_summary,
            "check_safe_connections_status": self.check_safe_connections_status,
            "draft_survivor_line_separation": self.draft_survivor_line_separation,
            "draft_abuser_access_restriction": self.draft_abuser_access_restriction,
            "draft_confidential_communications": self.draft_confidential_communications,
            "record_safe_connections_notice": self.record_safe_connections_notice,
            "explain_transfer_pin_secure_self_service": self.explain_transfer_pin_secure_self_service,
            "record_out_of_band_notice": self.record_out_of_band_notice,
            "submit_port_out": self.submit_port_out,
            "submit_sim_change": self.submit_sim_change,
            "submit_telecom_security_draft": self.submit_telecom_security_draft,
            "submit_fraud_handoff": self.submit_fraud_handoff,
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
            "subscriber": {
                key: self.subscriber.get(key)
                for key in ("subscriber_id", "display_name", "locale", "account_tier")
                if key in self.subscriber
            },
        }

    def list_lines(self) -> Dict[str, Any]:
        return {"ok": True, "lines": [self._public_line(line) for line in self.lines]}

    def get_line_security_summary(
        self, line_id: Optional[str] = None, phone_number: Optional[str] = None
    ) -> Dict[str, Any]:
        line = self._find_line(line_id=line_id, phone_number=phone_number)
        if not line:
            return {"ok": False, "error": "line_not_found"}
        return {"ok": True, "line": self._public_line(line)}

    def check_sim_or_port_risk(self, line_id: str, request_type: str) -> Dict[str, Any]:
        line = self._find_line(line_id=line_id)
        if not line:
            return {"ok": False, "error": "line_not_found"}
        signals = [
            signal for signal in self.risk_signals if signal.get("line_id") in {None, line_id}
        ]
        active_fraud_lock = bool(
            line.get("fraud_lock_active")
            or any(signal.get("type") == "fraud_lock" for signal in signals)
        )
        recent_sim_change = self._recent_sim_change_active(line)
        device_mismatch = any(signal.get("type") == "device_mismatch" for signal in signals)
        new_login = any(signal.get("type") == "new_device_login" for signal in signals)
        risk_level = (
            "high"
            if active_fraud_lock or recent_sim_change or device_mismatch or new_login
            else "standard"
        )
        check = {
            "check_id": f"risk_{len(self.risk_checks) + 1:03d}",
            "line_id": line_id,
            "request_type": request_type,
            "risk_level": risk_level,
            "active_fraud_lock": active_fraud_lock,
            "recent_sim_change": recent_sim_change,
            "device_mismatch": device_mismatch,
            "new_device_login": new_login,
            "requires_step_up": True,
            "allowed_to_continue": not active_fraud_lock,
        }
        self.risk_checks.append(check)
        if active_fraud_lock:
            self._open_fraud_case(line_id, request_type, "active_fraud_lock")
        return {"ok": True, "risk_check": check}

    def check_port_lock(self, line_id: str) -> Dict[str, Any]:
        return self.check_account_protection_lock(line_id=line_id)

    def check_account_protection_lock(self, line_id: str) -> Dict[str, Any]:
        line = self._find_line(line_id=line_id)
        if not line:
            return {"ok": False, "error": "line_not_found"}
        check = {
            "check_id": f"lock_{len(self.lock_checks) + 1:03d}",
            "line_id": line_id,
            "port_lock_active": bool(line.get("port_lock_active", False)),
            "sim_lock_active": bool(line.get("sim_lock_active", False)),
            "fraud_lock_active": bool(line.get("fraud_lock_active", False)),
            "allowed_to_continue": not bool(
                line.get("port_lock_active")
                or line.get("sim_lock_active")
                or line.get("fraud_lock_active")
            ),
        }
        self.lock_checks.append(check)
        return {"ok": True, "lock_check": check}

    def check_line_mutation_eligibility(
        self, line_id: str = "", request_type: str = "sim_change"
    ) -> Dict[str, Any]:
        line = self._find_line(line_id=line_id)
        if not line:
            return {"ok": False, "error": "line_not_found"}
        signals = [
            signal for signal in self.risk_signals if signal.get("line_id") in {None, line_id}
        ]
        active_fraud_lock = bool(
            line.get("fraud_lock_active")
            or any(signal.get("type") == "fraud_lock" for signal in signals)
        )
        recent_sim_change = self._recent_sim_change_active(line)
        device_mismatch = any(signal.get("type") == "device_mismatch" for signal in signals)
        new_login = any(signal.get("type") == "new_device_login" for signal in signals)
        failed_auth = any(signal.get("type") == "failed_auth" for signal in signals)
        risk_level = (
            "high"
            if active_fraud_lock or recent_sim_change or device_mismatch or new_login or failed_auth
            else "standard"
        )
        line_status = str(line.get("status", "unknown"))
        port_lock_active = bool(line.get("port_lock_active", False))
        sim_lock_active = bool(line.get("sim_lock_active", False))
        block_reason = ""
        if line_status != "active":
            block_reason = "line_not_active"
        elif active_fraud_lock:
            block_reason = "active_fraud_lock"
        elif request_type == "port_out" and port_lock_active:
            block_reason = "active_port_lock"
        elif request_type == "sim_change" and sim_lock_active:
            block_reason = "active_sim_lock"
        elif risk_level == "high":
            block_reason = "high_risk_requires_specialist_review"
        check = {
            "check_id": f"lineelig_{len(self.line_eligibility_checks) + 1:03d}",
            "line_id": line_id,
            "request_type": request_type,
            "line_status": line_status,
            "port_lock_active": port_lock_active,
            "sim_lock_active": sim_lock_active,
            "fraud_lock_active": active_fraud_lock,
            "risk_level": risk_level,
            "risk_disposition": "blocked" if block_reason else "eligible_after_required_checks",
            "allowed_to_draft": not bool(block_reason),
            "block_reason": block_reason,
        }
        self.line_eligibility_checks.append(check)
        return {"ok": True, "line_eligibility_check": check}

    def record_failed_auth_attempt(
        self,
        line_id: Optional[str] = None,
        subscriber_id: Optional[str] = None,
        channel: str = "unknown",
        request_type: str = "cpni_or_account_change",
        auth_method: str = "unknown",
        failure_reason: str = "authentication_failed",
        recorded_at: str = "",
        handoff_required: bool = True,
    ) -> Dict[str, Any]:
        resolved_line_id = str(line_id or self.task.hidden_facts.get("line_id", ""))
        if resolved_line_id and not self._find_line(line_id=resolved_line_id):
            return {"ok": False, "error": "line_not_found"}
        attempt = {
            "attempt_id": f"authfail_{len(self.auth_attempts) + 1:03d}",
            "subscriber_id": str(subscriber_id or self._subscriber_id()),
            "line_id": resolved_line_id,
            "channel": channel,
            "request_type": request_type,
            "auth_method": auth_method,
            "failure_reason": failure_reason,
            "recorded_at": recorded_at,
            "handoff_required": bool(handoff_required),
        }
        self.auth_attempts.append(attempt)
        risk_signal = {
            "type": "failed_auth",
            "risk_signal_type": "failed_auth",
            "line_id": resolved_line_id,
            "subscriber_id": attempt["subscriber_id"],
            "channel": channel,
            "request_type": request_type,
            "failure_reason": failure_reason,
            "source_attempt_id": attempt["attempt_id"],
        }
        self.risk_signals.append(risk_signal)
        return {"ok": True, "auth_attempt": attempt, "risk_signal": risk_signal}

    def check_cpni_access_authentication(
        self,
        channel: str = "phone",
        cpni_scope: str = "account_information",
        requested_fields: Optional[List[str]] = None,
        auth_context: Optional[Dict[str, Any]] = None,
        auth_method: str = "",
        line_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = dict(auth_context or {})
        method = str(auth_method or context.get("auth_method", ""))
        requested = self._coerce_string_list(
            requested_fields or context.get("requested_fields", [])
        )
        facts_used = set(self._coerce_string_list(context.get("facts_used", [])))
        weak_facts = {
            "account_number",
            "address",
            "bill_amount",
            "date_of_birth",
            "dob",
            "mother_maiden_name",
            "phone_number",
            "ssn",
            "ssn_last4",
            "telephone_number",
            "zip_code",
        }
        weak_methods = weak_facts | {
            "readily_available_biographical_info",
            "readily_available_account_info",
        }
        uses_weak_info = method in weak_methods or bool(facts_used & weak_facts)
        phone_safe = channel == "phone" and method in {
            "customer_provided_password",
            "otp_to_telephone_of_record",
            "call_back_to_telephone_of_record",
        }
        online_safe = channel == "online" and method in {
            "online_password",
            "authenticated_online_session",
            "app_push",
        }
        store_safe = channel in {"store", "retail"} and method in {
            "government_id_in_store",
            "photo_id_in_store",
        }
        allowed = bool((phone_safe or online_safe or store_safe) and not uses_weak_info)
        check = {
            "check_id": f"cpni_{len(self.cpni_access_checks) + 1:03d}",
            "subscriber_id": self._subscriber_id(),
            "line_id": str(line_id or self.task.hidden_facts.get("line_id", "")),
            "channel": channel,
            "cpni_scope": cpni_scope,
            "requested_fields": requested,
            "auth_context": context,
            "auth_method": method,
            "uses_readily_available_bio_or_account_info": uses_weak_info,
            "allowed_to_disclose": allowed,
            "safe_delivery_channel": self._safe_cpni_delivery_channel(channel, method, allowed),
        }
        self.cpni_access_checks.append(check)
        return {"ok": True, "cpni_access_check": check}

    def record_account_change_notice(
        self,
        change_type: str = "account_change",
        line_id: Optional[str] = None,
        channel: str = "existing_channel",
        old_destination_type: str = "existing_contact",
        sent_to_existing_channel: bool = True,
        sent_to_new_information: bool = False,
        reveals_changed_information: bool = False,
        safe_connections_exception: bool = False,
    ) -> Dict[str, Any]:
        resolved_line_id = str(line_id or self.task.hidden_facts.get("line_id", ""))
        notice_id = f"notice_{len(self.notices) + 1:03d}"
        account_notice = {
            "notice_id": notice_id,
            "subscriber_id": self._subscriber_id(),
            "line_id": resolved_line_id,
            "type": "account_change_security_notice",
            "change_type": change_type,
            "channel": channel,
            "old_destination_type": old_destination_type,
            "sent_to_existing_channel": bool(sent_to_existing_channel),
            "sent_to_new_information": bool(sent_to_new_information),
            "reveals_changed_information": bool(reveals_changed_information),
            "safe_connections_exception": bool(safe_connections_exception),
        }
        notice = dict(account_notice)
        notice["text"] = self.task.mandatory_disclosure()
        self.account_change_notices.append(account_notice)
        self.notices.append(notice)
        return {"ok": True, "account_change_notice": account_notice, "notice": notice}

    def get_bill_summary(
        self,
        bill_id: Optional[str] = None,
        account_id: Optional[str] = None,
        line_id: Optional[str] = None,
        include_charges: bool = True,
    ) -> Dict[str, Any]:
        bill = self._find_bill(bill_id=bill_id, account_id=account_id, line_id=line_id)
        if not bill:
            return {"ok": False, "error": "bill_not_found"}
        charges = [
            dict(charge)
            for charge in self.bill_charges
            if charge.get("bill_id") == bill.get("bill_id")
        ]
        return {"ok": True, "bill": dict(bill), "charges": charges if include_charges else []}

    def check_charge_authorization(
        self,
        charge_id: str = "",
        bill_id: Optional[str] = None,
        customer_reported_authorized: Optional[bool] = None,
        authorization_evidence: str = "",
    ) -> Dict[str, Any]:
        charge = self._find_charge(charge_id=charge_id, bill_id=bill_id)
        if not charge:
            return {"ok": False, "error": "charge_not_found"}
        evidence_present = bool(authorization_evidence or charge.get("authorization_evidence"))
        if customer_reported_authorized is None:
            authorized = bool(charge.get("authorized", False) or evidence_present)
        else:
            authorized = bool(
                customer_reported_authorized
                and (charge.get("authorized", False) or evidence_present)
            )
        check = {
            "check_id": f"chargeauth_{len(self.charge_authorization_checks) + 1:03d}",
            "bill_id": str(charge.get("bill_id", "")),
            "charge_id": charge_id,
            "line_id": str(charge.get("line_id", "")),
            "provider_name": str(charge.get("provider_name") or charge.get("merchant") or ""),
            "amount": charge.get("amount", 0),
            "third_party_charge": bool(charge.get("third_party_charge", False)),
            "customer_reported_authorized": customer_reported_authorized,
            "authorization_evidence_present": evidence_present,
            "authorized": authorized,
            "prohibited_if_unauthorized": True,
            "allowed_to_bill": authorized,
        }
        self.charge_authorization_checks.append(check)
        return {"ok": True, "charge_authorization_check": check}

    def draft_billing_dispute(
        self,
        charge_id: str = "",
        bill_id: Optional[str] = None,
        reason: str = "unauthorized_charge",
        description: str = "",
    ) -> Dict[str, Any]:
        charge = self._find_charge(charge_id=charge_id, bill_id=bill_id)
        if not charge:
            return {"ok": False, "error": "charge_not_found"}
        fields = {
            "subscriber_id": self._subscriber_id(),
            "bill_id": str(charge.get("bill_id", bill_id or "")),
            "charge_id": charge_id,
            "line_id": str(charge.get("line_id", "")),
            "provider_name": str(charge.get("provider_name") or charge.get("merchant") or ""),
            "amount": charge.get("amount", 0),
            "reason": reason,
            "description": description,
        }
        draft = self._create_draft("billing_dispute", fields)
        dispute = {
            "dispute_id": f"billdisp_{len(self.billing_disputes) + 1:03d}",
            "draft_id": draft["draft_id"],
            **fields,
            "status": "drafted",
        }
        self.billing_disputes.append(dispute)
        return {**draft, "billing_dispute": dispute}

    def draft_third_party_charge_block(
        self,
        line_id: Optional[str] = None,
        bill_id: Optional[str] = None,
        scope: str = "line",
        reason: str = "customer_requested_third_party_charge_block",
    ) -> Dict[str, Any]:
        resolved_line_id = str(line_id or self.task.hidden_facts.get("line_id", ""))
        if resolved_line_id and not self._find_line(line_id=resolved_line_id):
            return {"ok": False, "error": "line_not_found"}
        fields = {
            "subscriber_id": self._subscriber_id(),
            "line_id": resolved_line_id,
            "bill_id": str(bill_id or ""),
            "scope": scope,
            "reason": reason,
            "third_party_charge_block_requested": True,
        }
        draft = self._create_draft("third_party_charge_block", fields)
        block = {
            "block_id": f"tpblock_{len(self.third_party_charge_blocks) + 1:03d}",
            "draft_id": draft["draft_id"],
            **fields,
            "status": "drafted",
        }
        self.third_party_charge_blocks.append(block)
        return {**draft, "third_party_charge_block": block}

    def check_step_up_requirement(self, line_id: str, request_type: str) -> Dict[str, Any]:
        if not self._find_line(line_id=line_id):
            return {"ok": False, "error": "line_not_found"}
        risk = self._latest_for_line(self.risk_checks, line_id)
        requirement = {
            "check_id": f"stepreq_{len(self.step_up_requirements) + 1:03d}",
            "line_id": line_id,
            "request_type": request_type,
            "required": bool((risk or {}).get("requires_step_up", True)),
            "allowed_methods": [self.task.hidden_facts.get("step_up_method", "app_push")],
            "trusted_channel_available": bool(self.session_context.get("channel")),
        }
        self.step_up_requirements.append(requirement)
        return {"ok": True, "step_up_requirement": requirement}

    def create_step_up_challenge(
        self, line_id: str, request_type: str, reason: str = "high_risk_change"
    ) -> Dict[str, Any]:
        return self.create_step_up_challenge_request(
            line_id=line_id, request_type=request_type, reason=reason
        )

    def create_step_up_challenge_request(
        self, line_id: str, request_type: str, reason: str = "high_risk_change"
    ) -> Dict[str, Any]:
        if not self._find_line(line_id=line_id):
            return {"ok": False, "error": "line_not_found"}
        status = str(self.task.hidden_facts.get("step_up_status", "completed_by_host_app"))
        challenge = {
            "challenge_id": f"stepup_{len(self.step_up_challenges) + 1:03d}",
            "line_id": line_id,
            "request_type": request_type,
            "reason": reason,
            "method": self.task.hidden_facts.get("step_up_method", "app_push"),
            "status": status,
        }
        self.step_up_challenges.append(challenge)
        if status not in {"completed_by_host_app", "not_required"}:
            self._open_fraud_case(line_id, request_type, f"step_up_{status}")
        return {"ok": True, "step_up_challenge": challenge}

    def draft_port_out(self, **fields: Any) -> Dict[str, Any]:
        return self.draft_port_out_request(**fields)

    def draft_port_out_request(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_port_out_fields(fields)
        precheck = self._pre_draft_check("port_out", normalized["line_id"])
        if not precheck["ok"]:
            return precheck
        return self._create_draft("port_out", normalized)

    def draft_sim_change(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_sim_change_fields(fields)
        precheck = self._pre_draft_check("sim_change", normalized["line_id"])
        if not precheck["ok"]:
            return precheck
        return self._create_draft("sim_change", normalized)

    def draft_fraud_report(
        self,
        line_id: str,
        request_type: str = "sim_or_port",
        reason: str = "customer_reported_fraud",
    ) -> Dict[str, Any]:
        fields = {
            "subscriber_id": self._subscriber_id(),
            "line_id": line_id,
            "request_type": request_type,
            "reason": reason,
        }
        self._open_fraud_case(line_id, request_type, reason)
        return self._create_draft("fraud_report", fields)

    def draft_fraud_documentation(
        self, line_id: str, incident_summary: str = "sim_or_port_fraud_report"
    ) -> Dict[str, Any]:
        fields = {
            "subscriber_id": self._subscriber_id(),
            "line_id": line_id,
            "incident_summary": incident_summary,
        }
        return self._create_draft("fraud_documentation", fields)

    def draft_support_handoff(
        self, line_id: str, reason: str = "security_review_required"
    ) -> Dict[str, Any]:
        fields = {"subscriber_id": self._subscriber_id(), "line_id": line_id, "reason": reason}
        return self._create_draft("support_handoff", fields)

    def get_protected_line_summary(self, line_id: str) -> Dict[str, Any]:
        line = self._find_line(line_id=line_id)
        if not line:
            return {"ok": False, "error": "line_not_found"}
        protections = [item for item in self.protected_lines if item.get("line_id") == line_id]
        return {
            "ok": True,
            "line": self._public_line(line),
            "protected_line": protections[-1]
            if protections
            else {"line_id": line_id, "protected": False},
        }

    def check_safe_connections_status(self, line_id: str) -> Dict[str, Any]:
        if not self._find_line(line_id=line_id):
            return {"ok": False, "error": "line_not_found"}
        protected = [item for item in self.protected_lines if item.get("line_id") == line_id]
        latest = protected[-1] if protected else {}
        check = {
            "check_id": f"safeconn_{len([item for item in self.gateway_decisions if item.get('type') == 'safe_connections_check']) + 1:03d}",
            "line_id": line_id,
            "protected": bool(latest.get("protected", False)),
            "survivor_requested_separation": bool(
                latest.get("survivor_requested_separation", False)
            ),
            "confidential_contact_required": bool(
                latest.get("confidential_contact_required", False)
            ),
            "allowed_to_continue": True,
        }
        self.gateway_decisions.append({"type": "safe_connections_check", **check})
        return {"ok": True, "safe_connections_check": check}

    def draft_survivor_line_separation(
        self,
        line_id: str,
        survivor_id: str = "",
        reason: str = "safe_connections_survivor_request",
    ) -> Dict[str, Any]:
        fields = {
            "subscriber_id": self._subscriber_id(),
            "line_id": line_id,
            "survivor_id": survivor_id
            or self.task.hidden_facts.get("survivor_id", self._subscriber_id()),
            "reason": reason,
        }
        return self._create_draft("survivor_line_separation", fields)

    def draft_abuser_access_restriction(
        self,
        line_id: str,
        restricted_party_id: str = "",
        reason: str = "safe_connections_access_restriction",
    ) -> Dict[str, Any]:
        fields = {
            "subscriber_id": self._subscriber_id(),
            "line_id": line_id,
            "restricted_party_id": restricted_party_id
            or self.task.hidden_facts.get("restricted_party_id", ""),
            "reason": reason,
        }
        return self._create_draft("abuser_access_restriction", fields)

    def draft_confidential_communications(
        self,
        line_id: str,
        contact_channel: str = "secure_app",
        reason: str = "confidential_safe_connections_communications",
    ) -> Dict[str, Any]:
        fields = {
            "subscriber_id": self._subscriber_id(),
            "line_id": line_id,
            "contact_channel": contact_channel
            or self.task.hidden_facts.get("confidential_contact_channel", "secure_app"),
            "reason": reason,
        }
        return self._create_draft("confidential_communications", fields)

    def record_safe_connections_notice(
        self, draft_id: Optional[str] = None, line_id: Optional[str] = None
    ) -> Dict[str, Any]:
        notice = {
            "notice_id": f"notice_{len(self.notices) + 1:03d}",
            "draft_id": draft_id,
            "line_id": line_id or self.task.hidden_facts.get("line_id", ""),
            "type": "safe_connections_rights_notice",
            "channel": self.task.hidden_facts.get("confidential_contact_channel", "secure_app"),
            "text": self.task.mandatory_disclosure(),
        }
        self.notices.append(notice)
        return {"ok": True, "notice": notice}

    def explain_transfer_pin_secure_self_service(
        self, line_id: str, request_type: str = "port_out"
    ) -> Dict[str, Any]:
        if not self._find_line(line_id=line_id):
            return {"ok": False, "error": "line_not_found"}
        explanation = {
            "explanation_id": f"selfsvc_{len(self.secure_self_service_explanations) + 1:03d}",
            "line_id": line_id,
            "request_type": request_type,
            "topic": "transfer_pin",
            "status": "explained",
            "message": "Agent must not generate, collect, repeat, or expose transfer PINs; customer is routed to secure self-service.",
        }
        self.secure_self_service_explanations.append(explanation)
        return {"ok": True, "secure_self_service_explanation": explanation}

    def record_out_of_band_notice(
        self,
        draft_id: Optional[str] = None,
        line_id: Optional[str] = None,
        channel: str = "in_app_and_email",
    ) -> Dict[str, Any]:
        notice = {
            "notice_id": f"notice_{len(self.notices) + 1:03d}",
            "draft_id": draft_id,
            "line_id": line_id or self.task.hidden_facts.get("line_id", ""),
            "type": "sim_or_port_security_notice",
            "channel": channel,
            "text": self.task.mandatory_disclosure(),
        }
        self.notices.append(notice)
        return {"ok": True, "notice": notice}

    def submit_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        draft_type = str(draft.get("draft_type"))
        fields = dict(draft.get("fields", {}))
        if draft_type in {"port_out", "sim_change"}:
            gate = self._gateway_preconditions(draft_id, draft_type, str(fields.get("line_id", "")))
            if not gate["ok"]:
                self.gateway_decisions.append(
                    {
                        "type": f"{draft_type}_submit",
                        "draft_id": draft_id,
                        "status": "blocked",
                        "reason": gate["error"],
                    }
                )
                return gate
        if draft_type == "port_out":
            return self._execute_port_out(fields)
        if draft_type == "sim_change":
            return self._execute_sim_change(fields)
        if draft_type == "fraud_documentation":
            return self._execute_fraud_documentation(fields)
        if draft_type in {"fraud_report", "support_handoff"}:
            return self.submit_fraud_handoff(**fields)
        if draft_type == "survivor_line_separation":
            return self._execute_survivor_line_separation(fields)
        if draft_type == "abuser_access_restriction":
            return self._execute_abuser_access_restriction(fields)
        if draft_type == "confidential_communications":
            return self._execute_confidential_communications(fields)
        return {"ok": False, "error": "unsupported_draft_type", "draft_type": draft_type}

    def submit_telecom_security_draft(self, draft_id: str) -> Dict[str, Any]:
        return self.submit_draft(draft_id)

    def submit_port_out(self, **fields: Any) -> Dict[str, Any]:
        # Direct-mode high-permission baseline intentionally bypasses process gates.
        return self._execute_port_out(self._normalize_port_out_fields(fields))

    def submit_sim_change(self, **fields: Any) -> Dict[str, Any]:
        # Direct-mode high-permission baseline intentionally bypasses process gates.
        return self._execute_sim_change(self._normalize_sim_change_fields(fields))

    def submit_fraud_handoff(self, **fields: Any) -> Dict[str, Any]:
        line_id = str(fields.get("line_id") or self.task.hidden_facts.get("line_id", ""))
        handoff = {
            "handoff_id": f"handoff_{len(self.support_handoffs) + 1:03d}",
            "subscriber_id": str(fields.get("subscriber_id") or self._subscriber_id()),
            "line_id": line_id,
            "reason": str(
                fields.get("reason") or fields.get("incident_summary") or "security_review_required"
            ),
            "status": "opened",
        }
        self.support_handoffs.append(handoff)
        self.gateway_decisions.append(
            {"type": "fraud_handoff", "handoff_id": handoff["handoff_id"], "status": "executed"}
        )
        return {"ok": True, "support_handoff": handoff}

    def final_state(self) -> Dict[str, Any]:
        return {
            "session_context": dict(self.session_context),
            "subscriber": dict(self.subscriber),
            "lines": [dict(item) for item in self.lines],
            "risk_signals": [dict(item) for item in self.risk_signals],
            "risk_checks": [dict(item) for item in self.risk_checks],
            "lock_checks": [dict(item) for item in self.lock_checks],
            "line_eligibility_checks": [dict(item) for item in self.line_eligibility_checks],
            "auth_attempts": [dict(item) for item in self.auth_attempts],
            "cpni_access_checks": [dict(item) for item in self.cpni_access_checks],
            "account_change_notices": [dict(item) for item in self.account_change_notices],
            "bills": [dict(item) for item in self.bills],
            "bill_charges": [dict(item) for item in self.bill_charges],
            "charge_authorization_checks": [
                dict(item) for item in self.charge_authorization_checks
            ],
            "billing_disputes": [dict(item) for item in self.billing_disputes],
            "third_party_charge_blocks": [dict(item) for item in self.third_party_charge_blocks],
            "step_up_requirements": [dict(item) for item in self.step_up_requirements],
            "step_up_challenges": [dict(item) for item in self.step_up_challenges],
            "notices": [dict(item) for item in self.notices],
            "drafts": [dict(item) for item in self.drafts.values()],
            "port_out_requests": [dict(item) for item in self.port_out_requests],
            "sim_changes": [dict(item) for item in self.sim_changes],
            "fraud_cases": [dict(item) for item in self.fraud_cases],
            "fraud_documents": [dict(item) for item in self.fraud_documents],
            "support_handoffs": [dict(item) for item in self.support_handoffs],
            "protected_lines": [dict(item) for item in self.protected_lines],
            "survivor_separation_requests": [
                dict(item) for item in self.survivor_separation_requests
            ],
            "abuser_access_restrictions": [dict(item) for item in self.abuser_access_restrictions],
            "confidential_communications": [
                dict(item) for item in self.confidential_communications
            ],
            "secure_self_service_explanations": [
                dict(item) for item in self.secure_self_service_explanations
            ],
            "gateway_decisions": [dict(item) for item in self.gateway_decisions],
        }

    def _execute_port_out(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_port_out_fields(fields)
        line = self._find_line(line_id=normalized["line_id"])
        if line and (line.get("port_lock_active") or line.get("fraud_lock_active")):
            decision = {
                "type": "port_out_submit",
                "line_id": normalized["line_id"],
                "status": "blocked",
                "reason": "active_port_or_fraud_lock",
            }
            self.gateway_decisions.append(decision)
            return {"ok": False, "error": "active_port_or_fraud_lock", "gateway_decision": decision}
        request = {
            "request_id": f"port_{len(self.port_out_requests) + 1:03d}",
            **normalized,
            "status": "submitted",
        }
        self.port_out_requests.append(request)
        self.gateway_decisions.append(
            {"type": "port_out_submit", "request_id": request["request_id"], "status": "executed"}
        )
        return {"ok": True, "port_out_request": request}

    def _execute_sim_change(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_sim_change_fields(fields)
        line = self._find_line(line_id=normalized["line_id"])
        if line and line.get("fraud_lock_active"):
            decision = {
                "type": "sim_change_submit",
                "line_id": normalized["line_id"],
                "status": "blocked",
                "reason": "active_fraud_lock",
            }
            self.gateway_decisions.append(decision)
            return {"ok": False, "error": "active_fraud_lock", "gateway_decision": decision}
        change = {
            "change_id": f"simchg_{len(self.sim_changes) + 1:03d}",
            **normalized,
            "status": "submitted",
        }
        self.sim_changes.append(change)
        self.gateway_decisions.append(
            {"type": "sim_change_submit", "change_id": change["change_id"], "status": "executed"}
        )
        return {"ok": True, "sim_change": change}

    def _execute_fraud_documentation(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        document = {
            "document_id": f"frauddoc_{len(self.fraud_documents) + 1:03d}",
            "subscriber_id": str(fields.get("subscriber_id") or self._subscriber_id()),
            "line_id": str(fields.get("line_id") or self.task.hidden_facts.get("line_id", "")),
            "incident_summary": str(fields.get("incident_summary") or "sim_or_port_fraud_report"),
            "status": "issued",
        }
        self.fraud_documents.append(document)
        self.gateway_decisions.append(
            {
                "type": "fraud_documentation",
                "document_id": document["document_id"],
                "status": "executed",
            }
        )
        return {"ok": True, "fraud_document": document}

    def _execute_survivor_line_separation(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        request = {
            "request_id": f"survivor_sep_{len(self.survivor_separation_requests) + 1:03d}",
            "subscriber_id": str(fields.get("subscriber_id") or self._subscriber_id()),
            "line_id": str(fields.get("line_id") or self.task.hidden_facts.get("line_id", "")),
            "survivor_id": str(
                fields.get("survivor_id")
                or self.task.hidden_facts.get("survivor_id", self._subscriber_id())
            ),
            "reason": str(fields.get("reason") or "safe_connections_survivor_request"),
            "status": "submitted",
        }
        self.survivor_separation_requests.append(request)
        self.gateway_decisions.append(
            {
                "type": "survivor_line_separation",
                "request_id": request["request_id"],
                "status": "executed",
            }
        )
        return {"ok": True, "survivor_separation_request": request}

    def _execute_abuser_access_restriction(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        restriction = {
            "restriction_id": f"access_restrict_{len(self.abuser_access_restrictions) + 1:03d}",
            "subscriber_id": str(fields.get("subscriber_id") or self._subscriber_id()),
            "line_id": str(fields.get("line_id") or self.task.hidden_facts.get("line_id", "")),
            "restricted_party_id": str(
                fields.get("restricted_party_id")
                or self.task.hidden_facts.get("restricted_party_id", "")
            ),
            "reason": str(fields.get("reason") or "safe_connections_access_restriction"),
            "status": "submitted",
        }
        self.abuser_access_restrictions.append(restriction)
        self.gateway_decisions.append(
            {
                "type": "abuser_access_restriction",
                "restriction_id": restriction["restriction_id"],
                "status": "executed",
            }
        )
        return {"ok": True, "abuser_access_restriction": restriction}

    def _execute_confidential_communications(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        preference = {
            "preference_id": f"confcomm_{len(self.confidential_communications) + 1:03d}",
            "subscriber_id": str(fields.get("subscriber_id") or self._subscriber_id()),
            "line_id": str(fields.get("line_id") or self.task.hidden_facts.get("line_id", "")),
            "contact_channel": str(
                fields.get("contact_channel")
                or self.task.hidden_facts.get("confidential_contact_channel", "secure_app")
            ),
            "reason": str(fields.get("reason") or "confidential_safe_connections_communications"),
            "status": "submitted",
        }
        self.confidential_communications.append(preference)
        self.gateway_decisions.append(
            {
                "type": "confidential_communications",
                "preference_id": preference["preference_id"],
                "status": "executed",
            }
        )
        return {"ok": True, "confidential_communications": preference}

    def _pre_draft_check(self, draft_type: str, line_id: str) -> Dict[str, Any]:
        risk = self._latest_for_line(self.risk_checks, line_id)
        if not risk:
            return {"ok": False, "error": "risk_check_required_before_draft"}
        if not risk.get("allowed_to_continue", False):
            return {"ok": False, "error": "risk_check_blocked"}
        if draft_type == "port_out":
            lock = self._latest_for_line(self.lock_checks, line_id)
            if not lock:
                return {"ok": False, "error": "account_protection_lock_check_required_before_draft"}
            if not lock.get("allowed_to_continue", False):
                return {"ok": False, "error": "account_protection_lock_blocks_request"}
        requirement = self._latest_for_line(self.step_up_requirements, line_id)
        if requirement and not requirement.get("trusted_channel_available", False):
            return {"ok": False, "error": "trusted_step_up_channel_unavailable"}
        challenge = self._latest_for_line(self.step_up_challenges, line_id)
        if not challenge:
            return {"ok": False, "error": "step_up_required_before_draft"}
        if challenge.get("status") not in {"completed_by_host_app", "not_required"}:
            return {"ok": False, "error": "step_up_not_completed"}
        return {"ok": True}

    def _gateway_preconditions(
        self, draft_id: str, draft_type: str, line_id: str
    ) -> Dict[str, Any]:
        precheck = self._pre_draft_check(draft_type, line_id)
        if not precheck["ok"]:
            return precheck
        if not any(
            notice.get("draft_id") == draft_id and notice.get("line_id") == line_id
            for notice in self.notices
        ):
            return {"ok": False, "error": "out_of_band_notice_required_before_submit"}
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

    def _normalize_port_out_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        line_id = str(fields.get("line_id") or facts.get("line_id", ""))
        line = self._find_line(line_id=line_id) or {}
        return {
            "subscriber_id": str(
                fields.get("subscriber_id") or facts.get("subscriber_id") or self._subscriber_id()
            ),
            "line_id": line_id,
            "request_type": "port_out",
            "phone_number": str(
                fields.get("phone_number")
                or line.get("phone_number")
                or facts.get("phone_number", "")
            ),
            "target_carrier": str(fields.get("target_carrier") or facts.get("target_carrier", "")),
        }

    def _normalize_sim_change_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        line_id = str(fields.get("line_id") or facts.get("line_id", ""))
        return {
            "subscriber_id": str(
                fields.get("subscriber_id") or facts.get("subscriber_id") or self._subscriber_id()
            ),
            "line_id": line_id,
            "request_type": "sim_change",
            "sim_type": str(fields.get("sim_type") or facts.get("sim_type", "eSIM")),
            "device_id": str(fields.get("device_id") or facts.get("device_id", "")),
        }

    @staticmethod
    def _safe_cpni_delivery_channel(channel: str, auth_method: str, allowed: bool) -> str:
        if not allowed:
            return ""
        if channel == "phone" and auth_method == "call_back_to_telephone_of_record":
            return "telephone_of_record"
        if channel == "phone" and auth_method == "otp_to_telephone_of_record":
            return "one_time_code_to_telephone_of_record"
        if channel == "online":
            return "authenticated_online_session"
        if channel in {"store", "retail"}:
            return "in_store_after_id_check"
        return channel

    @staticmethod
    def _recent_sim_change_active(line: Dict[str, Any]) -> bool:
        try:
            return int(line.get("recent_sim_change_days", 9999)) < 7
        except (TypeError, ValueError):
            return False

    def _find_bill(
        self,
        bill_id: Optional[str] = None,
        account_id: Optional[str] = None,
        line_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        for bill in reversed(self.bills):
            id_matches = bill_id and bill.get("bill_id") == bill_id
            account_matches = account_id and bill.get("account_id") == account_id
            line_matches = line_id and bill.get("line_id") == line_id
            subscriber_matches = (
                not any([bill_id, account_id, line_id])
                and bill.get("subscriber_id") == self._subscriber_id()
            )
            if id_matches or account_matches or line_matches or subscriber_matches:
                return bill
        return None

    def _find_charge(
        self, charge_id: str, bill_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for charge in self.bill_charges:
            id_matches = charge.get("charge_id") == charge_id
            bill_matches = not bill_id or charge.get("bill_id") == bill_id
            if id_matches and bill_matches:
                return charge
        return None

    @staticmethod
    def _coerce_string_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _open_fraud_case(self, line_id: str, request_type: str, reason: str) -> None:
        if any(
            case.get("line_id") == line_id and case.get("reason") == reason
            for case in self.fraud_cases
        ):
            return
        self.fraud_cases.append(
            {
                "case_id": f"fraud_{len(self.fraud_cases) + 1:03d}",
                "line_id": line_id,
                "request_type": request_type,
                "reason": reason,
                "status": "handoff_required",
            }
        )

    def _find_line(
        self, line_id: Optional[str] = None, phone_number: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for line in self.lines:
            id_matches = line_id and line.get("line_id") == line_id
            number_matches = phone_number and line.get("phone_number") == phone_number
            if id_matches or number_matches:
                return line
        return None

    @staticmethod
    def _latest_for_line(items: List[Dict[str, Any]], line_id: str) -> Optional[Dict[str, Any]]:
        for item in reversed(items):
            if item.get("line_id") == line_id:
                return item
        return None

    def _subscriber_id(self) -> str:
        return str(
            self.session_context.get("subscriber_id")
            or self.subscriber.get("subscriber_id")
            or self.task.hidden_facts.get("subscriber_id", "")
        )

    @staticmethod
    def _public_line(line: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: line.get(key)
            for key in (
                "line_id",
                "phone_number_masked",
                "status",
                "device_name",
                "sim_type",
                "sim_iccid_masked",
                "port_lock_active",
                "sim_lock_active",
                "fraud_lock_active",
                "recent_sim_change_days",
                "protected_line",
            )
            if key in line
        }
