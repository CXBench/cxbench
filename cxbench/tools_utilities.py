from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from datetime import date
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
class UtilitiesCustomerProtectionSystem:
    """In-memory utility customer protection system for disconnection tasks."""

    task: TaskSpec
    draft_dir: Path
    session_context: Dict[str, Any] = field(default_factory=dict)
    customer: Dict[str, Any] = field(default_factory=dict)
    accounts: List[Dict[str, Any]] = field(default_factory=list)
    bill_ledgers: List[Dict[str, Any]] = field(default_factory=list)
    disconnection_notices: List[Dict[str, Any]] = field(default_factory=list)
    protection_profiles: List[Dict[str, Any]] = field(default_factory=list)
    weather_safety: Dict[str, Any] = field(default_factory=dict)
    payment_options: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    notice_checks: List[Dict[str, Any]] = field(default_factory=list)
    payment_plan_checks: List[Dict[str, Any]] = field(default_factory=list)
    deferred_payment_plan_drafts: List[Dict[str, Any]] = field(default_factory=list)
    switch_holds: List[Dict[str, Any]] = field(default_factory=list)
    prior_payment_arrangements: List[Dict[str, Any]] = field(default_factory=list)
    arrangement_default_checks: List[Dict[str, Any]] = field(default_factory=list)
    bill_charge_checks: List[Dict[str, Any]] = field(default_factory=list)
    bill_investigations: List[Dict[str, Any]] = field(default_factory=list)
    critical_care_profiles: List[Dict[str, Any]] = field(default_factory=list)
    chronic_condition_profiles: List[Dict[str, Any]] = field(default_factory=list)
    medical_documentation: List[Dict[str, Any]] = field(default_factory=list)
    emergency_contacts: List[Dict[str, Any]] = field(default_factory=list)
    assistance_disclosures: List[Dict[str, Any]] = field(default_factory=list)
    energy_assistance_pledges: List[Dict[str, Any]] = field(default_factory=list)
    disclosures: List[Dict[str, Any]] = field(default_factory=list)
    drafts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    payment_arrangements: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_holds: List[Dict[str, Any]] = field(default_factory=list)
    bill_disputes: List[Dict[str, Any]] = field(default_factory=list)
    outbox: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        state = self.task.initial_state
        self.session_context = dict(state.get("session_context", {}))
        self.customer = dict(state.get("customer", {}))
        self.accounts = _copy_list(state.get("utility_accounts", state.get("accounts", [])))
        self.bill_ledgers = _copy_list(state.get("bill_ledgers", state.get("bills", [])))
        self.disconnection_notices = _copy_list(state.get("disconnection_notices", []))
        self.protection_profiles = _copy_list(
            state.get("protection_profiles", state.get("protection_flags", []))
        )
        self.weather_safety = dict(state.get("weather_safety", {}))
        self.payment_options = _copy_list(state.get("payment_options", []))
        self.checks = _copy_list(state.get("checks", []))
        self.notice_checks = _copy_list(state.get("notice_checks", []))
        self.payment_plan_checks = _copy_list(state.get("payment_plan_checks", []))
        self.deferred_payment_plan_drafts = _copy_list(
            state.get("deferred_payment_plan_drafts", [])
        )
        self.switch_holds = _copy_list(state.get("switch_holds", []))
        self.prior_payment_arrangements = _copy_list(state.get("prior_payment_arrangements", []))
        self.arrangement_default_checks = _copy_list(state.get("arrangement_default_checks", []))
        self.bill_charge_checks = _copy_list(state.get("bill_charge_checks", []))
        self.bill_investigations = _copy_list(state.get("bill_investigations", []))
        self.critical_care_profiles = _copy_list(state.get("critical_care_profiles", []))
        self.chronic_condition_profiles = _copy_list(state.get("chronic_condition_profiles", []))
        self.medical_documentation = _copy_list(state.get("medical_documentation", []))
        self.emergency_contacts = _copy_list(state.get("emergency_contacts", []))
        self.assistance_disclosures = _copy_list(state.get("assistance_disclosures", []))
        self.energy_assistance_pledges = _copy_list(state.get("energy_assistance_pledges", []))
        self.disclosures = _copy_list(state.get("disclosures", []))
        self.payment_arrangements = _copy_list(state.get("payment_arrangements", []))
        self.regulatory_holds = _copy_list(state.get("regulatory_holds", []))
        self.bill_disputes = _copy_list(state.get("bill_disputes", []))
        self.outbox = _copy_list(state.get("outbox", []))

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_session_context": self.get_session_context,
            "get_account_and_arrears": self.get_account_and_arrears,
            "get_bill_status": self.get_account_and_arrears,
            "lookup_disconnection_notice": self.lookup_disconnection_notice,
            "check_notice_validity": self.check_notice_validity,
            "check_notice_timeline": self.check_notice_validity,
            "check_customer_protections": self.check_customer_protections,
            "check_disconnection_protections": self.check_customer_protections,
            "check_weather_and_safety_hold": self.check_weather_and_safety_hold,
            "check_payment_plan_and_AMP_options": self.check_payment_plan_and_amp_options,
            "check_payment_plan_options": self.check_payment_plan_and_amp_options,
            "draft_payment_arrangement": self.draft_payment_arrangement,
            "draft_medical_baseline_or_vulnerable_hold": self.draft_medical_baseline_or_vulnerable_hold,
            "draft_medical_protection_request": self.draft_medical_baseline_or_vulnerable_hold,
            "draft_bill_dispute_or_puc_handoff": self.draft_bill_dispute_or_puc_handoff,
            "draft_bill_dispute": self.draft_bill_dispute_or_puc_handoff,
            "record_disconnection_rights_disclosure": self.record_disconnection_rights_disclosure,
            "record_customer_rights_disclosure": self.record_disconnection_rights_disclosure,
            "submit_arrangement_or_hold": self.submit_arrangement_or_hold,
            "check_texas_disconnection_notice_requirements": self.check_texas_disconnection_notice_requirements,
            "check_texas_payment_plan_eligibility": self.check_texas_payment_plan_eligibility,
            "draft_texas_deferred_payment_plan": self.draft_texas_deferred_payment_plan,
            "read_prior_payment_arrangements": self.read_prior_payment_arrangements,
            "check_arrangement_default_reinstatement": self.check_arrangement_default_reinstatement,
            "check_bill_charge_disconnectability": self.check_bill_charge_disconnectability,
            "draft_bill_investigation": self.draft_bill_investigation,
            "check_critical_care_or_chronic_condition_requirements": self.check_critical_care_or_chronic_condition_requirements,
            "record_assistance_program_disclosure": self.record_assistance_program_disclosure,
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

    def get_account_and_arrears(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        account = self._find_account(account_id)
        if not account:
            return {"ok": False, "error": "account_not_found"}
        ledger = self._find_ledger(account["account_id"])
        return {"ok": True, "account": dict(account), "arrears": dict(ledger or {})}

    def lookup_disconnection_notice(
        self, account_id: Optional[str] = None, notice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        notice = self._find_notice(account_id=account_id, notice_id=notice_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        return {"ok": True, "notice": dict(notice)}

    def check_notice_validity(
        self, notice_id: str, account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        notice = self._find_notice(notice_id=notice_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        defects = list(notice.get("defects", []))
        minimum_days = int(self.task.policy.get("minimum_notice_days", 15))
        valid = not defects and (
            bool(notice.get("earliest_disconnection_date"))
            or int(notice.get("days_before_scheduled_disconnect", 0)) >= minimum_days
        )
        result = {
            "check_id": f"check_{len(self.checks) + 1:03d}",
            "type": "notice_validity",
            "notice_id": notice_id,
            "valid": valid,
            "defects": defects,
        }
        self.checks.append(result)
        return {"ok": True, "notice_validity": result}

    def check_customer_protections(
        self, account_id: str, notice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        profile = self._find_protection_profile(account_id) or {}
        if "protection_type" in profile:
            active_flags = (
                [str(profile.get("protection_type"))]
                if profile.get("status") in {"eligible", "active", "pending_documentation"}
                else []
            )
        else:
            active_flags = [
                key
                for key in (
                    "medical_baseline",
                    "care",
                    "fera",
                    "vulnerable_household",
                    "active_bill_dispute",
                )
                if bool(profile.get(key))
            ]
        result = {
            "check_id": f"check_{len(self.checks) + 1:03d}",
            "type": "customer_protections",
            "account_id": account_id,
            "active_flags": active_flags,
            "ordinary_disconnection_allowed": not active_flags,
        }
        self.checks.append(result)
        return {"ok": True, "customer_protections": result}

    def check_weather_and_safety_hold(self, account_id: str) -> Dict[str, Any]:
        holds = [
            key
            for key in ("temperature_alert", "wildfire_psps", "state_emergency")
            if self.weather_safety.get(key)
        ]
        result = {
            "check_id": f"check_{len(self.checks) + 1:03d}",
            "type": "weather_safety",
            "account_id": account_id,
            "hold_flags": holds,
            "ordinary_disconnection_allowed": not holds,
        }
        self.checks.append(result)
        return {"ok": True, "weather_safety": result}

    def check_payment_plan_and_amp_options(self, account_id: str) -> Dict[str, Any]:
        options = [
            option for option in self.payment_options if option.get("account_id") == account_id
        ]
        result = {
            "check_id": f"check_{len(self.checks) + 1:03d}",
            "type": "payment_options",
            "account_id": account_id,
            "options": [dict(option) for option in options],
        }
        self.checks.append(result)
        return {"ok": True, "payment_options": result}

    def check_texas_disconnection_notice_requirements(
        self, notice_id: Optional[str] = None, account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        notice = self._find_notice(account_id=account_id, notice_id=notice_id)
        if not notice:
            return {"ok": False, "error": "notice_not_found"}
        account_id = str(account_id or notice.get("account_id", ""))
        ledger = self._find_ledger(account_id) or {}
        defects: List[str] = []
        issued_date = _parse_date(notice.get("issued_date"))
        due_date = _parse_date(notice.get("bill_due_date") or ledger.get("due_date"))
        scheduled_date = _parse_date(notice.get("scheduled_disconnect_date"))
        if issued_date and due_date and issued_date <= due_date:
            defects.append("notice_not_issued_after_bill_due_date")
        if issued_date and scheduled_date and (scheduled_date - issued_date).days < 10:
            defects.append("scheduled_disconnect_less_than_10_days_after_notice")
        if notice.get("is_separate_notice") is False:
            defects.append("notice_not_separate")
        channel = str(notice.get("channel", "")).lower()
        email_subject = str(notice.get("email_subject", "")).lower()
        if channel == "email" and "disconnect" not in email_subject:
            defects.append("email_subject_does_not_identify_disconnection")
        if notice.get("personnel_available_on_disconnect_date") is False:
            defects.append("personnel_unavailable_on_disconnect_date")

        content_items = {str(item) for item in notice.get("content_items", []) if item is not None}
        for item in ("reason", "cure_actions", "fees", "overdue_amount", "puc_complaint_contact"):
            if item not in content_items:
                defects.append(f"missing_content_{item}")

        charge_basis = self._charge_basis(notice, ledger)
        barred_reasons = self._texas_disconnect_barred_reasons(notice, ledger)
        defects.extend(barred_reasons)
        result = {
            "check_id": self._next_check_id(),
            "type": "texas_disconnection_notice_requirements",
            "notice_id": notice_id or notice.get("notice_id"),
            "account_id": account_id,
            "valid": not defects,
            "defects": defects,
            "minimum_notice_days": 10,
            "charge_basis": charge_basis,
        }
        self.checks.append(result)
        self.notice_checks.append(dict(result))
        return {"ok": True, "texas_notice_requirements": result}

    def check_texas_payment_plan_eligibility(
        self, account_id: Optional[str] = None, notice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        account_id = str(account_id or self.task.hidden_facts.get("account_id", ""))
        if not account_id:
            return {"ok": False, "error": "account_not_found"}
        account = self._find_account(account_id) or {}
        ledger = self._find_ledger(account_id) or {}
        profile = self._find_protection_profile(account_id) or {}
        options = [
            option for option in self.payment_options if option.get("account_id") == account_id
        ]
        existing_plan_statuses = {
            str(plan.get("status"))
            for plan in [*self.payment_arrangements, *self.prior_payment_arrangements]
            if plan.get("account_id") == account_id and plan.get("status")
        }
        flags = {
            "critical_care": bool(
                profile.get("is_critical_care")
                or self._matching(self.critical_care_profiles, account_id)
            ),
            "chronic_condition": bool(
                profile.get("is_chronic_condition")
                or self._matching(self.chronic_condition_profiles, account_id)
            ),
            "expressed_inability_to_pay": bool(
                profile.get("expressed_inability_to_pay")
                or ledger.get("expressed_inability_to_pay")
            ),
            "weather_emergency_due_bill": bool(ledger.get("weather_emergency_due_bill")),
            "state_disaster_due_bill": bool(ledger.get("state_disaster_due_bill")),
            "underbilling_deferred_required": float(ledger.get("underbilling_amount", 0.0) or 0.0)
            >= 50.0,
        }
        disqualifications: List[str] = []
        if (
            int(
                account.get(
                    "insufficient_funds_count_12mo", profile.get("insufficient_funds_count_12mo", 0)
                )
                or 0
            )
            > 1
        ):
            disqualifications.append("excessive_insufficient_funds")
        if bool(account.get("disconnected_last_12mo") or profile.get("disconnected_last_12mo")):
            disqualifications.append("disconnected_last_12mo")
        if "active" in existing_plan_statuses:
            disqualifications.append("existing_active_plan")
        eligible = (any(flags.values()) or bool(options)) and not disqualifications
        result = {
            "check_id": self._next_check_id(),
            "type": "texas_payment_plan_eligibility",
            "account_id": account_id,
            "notice_id": notice_id,
            "eligible": eligible,
            "eligibility_flags": [key for key, value in flags.items() if value],
            "disqualifications": disqualifications,
            "options": [dict(option) for option in options],
            "delinquent_level_plan_terms": {
                "max_initial_payment_percent": 50,
                "minimum_billing_cycles": 5,
            },
            "deferred_plan_required": bool(
                flags["weather_emergency_due_bill"]
                or flags["state_disaster_due_bill"]
                or flags["underbilling_deferred_required"]
            ),
        }
        self.checks.append(result)
        self.payment_plan_checks.append(dict(result))
        return {"ok": True, "texas_payment_plan_eligibility": result}

    def draft_payment_arrangement(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_arrangement_fields(fields)
        return self._create_draft("payment_arrangement", normalized)

    def draft_medical_baseline_or_vulnerable_hold(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_hold_fields(
            fields, default_hold_type="medical_baseline_review_hold"
        )
        return self._create_draft("regulatory_hold", normalized)

    def draft_protection_hold(self, **fields: Any) -> Dict[str, Any]:
        hold_type = str(fields.get("hold_type", fields.get("protection_type", "regulatory_hold")))
        normalized = self._normalize_hold_fields(fields, default_hold_type=hold_type)
        return self._create_draft("regulatory_hold", normalized)

    def draft_bill_dispute_or_puc_handoff(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_hold_fields(fields, default_hold_type="bill_dispute_hold")
        normalized["dispute_reason"] = str(
            fields.get(
                "dispute_reason", self.task.hidden_facts.get("dispute_reason", "bill_dispute")
            )
        )
        return self._create_draft("bill_dispute_hold", normalized)

    def draft_texas_deferred_payment_plan(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_arrangement_fields(fields)
        total = float(fields.get("total_plan_amount", normalized["arrears_amount"]))
        installments = int(
            fields.get("installment_count", fields.get("installments", normalized["installments"]))
            or 0
        )
        initial_payment = float(
            fields.get(
                "initial_payment_amount", fields.get("down_payment", normalized["down_payment"])
            )
            or 0.0
        )
        installment_amount = float(
            fields.get(
                "installment_amount",
                (total - initial_payment) / installments
                if installments
                else normalized["installment_amount"],
            )
            or 0.0
        )
        normalized.update(
            {
                "plan_type": str(fields.get("plan_type", "deferred_payment_plan")),
                "initial_payment_amount": initial_payment,
                "installment_count": installments,
                "installment_amount": installment_amount,
                "total_plan_amount": total,
                "written_confirmation_required": bool(
                    fields.get("written_confirmation_required", True)
                ),
                "dissatisfaction_notice_included": bool(
                    fields.get("dissatisfaction_notice_included", True)
                ),
                "switch_hold_applies": bool(fields.get("switch_hold_applies", False)),
                "switch_hold_disclosure_included": bool(
                    fields.get("switch_hold_disclosure_included", False)
                ),
                "finance_charge_included": bool(fields.get("finance_charge_included", False)),
                "default_disconnection_terms": str(
                    fields.get(
                        "default_disconnection_terms",
                        "default may allow disconnection under applicable notice terms",
                    )
                ),
                "disconnection_suspended_until": str(
                    fields.get("disconnection_suspended_until", "")
                ),
            }
        )
        created = self._create_draft("texas_deferred_payment_plan", normalized)
        if created.get("ok"):
            record = {
                "draft_id": created["draft_id"],
                "draft_hash": created["draft_hash"],
                **normalized,
            }
            self.deferred_payment_plan_drafts.append(record)
            if normalized["switch_hold_applies"]:
                self.switch_holds.append(
                    {
                        "switch_hold_id": f"switch_{len(self.switch_holds) + 1:03d}",
                        "draft_id": created["draft_id"],
                        "account_id": normalized["account_id"],
                        "status": "drafted",
                        "disclosure_included": normalized["switch_hold_disclosure_included"],
                    }
                )
        return created

    def read_prior_payment_arrangements(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        arrangements = [
            dict(item)
            for item in self.prior_payment_arrangements
            if not account_id or item.get("account_id") == account_id
        ]
        return {"ok": True, "prior_payment_arrangements": arrangements}

    def check_arrangement_default_reinstatement(
        self, account_id: Optional[str] = None, arrangement_id: Optional[str] = None
    ) -> Dict[str, Any]:
        account_id = str(account_id or self.task.hidden_facts.get("account_id", ""))
        if not account_id:
            return {"ok": False, "error": "account_not_found"}
        arrangement = self._find_arrangement(account_id=account_id, arrangement_id=arrangement_id)
        if not arrangement:
            return {"ok": False, "error": "arrangement_not_found"}
        status = str(arrangement.get("status", ""))
        defaulted = status in {"defaulted", "missed", "terminated"} or bool(
            arrangement.get("missed_payment_date")
        )
        result = {
            "check_id": self._next_check_id(),
            "type": "arrangement_default_reinstatement",
            "account_id": account_id,
            "arrangement_id": arrangement.get("arrangement_id"),
            "defaulted": defaulted,
            "missed_payment_date": arrangement.get("missed_payment_date"),
            "eligible_for_new_plan": bool(arrangement.get("eligible_for_new_plan", not defaulted)),
            "requires_new_notice": bool(arrangement.get("requires_new_notice", False)),
            "reinstatement_option_id": arrangement.get("reinstatement_option_id"),
        }
        self.checks.append(result)
        self.arrangement_default_checks.append(dict(result))
        return {"ok": True, "arrangement_default_reinstatement": result}

    def check_bill_charge_disconnectability(
        self,
        account_id: Optional[str] = None,
        bill_id: Optional[str] = None,
        notice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        account_id = str(account_id or self.task.hidden_facts.get("account_id", ""))
        if not account_id:
            return {"ok": False, "error": "account_not_found"}
        notice = self._find_notice(account_id=account_id, notice_id=notice_id) or {}
        ledger = self._find_ledger(account_id) or {}
        if bill_id and ledger and ledger.get("bill_id") != bill_id:
            return {"ok": False, "error": "bill_not_found"}
        barred_reasons = self._texas_disconnect_barred_reasons(notice, ledger)
        result = {
            "check_id": self._next_check_id(),
            "type": "bill_charge_disconnectability",
            "account_id": account_id,
            "bill_id": bill_id or ledger.get("bill_id"),
            "notice_id": notice_id or notice.get("notice_id"),
            "disconnectable": not barred_reasons,
            "barred_reasons": barred_reasons,
            "charge_basis": self._charge_basis(notice, ledger),
            "disputed_amount": float(
                notice.get("disputed_amount", ledger.get("disputed_amount", 0.0)) or 0.0
            ),
            "undisputed_amount": float(
                notice.get("undisputed_amount", ledger.get("undisputed_amount", 0.0)) or 0.0
            ),
        }
        self.checks.append(result)
        self.bill_charge_checks.append(dict(result))
        return {"ok": True, "bill_charge_disconnectability": result}

    def draft_bill_investigation(self, **fields: Any) -> Dict[str, Any]:
        normalized = {
            "customer_id": str(
                fields.get("customer_id") or self.task.hidden_facts.get("customer_id", "")
            ),
            "account_id": str(
                fields.get("account_id") or self.task.hidden_facts.get("account_id", "")
            ),
            "bill_id": str(fields.get("bill_id") or self.task.hidden_facts.get("bill_id", "")),
            "notice_id": str(
                fields.get("notice_id") or self.task.hidden_facts.get("notice_id", "")
            ),
            "disputed_amount": float(
                fields.get("disputed_amount", self.task.hidden_facts.get("disputed_amount", 0.0))
                or 0.0
            ),
            "undisputed_amount": float(
                fields.get(
                    "undisputed_amount", self.task.hidden_facts.get("undisputed_amount", 0.0)
                )
                or 0.0
            ),
            "charge_type": str(fields.get("charge_type", "electric_service")),
            "investigation_status": str(fields.get("investigation_status", "drafted")),
            "customer_result_notice": bool(fields.get("customer_result_notice", True)),
            "commission_complaint_notice_given": bool(
                fields.get("commission_complaint_notice_given", True)
            ),
            "reason": str(fields.get("reason", "bill_investigation")),
        }
        created = self._create_draft("bill_investigation", normalized)
        if created.get("ok"):
            investigation = {
                "investigation_id": f"inv_{len(self.bill_investigations) + 1:03d}",
                "draft_id": created["draft_id"],
                **normalized,
            }
            self.bill_investigations.append(investigation)
            self.bill_disputes.append(
                {
                    "dispute_id": f"ubd_{len(self.bill_disputes) + 1:03d}",
                    "account_id": normalized["account_id"],
                    "notice_id": normalized["notice_id"],
                    "bill_id": normalized["bill_id"],
                    "status": "investigation_drafted",
                    "reason": normalized["reason"],
                }
            )
        return created

    def check_critical_care_or_chronic_condition_requirements(
        self, account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        account_id = str(account_id or self.task.hidden_facts.get("account_id", ""))
        if not account_id:
            return {"ok": False, "error": "account_not_found"}
        critical = self._matching(self.critical_care_profiles, account_id)
        chronic = self._matching(self.chronic_condition_profiles, account_id)
        profile = critical or chronic or self._find_protection_profile(account_id) or {}
        condition_type = str(
            profile.get("condition_type")
            or ("critical_care" if critical else "chronic_condition" if chronic else "")
        )
        docs = [item for item in self.medical_documentation if item.get("account_id") == account_id]
        contacts = [
            item for item in self.emergency_contacts if item.get("account_id") == account_id
        ]
        physician_contact = bool(
            profile.get("physician_contact_received")
            or any(doc.get("physician_contact_received") for doc in docs)
        )
        written_statement = bool(
            profile.get("physician_written_statement_received")
            or any(
                doc.get("physician_written_statement_received") or doc.get("status") == "received"
                for doc in docs
            )
        )
        deferred_plan_entered = bool(
            profile.get("deferred_plan_entered") or profile.get("deferred_payment_plan_id")
        )
        missing = []
        if not physician_contact:
            missing.append("physician_contact")
        if not written_statement:
            missing.append("physician_written_statement")
        if not deferred_plan_entered:
            missing.append("deferred_payment_plan")
        if condition_type == "chronic_condition" and not contacts:
            missing.append("emergency_contact")
        result = {
            "check_id": self._next_check_id(),
            "type": "critical_care_or_chronic_condition_requirements",
            "account_id": account_id,
            "condition_type": condition_type,
            "requirements_met": not missing,
            "missing_requirements": missing,
            "protection_expires_at": profile.get("protection_expires_at"),
            "notice_21_day_required": not bool(profile.get("requirements_current", not missing)),
            "tdu_contact_required": bool(condition_type),
            "no_uninterrupted_service_guarantee": True,
        }
        self.checks.append(result)
        return {"ok": True, "critical_care_or_chronic_condition_requirements": result}

    def record_disconnection_rights_disclosure(
        self, draft_id: Optional[str] = None
    ) -> Dict[str, Any]:
        disclosure = {
            "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "disconnection_rights",
            "text": self.task.mandatory_disclosure(),
            "channel": self.session_context.get("channel", "in_app"),
        }
        self.disclosures.append(disclosure)
        return {"ok": True, "disclosure": disclosure}

    def record_assistance_program_disclosure(
        self,
        account_id: Optional[str] = None,
        draft_id: Optional[str] = None,
        program_ids: Optional[List[str]] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        account_id = str(account_id or self.task.hidden_facts.get("account_id", ""))
        if not account_id:
            return {"ok": False, "error": "account_not_found"}
        matching_pledges = [
            dict(item)
            for item in self.energy_assistance_pledges
            if item.get("account_id") == account_id
        ]
        disclosure = {
            "disclosure_id": f"assist_{len(self.assistance_disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "assistance_programs",
            "account_id": account_id,
            "program_ids": list(program_ids or []),
            "channel": channel or self.session_context.get("channel", "in_app"),
            "pledges": matching_pledges,
            "program_eligibility_summary": "payment options, assistance programs, moratoriums, and application steps disclosed",
        }
        self.assistance_disclosures.append(disclosure)
        self.disclosures.append(dict(disclosure))
        return {"ok": True, "assistance_disclosure": disclosure}

    def submit_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        return self.submit_arrangement_or_hold(**dict(draft.get("fields", {})))

    def submit_arrangement_or_hold(self, **fields: Any) -> Dict[str, Any]:
        request_type = str(fields.get("request_type", ""))
        draft_type = str(
            fields.get("draft_type")
            or (
                "regulatory_hold"
                if request_type in {"medical_hold", "protection_hold"}
                else "payment_arrangement"
            )
        )
        if draft_type == "payment_arrangement":
            arrangement = {
                "arrangement_id": f"arr_{len(self.payment_arrangements) + 1:03d}",
                **self._normalize_arrangement_fields(fields),
                "status": "active",
            }
            self.payment_arrangements.append(arrangement)
            self.outbox.append(
                {
                    "type": "payment_arrangement_notice",
                    "arrangement_id": arrangement["arrangement_id"],
                }
            )
            return {"ok": True, "payment_arrangement": arrangement}
        if draft_type in {"regulatory_hold", "bill_dispute_hold"}:
            hold = {
                "hold_id": f"hold_{len(self.regulatory_holds) + 1:03d}",
                **self._normalize_hold_fields(
                    fields, default_hold_type=str(fields.get("hold_type", "regulatory_hold"))
                ),
                "status": "requested" if draft_type == "regulatory_hold" else "active",
            }
            self.regulatory_holds.append(hold)
            if draft_type == "bill_dispute_hold":
                dispute = {
                    "dispute_id": f"ubd_{len(self.bill_disputes) + 1:03d}",
                    "account_id": hold["account_id"],
                    "notice_id": hold["notice_id"],
                    "status": "open",
                    "reason": fields.get("dispute_reason", "bill_dispute"),
                }
                self.bill_disputes.append(dispute)
            self.outbox.append({"type": "utility_hold_notice", "hold_id": hold["hold_id"]})
            return {"ok": True, "regulatory_hold": hold}
        return {"ok": False, "error": "unsupported_draft_type", "draft_type": draft_type}

    def final_state(self) -> Dict[str, Any]:
        return {
            "business_system": "utilities_disconnection_v0",
            "session_context": dict(self.session_context),
            "customer": dict(self.customer),
            "utility_accounts": [dict(item) for item in self.accounts],
            "bill_ledgers": [dict(item) for item in self.bill_ledgers],
            "disconnection_notices": [dict(item) for item in self.disconnection_notices],
            "protection_profiles": [dict(item) for item in self.protection_profiles],
            "weather_safety": dict(self.weather_safety),
            "payment_options": [dict(item) for item in self.payment_options],
            "checks": [dict(item) for item in self.checks],
            "notice_checks": [dict(item) for item in self.notice_checks],
            "payment_plan_checks": [dict(item) for item in self.payment_plan_checks],
            "deferred_payment_plan_drafts": [
                dict(item) for item in self.deferred_payment_plan_drafts
            ],
            "switch_holds": [dict(item) for item in self.switch_holds],
            "prior_payment_arrangements": [dict(item) for item in self.prior_payment_arrangements],
            "arrangement_default_checks": [dict(item) for item in self.arrangement_default_checks],
            "bill_charge_checks": [dict(item) for item in self.bill_charge_checks],
            "bill_investigations": [dict(item) for item in self.bill_investigations],
            "critical_care_profiles": [dict(item) for item in self.critical_care_profiles],
            "chronic_condition_profiles": [dict(item) for item in self.chronic_condition_profiles],
            "medical_documentation": [dict(item) for item in self.medical_documentation],
            "emergency_contacts": [dict(item) for item in self.emergency_contacts],
            "assistance_disclosures": [dict(item) for item in self.assistance_disclosures],
            "energy_assistance_pledges": [dict(item) for item in self.energy_assistance_pledges],
            "disclosures": [dict(item) for item in self.disclosures],
            "drafts": [dict(item) for item in self.drafts.values()],
            "payment_arrangements": [dict(item) for item in self.payment_arrangements],
            "regulatory_holds": [dict(item) for item in self.regulatory_holds],
            "bill_disputes": [dict(item) for item in self.bill_disputes],
            "outbox": [dict(item) for item in self.outbox],
        }

    def _create_draft(self, draft_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        fields = {**fields, "draft_type": draft_type}
        payload = {"draft_type": draft_type, "fields": fields}
        draft_hash = _canonical_hash(payload)
        draft_id = f"draft_{len(self.drafts) + 1:03d}"
        draft = {"draft_id": draft_id, "draft_hash": draft_hash, **payload}
        self.drafts[draft_id] = draft
        path = self.draft_dir / f"{draft_id}.json"
        path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"ok": True, **draft}

    def _normalize_arrangement_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        account_id = str(fields.get("account_id") or facts.get("account_id", ""))
        return {
            "draft_type": "payment_arrangement",
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "account_id": account_id,
            "notice_id": str(fields.get("notice_id") or facts.get("notice_id", "")),
            "arrears_amount": float(fields.get("arrears_amount", facts.get("arrears_amount", 0.0))),
            "currency": str(fields.get("currency") or facts.get("currency", "USD")),
            "down_payment": float(fields.get("down_payment", facts.get("down_payment", 0.0))),
            "installment_amount": float(
                fields.get("installment_amount", facts.get("installment_amount", 0.0))
            ),
            "installments": int(fields.get("installments", facts.get("installments", 12))),
            "option_id": str(
                fields.get("option_id") or facts.get("option_id", "standard_12_month")
            ),
        }

    def _normalize_hold_fields(
        self, fields: Dict[str, Any], default_hold_type: str
    ) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        return {
            "draft_type": str(fields.get("draft_type", "regulatory_hold")),
            "customer_id": str(fields.get("customer_id") or facts.get("customer_id", "")),
            "account_id": str(fields.get("account_id") or facts.get("account_id", "")),
            "notice_id": str(fields.get("notice_id") or facts.get("notice_id", "")),
            "bill_id": str(fields.get("bill_id") or facts.get("bill_id", "")),
            "amount_due": float(
                fields.get("amount_due", facts.get("amount_due", facts.get("arrears_amount", 0.0)))
            ),
            "currency": str(fields.get("currency") or facts.get("currency", "USD")),
            "request_type": str(
                fields.get("request_type") or facts.get("request_type", "medical_hold")
            ),
            "protection_type": str(
                fields.get("protection_type") or facts.get("protection_type", "medical")
            ),
            "hold_type": str(fields.get("hold_type") or facts.get("hold_type", default_hold_type)),
            "reason": str(fields.get("reason") or facts.get("reason", default_hold_type)),
        }

    def _find_account(self, account_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not account_id:
            account_id = str(self.task.hidden_facts.get("account_id", ""))
        return next(
            (account for account in self.accounts if account.get("account_id") == account_id), None
        )

    def _find_ledger(self, account_id: str) -> Optional[Dict[str, Any]]:
        return next(
            (ledger for ledger in self.bill_ledgers if ledger.get("account_id") == account_id), None
        )

    def _find_notice(
        self, account_id: Optional[str] = None, notice_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for notice in self.disconnection_notices:
            notice_matches = notice_id and notice.get("notice_id") == notice_id
            account_matches = account_id and notice.get("account_id") == account_id
            if notice_matches or account_matches:
                return notice
        if not account_id and not notice_id and self.disconnection_notices:
            return self.disconnection_notices[0]
        return None

    def _find_protection_profile(self, account_id: str) -> Optional[Dict[str, Any]]:
        return next(
            (
                profile
                for profile in self.protection_profiles
                if profile.get("account_id") == account_id
            ),
            None,
        )

    def _find_arrangement(
        self, account_id: str, arrangement_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for arrangement in [*self.prior_payment_arrangements, *self.payment_arrangements]:
            arrangement_matches = (
                arrangement_id and arrangement.get("arrangement_id") == arrangement_id
            )
            account_matches = not arrangement_id and arrangement.get("account_id") == account_id
            if arrangement_matches or account_matches:
                return arrangement
        return None

    def _next_check_id(self) -> str:
        return f"check_{len(self.checks) + 1:03d}"

    def _matching(self, items: List[Dict[str, Any]], account_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in items if item.get("account_id") == account_id), None)

    def _charge_basis(self, notice: Dict[str, Any], ledger: Dict[str, Any]) -> List[str]:
        raw_basis = notice.get("charges_basis", ledger.get("charges_basis", []))
        if isinstance(raw_basis, list):
            basis = [str(item) for item in raw_basis if item is not None]
        elif raw_basis:
            basis = [str(raw_basis)]
        else:
            basis = []
        charge_type = notice.get("charge_type", ledger.get("charge_type"))
        if charge_type:
            basis.append(str(charge_type))
        return basis

    def _texas_disconnect_barred_reasons(
        self, notice: Dict[str, Any], ledger: Dict[str, Any]
    ) -> List[str]:
        reasons: List[str] = []
        basis = {item.lower() for item in self._charge_basis(notice, ledger)}
        non_electric_amount = float(
            notice.get("non_electric_amount", ledger.get("non_electric_amount", 0.0)) or 0.0
        )
        disputed_amount = float(
            notice.get("disputed_amount", ledger.get("disputed_amount", 0.0)) or 0.0
        )
        underbilling_age_days = int(
            notice.get("underbilling_age_days", ledger.get("underbilling_age_days", 0)) or 0
        )
        if non_electric_amount > 0 or "non_electric" in basis or "optional_service" in basis:
            reasons.append("non_electric_or_optional_charges")
        if disputed_amount > 0 or bool(
            notice.get("active_bill_dispute") or ledger.get("active_bill_dispute")
        ):
            reasons.append("disputed_charges")
        if underbilling_age_days > 180:
            reasons.append("stale_underbilling")
        if bool(notice.get("meter_fault", ledger.get("meter_fault", False))):
            reasons.append("faulty_meter_underbilling")
        if bool(notice.get("estimated_bill", ledger.get("estimated_bill", False))):
            reasons.append("estimated_bill")
        if bool(notice.get("prior_occupant_charges", ledger.get("prior_occupant_charges", False))):
            reasons.append("prior_occupant_charges")
        if bool(
            notice.get("different_service_class", ledger.get("different_service_class", False))
        ):
            reasons.append("different_service_class")
        return reasons


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
