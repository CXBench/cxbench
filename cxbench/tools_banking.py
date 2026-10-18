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
class BankingCustomerOpsSystem:
    """Small in-memory banking business system used by cxbench tasks.

    The system models a logged-in application session. Identity proofing is assumed
    to be handled by the host application, so agent tools operate from scoped
    session context and cannot directly submit high-risk mutations in the
    least-privilege runtime.
    """

    task: TaskSpec
    draft_dir: Path
    session_context: Dict[str, Any] = field(default_factory=dict)
    customer: Dict[str, Any] = field(default_factory=dict)
    accounts: List[Dict[str, Any]] = field(default_factory=list)
    cards: List[Dict[str, Any]] = field(default_factory=list)
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    pending_authorizations: List[Dict[str, Any]] = field(default_factory=list)
    disputes: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    disclosures: List[Dict[str, Any]] = field(default_factory=list)
    deadline_clocks: List[Dict[str, Any]] = field(default_factory=list)
    outbox: List[Dict[str, Any]] = field(default_factory=list)
    card_control_changes: List[Dict[str, Any]] = field(default_factory=list)
    written_confirmation_requests: List[Dict[str, Any]] = field(default_factory=list)
    provisional_credit_reviews: List[Dict[str, Any]] = field(default_factory=list)
    authorization_monitors: List[Dict[str, Any]] = field(default_factory=list)
    evidence_attachments: List[Dict[str, Any]] = field(default_factory=list)
    drafts: Dict[str, Draft] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        state = self.task.initial_state
        self.session_context = dict(state.get("session_context", {}))
        self.customer = dict(state.get("customer", {}))
        self.accounts = _copy_list(state.get("accounts", []))
        self.cards = _copy_list(state.get("cards", []))
        self.transactions = _copy_list(state.get("transactions", []))
        self.pending_authorizations = _copy_list(state.get("pending_authorizations", []))
        self.disputes = _copy_list(state.get("disputes", []))
        self.cases = _copy_list(state.get("cases", []))
        self.disclosures = _copy_list(state.get("disclosures", []))
        self.deadline_clocks = _copy_list(state.get("deadline_clocks", []))
        self.outbox = _copy_list(state.get("outbox", []))
        self.card_control_changes = _copy_list(state.get("card_control_changes", []))
        self.written_confirmation_requests = _copy_list(
            state.get("written_confirmation_requests", [])
        )
        self.provisional_credit_reviews = _copy_list(state.get("provisional_credit_reviews", []))
        self.authorization_monitors = _copy_list(state.get("authorization_monitors", []))
        self.evidence_attachments = _copy_list(state.get("evidence_attachments", []))

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_session_context": self.get_session_context,
            "list_customer_accounts": self.list_customer_accounts,
            "list_cards": self.list_cards,
            "get_card_status": self.get_card_status,
            "list_recent_transactions": self.list_recent_transactions,
            "get_transaction_detail": self.get_transaction_detail,
            "list_pending_authorizations": self.list_pending_authorizations,
            "get_authorization_detail": self.get_authorization_detail,
            "classify_eft_error": self.classify_eft_error,
            "check_written_confirmation_policy": self.check_written_confirmation_policy,
            "request_written_confirmation_if_required": self.request_written_confirmation_if_required,
            "check_provisional_credit_timeline": self.check_provisional_credit_timeline,
            "check_reg_e_notice_timing": self.check_reg_e_notice_timing,
            "check_access_device_reporting_window": self.check_access_device_reporting_window,
            "create_eft_dispute_draft": self.create_eft_dispute_draft,
            "create_multi_eft_dispute_draft": self.create_multi_eft_dispute_draft,
            "create_lost_card_dispute_draft": self.create_lost_card_dispute_draft,
            "create_pending_authorization_monitor_draft": self.create_pending_authorization_monitor_draft,
            "create_written_confirmation_followup": self.create_written_confirmation_followup,
            "create_provisional_credit_review_draft": self.create_provisional_credit_review_draft,
            "record_reg_e_timeline_disclosure": self.record_reg_e_timeline_disclosure,
            "record_written_confirmation_request": self.record_written_confirmation_request,
            "record_provisional_credit_timeline_notice": self.record_provisional_credit_timeline_notice,
            "record_no_outcome_promise_notice": self.record_no_outcome_promise_notice,
            "draft_card_freeze": self.draft_card_freeze,
            "upload_customer_evidence_stub": self.upload_customer_evidence_stub,
            "attach_evidence_to_dispute_draft": self.attach_evidence_to_dispute_draft,
            "submit_eft_dispute_case": self.submit_eft_dispute_case,
            "submit_multi_eft_dispute_case": self.submit_multi_eft_dispute_case,
            "submit_lost_card_dispute_case": self.submit_lost_card_dispute_case,
            "submit_pending_authorization_monitor": self.submit_pending_authorization_monitor,
            "submit_provisional_credit_review_case": self.submit_provisional_credit_review_case,
            "submit_card_control_change": self.submit_card_control_change,
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
                for key in ("customer_id", "display_name", "locale", "relationship_tier")
                if key in self.customer
            },
        }

    def list_customer_accounts(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        accounts = [
            account
            for account in self.accounts
            if account_id is None or account.get("account_id") == account_id
        ]
        return {"ok": True, "accounts": [self._public_account(account) for account in accounts]}

    def list_cards(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        cards = [
            card
            for card in self.cards
            if account_id is None or card.get("account_id") == account_id
        ]
        return {"ok": True, "cards": [self._public_card(card) for card in cards]}

    def get_card_status(
        self, card_id: Optional[str] = None, last4: Optional[str] = None
    ) -> Dict[str, Any]:
        card = self._find_card(card_id=card_id, last4=last4)
        if not card:
            return {"ok": False, "error": "card_not_found"}
        return {"ok": True, "card": self._public_card(card)}

    def list_recent_transactions(
        self,
        account_id: Optional[str] = None,
        card_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        transactions = [
            transaction
            for transaction in self.transactions
            if (account_id is None or transaction.get("account_id") == account_id)
            and (card_id is None or transaction.get("card_id") == card_id)
        ]
        return {"ok": True, "transactions": [dict(item) for item in transactions[:limit]]}

    def get_transaction_detail(
        self, transaction_id: Optional[str] = None, merchant: Optional[str] = None
    ) -> Dict[str, Any]:
        transaction = self._find_transaction(transaction_id=transaction_id, merchant=merchant)
        if not transaction:
            return {"ok": False, "error": "transaction_not_found"}
        return {"ok": True, "transaction": dict(transaction)}

    def list_pending_authorizations(
        self, account_id: Optional[str] = None, card_id: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        authorizations = [
            authorization
            for authorization in self.pending_authorizations
            if (account_id is None or authorization.get("account_id") == account_id)
            and (card_id is None or authorization.get("card_id") == card_id)
        ]
        return {
            "ok": True,
            "pending_authorizations": [dict(item) for item in authorizations[:limit]],
        }

    def get_authorization_detail(
        self,
        authorization_id: Optional[str] = None,
        merchant: Optional[str] = None,
    ) -> Dict[str, Any]:
        authorization = self._find_authorization(
            authorization_id=authorization_id, merchant=merchant
        )
        if not authorization:
            return {"ok": False, "error": "authorization_not_found"}
        return {"ok": True, "authorization": dict(authorization)}

    def classify_eft_error(self, transaction_id: str, reason: str) -> Dict[str, Any]:
        transaction = self._find_transaction(transaction_id=transaction_id)
        if not transaction:
            return {"ok": False, "error": "transaction_not_found"}
        is_card_or_eft = transaction.get("network") in {
            "visa_debit",
            "mastercard_debit",
            "ach",
            "atm",
        }
        is_error_notice = reason in {
            "unauthorized_transaction",
            "lost_card_unauthorized_transaction",
            "incorrect_amount",
            "not_received",
        }
        return {
            "ok": True,
            "classification": {
                "transaction_id": transaction_id,
                "reason": reason,
                "is_eft_error_notice": bool(is_card_or_eft and is_error_notice),
                "regime": "Reg E" if is_card_or_eft else "non_eft",
                "requires_timeline_disclosure": bool(is_card_or_eft and is_error_notice),
            },
        }

    def check_written_confirmation_policy(
        self,
        transaction_id: Optional[str] = None,
        notice_channel: str = "oral",
    ) -> Dict[str, Any]:
        policy = self._written_confirmation_policy(
            transaction_id=transaction_id, notice_channel=notice_channel
        )
        if policy.get("error"):
            return {"ok": False, "error": policy["error"]}
        return {"ok": True, "written_confirmation_policy": policy}

    def request_written_confirmation_if_required(
        self,
        transaction_id: Optional[str] = None,
        draft_id: Optional[str] = None,
        channel: Optional[str] = None,
        reason: Optional[str] = None,
        notice_channel: str = "oral",
    ) -> Dict[str, Any]:
        policy = self._written_confirmation_policy(
            transaction_id=transaction_id, notice_channel=notice_channel
        )
        if policy.get("error"):
            return {"ok": False, "error": policy["error"]}
        if not policy["required"]:
            return {
                "ok": True,
                "required": False,
                "request_recorded": False,
                "written_confirmation_policy": policy,
                "must_not_delay_investigation": True,
                "customer_notice": "Written confirmation is not required for this oral error notice; investigation intake should continue.",
            }

        request = self._find_written_confirmation_request(
            transaction_id=str(policy["transaction_id"]),
            draft_id=draft_id,
            channel=channel or str(self.session_context.get("channel", "in_app")),
        )
        idempotent = request is not None
        if request is None:
            request = self._append_written_confirmation_request(
                transaction_id=str(policy["transaction_id"]),
                draft_id=draft_id,
                due_business_days=int(policy["due_business_days"] or 10),
                address=str(policy["address"] or ""),
                channel=channel or str(self.session_context.get("channel", "in_app")),
                reason=reason,
            )
        notice = (
            f"Written confirmation is required within {request['due_business_days']} business days "
            f"at {request['address']}. We will not delay starting the investigation while waiting for it."
        )
        return {
            "ok": True,
            "required": True,
            "request_recorded": True,
            "idempotent": idempotent,
            "written_confirmation_policy": policy,
            "written_confirmation_request": request,
            "must_not_delay_investigation": True,
            "customer_notice": notice,
        }

    def check_provisional_credit_timeline(
        self,
        transaction_id: str,
        notice_received_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        transaction = self._find_transaction(transaction_id=transaction_id)
        if not transaction:
            return {"ok": False, "error": "transaction_not_found"}
        facts = self.task.hidden_facts
        return {
            "ok": True,
            "provisional_credit_timeline": {
                "transaction_id": transaction_id,
                "notice_received_date": notice_received_date or facts.get("notice_received_date"),
                "standard_investigation_business_days": int(
                    facts.get("investigation_business_days", 10)
                ),
                "provisional_credit_business_days": int(
                    facts.get("provisional_credit_due_business_days", 10)
                ),
                "extended_investigation_days": int(facts.get("extended_investigation_days", 45)),
                "eligible_for_review": True,
                "guaranteed_credit": False,
            },
        }

    def check_reg_e_notice_timing(
        self,
        transaction_id: str,
        notice_received_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        transaction = self._find_transaction(transaction_id=transaction_id)
        if not transaction:
            return {"ok": False, "error": "transaction_not_found"}
        facts = self.task.hidden_facts
        timing = str(facts.get("notice_timing", "timely"))
        return {
            "ok": True,
            "notice_timing": {
                "transaction_id": transaction_id,
                "transaction_date": transaction.get("date"),
                "notice_received_date": notice_received_date or facts.get("notice_received_date"),
                "timing": timing,
                "late": timing == "late",
                "extenuating_circumstances": facts.get("good_cause_claimed")
                or facts.get("extenuating_circumstances"),
                "intake_should_continue": True,
            },
        }

    def check_access_device_reporting_window(self, card_id: str) -> Dict[str, Any]:
        card = self._find_card(card_id=card_id)
        if not card:
            return {"ok": False, "error": "card_not_found"}
        facts = self.task.hidden_facts
        within = facts.get("reported_within_business_days")
        after = facts.get("reported_after_business_days")
        if within is not None:
            window = "within_two_business_days" if int(within) <= 2 else "after_two_business_days"
            days = int(within)
        elif after is not None:
            window = "after_two_business_days"
            days = int(after)
        else:
            window = "unknown"
            days = None
        return {
            "ok": True,
            "access_device_reporting_window": {
                "card_id": card_id,
                "reported_business_days": days,
                "window": window,
                "liability_context": facts.get("reg_e_liability_context", window),
                "intake_should_continue": True,
            },
        }

    def create_eft_dispute_draft(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_dispute_fields(fields)
        return self._create_draft("eft_dispute", normalized)

    def create_multi_eft_dispute_draft(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_multi_dispute_fields(fields)
        return self._create_draft("multi_eft_dispute", normalized)

    def create_lost_card_dispute_draft(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_dispute_fields(fields)
        card_id = str(
            fields.get("card_id")
            or normalized.get("card_id")
            or self.task.hidden_facts.get("card_id", "")
        )
        normalized["card_id"] = card_id
        normalized["card_action"] = str(fields.get("card_action", "freeze"))
        normalized["replacement_requested"] = bool(fields.get("replacement_requested", True))
        normalized["lost_card_reported"] = True
        return self._create_draft("lost_card_dispute", normalized)

    def create_pending_authorization_monitor_draft(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_authorization_monitor_fields(fields)
        return self._create_draft("pending_authorization_monitor", normalized)

    def create_written_confirmation_followup(self, **fields: Any) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        payload = {
            "customer_id": str(
                fields.get("customer_id") or facts.get("customer_id") or self._customer_id()
            ),
            "account_id": str(fields.get("account_id") or facts.get("account_id", "")),
            "transaction_id": str(fields.get("transaction_id") or facts.get("transaction_id", "")),
            "due_business_days": int(
                fields.get(
                    "due_business_days", facts.get("written_confirmation_due_business_days", 10)
                )
            ),
            "address": str(fields.get("address") or facts.get("written_confirmation_address", "")),
            "channel": str(fields.get("channel") or self.session_context.get("channel", "in_app")),
        }
        return self._create_draft("written_confirmation_followup", payload)

    def create_provisional_credit_review_draft(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_provisional_credit_fields(fields)
        return self._create_draft("provisional_credit_review", normalized)

    def draft_card_freeze(
        self, card_id: str, reason: str = "customer_reported_lost_card"
    ) -> Dict[str, Any]:
        card = self._find_card(card_id=card_id)
        if not card:
            return {"ok": False, "error": "card_not_found"}
        fields = {
            "customer_id": self._customer_id(),
            "card_id": card_id,
            "action": "freeze",
            "reason": reason,
        }
        return self._create_draft("card_freeze", fields)

    def record_reg_e_timeline_disclosure(self, draft_id: Optional[str] = None) -> Dict[str, Any]:
        disclosure = {
            "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "reg_e_timeline",
            "text": self.task.mandatory_disclosure(),
            "channel": self.session_context.get("channel", "in_app"),
        }
        self.disclosures.append(disclosure)
        return {"ok": True, "disclosure": disclosure}

    def record_written_confirmation_request(
        self,
        draft_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        due_business_days: Optional[int] = None,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        policy = self._written_confirmation_policy(transaction_id=transaction_id)
        if policy.get("error"):
            return {"ok": False, "error": policy["error"]}
        if not policy["required"]:
            return {
                "ok": True,
                "required": False,
                "request_recorded": False,
                "written_confirmation_policy": policy,
                "must_not_delay_investigation": True,
            }
        request = self._find_written_confirmation_request(
            transaction_id=str(policy["transaction_id"]),
            draft_id=draft_id,
            channel=str(self.session_context.get("channel", "in_app")),
        )
        if request is None:
            request = self._append_written_confirmation_request(
                transaction_id=str(policy["transaction_id"]),
                draft_id=draft_id,
                due_business_days=int(due_business_days or policy["due_business_days"] or 10),
                address=str(address or policy["address"] or ""),
                channel=str(self.session_context.get("channel", "in_app")),
            )
        return {
            "ok": True,
            "required": True,
            "request_recorded": True,
            "written_confirmation_request": request,
        }

    def _written_confirmation_policy(
        self,
        transaction_id: Optional[str] = None,
        notice_channel: str = "oral",
    ) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        resolved_id = str(
            transaction_id
            or facts.get("transaction_id")
            or ((facts.get("transaction_ids") or [""])[0])
        )
        transaction = self._find_transaction(transaction_id=resolved_id) if resolved_id else None
        authorization = (
            self._find_authorization(authorization_id=resolved_id) if resolved_id else None
        )
        if resolved_id and not transaction and not authorization:
            return {"error": "transaction_not_found"}
        required = bool(facts.get("written_confirmation_required"))
        due_days = int(facts.get("written_confirmation_due_business_days", 10))
        address = str(facts.get("written_confirmation_address", ""))
        return {
            "transaction_id": resolved_id or facts.get("transaction_id"),
            "notice_channel": notice_channel,
            "required": required,
            "due_business_days": due_days if required else None,
            "address": address if required else "",
            "investigation_can_start_from_oral_notice": True,
            "must_not_delay_investigation": True,
        }

    def _find_written_confirmation_request(
        self,
        transaction_id: str,
        draft_id: Optional[str],
        channel: str,
    ) -> Optional[Dict[str, Any]]:
        attachable: Optional[Dict[str, Any]] = None
        for request in self.written_confirmation_requests:
            if request.get("transaction_id") != transaction_id:
                continue
            if request.get("channel") != channel:
                continue
            if request.get("request_type", "written_confirmation") != "written_confirmation":
                continue
            if request.get("draft_id") == draft_id:
                return request
            if draft_id and not request.get("draft_id"):
                attachable = request
        if attachable is not None:
            attachable["draft_id"] = draft_id
            return attachable
        return None

    def _append_written_confirmation_request(
        self,
        *,
        transaction_id: str,
        draft_id: Optional[str],
        due_business_days: int,
        address: str,
        channel: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = {
            "request_id": f"wconf_{len(self.written_confirmation_requests) + 1:03d}",
            "request_type": "written_confirmation",
            "draft_id": draft_id,
            "transaction_id": transaction_id,
            "due_business_days": due_business_days,
            "address": address,
            "channel": channel,
            "status": "requested",
            "must_not_delay_investigation": True,
        }
        if reason:
            request["reason"] = reason
        self.written_confirmation_requests.append(request)
        self.disclosures.append(
            {
                "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
                "type": "written_confirmation_request",
                **request,
            }
        )
        self.outbox.append(
            {"type": "written_confirmation_request", "request_id": request["request_id"]}
        )
        return request

    def record_provisional_credit_timeline_notice(
        self,
        draft_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        notice = {
            "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
            "draft_id": draft_id,
            "transaction_id": transaction_id or self.task.hidden_facts.get("transaction_id"),
            "type": "provisional_credit_timeline",
            "guaranteed_credit": False,
            "channel": self.session_context.get("channel", "in_app"),
        }
        self.disclosures.append(notice)
        return {"ok": True, "disclosure": notice}

    def record_no_outcome_promise_notice(
        self,
        draft_id: Optional[str] = None,
        scope: str = "dispute",
        **_: Any,
    ) -> Dict[str, Any]:
        notice = {
            "disclosure_id": f"disc_{len(self.disclosures) + 1:03d}",
            "draft_id": draft_id,
            "type": "no_outcome_promise",
            "scope": scope,
            "channel": self.session_context.get("channel", "in_app"),
        }
        self.disclosures.append(notice)
        return {"ok": True, "disclosure": notice}

    def upload_customer_evidence_stub(
        self,
        evidence_type: str,
        description: str,
        transaction_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        evidence = {
            "evidence_id": f"evidence_{len(self.evidence_attachments) + 1:03d}",
            "customer_id": self._customer_id(),
            "transaction_id": transaction_id or self.task.hidden_facts.get("transaction_id"),
            "evidence_type": evidence_type,
            "description": description,
            "file_name": file_name or f"{evidence_type}.txt",
            "content_storage": "stub_metadata_only",
            "status": "attached",
        }
        self.evidence_attachments.append(evidence)
        return {"ok": True, "evidence": evidence}

    def attach_evidence_to_dispute_draft(self, draft_id: str, evidence_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        if not any(item.get("evidence_id") == evidence_id for item in self.evidence_attachments):
            return {"ok": False, "error": "evidence_not_found"}
        fields = dict(draft.get("fields", {}))
        evidence_ids = (
            list(fields.get("evidence_ids", []))
            if isinstance(fields.get("evidence_ids", []), list)
            else []
        )
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
        fields["evidence_ids"] = evidence_ids
        draft["fields"] = fields
        draft["draft_hash"] = _canonical_hash(
            {"draft_type": draft.get("draft_type"), "fields": fields}
        )
        (self.draft_dir / f"{draft_id}.json").write_text(
            json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return {"ok": True, **draft}

    def submit_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": "draft_not_found"}
        draft_type = draft.get("draft_type")
        fields = dict(draft.get("fields", {}))
        if draft_type == "eft_dispute":
            return self.submit_eft_dispute_case(**fields)
        if draft_type == "multi_eft_dispute":
            return self.submit_multi_eft_dispute_case(**fields)
        if draft_type == "lost_card_dispute":
            return self.submit_lost_card_dispute_case(**fields)
        if draft_type == "pending_authorization_monitor":
            return self.submit_pending_authorization_monitor(**fields)
        if draft_type == "provisional_credit_review":
            return self.submit_provisional_credit_review_case(**fields)
        if draft_type == "card_freeze":
            return self.submit_card_control_change(**fields)
        return {"ok": False, "error": "unsupported_draft_type", "draft_type": draft_type}

    def submit_eft_dispute_case(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_dispute_fields(fields)
        dispute = self._open_dispute(normalized)
        return {"ok": True, "dispute": dispute}

    def submit_multi_eft_dispute_case(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_multi_dispute_fields(fields)
        dispute_id = f"dispute_{len(self.disputes) + 1:03d}"
        dispute = {
            "dispute_id": dispute_id,
            **normalized,
            "dispute_type": "multi_eft_dispute",
            "status": "submitted",
        }
        self.disputes.append(dispute)
        case = {
            "case_id": f"case_{len(self.cases) + 1:03d}",
            "case_type": "multi_eft_dispute",
            "dispute_id": dispute_id,
            "status": "open",
            "customer_id": dispute.get("customer_id"),
        }
        self.cases.append(case)
        self.deadline_clocks.append(
            {
                "clock_id": f"clock_{len(self.deadline_clocks) + 1:03d}",
                "case_id": case["case_id"],
                "type": "reg_e_investigation",
                "status": "running",
            }
        )
        self.outbox.append(
            {"type": "dispute_case_notice", "case_id": case["case_id"], "dispute_id": dispute_id}
        )
        return {"ok": True, "dispute": dispute}

    def submit_lost_card_dispute_case(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_dispute_fields(fields)
        card_result = self.submit_card_control_change(
            card_id=str(fields.get("card_id") or normalized.get("card_id") or ""),
            action=str(fields.get("card_action", "freeze")),
            reason="lost_card_reported_with_unauthorized_transaction",
        )
        dispute = self._open_dispute(normalized)
        return {"ok": bool(card_result.get("ok")), "card_control": card_result, "dispute": dispute}

    def submit_pending_authorization_monitor(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_authorization_monitor_fields(fields)
        monitor = {
            "monitor_id": f"authmon_{len(self.authorization_monitors) + 1:03d}",
            **normalized,
            "status": "opened",
        }
        self.authorization_monitors.append(monitor)
        self.outbox.append(
            {"type": "pending_authorization_monitor_notice", "monitor_id": monitor["monitor_id"]}
        )
        return {"ok": True, "authorization_monitor": monitor}

    def submit_provisional_credit_review_case(self, **fields: Any) -> Dict[str, Any]:
        normalized = self._normalize_provisional_credit_fields(fields)
        review = {
            "review_id": f"pcredit_{len(self.provisional_credit_reviews) + 1:03d}",
            **normalized,
            "status": "submitted",
        }
        self.provisional_credit_reviews.append(review)
        self.outbox.append(
            {"type": "provisional_credit_review_notice", "review_id": review["review_id"]}
        )
        return {"ok": True, "provisional_credit_review": review}

    def submit_card_control_change(
        self, card_id: str, action: str = "freeze", reason: str = "customer_request", **_: Any
    ) -> Dict[str, Any]:
        card = self._find_card(card_id=card_id)
        if not card:
            return {"ok": False, "error": "card_not_found"}
        previous_status = card.get("status")
        if action == "freeze":
            card["status"] = "frozen"
        elif action == "unfreeze":
            card["status"] = "active"
        else:
            return {"ok": False, "error": "unsupported_card_action"}
        change = {
            "change_id": f"cardchg_{len(self.card_control_changes) + 1:03d}",
            "customer_id": self._customer_id(),
            "card_id": card_id,
            "action": action,
            "reason": reason,
            "previous_status": previous_status,
            "new_status": card.get("status"),
        }
        self.card_control_changes.append(change)
        self.outbox.append({"type": "card_control_notice", "card_id": card_id, "action": action})
        return {"ok": True, "card_control_change": change, "card": self._public_card(card)}

    def final_state(self) -> Dict[str, Any]:
        return {
            "session_context": dict(self.session_context),
            "customer": dict(self.customer),
            "accounts": [dict(item) for item in self.accounts],
            "cards": [dict(item) for item in self.cards],
            "transactions": [dict(item) for item in self.transactions],
            "pending_authorizations": [dict(item) for item in self.pending_authorizations],
            "drafts": [dict(item) for item in self.drafts.values()],
            "disputes": [dict(item) for item in self.disputes],
            "cases": [dict(item) for item in self.cases],
            "disclosures": [dict(item) for item in self.disclosures],
            "deadline_clocks": [dict(item) for item in self.deadline_clocks],
            "outbox": [dict(item) for item in self.outbox],
            "card_control_changes": [dict(item) for item in self.card_control_changes],
            "written_confirmation_requests": [
                dict(item) for item in self.written_confirmation_requests
            ],
            "provisional_credit_reviews": [dict(item) for item in self.provisional_credit_reviews],
            "authorization_monitors": [dict(item) for item in self.authorization_monitors],
            "evidence_attachments": [dict(item) for item in self.evidence_attachments],
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

    def _open_dispute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        dispute_id = f"dispute_{len(self.disputes) + 1:03d}"
        dispute = {"dispute_id": dispute_id, **fields, "status": "submitted"}
        self.disputes.append(dispute)
        case = {
            "case_id": f"case_{len(self.cases) + 1:03d}",
            "case_type": "eft_dispute",
            "dispute_id": dispute_id,
            "status": "open",
            "customer_id": dispute.get("customer_id"),
        }
        self.cases.append(case)
        self.deadline_clocks.append(
            {
                "clock_id": f"clock_{len(self.deadline_clocks) + 1:03d}",
                "case_id": case["case_id"],
                "type": "reg_e_investigation",
                "status": "running",
            }
        )
        self.outbox.append(
            {"type": "dispute_case_notice", "case_id": case["case_id"], "dispute_id": dispute_id}
        )
        return dispute

    def _normalize_dispute_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        transaction_id = str(fields.get("transaction_id") or facts.get("transaction_id", ""))
        transaction = self._find_transaction(transaction_id=transaction_id) or {}
        account_id = str(
            fields.get("account_id") or transaction.get("account_id") or facts.get("account_id", "")
        )
        card_id = str(
            fields.get("card_id") or transaction.get("card_id") or facts.get("card_id", "")
        )
        return {
            "customer_id": str(
                fields.get("customer_id") or facts.get("customer_id") or self._customer_id()
            ),
            "account_id": account_id,
            "card_id": card_id,
            "transaction_id": transaction_id,
            "merchant": str(
                fields.get("merchant") or transaction.get("merchant") or facts.get("merchant", "")
            ),
            "amount": float(
                fields.get("amount", transaction.get("amount", facts.get("amount", 0.0)))
            ),
            "currency": str(
                fields.get("currency")
                or transaction.get("currency")
                or facts.get("currency", "USD")
            ),
            "reason": str(fields.get("reason") or facts.get("reason", "customer_dispute")),
        }

    def _normalize_multi_dispute_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        raw_ids = (
            fields.get("transaction_ids")
            or facts.get("transaction_ids")
            or facts.get("expected_transaction_ids")
            or []
        )
        if isinstance(raw_ids, str):
            transaction_ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
        elif isinstance(raw_ids, list):
            transaction_ids = [str(item) for item in raw_ids]
        else:
            transaction_ids = []
        if not transaction_ids and facts.get("transaction_id"):
            transaction_ids = [str(facts["transaction_id"])]
        transactions = [
            self._find_transaction(transaction_id=item) or {"transaction_id": item}
            for item in transaction_ids
        ]
        account_id = str(
            fields.get("account_id")
            or facts.get("account_id")
            or (transactions[0].get("account_id") if transactions else "")
        )
        currency = str(
            fields.get("currency")
            or facts.get("currency")
            or (transactions[0].get("currency") if transactions else "USD")
        )
        total_amount = fields.get("total_amount", facts.get("total_amount"))
        if total_amount is None:
            total_amount = sum(float(item.get("amount", 0.0)) for item in transactions)
        return {
            "customer_id": str(
                fields.get("customer_id") or facts.get("customer_id") or self._customer_id()
            ),
            "account_id": account_id,
            "transaction_ids": transaction_ids,
            "transactions": [
                {
                    "transaction_id": item.get("transaction_id"),
                    "merchant": item.get("merchant"),
                    "amount": item.get("amount"),
                    "currency": item.get("currency", currency),
                }
                for item in transactions
            ],
            "merchant": str(
                fields.get("merchant")
                or facts.get("merchant")
                or (transactions[0].get("merchant") if transactions else "")
            ),
            "total_amount": float(total_amount or 0.0),
            "currency": currency,
            "reason": str(fields.get("reason") or facts.get("reason", "customer_dispute")),
        }

    def _normalize_authorization_monitor_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.task.hidden_facts
        authorization_id = str(
            fields.get("authorization_id") or facts.get("authorization_id") or ""
        )
        authorization = self._find_authorization(authorization_id=authorization_id) or {}
        return {
            "customer_id": str(
                fields.get("customer_id") or facts.get("customer_id") or self._customer_id()
            ),
            "account_id": str(
                fields.get("account_id")
                or authorization.get("account_id")
                or facts.get("account_id", "")
            ),
            "card_id": str(
                fields.get("card_id") or authorization.get("card_id") or facts.get("card_id", "")
            ),
            "authorization_id": authorization_id,
            "merchant": str(
                fields.get("merchant") or authorization.get("merchant") or facts.get("merchant", "")
            ),
            "amount": float(
                fields.get("amount", authorization.get("amount", facts.get("amount", 0.0)))
            ),
            "currency": str(
                fields.get("currency")
                or authorization.get("currency")
                or facts.get("currency", "USD")
            ),
            "reason": str(
                fields.get("reason") or facts.get("reason", "unauthorized_pending_authorization")
            ),
            "next_step": str(
                fields.get("next_step") or facts.get("next_step", "monitor_until_posted")
            ),
        }

    def _normalize_provisional_credit_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        dispute_fields = self._normalize_dispute_fields(fields)
        return {
            **dispute_fields,
            "requested_review": True,
            "guaranteed_credit": False,
            "notice_received_date": str(
                fields.get("notice_received_date")
                or self.task.hidden_facts.get("notice_received_date", "")
            ),
        }

    def _find_transaction(
        self, transaction_id: Optional[str] = None, merchant: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for transaction in self.transactions:
            id_matches = transaction_id and transaction.get("transaction_id") == transaction_id
            merchant_matches = (
                merchant and str(transaction.get("merchant", "")).lower() == merchant.lower()
            )
            if id_matches or merchant_matches:
                return transaction
        return None

    def _find_authorization(
        self, authorization_id: Optional[str] = None, merchant: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for authorization in self.pending_authorizations:
            id_matches = (
                authorization_id and authorization.get("authorization_id") == authorization_id
            )
            merchant_matches = (
                merchant and str(authorization.get("merchant", "")).lower() == merchant.lower()
            )
            if id_matches or merchant_matches:
                return authorization
        return None

    def _find_card(
        self, card_id: Optional[str] = None, last4: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for card in self.cards:
            id_matches = card_id and card.get("card_id") == card_id
            last4_matches = last4 and str(card.get("last4")) == str(last4)
            if id_matches or last4_matches:
                return card
        return None

    def _customer_id(self) -> str:
        return str(
            self.session_context.get("customer_id")
            or self.customer.get("customer_id")
            or self.task.hidden_facts.get("customer_id", "")
        )

    @staticmethod
    def _public_account(account: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: account.get(key)
            for key in (
                "account_id",
                "type",
                "currency",
                "status",
                "available_balance",
                "masked_number",
            )
            if key in account
        }

    @staticmethod
    def _public_card(card: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: card.get(key)
            for key in (
                "card_id",
                "account_id",
                "last4",
                "network",
                "status",
                "controls",
                "replacement_status",
            )
            if key in card
        }
