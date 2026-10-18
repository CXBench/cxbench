from __future__ import annotations

import builtins
import json
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .openai_compat import OpenAICompatibleClient, OpenAICompatConfig
from .task_schema import TaskSpec

UserSimKind = Literal["deterministic", "human", "llm", "react", "verify", "reflection"]
ObservationKind = Literal["message", "confirm", "reject", "stop"]


def _amount_list_matches(fields: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    expected = facts.get("amounts")
    if isinstance(expected, dict):
        transactions = fields.get("transactions")
        if not isinstance(transactions, list):
            return False
        actual_by_id = {
            str(item.get("transaction_id")): item.get("amount") for item in transactions
        }
        return all(
            str(transaction_id) in actual_by_id
            and float(actual_by_id[str(transaction_id)]) == float(amount)
            for transaction_id, amount in expected.items()
        )
    if not isinstance(expected, list):
        return False
    actual = fields.get("amounts")
    if actual is None and isinstance(fields.get("transactions"), list):
        actual = [item.get("amount") for item in fields["transactions"]]
    if actual is None:
        return False
    if len(actual) != len(expected):
        return False
    return all(float(got) == float(want) for got, want in zip(actual, expected))


def _asks_for_confirmation(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "confirm",
            "confirmation",
            "please approve",
            "do you approve",
        )
    )


def _loose_code_matches(field_name: str, actual: Any, expected: Any, task: TaskSpec) -> bool:
    """Accept narrow natural-language aliases for codes while keeping IDs exact."""
    if str(actual) == str(expected):
        return True
    if field_name.endswith("_code"):
        return _normalized_code(actual) == _normalized_code(expected)
    if field_name in {
        "appeal_summary",
        "appeal_narrative",
        "customer_statement",
        "reason_narrative",
    }:
        return _summary_matches(actual, expected, task)
    if field_name == "reason" and task.domain == "banking":
        return _banking_reason_matches(actual, expected)
    if field_name == "reason_code" and task.domain == "banking":
        return _banking_reason_matches(actual, expected)
    if field_name == "requested_action" and task.domain == "healthcare":
        return _healthcare_action_matches(actual, expected)
    if field_name == "requested_action_code" and task.domain == "healthcare":
        return _healthcare_action_matches(actual, expected)
    if field_name == "appeal_reason" and task.domain == "healthcare":
        return _healthcare_appeal_reason_matches(actual, expected)
    if field_name == "appeal_basis_code" and task.domain == "healthcare":
        return _healthcare_appeal_reason_matches(actual, expected)
    return False


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").replace("_", " ").split())


def _normalized_code(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", "_").split())


def _banking_reason_matches(actual: Any, expected: Any) -> bool:
    actual_text = _normalized_text(actual)
    expected_text = _normalized_text(expected)
    if not actual_text:
        return False
    if expected_text == "unauthorized transaction":
        return any(
            marker in actual_text
            for marker in (
                "unauthorized",
                "not authorized",
                "did not authorize",
                "not my",
                "does not recognize",
                "do not recognize",
                "unrecognized",
            )
        )
    if expected_text == "lost card unauthorized transaction":
        return (
            "lost" in actual_text
            and ("card" in actual_text or "debit" in actual_text)
            and _banking_reason_matches(actual_text, "unauthorized_transaction")
        )
    if expected_text == "incorrect amount":
        return any(
            marker in actual_text
            for marker in ("incorrect amount", "wrong amount", "charged", "overcharged")
        )
    if expected_text == "unauthorized pending authorization":
        return "pending" in actual_text and _banking_reason_matches(
            actual_text, "unauthorized_transaction"
        )
    return actual_text == expected_text


def _healthcare_action_matches(actual: Any, expected: Any) -> bool:
    actual_text = _normalized_text(actual)
    expected_text = _normalized_text(expected)
    if not actual_text:
        return False
    if expected_text == "internal appeal":
        return (
            "internal appeal" in actual_text or "appeal" in actual_text or "overturn" in actual_text
        )
    if expected_text == "external review request":
        return "external review" in actual_text or "independent review" in actual_text
    return actual_text == expected_text


def _healthcare_appeal_reason_matches(actual: Any, expected: Any) -> bool:
    actual_text = _normalized_text(actual)
    expected_text = _normalized_text(expected)
    if not actual_text:
        return False
    if actual_text == expected_text:
        return True
    keywords = [
        token
        for token in expected_text.split()
        if len(token) >= 5
        and token
        not in {
            "appeal",
            "review",
            "request",
            "records",
            "record",
            "available",
            "supporting",
            "submitted",
            "internal",
            "external",
            "denial",
            "final",
        }
    ]
    return bool(keywords and any(token in actual_text for token in keywords))


def _summary_matches(actual: Any, expected: Any, task: TaskSpec) -> bool:
    actual_text = _normalized_text(actual)
    expected_text = _normalized_text(expected)
    if not actual_text:
        return False
    if expected_text and expected_text in actual_text:
        return True
    facts = task.hidden_facts
    if task.domain == "healthcare":
        action = facts.get("requested_action_code") or facts.get("requested_action")
        basis = facts.get("appeal_basis_code") or facts.get("appeal_reason")
        action_ok = not action or _healthcare_action_matches(actual_text, action)
        basis_ok = not basis or _healthcare_appeal_reason_matches(actual_text, basis)
        identifiers = [
            facts.get("denial_id"),
            facts.get("claim_id"),
        ]
        identifier_ok = any(
            str(item).lower() in str(actual).lower() for item in identifiers if item
        )
        no_promise = not any(
            marker in actual_text
            for marker in ("guarantee", "will approve", "will cover", "will pay")
        )
        return (
            action_ok
            and basis_ok
            and (identifier_ok or "denial" in actual_text or "appeal" in actual_text)
            and no_promise
        )
    return bool(
        expected_text
        and any(token in actual_text for token in expected_text.split() if len(token) >= 5)
    )


def _expected_fact_value(field_name: str, facts: Dict[str, Any]) -> Any:
    if field_name in facts:
        return facts[field_name]
    aliases = {
        "requested_action_code": "requested_action",
        "appeal_basis_code": "appeal_reason",
        "reason_code": "reason",
        "reason_narrative": "reason",
        "appeal_summary": "appeal_reason",
        "appeal_narrative": "appeal_reason",
        "customer_statement": "appeal_reason",
    }
    alias = aliases.get(field_name)
    if alias and alias in facts:
        return facts[alias]
    raise KeyError(field_name)


def draft_mismatch_feedback(task: TaskSpec, draft: Dict[str, Any]) -> str:
    """Return field-aware natural customer feedback without exposing a JSON diff."""
    fields = draft.get("fields", {}) if isinstance(draft, dict) else {}
    facts = task.hidden_facts
    for field_name in task.confirmation_contract:
        expected = _expected_fact_value(field_name, facts)
        actual = fields.get(field_name)
        if _field_matches_for_feedback(task, field_name, actual, expected, fields, facts):
            continue
        if task.domain == "healthcare":
            return _actionable_mismatch_feedback(
                field_name, expected, _healthcare_mismatch_feedback(field_name, actual, expected)
            )
        if task.domain == "banking":
            return _actionable_mismatch_feedback(
                field_name, expected, _banking_mismatch_feedback(field_name)
            )
        return _actionable_mismatch_feedback(
            field_name, expected, _generic_mismatch_feedback(field_name)
        )
    return "This draft does not match what I need. Please correct it based on what I said earlier and then let me confirm."


def _field_matches_for_feedback(
    task: TaskSpec,
    field_name: str,
    actual: Any,
    expected: Any,
    fields: Dict[str, Any],
    facts: Dict[str, Any],
) -> bool:
    if field_name in {"amount", "total_amount"}:
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    if field_name == "amounts":
        return _amount_list_matches(fields, facts)
    if isinstance(expected, list):
        return [str(item) for item in fields.get(field_name, [])] == [
            str(item) for item in expected
        ]
    return _loose_code_matches(field_name, actual, expected, task)


def _actionable_mismatch_feedback(field_name: str, expected: Any, message: str) -> str:
    if field_name in {
        "appeal_summary",
        "appeal_narrative",
        "customer_statement",
        "reason_narrative",
    }:
        return f"{field_name} does not match my situation. {message}"
    if isinstance(expected, (dict, list)):
        expected_text = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    else:
        expected_text = str(expected)
    return f"{field_name} is incorrect. Please change it to {expected_text}. {message}"


def _healthcare_mismatch_feedback(field_name: str, actual: Any, expected: Any) -> str:
    normalized_expected = _normalized_text(expected)
    normalized_actual = _normalized_text(actual)
    if field_name in {"requested_action", "requested_action_code"}:
        if "external review" in normalized_expected:
            return "This draft does not yet express an external/independent review request. Please change the request type to external review and then let me confirm."
        if "expedited" in normalized_expected or "urgent" in normalized_expected:
            return "This draft does not clearly express my need for expedited internal appeal. The doctor said delay could cause serious harm. Please make the expedited request clear."
        if "internal appeal" in normalized_expected:
            return "This draft does not look like the internal appeal process I need. I want to file an internal appeal first, not an external review or other request."
        return "The request type in this draft is wrong. Please change it to the appeal/review request I am submitting."
    if field_name in {"appeal_reason", "appeal_basis_code"}:
        if "urgent" in normalized_expected or "continuity" in normalized_expected:
            return "The appeal reason does not reflect the delay risk or continuity-of-care need. Please correct it in that direction and then let me confirm."
        if "medical necessity" in normalized_expected or "medical" in normalized_expected:
            return "The appeal reason does not accurately express the medical necessity and available supporting materials. Please correct this part and then let me confirm."
        if "denial" in normalized_expected or "appeal" in normalized_expected:
            return "The appeal reason is written incorrectly. I am challenging this denial/coverage decision. Please revise accordingly."
        return "The appeal reason does not match my situation. Please correct the reason and then let me confirm."
    if field_name in {"appeal_summary", "appeal_narrative"}:
        if any(
            marker in normalized_actual
            for marker in ("approve coverage", "guarantee", "will cover", "will pay")
        ):
            return "The appeal summary should not promise approval or coverage. Please change it to request a review of this denial under the rules."
        return "The appeal summary does not accurately reflect what I am applying for. Please make the summary more aligned with this denial and the appeal request."
    if field_name == "customer_statement":
        return "The customer statement does not need to exactly match internal codes. Please save my exact words separately instead of rewriting them as medical or coverage conclusions."
    if field_name == "urgency_requested":
        return "The expedited flag is wrong. Please set whether it is expedited correctly based on the delay risk I mentioned."
    if field_name == "representative_id":
        return "The representative information is wrong. Please confirm whether this is being submitted by the member or an authorized representative."
    if "phi" in field_name or "scope" in field_name:
        return "Please use only the minimum information needed for this appeal. Do not expand to the full medical record or unrelated materials."
    return _generic_mismatch_feedback(field_name)


def _banking_mismatch_feedback(field_name: str) -> str:
    if field_name in {"reason", "reason_code", "reason_narrative"}:
        return "The dispute reason does not match what I described. Please correct it based on the actual situation I mentioned: unauthorized, lost card, amount error, etc."
    if field_name in {"transaction_id", "merchant", "amount", "currency", "account_id", "card_id"}:
        return "The transaction information is wrong. Please verify the merchant, amount, account, or card and then let me confirm."
    return _generic_mismatch_feedback(field_name)


def _generic_mismatch_feedback(field_name: str) -> str:
    readable = field_name.replace("_", " ")
    return f"The {readable} in the draft does not match my situation. Please correct it and then let me confirm."


@dataclass(frozen=True)
class UserStepInput:
    assistant_message: str
    latest_draft: Optional[Dict[str, Any]]
    final_state: Dict[str, Any]
    trace_tail: List[Dict[str, Any]]


@dataclass(frozen=True)
class UserObservation:
    kind: ObservationKind
    content: str
    draft_id: Optional[str] = None
    draft_hash: Optional[str] = None
    decision_reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def decision_event(self, strategy: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "strategy": strategy,
            "kind": self.kind,
            "decision_reason": self.decision_reason,
        }
        if self.draft_id:
            payload["draft_id"] = self.draft_id
        if self.draft_hash:
            payload["draft_hash"] = self.draft_hash
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        return payload


# Backward-compatible name used by Runtime before the SPI refactor.
UserReply = UserObservation


class BaseUserSim:
    strategy = "base"

    def reset(self) -> UserObservation:
        raise NotImplementedError

    def step(self, input: UserStepInput) -> UserObservation:
        raise NotImplementedError

    def get_total_cost(self) -> float:
        return 0.0


class DeterministicUserSim(BaseUserSim):
    """Deterministic customer simulator for business-field confirmation."""

    strategy = "deterministic"

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.confirmed = False
        self.stopped = False

    def reset(self) -> UserObservation:
        return UserObservation(
            kind="message", content=self.task.seed_message, decision_reason="seed_message"
        )

    def step(self, input: UserStepInput) -> UserObservation:
        if _has_terminal_submission(input.final_state):
            self.stopped = True
            return UserObservation(
                kind="stop", content="###STOP###", decision_reason="goal_satisfied"
            )

        if not input.latest_draft:
            return UserObservation(
                kind="message",
                content="Sure, please first tell me which specific fields need confirmation.",
                decision_reason="waiting_for_draft",
            )

        if not _asks_for_confirmation(input.assistant_message):
            return UserObservation(
                kind="message",
                content="Please continue; let me know when you have prepared the draft fields that need my confirmation.",
                decision_reason="waiting_for_explicit_confirmation_request",
            )

        if self._draft_matches(input.latest_draft):
            self.confirmed = True
            return UserObservation(
                kind="confirm",
                content="CONFIRM: I confirm submission of this dispute draft.",
                draft_id=input.latest_draft["draft_id"],
                draft_hash=input.latest_draft["draft_hash"],
                decision_reason="draft_matches_hidden_facts",
            )
        return UserObservation(
            kind="message",
            content=draft_mismatch_feedback(self.task, input.latest_draft),
            decision_reason="draft_mismatch",
        )

    def _draft_matches(self, draft: Dict[str, Any]) -> bool:
        fields = draft.get("fields", {})
        facts = self.task.hidden_facts
        for field_name in self.task.confirmation_contract:
            expected = _expected_fact_value(field_name, facts)
            if field_name == "amount":
                if float(fields.get(field_name, -1)) != float(expected):
                    return False
            elif field_name == "total_amount":
                if float(fields.get(field_name, -1)) != float(expected):
                    return False
            elif field_name == "amounts":
                if not _amount_list_matches(fields, facts):
                    return False
            elif isinstance(expected, list):
                if [str(item) for item in fields.get(field_name, [])] != [
                    str(item) for item in expected
                ]:
                    return False
            elif not _loose_code_matches(field_name, fields.get(field_name), expected, self.task):
                return False
        return True

    def _has_disclosure(self, assistant_message: str) -> bool:
        disclosure = self.task.mandatory_disclosure()
        return not disclosure or disclosure in assistant_message


class HumanUserSim(BaseUserSim):
    strategy = "human"

    def __init__(self, task: TaskSpec) -> None:
        self.task = task

    def reset(self) -> UserObservation:
        content = builtins.input(f"{self.task.customer_goal}\n")
        return UserObservation(kind="message", content=content, decision_reason="human_input")

    def step(self, step_input: UserStepInput) -> UserObservation:
        content = builtins.input(f"{step_input.assistant_message}\n")
        return UserObservation(
            kind="stop" if "###STOP###" in content else "message",
            content=content,
            decision_reason="human_input",
        )


_ECHO_FALLBACK = (
    "I've already provided the information you asked for. Please proceed with what you have."
)

_AGENT_ECHO_PATTERNS = (
    "i need you to",
    "please provide me with",
    "can you help me with",
    "could you please provide",
    "i would need you to",
    "please share your",
    "can you share your",
    "could you share your",
    "i'll need you to",
    "may i have your",
    "can i have your",
    "could i have your",
    "let me help you with",
    "i can help you with",
    "i'd be happy to help",
    "i'd be happy to assist",
)


def _normalize_for_similarity(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _word_set(text: str) -> set[str]:
    return set(_normalize_for_similarity(text).split())


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_substring_echo(response: str, agent_message: str) -> bool:
    norm_resp = _normalize_for_similarity(response)
    norm_agent = _normalize_for_similarity(agent_message)
    if len(norm_resp) < 10 or len(norm_agent) < 10:
        return False
    return norm_resp in norm_agent or norm_agent in norm_resp


def _looks_like_agent_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _AGENT_ECHO_PATTERNS)


class LLMUserSim(BaseUserSim):
    strategy = "llm"

    def __init__(
        self, task: TaskSpec, model: str, client: Any | None = None, max_attempts: int = 3
    ) -> None:
        self.task = task
        self.model = model
        self.max_attempts = max_attempts
        self.client = client or OpenAICompatibleClient(
            OpenAICompatConfig.from_model_string(model, role="user-sim")
        )
        self.messages: List[Dict[str, Any]] = []
        self.total_cost = 0.0
        self._recent_messages: List[str] = []

    def reset(self) -> UserObservation:
        self._recent_messages = []
        self.messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._task_context_prompt()},
        ]
        seed_message = self.task.seed_message
        if not seed_message.strip():
            raise ValueError(
                f"task {self.task.id} must define a non-empty seed_message for LLM user simulation"
            )
        observation = UserObservation(
            kind="message", content=seed_message, decision_reason="seed_message"
        )
        self._append_user_message(observation.content)
        return observation

    def step(self, input: UserStepInput) -> UserObservation:
        if _has_terminal_submission(input.final_state):
            return UserObservation(
                kind="stop", content="###STOP###", decision_reason="goal_satisfied"
            )
        self.messages.append({"role": "user", "content": f"Agent said: {input.assistant_message}"})
        observation = self._generate_observation(
            self.messages, input=input, context="user-sim step"
        )
        observation = self._guard_observation(observation, input)
        observation = self._check_repetition(observation, input)
        self._append_user_message(observation.content)
        self._recent_messages.append(observation.content)
        if len(self._recent_messages) > 3:
            self._recent_messages = self._recent_messages[-3:]
        return observation

    def get_total_cost(self) -> float:
        return self.total_cost

    def _generate_observation(
        self, messages: List[Dict[str, Any]], input: Optional[UserStepInput], context: str
    ) -> UserObservation:
        response, obj = self.client.create_validated(
            messages=[*messages, {"role": "user", "content": self._output_contract(input)}],
            response_format={"type": "json_object"},
            validator=lambda candidate: self.client.validate_json_object(
                candidate,
                required_keys=["kind"],
                context=context,
            ),
            validation_context=context,
        )
        self._accumulate_cost(response)
        return self._observation_from_json(obj)

    def _system_prompt(self) -> str:
        return (
            "You are a customer interacting with a regulated CX agent in cxbench. "
            "You are not a free-form chat bot; you choose one structured customer action each turn. "
            "Do not reveal the full hidden goal at once. Only provide information needed for the current step. "
            "Do not invent facts. Do not evaluate regulatory disclosure quality; cxbench grades compliance separately. "
            "Return only one JSON object. The required key is kind. kind must be one of message, confirm, reject, stop. "
            "Use kind=message only when you are providing requested facts, asking a question, or continuing the conversation; "
            "message requires content. "
            "Use kind=confirm as the only way to approve a visible draft for submission. Confirm is a runtime control signal, "
            "not a chat sentence: do not include content for confirm. If the agent asks you to confirm a draft and the draft "
            "matches your hidden facts, return kind=confirm immediately. Do not return kind=message with words like yes, "
            "submit, please submit, looks correct, go ahead, approve, I confirm, or equivalent wording. "
            "Use kind=reject when a visible draft is wrong or you cannot approve it; reject requires reject_reason and should "
            "include correction when a specific field is wrong. "
            "Use kind=stop only when the final state shows the goal is satisfied; stop may use content ###STOP###. "
            "decision_reason is optional diagnostic text for cxbench, not a customer message. "
            "Do not include hidden facts unless the agent asked for that specific detail. "
            "The task-specific customer goal, seed message, and hidden facts are provided after this fixed instruction block."
        )

    def _task_context_prompt(self) -> str:
        facts = json.dumps(self.task.hidden_facts, ensure_ascii=False, sort_keys=True)
        return (
            "<CXBENCH_USER_SIM_TASK_CONTEXT>\n"
            f"Customer goal: {self.task.customer_goal}\n"
            f"Initial seed message: {self.task.seed_message}\n"
            f"Hidden facts: {facts}\n"
            "</CXBENCH_USER_SIM_TASK_CONTEXT>"
        )

    def _output_contract(self, input: Optional[UserStepInput]) -> str:
        draft = json.dumps(
            input.latest_draft if input else None, ensure_ascii=False, sort_keys=True
        )
        final_state = json.dumps(
            input.final_state if input else {}, ensure_ascii=False, sort_keys=True
        )
        asked_confirmation = bool(input and _asks_for_confirmation(input.assistant_message))
        if input and input.latest_draft and asked_confirmation:
            action_contract = (
                "The agent is asking you to approve a visible draft. Choose exactly one of these actions:\n"
                '- If every field in Latest draft matches your hidden facts, return {"kind":"confirm", '
                '"decision_reason":"draft_matches_hidden_facts"}. Do not include content.\n'
                '- If any field is wrong or missing, return {"kind":"reject", "reject_reason":"...", '
                '"correction":{"field":"...","expected":"..."}}.\n'
                "- Do not return kind=message to approve, acknowledge, or ask the agent to submit this draft.\n"
            )
        else:
            action_contract = (
                "If the agent has not asked you to approve a visible draft, do not use kind=confirm. "
                'For normal conversation return {"kind":"message", "content":"..."}. '
                'If final_state shows the goal is complete, return {"kind":"stop", "content":"###STOP###"}.\n'
            )
        return (
            "Current dynamic state for your next customer reply. Return the next JSON object now. "
            f"Agent asked for draft confirmation: {asked_confirmation}\n"
            f"Latest draft visible to you: {draft}\n"
            f"Final state: {final_state}\n"
            f"{action_contract}"
        )

    def _observation_from_json(self, obj: Dict[str, Any]) -> UserObservation:
        kind = str(obj.get("kind", "message")).lower()
        if kind not in {"message", "confirm", "reject", "stop"}:
            kind = "message"
        content = str(obj.get("content") or "")
        reject_reason = str(obj.get("reject_reason") or "")
        if kind == "confirm":
            content = "CONFIRM"
        elif kind == "reject":
            content = (
                reject_reason
                or content
                or "I cannot confirm this draft. Please correct it and then let me confirm."
            )
        if kind == "stop" and "###STOP###" not in content:
            content = "###STOP###"
        if kind == "message" and not content:
            content = "Please continue."
        return UserObservation(
            kind=kind,  # type: ignore[arg-type]
            content=content,
            decision_reason=str(obj.get("decision_reason") or "llm_generated"),
            diagnostics={
                k: v for k, v in obj.items() if k not in {"kind", "content", "decision_reason"}
            },
        )

    def _check_repetition(
        self, observation: UserObservation, input: UserStepInput
    ) -> UserObservation:
        if observation.kind != "message":
            return observation

        content = observation.content
        response_words = _word_set(content)

        agent_words = _word_set(input.assistant_message)
        if (
            response_words
            and agent_words
            and _jaccard_similarity(response_words, agent_words) > 0.7
        ):
            return UserObservation(
                kind="message",
                content=_ECHO_FALLBACK,
                decision_reason="repetition_detected_agent_echo",
                diagnostics={**observation.diagnostics, "blocked_content": content},
            )

        if _is_substring_echo(content, input.assistant_message):
            return UserObservation(
                kind="message",
                content=_ECHO_FALLBACK,
                decision_reason="repetition_detected_substring_echo",
                diagnostics={**observation.diagnostics, "blocked_content": content},
            )

        for recent in self._recent_messages[-3:]:
            recent_words = _word_set(recent)
            if (
                response_words
                and recent_words
                and _jaccard_similarity(response_words, recent_words) > 0.7
            ):
                return UserObservation(
                    kind="message",
                    content=_ECHO_FALLBACK,
                    decision_reason="repetition_detected_self_repeat",
                    diagnostics={**observation.diagnostics, "blocked_content": content},
                )

        consecutive_agent_instructions = 0
        for recent in reversed(self._recent_messages[-2:]):
            if _looks_like_agent_instruction(recent):
                consecutive_agent_instructions += 1
            else:
                break
        if consecutive_agent_instructions >= 2 and _looks_like_agent_instruction(content):
            return UserObservation(
                kind="message",
                content=_ECHO_FALLBACK,
                decision_reason="repetition_detected_agent_instruction_pattern",
                diagnostics={**observation.diagnostics, "blocked_content": content},
            )

        return observation

    def _guard_observation(
        self, observation: UserObservation, input: UserStepInput
    ) -> UserObservation:
        return self._guard_observation_once(observation, input, allow_repair=True)

    def _guard_observation_once(
        self, observation: UserObservation, input: UserStepInput, *, allow_repair: bool
    ) -> UserObservation:
        if observation.kind == "stop" and not _has_terminal_submission(input.final_state):
            if allow_repair:
                repaired = self._repair_observation(
                    observation, input, reason="premature_stop_nonterminal"
                )
                return self._guard_observation_once(repaired, input, allow_repair=False)
            return UserObservation(
                kind="message",
                content="I have not completed this yet. Please continue helping me prepare the draft that needs confirmation, or explain why a handoff is necessary.",
                decision_reason="premature_stop_blocked",
                diagnostics={**observation.diagnostics, "blocked_kind": observation.kind},
            )

        if observation.kind == "reject" and not self._has_actionable_rejection(observation):
            if allow_repair:
                repaired = self._repair_observation(
                    observation, input, reason="reject_missing_actionable_correction"
                )
                return self._guard_observation_once(repaired, input, allow_repair=False)
            return UserObservation(
                kind="message",
                content=self._fallback_rejection_feedback(input),
                decision_reason="reject_replaced_with_actionable_feedback",
                diagnostics={**observation.diagnostics, "blocked_kind": observation.kind},
            )

        if observation.kind != "confirm":
            return observation
        if not input.latest_draft:
            return UserObservation(
                kind="message",
                content="Please first give me a draft I can confirm.",
                decision_reason="confirm_blocked_no_draft",
            )
        if not _asks_for_confirmation(input.assistant_message):
            return UserObservation(
                kind="message",
                content="Please continue; let me know when you have prepared the draft fields that need my confirmation.",
                decision_reason="confirm_blocked_no_confirmation_request",
            )
        if not self._draft_matches(input.latest_draft):
            return UserObservation(
                kind="message",
                content=draft_mismatch_feedback(self.task, input.latest_draft),
                decision_reason="confirm_blocked_draft_mismatch",
            )
        return UserObservation(
            kind="confirm",
            content=observation.content,
            draft_id=input.latest_draft["draft_id"],
            draft_hash=input.latest_draft["draft_hash"],
            decision_reason=observation.decision_reason or "llm_confirmed_matching_draft",
            diagnostics=observation.diagnostics,
        )

    def _repair_observation(
        self, observation: UserObservation, input: UserStepInput, *, reason: str
    ) -> UserObservation:
        prompt = (
            "Your previous simulated customer reply violates cxbench user-sim rules. "
            f"Violation: {reason}. "
            "If the task is not complete, do not stop. If the agent asks you to approve a matching visible draft, "
            "return kind=confirm with no content. If rejecting a draft, include reject_reason plus the exact field name "
            "and the correction needed, or say you cannot confirm and ask for handoff/defer. "
            f"Previous reply: {json.dumps(observation.decision_event(self.strategy), ensure_ascii=False, sort_keys=True)}"
        )
        repaired = self._generate_observation(
            [*self.messages, {"role": "user", "content": prompt}],
            input=input,
            context=f"user-sim repair: {reason}",
        )
        return UserObservation(
            kind=repaired.kind,
            content=repaired.content,
            draft_id=repaired.draft_id,
            draft_hash=repaired.draft_hash,
            decision_reason=repaired.decision_reason or f"repaired_{reason}",
            diagnostics={
                **repaired.diagnostics,
                "repair_reason": reason,
                "initial_kind": observation.kind,
            },
        )

    def _has_actionable_rejection(self, observation: UserObservation) -> bool:
        correction = observation.diagnostics.get("correction")
        if isinstance(correction, dict) and correction.get("field"):
            return True
        content = _normalized_text(
            " ".join(
                part
                for part in (
                    observation.content,
                    str(observation.diagnostics.get("reject_reason") or ""),
                )
                if part
            )
        )
        if any(
            marker in content
            for marker in (
                "handoff",
                "defer",
                "cannot confirm",
                "unable to confirm",
                "transfer to agent",
                "unable to verify",
            )
        ):
            return True
        return any(
            _normalized_text(field_name) in content
            for field_name in self.task.confirmation_contract
        )

    def _fallback_rejection_feedback(self, input: UserStepInput) -> str:
        if input.latest_draft:
            return draft_mismatch_feedback(self.task, input.latest_draft)
        return "I cannot confirm the current content. Please first provide draft fields that can be confirmed, or transfer to a human agent."

    def _draft_matches(self, draft: Dict[str, Any]) -> bool:
        return DeterministicUserSim(self.task)._draft_matches(draft)

    def _append_user_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def _accumulate_cost(self, response: Dict[str, Any]) -> None:
        usage = response.get("_cxbench", {}).get("usage", {})
        if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
            self.total_cost += float(usage["cost"])


class ReactUserSim(LLMUserSim):
    strategy = "react"

    def _output_contract(self, input: Optional[UserStepInput]) -> str:
        return (
            super()._output_contract(input)
            + " Also include key private_thought. private_thought is for trace diagnostics only and is never sent to the agent."
        )

    def _observation_from_json(self, obj: Dict[str, Any]) -> UserObservation:
        observation = super()._observation_from_json(obj)
        diagnostics = dict(observation.diagnostics)
        if "private_thought" in obj:
            diagnostics["private_thought"] = str(obj["private_thought"])
        return UserObservation(
            kind=observation.kind,
            content=observation.content,
            draft_id=observation.draft_id,
            draft_hash=observation.draft_hash,
            decision_reason=observation.decision_reason,
            diagnostics=diagnostics,
        )


class VerifyUserSim(LLMUserSim):
    strategy = "verify"

    def _generate_observation(
        self, messages: List[Dict[str, Any]], input: Optional[UserStepInput], context: str
    ) -> UserObservation:
        last_observation: Optional[UserObservation] = None
        verification_errors: List[str] = []
        for attempt in range(self.max_attempts):
            candidate = super()._generate_observation(messages, input, context)
            valid, reason = self._verify_candidate(messages, candidate)
            diagnostics = {
                **candidate.diagnostics,
                "verify_attempt": attempt + 1,
                "verify_reason": reason,
            }
            candidate = UserObservation(
                kind=candidate.kind,
                content=candidate.content,
                draft_id=candidate.draft_id,
                draft_hash=candidate.draft_hash,
                decision_reason=candidate.decision_reason,
                diagnostics=diagnostics,
            )
            if valid:
                return candidate
            verification_errors.append(reason)
            last_observation = candidate
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": f"Verifier rejected the last user reply: {reason}. Try again.",
                },
            ]
        return UserObservation(
            kind="message",
            content="Please continue, but do not assume information I have not provided.",
            decision_reason="verify_failed_max_attempts",
            diagnostics={
                "verification_errors": verification_errors,
                "last_content": last_observation.content if last_observation else "",
            },
        )

    def _verify_candidate(
        self, messages: List[Dict[str, Any]], candidate: UserObservation
    ) -> tuple[bool, str]:
        prompt = (
            "You are verifying a simulated customer reply. Return JSON with valid and reason. "
            "The reply is invalid if it reveals the complete hidden goal at once, invents facts, "
            "confirms without basis, or ignores the current transcript.\n"
            f"Transcript: {json.dumps(messages[-8:], ensure_ascii=False)}\n"
            f"Candidate: {candidate.content}"
        )
        response, obj = self.client.create_validated(
            [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._task_context_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            validator=lambda response: self.client.validate_json_object(
                response,
                required_keys=["valid", "reason"],
                context="user-sim verifier",
            ),
            validation_context="user-sim verifier",
        )
        self._accumulate_cost(response)
        return bool(obj.get("valid", False)), str(obj.get("reason", ""))


class ReflectionUserSim(VerifyUserSim):
    strategy = "reflection"

    def _generate_observation(
        self, messages: List[Dict[str, Any]], input: Optional[UserStepInput], context: str
    ) -> UserObservation:
        candidate = LLMUserSim._generate_observation(self, messages, input, context)
        valid, reason = self._verify_candidate(messages, candidate)
        if valid:
            return candidate
        reflection = self._reflect(messages, candidate, reason)
        repaired_messages = [*messages, {"role": "user", "content": reflection}]
        repaired = LLMUserSim._generate_observation(self, repaired_messages, input, context)
        valid_after, reason_after = self._verify_candidate(messages, repaired)
        diagnostics = {
            **repaired.diagnostics,
            "reflection": reflection,
            "initial_content": candidate.content,
            "initial_verify_reason": reason,
            "verify_after_reflection": valid_after,
            "verify_after_reflection_reason": reason_after,
        }
        if valid_after:
            return UserObservation(
                kind=repaired.kind,
                content=repaired.content,
                draft_id=repaired.draft_id,
                draft_hash=repaired.draft_hash,
                decision_reason=repaired.decision_reason,
                diagnostics=diagnostics,
            )
        return UserObservation(
            kind="message",
            content="Please continue, but first clarify what you need me to confirm.",
            decision_reason="reflection_failed_verification",
            diagnostics=diagnostics,
        )

    def _reflect(
        self, messages: List[Dict[str, Any]], candidate: UserObservation, reason: str
    ) -> str:
        prompt = (
            "The simulated customer reply was rejected. Return JSON with reflection and revised_instruction.\n"
            f"Reason: {reason}\n"
            f"Candidate: {candidate.content}\n"
            f"Transcript: {json.dumps(messages[-8:], ensure_ascii=False)}"
        )
        response, obj = self.client.create_validated(
            [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._task_context_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            validator=lambda response: self.client.validate_json_object(
                response,
                required_keys=["reflection", "revised_instruction"],
                context="user-sim reflection",
            ),
            validation_context="user-sim reflection",
        )
        self._accumulate_cost(response)
        return str(
            obj.get("revised_instruction")
            or obj.get("reflection")
            or "Generate a safer customer reply."
        )


def build_user_sim(
    strategy: str, task: TaskSpec, model: Optional[str] = None, max_attempts: int = 3
) -> BaseUserSim:
    normalized = strategy.lower().replace("_", "-")
    if normalized == "deterministic":
        return DeterministicUserSim(task)
    if normalized == "human":
        return HumanUserSim(task)
    if normalized in {"llm", "react", "verify", "reflection"} and not model:
        raise ValueError(f"user-sim strategy {strategy!r} requires --user-model openai:<model>")
    if normalized == "llm":
        return LLMUserSim(task, model=model or "", max_attempts=max_attempts)
    if normalized == "react":
        return ReactUserSim(task, model=model or "", max_attempts=max_attempts)
    if normalized == "verify":
        return VerifyUserSim(task, model=model or "", max_attempts=max_attempts)
    if normalized == "reflection":
        return ReflectionUserSim(task, model=model or "", max_attempts=max_attempts)
    raise ValueError(
        "unknown user-sim strategy; expected deterministic, human, llm, react, verify, or reflection"
    )


def _has_terminal_submission(final_state: Dict[str, Any]) -> bool:
    terminal_keys = (
        "disputes",
        "card_control_changes",
        "payment_arrangements",
        "regulatory_holds",
        "utility_disputes",
        "appeal_cases",
        "hearing_cases",
        "port_out_requests",
        "sim_changes",
        "support_handoffs",
        "fraud_documents",
        "survivor_separation_requests",
        "abuser_access_restrictions",
        "confidential_communications",
        "secure_self_service_explanations",
    )
    return any(bool(final_state.get(key)) for key in terminal_keys)
