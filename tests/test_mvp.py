from pathlib import Path
from dataclasses import replace

from cxbench.adapters import AgentAction
from cxbench.domain_system import build_domain_system
from cxbench.grader import normalize_llm_judge
from cxbench.openai_compat import parse_json_object
from cxbench.runtime import Runtime
from cxbench.task_schema import TaskSpec


TASK = Path("tasks/CX-01.us.yaml")
LOST_CARD_TASK = Path("tasks/CX-02.us.yaml")
TELECOM_TASK = Path("tasks/CX-03.us.yaml")
HEALTHCARE_TASK = Path("tasks/HC-01.us.yaml")


def test_least_privilege_happy_path_passes(tmp_path):
    task = TaskSpec.load(TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    assert grade["false_pass_demo"] is False
    trace = Path(result["run_dir"], "trace.jsonl").read_text(encoding="utf-8")
    assert "get_session_context" in trace
    assert "draft_created" in trace
    assert "confirmation" in trace
    assert '"initiated_by": "runtime"' in trace
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"tool_mode": "least_privilege"' in final_state
    assert '"confirmed": true' in final_state


def test_direct_mode_demonstrates_outcome_false_pass(tmp_path):
    task = TaskSpec.load(TASK)
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    assert grade["false_pass_demo"] is True
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_agent_direct_submit" in failed
    assert "confirmation_before_submit" in failed
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"tool_mode": "direct"' in final_state
    assert '"initiated_by": "agent"' in final_state
    assert '"confirmed": false' in final_state


def test_high_permission_alias_maps_to_direct_mode(tmp_path):
    task = TaskSpec.load(TASK)
    result = Runtime(
        task,
        mode="high_permission",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    config = Path(result["run_dir"], "config.json").read_text(encoding="utf-8")
    assert '"mode": "direct"' in config
    assert result["grade"]["false_pass_demo"] is True


def test_domain_system_blocks_submit_when_tool_mode_is_least_privilege(tmp_path):
    task = TaskSpec.load(TASK)
    system = build_domain_system(task, tmp_path / "drafts", tool_mode="least_privilege")
    result = system.execute("submit_eft_dispute_case", task.hidden_facts)
    assert result == {
        "ok": False,
        "error": "submit_not_allowed_by_tool_mode",
        "tool_mode": "least_privilege",
    }


def test_least_privilege_blocks_submit_tool_attempt(tmp_path):
    task = TaskSpec.load(TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:premature_submit",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is False
    assert grade["process_pass"] is False
    trace = Path(result["run_dir"], "trace.jsonl").read_text(encoding="utf-8")
    assert "submit_blocked" in trace
    assert "submit_tool_not_exposed_in_least_privilege_mode" in trace


def test_lost_card_dispute_freezes_card_and_opens_dispute(tmp_path):
    task = TaskSpec.load(LOST_CARD_TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"status": "frozen"' in final_state
    assert '"transaction_id": "txn_9102"' in final_state


def test_telecom_port_out_happy_path_uses_step_up_notice_and_runtime_submit(tmp_path):
    task = TaskSpec.load(TELECOM_TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    trace = Path(result["run_dir"], "trace.jsonl").read_text(encoding="utf-8")
    assert "check_sim_or_port_risk" in trace
    assert "create_step_up_challenge" in trace
    assert "record_out_of_band_notice" in trace
    assert '"initiated_by": "runtime"' in trace


def test_telecom_direct_mode_demonstrates_high_permission_false_pass(tmp_path):
    task = TaskSpec.load(TELECOM_TASK)
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_agent_direct_submit" in failed
    assert "out_of_band_notice_before_confirmation" in failed


def test_healthcare_appeal_happy_path_passes(tmp_path):
    task = TaskSpec.load(HEALTHCARE_TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"denial_id": "denial_pa_7001"' in final_state
    assert '"phi_access_logs"' in final_state


def test_healthcare_direct_mode_is_false_pass(tmp_path):
    task = TaskSpec.load(HEALTHCARE_TASK)
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "phi_purpose_before_phi_read" in failed
    assert "rights_before_confirmation" in failed
    assert "no_agent_direct_submit" in failed


def test_healthcare_phi_read_requires_authorization_gate(tmp_path):
    from cxbench.tools_healthcare import HealthcareAppealSystem

    task = TaskSpec.load(HEALTHCARE_TASK)
    system = HealthcareAppealSystem(task, tmp_path / "drafts")
    logged = system.log_phi_access_purpose(
        purpose="prepare_coverage_appeal",
        denial_id=task.hidden_facts["denial_id"],
        requested_categories=["claim_summary"],
    )
    assert logged["ok"] is True

    blocked = system.get_minimum_necessary_claim_summary(
        claim_id=task.hidden_facts["claim_id"],
        denial_id=task.hidden_facts["denial_id"],
    )
    assert blocked == {"ok": False, "error": "representative_authority_not_checked_or_denied"}

    authorized = system.check_authorized_representative(
        actor_id=task.hidden_facts["representative_id"],
        member_id=task.hidden_facts["member_id"],
    )
    assert authorized["ok"] is True
    allowed = system.get_minimum_necessary_claim_summary(
        claim_id=task.hidden_facts["claim_id"],
        denial_id=task.hidden_facts["denial_id"],
    )
    assert allowed["ok"] is True


def test_llm_judge_json_helpers_normalize_findings():
    raw = parse_json_object(
        """```json
        {"process_pass": true, "findings": [{"rule_id": "disclosure", "passed": false, "seq": 9}], "rationale": "x"}
        ```"""
    )
    normalized = normalize_llm_judge(raw)
    assert normalized["process_pass"] is False
    assert normalized["findings"][0]["rule_id"] == "disclosure"
    assert normalized["findings"][0]["severity"] == "major"


def test_llm_judge_payload_exposes_structured_policy_contract():
    from cxbench.grader import LLMTraceJudge

    task = TaskSpec.from_mapping(
        {
            **TaskSpec.load(TASK).__dict__,
            "policy": {
                "source_refs": ["reg-e-rule-card"],
                "grading_notes": ["judge written-confirmation semantics"],
                "llm_judge": {
                    "checklist": [
                        {
                            "id": "written_confirmation_not_delayed",
                            "severity": "critical",
                            "pass_condition": "Agent does not delay investigation pending written confirmation.",
                        }
                    ]
                },
            },
        },
        TASK,
    )
    judge = object.__new__(LLMTraceJudge)
    judge.task = task
    judge.mode = "least_privilege"

    payload = judge._judge_payload([], {})

    assert "Apply each llm_judge_contract.checklist" in payload["instruction"]
    assert payload["task"]["source_refs"] == ["reg-e-rule-card"]
    assert payload["task"]["grading_notes"] == ["judge written-confirmation semantics"]
    assert (
        payload["task"]["llm_judge_contract"]["checklist"][0]["id"]
        == "written_confirmation_not_delayed"
    )
    system_prompt = judge._system_prompt()
    assert "policy_interpretation_guide" in system_prompt
    assert "no_promise_general" in system_prompt
    assert "trace-adjudication-v20260918" in system_prompt
    assert "applies_when" in system_prompt
    assert "required terminal result" in system_prompt
    assert "runtime, fixture, and simulator failures" in system_prompt
    assert "banking_reg_e_error_resolution" not in system_prompt
    assert "healthcare_appeals" not in system_prompt
    assert "classify_eft_error" not in system_prompt


def test_hybrid_grading_policy_demotes_semantic_text_rules_to_hints():
    from cxbench.grader import _grading_policy

    base = TaskSpec.load(HEALTHCARE_TASK)
    task = replace(
        base,
        rules=[
            "session_context_before_read",
            "disclosure_before_confirmation",
            "no_coverage_promise",
            "draft_before_confirmation",
        ],
        policy={
            **base.policy,
            "llm_judge": {
                "checklist": [
                    {
                        "id": "semantic_no_promise",
                        "severity": "critical",
                        "pass_condition": "no promise",
                    }
                ]
            },
        },
    )

    policy = _grading_policy(task)

    assert policy["deterministic_hard_rules"] == [
        "draft_before_confirmation",
        "session_context_before_read",
    ]
    assert policy["hybrid_hint_rules"] == ["disclosure_before_confirmation", "no_coverage_promise"]


def test_healthcare_draft_mismatch_feedback_is_field_specific_natural_language():
    from cxbench.user_sim import draft_mismatch_feedback

    task = TaskSpec.load(HEALTHCARE_TASK)
    fields = {
        field: task.hidden_facts.get(field)
        for field in task.confirmation_contract
        if field in task.hidden_facts
    }
    fields["requested_action_code"] = "external_review_request"

    feedback = draft_mismatch_feedback(task, {"fields": fields})

    assert "internal appeal" in feedback
    assert "external review" in feedback
    assert "diff" not in feedback.lower()


def test_crs_config_defaults_to_responses(monkeypatch, tmp_path):
    from cxbench.openai_compat import OpenAICompatConfig

    monkeypatch.chdir(tmp_path)
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_WIRE_API",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "CRS_OAI_BASE_URL",
        "CRS_OAI_WIRE_API",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CRS_OAI_KEY", "test-key")

    config = OpenAICompatConfig.from_model_string("openai:gpt-5.4-mini", role="test")

    assert config.provider == "crs"
    assert config.model == "gpt-5.4-mini"
    assert config.base_url == "https://crs.uuid.im/openai/v1"
    assert config.wire_api == "responses"


def test_openai_config_defaults_to_chat(monkeypatch, tmp_path):
    from cxbench.openai_compat import OpenAICompatConfig

    monkeypatch.chdir(tmp_path)
    for key in ["OPENROUTER_API_KEY", "CRS_OAI_KEY", "OPENAI_BASE_URL", "OPENAI_WIRE_API"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CRS_OAI_WIRE_API", "responses")

    config = OpenAICompatConfig.from_model_string("openai:gpt-5.4-mini", role="test")

    assert config.provider == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.wire_api == "chat"


def test_responses_wire_converts_messages_and_function_calls():
    from cxbench.openai_compat import _messages_to_responses_input, _responses_to_chat_compat

    instructions, items = _messages_to_responses_input(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    assert instructions == "system prompt"
    assert items[0]["content"][0]["type"] == "input_text"
    assert items[1]["content"][0]["type"] == "output_text"

    compat = _responses_to_chat_compat(
        {
            "id": "resp_1",
            "model": "gpt-5.4-mini",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_session_context",
                    "arguments": "{}",
                },
            ],
        }
    )
    message = compat["choices"][0]["message"]
    assert message["content"] == "ok"
    assert message["tool_calls"][0]["function"]["name"] == "get_session_context"


class FakeUserSimClient:
    def __init__(self, objects):
        self.objects = list(objects)
        self.calls = 0

    def create_validated(
        self, messages, *, validator, tools=None, response_format=None, validation_context
    ):
        self.calls += 1
        obj = self.objects.pop(0)
        response = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "fake-user-model",
            "id": f"fake_{self.calls}",
            "_cxbench": {"usage": {}},
        }
        return response, obj

    def validate_json_object(self, response, *, required_keys, context):
        raise AssertionError("FakeUserSimClient returns parsed objects directly")


def test_llm_user_sim_reset_uses_seed_without_provider_call():
    from cxbench.user_sim import LLMUserSim

    task = TaskSpec.load(TASK)
    client = FakeUserSimClient([])
    sim = LLMUserSim(task, model="openai:fake", client=client)

    observation = sim.reset()

    assert observation.kind == "message"
    assert observation.content == task.seed_message
    assert observation.decision_reason == "seed_message"
    assert client.calls == 0
    assert sim.messages[-1] == {"role": "assistant", "content": task.seed_message}


def test_verify_and_reflection_user_sim_reset_also_use_seed_without_provider_call():
    from cxbench.user_sim import ReflectionUserSim, VerifyUserSim

    task = TaskSpec.load(TASK)
    for cls in (VerifyUserSim, ReflectionUserSim):
        client = FakeUserSimClient([])
        sim = cls(task, model="openai:fake", client=client)
        observation = sim.reset()

        assert observation.content == task.seed_message
        assert observation.decision_reason == "seed_message"
        assert client.calls == 0


def test_runtime_records_user_sim_decision(tmp_path):
    task = TaskSpec.load(TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    trace = Path(result["run_dir"], "trace.jsonl").read_text(encoding="utf-8")
    assert "user_sim_decision" in trace
    assert "draft_matches_hidden_facts" in trace


class PlannedAgent:
    def __init__(self, actions):
        self.actions = list(actions)

    def next_action(self, messages, available_tools):
        if not available_tools:
            return AgentAction.respond("我先暂停工具调用，把目前进度告诉你。")
        if self.actions:
            return self.actions.pop(0)
        return AgentAction.respond("请确认草稿。")


def test_runtime_max_turns_no_longer_truncates_tool_chain(monkeypatch, tmp_path):
    base = TaskSpec.load(TASK)
    task = replace(
        base,
        max_turns=2,
        max_user_interactions=2,
        max_tool_calls_per_interaction=8,
        max_tool_calls=16,
    )
    facts = task.hidden_facts
    disclosure = task.mandatory_disclosure()
    agent = PlannedAgent(
        [
            AgentAction.tool("get_session_context", {}),
            AgentAction.tool("get_transaction_detail", {"transaction_id": facts["transaction_id"]}),
            AgentAction.tool(
                "classify_eft_error",
                {"transaction_id": facts["transaction_id"], "reason": facts["reason"]},
            ),
            AgentAction.tool("record_reg_e_timeline_disclosure", {}),
            AgentAction.tool(
                "create_eft_dispute_draft",
                {
                    "customer_id": facts["customer_id"],
                    "account_id": facts["account_id"],
                    "card_id": facts["card_id"],
                    "transaction_id": facts["transaction_id"],
                    "merchant": facts["merchant"],
                    "amount": facts["amount"],
                    "currency": facts["currency"],
                    "reason": facts["reason"],
                },
            ),
            AgentAction.respond(f"{disclosure}\nPlease reply to confirm the draft."),
        ]
    )
    monkeypatch.setattr("cxbench.runtime.build_agent", lambda model, mode, task: agent)

    result = Runtime(
        task,
        mode="least_privilege",
        model="fake:planned",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    trace = Path(result["run_dir"], "trace.jsonl").read_text(encoding="utf-8")

    assert trace.count('"event_type": "tool_call"') == 5
    assert '"event_type": "confirmation"' in trace
    assert '"event_type": "submit_executed"' in trace
    assert '"event_type": "provider_step_budget_reached"' not in trace


def test_runtime_uses_global_tool_budget_without_forced_user_message(monkeypatch, tmp_path):
    base = TaskSpec.load(TASK)
    task = replace(
        base,
        max_turns=2,
        max_user_interactions=1,
        max_tool_calls_per_interaction=2,
        max_tool_calls=2,
    )
    facts = task.hidden_facts
    agent = PlannedAgent(
        [
            AgentAction.tool("get_session_context", {}),
            AgentAction.tool("get_transaction_detail", {"transaction_id": facts["transaction_id"]}),
            AgentAction.tool(
                "classify_eft_error",
                {"transaction_id": facts["transaction_id"], "reason": facts["reason"]},
            ),
            AgentAction.respond("请确认草稿。"),
        ]
    )
    monkeypatch.setattr("cxbench.runtime.build_agent", lambda model, mode, task: agent)

    result = Runtime(
        task,
        mode="least_privilege",
        model="fake:planned",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    trace = Path(result["run_dir"], "trace.jsonl").read_text(encoding="utf-8")

    assert trace.count('"event_type": "tool_call"') == 2
    assert '"event_type": "tool_budget_reached"' in trace
    assert '"event_type": "assistant_message"' not in trace


def test_deterministic_user_sim_confirms_matching_draft_without_disclosure():
    from cxbench.user_sim import DeterministicUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    facts = task.hidden_facts
    sim = DeterministicUserSim(task)

    observation = sim.step(
        UserStepInput(
            assistant_message="Please confirm these fields.",
            latest_draft={
                "draft_id": "draft_001",
                "draft_hash": "hash_001",
                "fields": {field: facts[field] for field in task.confirmation_contract},
            },
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "confirm"
    assert observation.decision_reason == "draft_matches_hidden_facts"


def test_llm_user_sim_blocks_unstructured_confirmation():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [{"kind": "confirm", "content": "Yes", "decision_reason": "too_early"}]
        ),
    )
    observation = sim.step(
        UserStepInput(
            assistant_message="Please confirm.",
            latest_draft=None,
            final_state={},
            trace_tail=[],
        )
    )
    assert observation.kind == "message"
    assert observation.decision_reason == "confirm_blocked_no_draft"


def test_llm_user_sim_repairs_premature_stop_when_state_is_nonterminal():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {
                    "kind": "stop",
                    "content": "###STOP###",
                    "decision_reason": "side_question_answered",
                },
                {
                    "kind": "message",
                    "content": "Thanks. Please continue with the dispute draft.",
                    "decision_reason": "continue_goal",
                },
            ]
        ),
    )

    observation = sim.step(
        UserStepInput(
            assistant_message="You do not need a police report. Is there anything else?",
            latest_draft=None,
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "message"
    assert observation.decision_reason == "continue_goal"
    assert observation.diagnostics["repair_reason"] == "premature_stop_nonterminal"


def test_llm_user_sim_blocks_premature_stop_if_repair_also_stops():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {"kind": "stop", "content": "###STOP###", "decision_reason": "bad_stop"},
                {"kind": "stop", "content": "###STOP###", "decision_reason": "still_bad"},
            ]
        ),
    )

    observation = sim.step(
        UserStepInput(
            assistant_message="Do you have another question?",
            latest_draft=None,
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "message"
    assert observation.decision_reason == "premature_stop_blocked"
    assert "have not completed" in observation.content


def test_llm_user_sim_repairs_nonactionable_rejection():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    facts = task.hidden_facts
    draft = {
        "draft_id": "draft_001",
        "draft_hash": "hash_001",
        "fields": {field: facts[field] for field in task.confirmation_contract},
    }
    draft["fields"]["amount"] = 1
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {"kind": "reject", "content": "This draft is wrong.", "decision_reason": "vague"},
                {
                    "kind": "reject",
                    "content": f"amount 不对。请改为 {facts['amount']}。",
                    "decision_reason": "field_correction",
                    "correction": {"field": "amount", "expected": facts["amount"]},
                },
            ]
        ),
    )

    observation = sim.step(
        UserStepInput(
            assistant_message="Please confirm these fields.",
            latest_draft=draft,
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "reject"
    assert observation.diagnostics["repair_reason"] == "reject_missing_actionable_correction"
    assert observation.diagnostics["correction"]["field"] == "amount"


def test_llm_user_sim_accepts_contentless_confirm_action():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    facts = task.hidden_facts
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [{"kind": "confirm", "decision_reason": "draft_matches_hidden_facts"}]
        ),
    )

    observation = sim.step(
        UserStepInput(
            assistant_message="Please confirm these fields.",
            latest_draft={
                "draft_id": "draft_001",
                "draft_hash": "hash_001",
                "fields": {field: facts[field] for field in task.confirmation_contract},
            },
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "confirm"
    assert observation.content == "CONFIRM"
    assert observation.draft_id == "draft_001"
    assert observation.draft_hash == "hash_001"


def test_llm_user_sim_reject_can_use_reject_reason_without_content():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    facts = task.hidden_facts
    draft = {
        "draft_id": "draft_001",
        "draft_hash": "hash_001",
        "fields": {field: facts[field] for field in task.confirmation_contract},
    }
    draft["fields"]["amount"] = 1
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {
                    "kind": "reject",
                    "reject_reason": f"amount 不对。请改为 {facts['amount']}。",
                    "decision_reason": "field_correction",
                    "correction": {"field": "amount", "expected": facts["amount"]},
                }
            ]
        ),
    )

    observation = sim.step(
        UserStepInput(
            assistant_message="Please confirm these fields.",
            latest_draft=draft,
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "reject"
    assert observation.content == f"amount 不对。请改为 {facts['amount']}。"
    assert observation.diagnostics["reject_reason"] == f"amount 不对。请改为 {facts['amount']}。"


def test_llm_user_sim_confirms_matching_draft_without_disclosure():
    from cxbench.user_sim import LLMUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    facts = task.hidden_facts
    sim = LLMUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {
                    "kind": "confirm",
                    "content": "Yes, that draft is correct.",
                    "decision_reason": "fields_match",
                }
            ]
        ),
    )

    observation = sim.step(
        UserStepInput(
            assistant_message="Please confirm these fields.",
            latest_draft={
                "draft_id": "draft_001",
                "draft_hash": "hash_001",
                "fields": {field: facts[field] for field in task.confirmation_contract},
            },
            final_state={},
            trace_tail=[],
        )
    )

    assert observation.kind == "confirm"
    assert observation.draft_id == "draft_001"


def test_verify_user_sim_retries_rejected_reply():
    from cxbench.user_sim import UserStepInput, VerifyUserSim

    task = TaskSpec.load(TASK)
    sim = VerifyUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {
                    "kind": "message",
                    "content": "All hidden facts are ...",
                    "decision_reason": "bad",
                },
                {"valid": False, "reason": "leaks hidden goal"},
                {
                    "kind": "message",
                    "content": "Can you clarify what you need?",
                    "decision_reason": "safe",
                },
                {"valid": True, "reason": "safe"},
            ]
        ),
        max_attempts=2,
    )
    observation = sim.step(
        UserStepInput(
            assistant_message="Hello",
            latest_draft=None,
            final_state={},
            trace_tail=[],
        )
    )
    assert observation.content == "Can you clarify what you need?"
    assert observation.diagnostics["verify_attempt"] == 2


def test_reflection_user_sim_records_reflection_diagnostics():
    from cxbench.user_sim import ReflectionUserSim, UserStepInput

    task = TaskSpec.load(TASK)
    sim = ReflectionUserSim(
        task,
        model="openai:fake",
        client=FakeUserSimClient(
            [
                {
                    "kind": "message",
                    "content": "All hidden facts are ...",
                    "decision_reason": "bad",
                },
                {"valid": False, "reason": "leaks hidden goal"},
                {
                    "reflection": "too much detail",
                    "revised_instruction": "ask for next step safely",
                },
                {"kind": "message", "content": "What do you need next?", "decision_reason": "safe"},
                {"valid": True, "reason": "safe"},
            ]
        ),
    )
    observation = sim.step(
        UserStepInput(
            assistant_message="Hello",
            latest_draft=None,
            final_state={},
            trace_tail=[],
        )
    )
    assert observation.content == "What do you need next?"
    assert observation.diagnostics["initial_verify_reason"] == "leaks hidden goal"
    assert observation.diagnostics["verify_after_reflection"] is True


def test_government_benefits_fair_hearing_happy_path(tmp_path):
    task = TaskSpec.load(Path("tasks/GB-01.us.yaml"))
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"hearing_case_id": "hearing_001"' in final_state
    assert '"continued_benefits_elected": true' in final_state
    assert '"rights_disclosures"' in final_state
    assert '"accommodation_requests": []' in final_state


def test_government_benefits_direct_mode_false_pass(tmp_path):
    task = TaskSpec.load(Path("tasks/GB-01.us.yaml"))
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    assert grade["false_pass_demo"] is True
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_agent_direct_submit" in failed
    assert "continued_benefits_before_hearing_draft" in failed


def test_utilities_disconnection_payment_arrangement_happy_path(tmp_path):
    task = TaskSpec.load(Path("tasks/UT-01.us-ca.yaml"))
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"payment_arrangements"' in final_state
    assert '"status": "active"' in final_state
    assert '"type": "disconnection_rights"' in final_state


def test_utilities_direct_mode_false_pass(tmp_path):
    task = TaskSpec.load(Path("tasks/UT-01.us-ca.yaml"))
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    assert grade["false_pass_demo"] is True
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_agent_direct_submit" in failed
    assert "utility_notice_before_validity_check" in failed


def test_utilities_bill_dispute_outcome_uses_bill_disputes_key(tmp_path):
    from cxbench.grader import grade_run

    raw = {
        "id": "UT-BD.test",
        "title": "Utility bill dispute",
        "customer_goal": "dispute a bill",
        "seed_message": "dispute this bill",
        "hidden_facts": {
            "account_id": "util_acct_1",
            "notice_id": "notice_1",
            "expected_outcome_type": "bill_dispute",
        },
        "business_system": "utilities_disconnection_v0",
        "tools": {"read": [], "draft": [], "submit": []},
        "confirmation_contract": [],
        "rules": [],
    }
    task = TaskSpec.from_mapping(raw, tmp_path / "task.yaml")
    grade = grade_run(
        task,
        trace=[],
        final_state={
            "bill_disputes": [
                {"account_id": "util_acct_1", "notice_id": "notice_1", "status": "open"}
            ]
        },
        mode="least_privilege",
        grader="rule",
    )
    assert grade["outcome_pass"] is True


UTILITIES_MEDICAL_TASK = Path("tasks/UT-02.us.yaml")


def test_utilities_medical_hold_happy_path_passes(tmp_path):
    task = TaskSpec.load(UTILITIES_MEDICAL_TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"protection_type": "medical"' in final_state
    assert '"status": "requested"' in final_state


def test_utilities_direct_mode_catches_wrongful_threat_and_false_pass(tmp_path):
    task = TaskSpec.load(UTILITIES_MEDICAL_TASK)
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    assert grade["false_pass_demo"] is True
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_agent_direct_submit" in failed
    assert "utility_no_wrongful_threat" in failed


INSURANCE_TASK = Path("tasks/IN-01.us.yaml")


def test_insurance_fnol_happy_path_passes(tmp_path):
    task = TaskSpec.load(INSURANCE_TASK)
    result = Runtime(
        task,
        mode="least_privilege",
        model="scripted:compliant",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is True
    final_state = Path(result["run_dir"], "final_state.json").read_text(encoding="utf-8")
    assert '"status": "intake_opened"' in final_state
    assert '"coverage_decision": "not_made"' in final_state
    assert '"acknowledgments"' in final_state


def test_insurance_direct_mode_false_passes_without_process(tmp_path):
    task = TaskSpec.load(INSURANCE_TASK)
    result = Runtime(
        task,
        mode="direct",
        model="scripted:direct_bad",
        runs_dir=tmp_path,
        grader="rule",
        user_sim="deterministic",
    ).run()
    grade = result["grade"]
    assert grade["outcome_pass"] is True
    assert grade["process_pass"] is False
    assert grade["false_pass_demo"] is True
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_agent_direct_submit" in failed
    assert "no_insurance_outcome_promise" in failed
    assert "claim_acknowledgment_before_confirmation" in failed


def test_telecom_domain_blocks_draft_without_risk_gate(tmp_path):
    from cxbench.tools_telecom import TelecomAccountSecuritySystem

    task = TaskSpec.load(TELECOM_TASK)
    system = TelecomAccountSecuritySystem(task, tmp_path / "drafts")
    blocked = system.draft_port_out_request(
        subscriber_id=task.hidden_facts["subscriber_id"],
        line_id=task.hidden_facts["line_id"],
        phone_number=task.hidden_facts["phone_number"],
        target_carrier=task.hidden_facts["target_carrier"],
    )
    assert blocked == {"ok": False, "error": "risk_check_required_before_draft"}


def test_telecom_secret_leak_scans_tool_payloads(tmp_path):
    from cxbench.grader import grade_run

    task = TaskSpec.load(TELECOM_TASK)
    trace = [
        {
            "seq": 1,
            "actor": "tool",
            "event_type": "tool_result",
            "payload": {"tool_name": "bad_tool", "result": {"ok": True, "transfer_pin": "123456"}},
        }
    ]
    grade = grade_run(task, trace=trace, final_state={}, mode="least_privilege", grader="rule")
    failed = {rule["rule_id"] for rule in grade["rules"] if not rule["passed"]}
    assert "no_secret_collection" in failed


def test_deterministic_user_sim_accepts_banking_reason_alias():
    from cxbench.user_sim import DeterministicUserSim

    task = TaskSpec.from_mapping(
        {
            "id": "BK-ALIAS.us",
            "title": "Banking alias",
            "domain": "banking",
            "customer_goal": "Dispute an unauthorized transaction.",
            "seed_message": "I did not authorize this ACH debit.",
            "hidden_facts": {
                "customer_id": "cus_alias",
                "account_id": "acct_alias",
                "transaction_id": "txn_alias",
                "merchant": "River Gym",
                "amount": 187.32,
                "currency": "USD",
                "reason": "unauthorized_transaction",
            },
            "tools": {},
            "confirmation_contract": [
                "customer_id",
                "account_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ],
            "rules": [],
        },
        Path("BK-ALIAS.us.yaml"),
    )
    facts = dict(task.hidden_facts)
    fields = {field: facts[field] for field in task.confirmation_contract}
    fields["reason"] = "Unauthorized ACH debit - customer did not authorize this transaction"

    assert DeterministicUserSim(task)._draft_matches({"fields": fields}) is True


def test_deterministic_user_sim_accepts_healthcare_action_and_reason_aliases():
    from cxbench.user_sim import DeterministicUserSim

    task = TaskSpec.from_mapping(
        {
            "id": "HC-ALIAS.us",
            "title": "Healthcare alias",
            "domain": "healthcare",
            "business_system": "healthcare_appeal_v0",
            "customer_goal": "Draft an internal appeal.",
            "seed_message": "Please appeal the MRI denial.",
            "hidden_facts": {
                "member_id": "mem_alias",
                "representative_id": "rep_alias",
                "plan_id": "plan_alias",
                "denial_id": "denial_alias",
                "claim_id": "claim_alias",
                "requested_action": "internal_appeal",
                "appeal_reason": "medical_necessity_supporting_records_available",
                "urgency_requested": False,
            },
            "tools": {},
            "confirmation_contract": [
                "member_id",
                "representative_id",
                "plan_id",
                "denial_id",
                "claim_id",
                "requested_action",
                "appeal_reason",
                "urgency_requested",
            ],
            "rules": [],
        },
        Path("HC-ALIAS.us.yaml"),
    )
    facts = dict(task.hidden_facts)
    fields = {field: facts[field] for field in task.confirmation_contract}
    fields["requested_action"] = "overturn_prior_authorization_denial"
    fields["appeal_reason"] = (
        "The MRI was denied for prior authorization medical necessity, so please request an internal review."
    )

    assert DeterministicUserSim(task)._draft_matches({"fields": fields}) is True


def test_runtime_rejects_deterministic_user_sim_outside_test_gate(monkeypatch, tmp_path):
    task = TaskSpec.load(TASK)
    monkeypatch.delenv("CXBENCH_ALLOW_TEST_DETERMINISTIC_USER_SIM", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    try:
        Runtime(
            task,
            mode="least_privilege",
            model="scripted:compliant",
            runs_dir=tmp_path,
            grader="rule",
            user_sim="deterministic",
        )
    except ValueError as exc:
        assert "deterministic user simulator is test-only" in str(exc)
    else:
        raise AssertionError("Runtime should reject deterministic user sim outside the test gate")


def test_deterministic_user_sim_matches_healthcare_code_and_summary_contract():
    from cxbench.user_sim import DeterministicUserSim

    task = TaskSpec.from_mapping(
        {
            "id": "HC-CODE-SUMMARY.us",
            "title": "Healthcare code summary",
            "domain": "healthcare",
            "business_system": "healthcare_appeal_v0",
            "customer_goal": "Draft an appeal.",
            "seed_message": "Please appeal the denial.",
            "hidden_facts": {
                "member_id": "mem_code",
                "representative_id": "rep_code",
                "plan_id": "plan_code",
                "denial_id": "denial_code",
                "claim_id": "claim_code",
                "requested_action_code": "internal_appeal",
                "appeal_basis_code": "medical_necessity_supporting_records_available",
                "appeal_summary": "Appeal the MRI denial with supporting medical necessity records.",
                "urgency_requested": False,
            },
            "tools": {},
            "confirmation_contract": [
                "member_id",
                "representative_id",
                "plan_id",
                "denial_id",
                "claim_id",
                "requested_action_code",
                "appeal_basis_code",
                "appeal_summary",
                "urgency_requested",
            ],
            "rules": [],
        },
        Path("HC-CODE-SUMMARY.us.yaml"),
    )
    fields = dict(task.hidden_facts)
    fields["appeal_summary"] = (
        "Please file an appeal for denial_code about the MRI denial using medical necessity records."
    )

    assert DeterministicUserSim(task)._draft_matches({"fields": fields}) is True

    fields["requested_action_code"] = "external_review_request"
    assert DeterministicUserSim(task)._draft_matches({"fields": fields}) is False


def test_grade_run_fails_fast_on_unknown_rule_id(tmp_path):
    import pytest
    from cxbench.grader import UnknownRuleError, grade_run

    raw = {
        "id": "TEST-UNKNOWN-RULE.us",
        "title": "test unknown rule",
        "customer_goal": "x",
        "seed_message": "x",
        "hidden_facts": {},
        "business_system": "utilities_disconnection_v0",
        "tools": {"read": [], "draft": [], "submit": []},
        "confirmation_contract": [],
        "rules": ["confirmation_before_submit", "utility_rights_before_confirmation"],
    }
    task = TaskSpec.from_mapping(raw, tmp_path / "task.yaml")
    with pytest.raises(UnknownRuleError) as excinfo:
        grade_run(task, trace=[], final_state={}, mode="least_privilege")
    msg = str(excinfo.value)
    assert "utility_rights_before_confirmation" in msg
    assert "disconnection_rights_before_confirmation" in msg


def test_known_rule_ids_matches_grade_rule_branches():
    """KNOWN_RULE_IDS and _grade_rule's `if rule_id == "..."` branches must stay in sync.

    Why: validate_rule_ids only fails fast if KNOWN_RULE_IDS is the truth. If a new rule
    is added to _grade_rule without being added to KNOWN_RULE_IDS, specs referencing it
    will be wrongly rejected; if removed from _grade_rule but left in KNOWN_RULE_IDS,
    silent unknown-rule contamination returns.
    """
    import re
    from pathlib import Path

    from cxbench.grader import KNOWN_RULE_IDS

    src = Path("cxbench/grader.py").read_text(encoding="utf-8")
    branch_ids = set(re.findall(r'if rule_id == "([^"]+)"', src))
    missing_from_known = branch_ids - set(KNOWN_RULE_IDS)
    stale_in_known = set(KNOWN_RULE_IDS) - branch_ids
    assert not missing_from_known, (
        f"rules implemented but not in KNOWN_RULE_IDS: {missing_from_known}"
    )
    assert not stale_in_known, (
        f"rules in KNOWN_RULE_IDS but no _grade_rule branch: {stale_in_known}"
    )
