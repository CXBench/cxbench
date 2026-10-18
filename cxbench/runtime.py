from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import AgentAction, AgentAdapter, build_agent, get_tool_schemas_for_task
from .domain_policy import domain_policy_context_for_task
from .eval_config import RuntimeConfig
from .grader import grade_run
from .report import write_report
from .task_schema import TaskSpec
from .domain_system import build_domain_system
from .user_sim import BaseUserSim, UserObservation, UserStepInput, build_user_sim


class RuntimeTimeoutError(RuntimeError):
    pass


class Runtime:
    def __init__(
        self,
        task: TaskSpec,
        mode: str,
        model: str,
        runs_dir: str | Path,
        grader: str,
        user_sim: str,
        judge_model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_max_attempts: int = 3,
        max_elapsed_seconds: Optional[int] = None,
        agent: Optional[AgentAdapter] = None,
    ) -> None:
        config = RuntimeConfig.parse(
            mode=mode,
            model=model,
            runs_dir=runs_dir,
            grader=grader,
            judge_model=judge_model,
            user_sim=user_sim,
            user_model=user_model,
            user_max_attempts=user_max_attempts,
            max_elapsed_seconds=max_elapsed_seconds,
        )
        self.task = task
        self.mode = config.mode
        self.model = config.model
        self.grader = config.grader
        self.judge_model = config.judge_model
        self.user_sim_strategy = config.user_sim
        self.user_model = config.user_model
        self.user_max_attempts = config.user_max_attempts
        self.max_elapsed_seconds = config.max_elapsed_seconds
        self.runs_dir = config.runs_dir
        self.run_dir = self._create_run_dir()
        self.trace: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        self.latest_draft: Optional[Dict[str, Any]] = None
        self.confirmation: Optional[Dict[str, Any]] = None
        self.tools = build_domain_system(task, self.run_dir / "drafts", tool_mode=self.mode)
        self.user: BaseUserSim = build_user_sim(
            config.user_sim,
            task,
            model=config.user_model,
            max_attempts=config.user_max_attempts,
        )
        if agent is not None:
            self.agent = agent
        else:
            self.agent = build_agent(config.model, config.mode, task)
        self._prepare_agent()

    def _prepare_agent(self) -> None:
        """Notify the agent about the task and provide tool schemas.

        Calls ``on_task_start`` and ``set_tool_registry`` hooks so
        company-side agents can pre-ingest context.  ``hidden_facts``
        are intentionally excluded.
        """
        policy = self.task.policy or {}
        process_reqs = list(policy.get("process_requirements", []))
        source_refs = list(policy.get("source_refs", []))
        domain_ctx = domain_policy_context_for_task(self.task)
        try:
            self.agent.on_task_start(
                task_id=self.task.id,
                task_title=self.task.title,
                domain=self.task.domain,
                mode=self.mode,
                process_requirements=process_reqs,
                source_refs=source_refs,
                domain_policy_context=domain_ctx,
            )
        except AttributeError:
            pass  # base class no-op is fine; older subclasses may lack it
        try:
            schemas = get_tool_schemas_for_task(self.task)
            self.agent.set_tool_registry(schemas)
        except AttributeError:
            pass

    def run(self) -> Dict[str, Any]:
        self._write_config()
        run_start = time.monotonic()
        first = self.user.reset()
        self._record_user_reply(first)

        user_interactions = 0
        tool_calls_total = 0
        provider_steps = 0
        provider_step_limit = self._provider_step_limit()
        tool_call_limit = self._tool_call_limit()
        elapsed_timeout = False
        while (
            user_interactions < self.task.max_user_interactions
            and provider_steps < provider_step_limit
            and tool_calls_total < tool_call_limit
        ):
            if (
                self.max_elapsed_seconds
                and (time.monotonic() - run_start) >= self.max_elapsed_seconds
            ):
                elapsed_timeout = True
                break
            available_tools = self._available_tools()
            action = self.agent.next_action(self.messages, available_tools)
            provider_steps += 1
            self._record_provider_meta(action)
            if action.kind == "tool":
                self._handle_tool(action)
                tool_calls_total += 1
            elif action.kind == "message":
                self._handle_agent_message(action)
                user_interactions += 1
            else:
                raise ValueError(f"Unsupported agent action kind: {action.kind}")
            if self._should_stop():
                break
        if elapsed_timeout:
            elapsed = round(time.monotonic() - run_start, 1)
            self._append(
                "runtime",
                "elapsed_timeout",
                {
                    "elapsed_seconds": elapsed,
                    "max_elapsed_seconds": self.max_elapsed_seconds,
                    "provider_steps": provider_steps,
                    "user_interactions": user_interactions,
                },
            )
            self._write_trace()
            raise RuntimeTimeoutError(
                f"task exceeded {self.max_elapsed_seconds}s wall-clock limit "
                f"after {provider_steps} provider steps and {user_interactions} user interactions"
            )
        if not self._should_stop():
            if user_interactions >= self.task.max_user_interactions:
                self._append(
                    "runtime",
                    "interaction_budget_reached",
                    {
                        "user_interactions": user_interactions,
                        "max_user_interactions": self.task.max_user_interactions,
                    },
                )
            elif provider_steps >= provider_step_limit:
                self._append(
                    "runtime",
                    "provider_step_budget_reached",
                    {"provider_steps": provider_steps, "provider_step_limit": provider_step_limit},
                )
            elif tool_calls_total >= tool_call_limit:
                self._append(
                    "runtime",
                    "tool_budget_reached",
                    {"tool_calls_total": tool_calls_total, "max_tool_calls": tool_call_limit},
                )

        final_state = self._final_state_with_runtime_metadata()
        (self.run_dir / "final_state.json").write_text(
            json.dumps(final_state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._write_trace()
        grade = grade_run(
            self.task,
            self.trace,
            final_state,
            self.mode,
            grader=self.grader,
            judge_model=self.judge_model,
        )
        (self.run_dir / "grade.json").write_text(
            json.dumps(grade, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report_path = write_report(self.run_dir, grade)
        return {"run_dir": str(self.run_dir), "grade": grade, "report": str(report_path)}

    def _handle_tool(self, action: AgentAction) -> None:
        tool_name = action.tool_name
        args = dict(action.tool_args or {})
        self._append(
            "agent",
            "tool_call",
            {"tool_name": tool_name, "args": args, "tool_call_id": action.tool_call_id},
        )

        if tool_name in self.task.tools.get("submit", []) and self.mode == "least_privilege":
            result = {"ok": False, "error": "submit_tool_not_exposed_in_least_privilege_mode"}
            self._append(
                "runtime", "submit_blocked", {"tool_name": tool_name, "reason": result["error"]}
            )
            self._append("tool", "tool_result", {"tool_name": tool_name, "result": result})
            self._record_tool_message(action, result)
            return

        if tool_name not in self._available_tools():
            result = {"ok": False, "error": "tool_not_available"}
        else:
            result = self.tools.execute(tool_name, args)

        if result.get("ok") and {"draft_id", "draft_hash", "fields"}.issubset(result):
            self.latest_draft = {
                "draft_id": result["draft_id"],
                "draft_hash": result["draft_hash"],
                "draft_type": result.get("draft_type"),
                "fields": result["fields"],
            }
            self._append("runtime", "draft_created", self.latest_draft)

        if tool_name in self.task.tools.get("submit", []) and self.mode == "direct":
            draft_hash = self.latest_draft["draft_hash"] if self.latest_draft else None
            self._append(
                "runtime",
                "submit_executed",
                {
                    "initiated_by": "agent",
                    "tool_name": tool_name,
                    "draft_id": None,
                    "draft_hash": draft_hash,
                    "confirmed": False,
                    "result": result,
                },
            )

        self._append("tool", "tool_result", {"tool_name": tool_name, "result": result})
        self._record_tool_message(action, result)

    def _handle_agent_message(self, action: AgentAction) -> None:
        content = action.content
        self._append("agent", "assistant_message", {"content": content})
        self.messages.append({"role": "assistant", "content": content})
        reply = self.user.step(
            UserStepInput(
                assistant_message=content,
                latest_draft=self.latest_draft,
                final_state=self.tools.final_state(),
                trace_tail=self.trace[-12:],
            )
        )
        self._record_user_reply(reply)
        if reply.kind == "confirm" and self.latest_draft:
            self.confirmation = {
                "token": f"confirm_{uuid.uuid4().hex[:10]}",
                "draft_id": reply.draft_id,
                "draft_hash": reply.draft_hash,
            }
            self._append("customer", "confirmation", self.confirmation)
            if self.mode == "least_privilege":
                self._submit_confirmed_draft()

    def _submit_confirmed_draft(self) -> None:
        if not self.confirmation:
            return
        draft_id = self.confirmation["draft_id"]
        result = self.tools.submit_draft(draft_id)
        draft_type = self.latest_draft.get("draft_type") if self.latest_draft else None
        self._append(
            "runtime",
            "submit_executed",
            {
                "initiated_by": "runtime",
                "confirmation_token": self.confirmation["token"],
                "draft_id": draft_id,
                "draft_hash": self.confirmation["draft_hash"],
                "draft_type": draft_type,
                "confirmed": True,
                "result": result,
            },
        )

    def _record_user_reply(self, reply: UserObservation) -> None:
        self._append("user_sim", "user_sim_decision", reply.decision_event(self.user.strategy))
        if reply.kind == "stop":
            self._append("customer", "stop", {"content": reply.content})
            self.messages.append({"role": "user", "content": reply.content})
            return
        if reply.kind == "confirm":
            self.messages.append({"role": "user", "content": reply.content})
            return
        self._append("customer", "customer_message", {"content": reply.content})
        self.messages.append({"role": "user", "content": reply.content})

    def _record_tool_message(self, action: AgentAction, result: Dict[str, Any]) -> None:
        if action.raw_message and action.tool_call_id:
            self.messages.append(action.raw_message)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": action.tool_call_id,
                    "name": action.tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            return
        args = dict(action.tool_args or {})
        self.messages.append(
            {
                "role": "assistant",
                "content": f"Internal tool action completed for {action.tool_name} with args {json.dumps(args, ensure_ascii=False)}.",
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": f"Internal tool observation for {action.tool_name}: {json.dumps(result, ensure_ascii=False)}",
            }
        )

    def _record_provider_meta(self, action: AgentAction) -> None:
        if action.provider_meta:
            self._append("provider", "provider_call", action.provider_meta)

    def _available_tools(self) -> List[str]:
        tools = (
            list(self.task.tools.get("read", []))
            + list(self.task.tools.get("check", []))
            + list(self.task.tools.get("draft", []))
            + list(self.task.tools.get("notice", []))
        )
        if self.mode == "direct":
            tools += list(self.task.tools.get("submit", []))
        return tools

    def _provider_step_limit(self) -> int:
        budget_limit = self.task.max_user_interactions + self._tool_call_limit() + 5
        return max(self.task.max_turns, budget_limit)

    def _tool_call_limit(self) -> int:
        return self.task.max_tool_calls

    def _should_stop(self) -> bool:
        final_state = self.tools.final_state()
        return any(event.get("event_type") == "stop" for event in self.trace) or bool(
            self.mode == "least_privilege"
            and self.confirmation
            and _has_terminal_submission(final_state)
        )

    def _append(self, actor: str, event_type: str, payload: Dict[str, Any]) -> None:
        self.trace.append(
            {
                "seq": len(self.trace) + 1,
                "ts": datetime.now(timezone.utc).isoformat(),
                "actor": actor,
                "event_type": event_type,
                "payload": payload,
            }
        )

    def _write_trace(self) -> None:
        with (self.run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
            for event in self.trace:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_config(self) -> None:
        config = {
            "task_path": str(self.task.path),
            "task_id": self.task.id,
            "mode": self.mode,
            "tool_mode": self.mode,
            "model": self.model,
            "grader": self.grader,
            "judge_model": self.judge_model,
            "user_sim": self.user_sim_strategy,
            "user_model": self.user_model,
            "user_max_attempts": self.user_max_attempts,
            "max_turns": self.task.max_turns,
            "max_user_interactions": self.task.max_user_interactions,
            "max_tool_calls_per_interaction": self.task.max_tool_calls_per_interaction,
            "max_tool_calls": self.task.max_tool_calls,
            "provider_step_limit": self._provider_step_limit(),
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _final_state_with_runtime_metadata(self) -> Dict[str, Any]:
        final_state = dict(self.tools.final_state())
        final_state["tool_mode"] = self.mode
        final_state["submissions"] = self._submission_records()
        return final_state

    def _submission_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for event in self.trace:
            if event.get("event_type") != "submit_executed":
                continue
            payload = dict(event.get("payload", {}))
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            records.append(
                {
                    "seq": event.get("seq"),
                    "initiated_by": payload.get("initiated_by"),
                    "tool_name": payload.get("tool_name"),
                    "draft_id": payload.get("draft_id"),
                    "draft_hash": payload.get("draft_hash"),
                    "draft_type": payload.get("draft_type"),
                    "confirmed": bool(
                        payload.get("confirmed") or payload.get("confirmation_token")
                    ),
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
                }
            )
        return records

    def _create_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{self.task.id}-{self.mode}-{stamp}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def _has_terminal_submission(final_state: Dict[str, Any]) -> bool:
    terminal_keys = (
        "disputes",
        "card_control_changes",
        "payment_arrangements",
        "regulatory_holds",
        "utility_disputes",
        "port_out_requests",
        "sim_changes",
        "appeal_cases",
        "hearing_cases",
        "claim_cases",
        "duplicate_review_cases",
        "siu_handoffs",
        "support_handoffs",
        "policy_term_reviews",
        "subrogation_referrals",
        "vehicle_loss_assessments",
        "tow_storage_requests",
        "claim_withdrawal_requests",
        "claim_close_requests",
        "claim_reopen_requests",
        "claim_status_inquiries",
        "fraud_documents",
        "survivor_separation_requests",
        "abuser_access_restrictions",
        "confidential_communications",
        "secure_self_service_explanations",
        "authorization_monitors",
        "provisional_credit_reviews",
    )
    return any(bool(final_state.get(key)) for key in terminal_keys)
