from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .task_schema import TaskSpec
from .tools_banking import BankingCustomerOpsSystem
from .tools_benefits import GovernmentBenefitsAppealSystem
from .tools_healthcare import HealthcareAppealSystem
from .tools_insurance import InsuranceClaimIntakeSystem
from .tools_telecom import TelecomAccountSecuritySystem
from .tools_utilities import UtilitiesCustomerProtectionSystem


class DomainSystem(Protocol):
    def execute(self, tool_name: str, args: dict) -> dict: ...

    def submit_draft(self, draft_id: str) -> dict: ...

    def final_state(self) -> dict: ...


@dataclass
class ToolModeDomainSystem:
    """Runtime-facing guardrail around a domain implementation.

    Runtime is still responsible for hiding submit tools, but this wrapper keeps
    the domain layer defensive if a caller bypasses the normal tool exposure path.
    """

    inner: DomainSystem
    submit_tools: set[str]
    tool_mode: str

    def execute(self, tool_name: str, args: dict) -> dict:
        if tool_name in self.submit_tools and self.tool_mode != "direct":
            return {
                "ok": False,
                "error": "submit_not_allowed_by_tool_mode",
                "tool_mode": self.tool_mode,
            }
        return self.inner.execute(tool_name, args)

    def submit_draft(self, draft_id: str) -> dict:
        return self.inner.submit_draft(draft_id)

    def final_state(self) -> dict:
        state = dict(self.inner.final_state())
        state.setdefault("tool_mode", self.tool_mode)
        return state


def build_domain_system(
    task: TaskSpec, draft_dir: Path, tool_mode: str = "least_privilege"
) -> DomainSystem:
    tool_mode = _normalize_tool_mode(tool_mode)
    business_system = task.business_system or str(task.initial_state.get("business_system", ""))
    submit_tools = set(task.tools.get("submit", []))
    if business_system.startswith("telecom_") or submit_tools & {
        "submit_port_out",
        "submit_sim_change",
    }:
        inner: DomainSystem = TelecomAccountSecuritySystem(task=task, draft_dir=draft_dir)
    elif business_system.startswith("utilities_") or "submit_arrangement_or_hold" in submit_tools:
        inner = UtilitiesCustomerProtectionSystem(task=task, draft_dir=draft_dir)
    elif (
        business_system.startswith("government_benefits_")
        or "submit_hearing_request" in submit_tools
    ):
        inner = GovernmentBenefitsAppealSystem(task=task, draft_dir=draft_dir)
    elif business_system.startswith("healthcare_appeal") or "submit_appeal_case" in submit_tools:
        inner = HealthcareAppealSystem(task=task, draft_dir=draft_dir)
    elif (
        business_system.startswith("insurance_claim_intake")
        or "submit_claim_intake" in submit_tools
    ):
        inner = InsuranceClaimIntakeSystem(task=task, draft_dir=draft_dir)
    else:
        inner = BankingCustomerOpsSystem(task=task, draft_dir=draft_dir)
    return ToolModeDomainSystem(inner=inner, submit_tools=submit_tools, tool_mode=tool_mode)


def _normalize_tool_mode(tool_mode: str) -> str:
    normalized = tool_mode.lower().replace("-", "_")
    if normalized == "high_permission":
        return "direct"
    return normalized
