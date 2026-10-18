from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


def infer_task_domain(raw_domain: Any, business_system: str) -> str:
    if raw_domain:
        return str(raw_domain)
    for prefix, domain in {
        "telecom_": "telecom",
        "utilities_": "utilities",
        "government_benefits_": "government_benefits",
        "healthcare_": "healthcare",
        "insurance_": "insurance",
        "banking_": "banking",
    }.items():
        if business_system.startswith(prefix):
            return domain
    return "banking"


@dataclass(frozen=True)
class TaskSpec:
    """Small task schema for the cxbench MVP vertical slice."""

    id: str
    title: str
    domain: str
    scenario: str
    business_system: str
    customer_goal: str
    seed_message: str
    hidden_facts: Dict[str, Any]
    tools: Dict[str, List[str]]
    confirmation_contract: List[str]
    rules: List[str]
    initial_state: Dict[str, Any]
    policy: Dict[str, Any]
    max_turns: int
    max_user_interactions: int
    max_tool_calls_per_interaction: int
    max_tool_calls: int
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        task_path = Path(path)
        with task_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Task YAML must be a mapping: {task_path}")
        return cls.from_mapping(raw, task_path)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: Path) -> "TaskSpec":
        required = [
            "id",
            "title",
            "customer_goal",
            "seed_message",
            "hidden_facts",
            "tools",
            "confirmation_contract",
            "rules",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Task YAML missing required fields: {', '.join(missing)}")
        business_system = str(
            raw.get("business_system")
            or dict(raw.get("initial_state", {})).get("business_system")
            or dict(raw.get("policy", {})).get("business_system")
            or "banking_customer_ops_v0"
        )
        max_user_interactions = int(raw.get("max_user_interactions", 20))
        legacy_tool_burst_default = int(raw.get("max_tool_calls_per_interaction", 10))
        max_tool_calls = int(
            raw.get("max_tool_calls", legacy_tool_burst_default * max_user_interactions)
        )
        return cls(
            id=str(raw["id"]),
            title=str(raw["title"]),
            domain=infer_task_domain(raw.get("domain"), business_system),
            scenario=str(raw.get("scenario", raw.get("title", ""))),
            business_system=business_system,
            customer_goal=str(raw["customer_goal"]),
            seed_message=str(raw["seed_message"]),
            hidden_facts=dict(raw["hidden_facts"]),
            tools={key: list(value or []) for key, value in dict(raw["tools"]).items()},
            confirmation_contract=list(raw["confirmation_contract"]),
            rules=list(raw["rules"]),
            initial_state=dict(raw.get("initial_state", {})),
            policy=dict(raw.get("policy", {})),
            max_turns=int(raw.get("max_turns", 12)),
            max_user_interactions=max_user_interactions,
            max_tool_calls_per_interaction=legacy_tool_burst_default,
            max_tool_calls=max_tool_calls,
            path=path,
        )

    def mandatory_disclosure(self) -> str:
        return str(self.policy.get("mandatory_disclosure", "")).strip()
