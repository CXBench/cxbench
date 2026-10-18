from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LLM_USER_SIM_STRATEGIES = {"llm", "react", "verify", "reflection"}
GRADERS = {"rule", "llm", "hybrid"}
PROVIDER_PREFIXES = ("openai:", "crs:", "deepseek:", "openrouter:", "siliconflow:", "shubiaobiao:")


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    model: str
    runs_dir: Path
    grader: str
    user_sim: str
    judge_model: Optional[str] = None
    user_model: Optional[str] = None
    user_max_attempts: int = 3
    max_elapsed_seconds: Optional[int] = None

    @classmethod
    def parse(
        cls,
        *,
        mode: str,
        model: str,
        runs_dir: str | Path,
        grader: str,
        user_sim: str,
        judge_model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_max_attempts: int = 3,
        max_elapsed_seconds: Optional[int] = None,
    ) -> "RuntimeConfig":
        normalized_mode = normalize_tool_mode(mode)
        if normalized_mode not in {"least_privilege", "direct"}:
            raise ValueError("mode must be 'least_privilege', 'direct', or 'high_permission'")

        normalized_grader = grader.lower().replace("-", "_")
        if normalized_grader not in GRADERS:
            raise ValueError("grader must be 'rule', 'llm', or 'hybrid'")
        if normalized_grader in {"llm", "hybrid"} and not judge_model:
            raise ValueError(f"grader {normalized_grader!r} requires judge_model")

        normalized_user_sim = user_sim.lower().replace("_", "-")
        validate_user_sim(normalized_user_sim, user_model)

        return cls(
            mode=normalized_mode,
            model=model,
            runs_dir=Path(runs_dir),
            grader=normalized_grader,
            judge_model=judge_model,
            user_sim=normalized_user_sim,
            user_model=user_model,
            user_max_attempts=user_max_attempts,
            max_elapsed_seconds=max_elapsed_seconds,
        )


def normalize_tool_mode(mode: str) -> str:
    normalized = mode.lower().replace("-", "_")
    if normalized == "high_permission":
        return "direct"
    return normalized


def validate_user_sim(user_sim: str, user_model: Optional[str]) -> None:
    normalized = user_sim.lower().replace("_", "-")
    if normalized == "deterministic":
        if deterministic_user_sim_allowed_for_tests():
            return
        raise ValueError(
            "deterministic user simulator is test-only and cannot be used for runtime evaluation; "
            "use user_sim=llm|react|verify|reflection with user_model=<provider>:<model>."
        )
    if normalized == "human":
        raise ValueError(
            "human user simulator is not allowed for benchmark evaluation; "
            "use user_sim=llm|react|verify|reflection with user_model=<provider>:<model>."
        )
    if normalized not in LLM_USER_SIM_STRATEGIES:
        raise ValueError("unknown user simulator; expected llm, react, verify, or reflection")
    if not user_model:
        raise ValueError(f"user simulator strategy {user_sim!r} requires user_model")


def deterministic_user_sim_allowed_for_tests() -> bool:
    return (
        bool(os.environ.get("PYTEST_CURRENT_TEST"))
        and os.environ.get("CXBENCH_ALLOW_TEST_DETERMINISTIC_USER_SIM") == "1"
    )


def model_string(value: str, *, default_provider: str = "openai") -> str:
    if value.startswith(PROVIDER_PREFIXES):
        return value
    return f"{default_provider}:{value}"
