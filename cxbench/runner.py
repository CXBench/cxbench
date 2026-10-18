"""High-level runner helpers for company-side agent evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Sequence

from .adapters import AgentAdapter
from .eval_config import RuntimeConfig
from .retry import retry_until_completed
from .runtime import Runtime
from .task_schema import TaskSpec


@dataclass(frozen=True)
class EvalResult:
    """Result of a single task evaluation."""

    task_id: str
    run_dir: str
    grade: Dict[str, Any]
    report: str
    error: Optional[str] = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "error"
        return str(self.grade.get("verdict", "unknown"))

    @property
    def outcome_pass(self) -> Optional[bool]:
        return self.grade.get("outcome_pass") if not self.error else None

    @property
    def process_pass(self) -> Optional[bool]:
        return self.grade.get("process_pass") if not self.error else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_dir": self.run_dir,
            "report": self.report,
            "verdict": self.verdict,
            "outcome_pass": self.outcome_pass,
            "process_pass": self.process_pass,
            "error": self.error,
            "grade": self.grade,
        }


def run_eval(
    task: TaskSpec | str | Path,
    agent: AgentAdapter,
    *,
    mode: str = "least_privilege",
    grader: str = "hybrid",
    judge_model: str | None = None,
    user_sim: str = "llm",
    user_model: str | None = None,
    user_max_attempts: int = 3,
    runs_dir: str | Path = "runs",
    max_elapsed_seconds: int | None = None,
    _stop_event: Event | None = None,
) -> EvalResult:
    """Evaluate a single agent on a single task.

    Execution errors retry with a fresh runtime until evaluation completes.
    A valid failing verdict is returned immediately. Task files and basic
    runtime options are validated before execution.

    Parameters
    ----------
    task:
        A ``TaskSpec`` instance or a path to a task YAML file.
    agent:
        An ``AgentAdapter`` subclass instance implementing ``next_action``.
    mode:
        Tool permission mode: ``"least_privilege"`` (default) or ``"direct"``.
    grader:
        Grader strategy. ``"hybrid"`` is the only formally-validated mode.
    judge_model:
        LLM judge model string. Required when ``grader`` is ``"llm"`` or ``"hybrid"``.
    user_sim:
        Customer simulator strategy: ``"llm"``, ``"react"``, ``"verify"``, or ``"reflection"``.
    user_model:
        LLM model for the user simulator.
    user_max_attempts:
        Maximum verify/reflection attempts for the user simulator.
    runs_dir:
        Directory where run artifacts are written.
    max_elapsed_seconds:
        Wall-clock timeout for the run. ``None`` means no limit.

    Returns
    -------
    EvalResult
        Frozen dataclass with ``task_id``, ``run_dir``, ``grade`` dict,
        ``report`` path, and convenience properties.
    """
    task_spec = task if isinstance(task, TaskSpec) else TaskSpec.load(task)
    runtime_options = dict(
        mode=mode,
        model="external",  # not used when agent is provided
        runs_dir=runs_dir,
        grader=grader,
        judge_model=judge_model,
        user_sim=user_sim,
        user_model=user_model,
        user_max_attempts=user_max_attempts,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    RuntimeConfig.parse(**runtime_options)
    raw = retry_until_completed(
        lambda: Runtime(task=task_spec, agent=agent, **runtime_options).run(),
        label=f"task {task_spec.id}",
        stop_event=_stop_event,
    )
    return EvalResult(
        task_id=task_spec.id,
        run_dir=str(raw["run_dir"]),
        grade=raw["grade"],
        report=str(raw["report"]),
    )


def run_suite(
    tasks_dir: str | Path,
    agent: AgentAdapter,
    *,
    task_ids: Sequence[str] | None = None,
    workers: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate an agent across a directory of tasks.

    Parameters
    ----------
    tasks_dir:
        Directory containing ``*.yaml`` task files.
    agent:
        An ``AgentAdapter`` subclass instance. The same agent instance is
        reused across tasks; ``on_task_start`` and ``set_tool_registry``
        are called per-task by ``Runtime``.
    task_ids:
        Optional subset of task IDs to run. If ``None``, all YAML files
        in ``tasks_dir`` are evaluated.
    workers:
        Number of parallel workers. Default is 1 (sequential).
    **kwargs:
        Forwarded to :func:`run_eval`.

    Returns
    -------
    dict
        Aggregate report with ``total``, ``passed``, ``failed``,
        ``errors``, ``pass_rate``, and ``results`` list.
    """
    tasks_path = Path(tasks_dir)
    yaml_files = sorted(tasks_path.glob("*.yaml"))
    if task_ids is not None:
        id_set = set(task_ids)
        yaml_files = [f for f in yaml_files if TaskSpec.load(f).id in id_set]

    if not yaml_files:
        return {"total": 0, "passed": 0, "failed": 0, "errors": 0, "pass_rate": 0.0, "results": []}

    results: List[EvalResult] = []
    stop_event = Event() if workers > 1 else None

    def _run_one(yaml_path: Path) -> EvalResult:
        return run_eval(yaml_path, agent, _stop_event=stop_event, **kwargs)

    if workers <= 1:
        for yaml_path in yaml_files:
            results.append(_run_one(yaml_path))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                futures = {pool.submit(_run_one, yp): yp for yp in yaml_files}
                for future in as_completed(futures):
                    results.append(future.result())
            except BaseException:
                stop_event.set()
                raise

    results.sort(key=lambda r: r.task_id)

    passed = sum(1 for r in results if r.verdict == "pass")
    failed = sum(1 for r in results if r.verdict == "fail")
    errors = sum(1 for r in results if r.verdict == "error")
    total = len(results)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / total if total else 0.0,
        "results": [r.to_dict() for r in results],
    }
