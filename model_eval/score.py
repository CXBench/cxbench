#!/usr/bin/env python3
"""Score existing cxbench traces without running an agent or user simulator."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from cxbench.grader import grade_run, load_trace
from cxbench.task_schema import TaskSpec

from .reporting import write_model_report
from .run import ENV_FILE, REPO_ROOT, _load_env_file


DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_OUTPUT_TOKENS = 8192
SCORE_SCHEMA_VERSION = 1
_PROVIDER_PREFIXES = (
    "openai",
    "deepseek",
    "openrouter",
    "siliconflow",
    "shubiaobiao",
)
_MANAGED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_WIRE_API",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_MAX_OUTPUT_TOKENS",
    "PYTHONUTF8",
    "PYTHONUNBUFFERED",
)


class ScoreError(ValueError):
    """Raised when existing run artifacts cannot be scored safely."""


@dataclass(frozen=True)
class ScoreInput:
    task_id: str
    task: TaskSpec
    task_path: Path
    run_dir: Path | None
    trace_path: Path | None
    final_state_path: Path | None
    mode: str
    judge_model: str | None
    config: Mapping[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class ScorePlan:
    source: Path
    source_dir: Path
    source_kind: str
    source_manifest: Mapping[str, Any]
    inputs: tuple[ScoreInput, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScoreError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScoreError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScoreError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ScoreError(f"required file does not exist: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoreError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ScoreError(f"expected a JSON object on {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_name(raw_path: str) -> str:
    return raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _resolve_explicit_path(raw_path: str | Path) -> Path:
    supplied = Path(raw_path).expanduser()
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.append(REPO_ROOT / supplied)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ScoreError(f"file does not exist: {raw_path}")


def _resolve_recorded_task_path(raw_path: str, *, benchmark: str | None = None) -> Path:
    supplied = Path(raw_path).expanduser()
    exact_candidates = [supplied]
    if not supplied.is_absolute():
        exact_candidates.append(REPO_ROOT / supplied)

    portable_parts = [part for part in raw_path.replace("\\", "/").split("/") if part]
    root_indexes = [
        index
        for index, part in enumerate(portable_parts)
        if part.casefold() == REPO_ROOT.name.casefold()
    ]
    if root_indexes:
        exact_candidates.append(REPO_ROOT.joinpath(*portable_parts[root_indexes[-1] + 1 :]))

    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate.resolve()

    name = _portable_name(raw_path)
    # A renamed checkout must retain the recorded release, even when IDs overlap.
    release_parts = portable_parts[-4:]
    if len(release_parts) == 4 and release_parts[0] == "benchmarks" and release_parts[2] == "tasks":
        release_path = REPO_ROOT.joinpath(*release_parts)
        if release_path.is_file():
            return release_path.resolve()
        raise ScoreError(f"recorded task release cannot be found on this platform: {raw_path}")
    if benchmark:
        release_path = REPO_ROOT / "benchmarks" / benchmark / "tasks" / name
        if release_path.is_file():
            return release_path.resolve()
        raise ScoreError(f"recorded benchmark task cannot be found: {benchmark}/{name}")
    fallback_candidates = [REPO_ROOT / "tasks" / name]
    benchmarks_dir = REPO_ROOT / "benchmarks"
    if benchmarks_dir.is_dir():
        fallback_candidates.extend(path / name for path in sorted(benchmarks_dir.glob("*/tasks")))

    matches: dict[str, Path] = {}
    for candidate in fallback_candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            matches[str(resolved).casefold()] = resolved
    if len(matches) == 1:
        return next(iter(matches.values()))
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches.values())
        raise ScoreError(
            f"recorded task path is ambiguous on this platform: {raw_path}; matches: {rendered}"
        )
    raise ScoreError(f"recorded task cannot be found on this platform: {raw_path}")


def _normalize_judge_model(raw_model: str | None, provider: str | None = None) -> str | None:
    if not raw_model or not raw_model.strip():
        return None
    model = raw_model.strip()
    if any(model.startswith(f"{prefix}:") for prefix in _PROVIDER_PREFIXES):
        return model
    normalized_provider = (provider or "openai").strip().lower()
    if normalized_provider == "same":
        normalized_provider = "openai"
    if normalized_provider not in _PROVIDER_PREFIXES:
        normalized_provider = "openai"
    return f"{normalized_provider}:{model}"


def _manifest_tasks(manifest: Mapping[str, Any]) -> dict[str, tuple[TaskSpec, Path]]:
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ScoreError("batch manifest contains no tasks")
    benchmark = str(manifest.get("benchmark") or "") or None
    tasks: dict[str, tuple[TaskSpec, Path]] = {}
    for entry in raw_tasks:
        if isinstance(entry, str):
            raw_path = entry
            expected_hash = None
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            raw_path = entry["path"]
            expected_hash = entry.get("sha256")
        else:
            raise ScoreError("manifest task entries must be paths or objects containing a path")
        task_path = _resolve_recorded_task_path(raw_path, benchmark=benchmark)
        if expected_hash and _sha256(task_path) != expected_hash:
            raise ScoreError(f"task changed since the original run: {task_path}")
        task = TaskSpec.load(task_path)
        if task.id in tasks:
            raise ScoreError(f"manifest contains duplicate task id: {task.id}")
        tasks[task.id] = (task, task_path)
    expected_count = manifest.get("task_count")
    if isinstance(expected_count, int) and expected_count != len(tasks):
        raise ScoreError(f"manifest declares {expected_count} tasks but {len(tasks)} were loaded")
    return tasks


def _load_optional_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _task_batch_dir(run_dir: Path) -> Path | None:
    if run_dir.parent.name == "task_runs" and (run_dir.parent.parent / "manifest.json").is_file():
        return run_dir.parent.parent
    return None


def _source_location(source: Path) -> tuple[str, Path, Path]:
    if source.is_file():
        if source.name in {
            "trace.jsonl",
            "config.json",
            "final_state.json",
            "grade.json",
        }:
            return "task_run", source.parent, source
        if (
            source.name in {"manifest.json", "results.jsonl", "results_partial.jsonl"}
            and (source.parent / "task_runs").is_dir()
        ):
            return "batch", source.parent, source
        raise ScoreError(
            "input file must be trace.jsonl, a task-run artifact, or a batch manifest/results JSONL"
        )
    if not source.is_dir():
        raise ScoreError(f"input does not exist: {source}")
    if (source / "trace.jsonl").is_file():
        return "task_run", source, source
    if (source / "task_runs").is_dir():
        return "batch", source, source
    if source.name == "task_runs" and source.parent.is_dir():
        return "batch", source.parent, source
    raise ScoreError(f"input is neither a task run nor a batch run: {source}")


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    return _read_json(config_path) if config_path.is_file() else {}


def _input_from_run_dir(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_tasks: Mapping[str, tuple[TaskSpec, Path]],
    explicit_task: Path | None,
    explicit_final_state: Path | None,
    mode_override: str | None,
    judge_model_override: str | None,
) -> ScoreInput:
    config = _load_config(run_dir)
    configured_task_id = str(config.get("task_id") or "")

    if explicit_task:
        task_path = explicit_task
        task = TaskSpec.load(task_path)
    elif configured_task_id and configured_task_id in manifest_tasks:
        task, task_path = manifest_tasks[configured_task_id]
    else:
        raw_task_path = config.get("task_path")
        if not isinstance(raw_task_path, str) or not raw_task_path:
            raise ScoreError(f"cannot resolve task for {run_dir}; provide --task")
        task_path = _resolve_recorded_task_path(
            raw_task_path,
            benchmark=str(manifest.get("benchmark") or "") or None,
        )
        task = TaskSpec.load(task_path)

    if configured_task_id and configured_task_id != task.id:
        raise ScoreError(
            f"task id mismatch in {run_dir}: config has {configured_task_id!r}, task has {task.id!r}"
        )

    trace_path = run_dir / "trace.jsonl"
    final_state_path = explicit_final_state or (run_dir / "final_state.json")
    mode = mode_override or str(config.get("mode") or manifest.get("mode") or "")
    if mode == "high_permission":
        mode = "direct"
    if mode not in {"least_privilege", "direct"}:
        raise ScoreError(f"cannot determine a valid runtime mode for {task.id}; provide --mode")

    source_provider = str(manifest.get("judge_provider") or "")
    if source_provider == "same":
        source_provider = str(manifest.get("agent_provider") or "openai")
    judge_model = _normalize_judge_model(
        judge_model_override or str(config.get("judge_model") or manifest.get("judge_model") or ""),
        provider=source_provider,
    )
    error = None
    missing = [path.name for path in (trace_path, final_state_path) if not path.is_file()]
    if missing:
        error = f"missing required artifact(s): {', '.join(missing)}"
    return ScoreInput(
        task_id=task.id,
        task=task,
        task_path=task_path,
        run_dir=run_dir.resolve(),
        trace_path=trace_path.resolve(),
        final_state_path=final_state_path.resolve(),
        mode=mode,
        judge_model=judge_model,
        config=config,
        error=error,
    )


def _result_run_dirs(batch_dir: Path) -> dict[str, Path]:
    rows: dict[str, dict[str, Any]] = {}
    for name in ("results.jsonl", "results_partial.jsonl"):
        path = batch_dir / name
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            task_id = row.get("task_id")
            if isinstance(task_id, str) and task_id:
                rows[task_id] = row
        if rows:
            break

    task_runs_dir = batch_dir / "task_runs"
    resolved: dict[str, Path] = {}
    for task_id, row in rows.items():
        raw_run_dir = row.get("run_dir")
        if not isinstance(raw_run_dir, str) or not raw_run_dir:
            continue
        candidate = task_runs_dir / _portable_name(raw_run_dir)
        if candidate.is_dir():
            resolved[task_id] = candidate.resolve()

    for candidate in sorted(task_runs_dir.iterdir()):
        if not candidate.is_dir():
            continue
        config_path = candidate / "config.json"
        if not config_path.is_file():
            continue
        config = _read_json(config_path)
        task_id = config.get("task_id")
        if isinstance(task_id, str) and task_id:
            previous = resolved.get(task_id)
            if previous and previous != candidate.resolve():
                raise ScoreError(f"multiple task runs found for {task_id} in {task_runs_dir}")
            resolved[task_id] = candidate.resolve()
    return resolved


def build_score_plan(
    source: str | Path,
    *,
    task_path: str | Path | None = None,
    final_state_path: str | Path | None = None,
    mode: str | None = None,
    judge_model: str | None = None,
) -> ScorePlan:
    resolved_source = Path(source).expanduser().resolve()
    source_kind, source_dir, _ = _source_location(resolved_source)
    if source_kind == "batch" and (task_path or final_state_path):
        raise ScoreError("--task and --final-state can only be used with one task run")

    explicit_task = _resolve_explicit_path(task_path) if task_path else None
    explicit_final_state = _resolve_explicit_path(final_state_path) if final_state_path else None
    batch_dir = source_dir if source_kind == "batch" else _task_batch_dir(source_dir)
    manifest = _load_optional_manifest(batch_dir / "manifest.json") if batch_dir else {}
    manifest_tasks = _manifest_tasks(manifest) if manifest.get("tasks") else {}

    if source_kind == "task_run":
        score_input = _input_from_run_dir(
            source_dir,
            manifest=manifest,
            manifest_tasks=manifest_tasks,
            explicit_task=explicit_task,
            explicit_final_state=explicit_final_state,
            mode_override=mode,
            judge_model_override=judge_model,
        )
        return ScorePlan(
            source=resolved_source,
            source_dir=source_dir.resolve(),
            source_kind=source_kind,
            source_manifest=manifest,
            inputs=(score_input,),
        )

    run_dirs = _result_run_dirs(source_dir)
    if manifest_tasks:
        unexpected = sorted(set(run_dirs) - set(manifest_tasks))
        if unexpected:
            raise ScoreError(f"task runs absent from the manifest: {', '.join(unexpected)}")
        task_ids = list(manifest_tasks)
    else:
        if not run_dirs:
            raise ScoreError(f"no reusable task runs found in {source_dir / 'task_runs'}")
        task_ids = sorted(run_dirs)

    inputs: list[ScoreInput] = []
    for task_id in task_ids:
        run_dir = run_dirs.get(task_id)
        if run_dir is None:
            task, loaded_task_path = manifest_tasks[task_id]
            resolved_mode = mode or str(manifest.get("mode") or "")
            if resolved_mode == "high_permission":
                resolved_mode = "direct"
            if resolved_mode not in {"least_privilege", "direct"}:
                resolved_mode = "least_privilege"
            source_provider = str(manifest.get("judge_provider") or "")
            if source_provider == "same":
                source_provider = str(manifest.get("agent_provider") or "openai")
            inputs.append(
                ScoreInput(
                    task_id=task_id,
                    task=task,
                    task_path=loaded_task_path,
                    run_dir=None,
                    trace_path=None,
                    final_state_path=None,
                    mode=resolved_mode,
                    judge_model=_normalize_judge_model(
                        judge_model or str(manifest.get("judge_model") or ""),
                        provider=source_provider,
                    ),
                    config={},
                    error="no task run with a reusable trace and final state",
                )
            )
            continue
        inputs.append(
            _input_from_run_dir(
                run_dir,
                manifest=manifest,
                manifest_tasks=manifest_tasks,
                explicit_task=None,
                explicit_final_state=None,
                mode_override=mode,
                judge_model_override=judge_model,
            )
        )
    return ScorePlan(
        source=resolved_source,
        source_dir=source_dir.resolve(),
        source_kind=source_kind,
        source_manifest=manifest,
        inputs=tuple(inputs),
    )


def validate_score_plan(plan: ScorePlan, *, grader: str) -> None:
    if grader not in {"rule", "llm", "hybrid"}:
        raise ScoreError("grader must be 'rule', 'llm', or 'hybrid'")
    seen: set[str] = set()
    for score_input in plan.inputs:
        if score_input.task_id in seen:
            raise ScoreError(f"duplicate task id in score plan: {score_input.task_id}")
        seen.add(score_input.task_id)
        if score_input.error:
            continue
        assert score_input.trace_path is not None
        assert score_input.final_state_path is not None
        load_trace(score_input.trace_path)
        _read_json(score_input.final_state_path)
        if grader in {"llm", "hybrid"} and not score_input.judge_model:
            raise ScoreError(f"{grader} scoring requires --judge-model for {score_input.task_id}")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "task"


def _new_score_id(grader: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{grader}"


def default_output_dir(plan: ScorePlan, *, grader: str) -> Path:
    base = plan.source_dir
    score_id = _new_score_id(grader)
    candidate = base / "scores" / score_id
    suffix = 2
    while candidate.exists():
        candidate = base / "scores" / f"{score_id}-{suffix}"
        suffix += 1
    return candidate


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _error_result(
    score_input: ScoreInput, message: str, *, grader: str, elapsed: float = 0.0
) -> dict[str, Any]:
    return {
        "task_id": score_input.task_id,
        "domain": score_input.task.domain,
        "scenario": score_input.task.scenario,
        "task_path": str(score_input.task_path),
        "run_dir": str(score_input.run_dir) if score_input.run_dir else None,
        "trace_path": str(score_input.trace_path) if score_input.trace_path else None,
        "grade_path": None,
        "ok": False,
        "execution_grader": score_input.config.get("grader"),
        "final_grader": grader,
        "outcome_pass": False,
        "process_pass": False,
        "verdict": "score_error",
        "false_pass_demo": False,
        "rule_failures": [],
        "llm_judge_process_pass": None,
        "judge_model": score_input.judge_model,
        "score_elapsed_seconds": round(elapsed, 3),
        "error": message,
    }


def _score_one(score_input: ScoreInput, output_dir: Path, grader: str) -> dict[str, Any]:
    started = time.perf_counter()
    if score_input.error:
        return _error_result(score_input, score_input.error, grader=grader)
    assert score_input.trace_path is not None
    assert score_input.final_state_path is not None
    try:
        trace = load_trace(score_input.trace_path)
        final_state = _read_json(score_input.final_state_path)
        grade = grade_run(
            score_input.task,
            trace,
            final_state,
            score_input.mode,
            grader=grader,
            judge_model=score_input.judge_model,
        )
        grade_path = output_dir / "grades" / f"{_safe_component(score_input.task_id)}.json"
        _atomic_write_text(grade_path, json.dumps(grade, indent=2, ensure_ascii=False) + "\n")
        return {
            "task_id": score_input.task_id,
            "domain": score_input.task.domain,
            "scenario": score_input.task.scenario,
            "task_path": str(score_input.task_path),
            "task_sha256": _sha256(score_input.task_path),
            "run_dir": str(score_input.run_dir) if score_input.run_dir else None,
            "trace_path": str(score_input.trace_path),
            "trace_sha256": _sha256(score_input.trace_path),
            "final_state_path": str(score_input.final_state_path),
            "final_state_sha256": _sha256(score_input.final_state_path),
            "grade_path": str(grade_path.resolve()),
            "ok": True,
            "execution_grader": score_input.config.get(
                "execution_grader", score_input.config.get("grader")
            ),
            "final_grader": grader,
            "outcome_pass": bool(grade.get("outcome_pass")),
            "process_pass": bool(grade.get("process_pass")),
            "verdict": grade.get("verdict"),
            "false_pass_demo": bool(grade.get("false_pass_demo")),
            "rule_failures": [rule for rule in grade.get("rules", []) if not rule.get("passed")],
            "llm_judge_process_pass": (grade.get("llm_judge") or {}).get("process_pass"),
            "judge_model": score_input.judge_model,
            "score_elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - batch scoring records per-task failures.
        result = _error_result(
            score_input,
            f"{type(exc).__name__}: {exc}",
            grader=grader,
            elapsed=time.perf_counter() - started,
        )
        result["traceback"] = traceback.format_exc()
        return result


def _common_value(values: Sequence[str]) -> str:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "mixed"


def _score_manifest(
    plan: ScorePlan,
    *,
    grader: str,
    output_dir: Path,
    concurrency: int,
) -> dict[str, Any]:
    source = plan.source_manifest
    judge_models = [item.judge_model for item in plan.inputs if item.judge_model]
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": source.get("benchmark"),
        "model": source.get("model"),
        "wire_api": os.environ.get("OPENAI_WIRE_API") or source.get("wire_api"),
        "reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT"),
        "mode": _common_value([item.mode for item in plan.inputs]),
        "grader": grader,
        "judge_model": _common_value([str(value) for value in judge_models])
        if judge_models
        else None,
        "executor": "thread" if concurrency > 1 else "serial",
        "concurrency": concurrency,
        "task_count": len(plan.inputs),
        "source_kind": plan.source_kind,
        "source": str(plan.source),
        "source_dir": str(plan.source_dir),
        "source_manifest": str(plan.source_dir / "manifest.json")
        if (plan.source_dir / "manifest.json").is_file()
        else None,
        "source_artifacts_modified": False,
        "api_key_stored": False,
        "tasks": [
            {
                "task_id": item.task_id,
                "task_path": str(item.task_path),
                "task_sha256": _sha256(item.task_path),
                "trace_path": str(item.trace_path) if item.trace_path else None,
                "final_state_path": str(item.final_state_path) if item.final_state_path else None,
            }
            for item in plan.inputs
        ],
    }


def score_existing(
    plan: ScorePlan,
    output_dir: str | Path,
    *,
    grader: str = "hybrid",
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if concurrency < 1:
        raise ScoreError("concurrency must be at least 1")
    validate_score_plan(plan, grader=grader)
    destination = Path(output_dir).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ScoreError(f"output directory already exists: {destination}") from exc
    (destination / "grades").mkdir()

    manifest = _score_manifest(plan, grader=grader, output_dir=destination, concurrency=concurrency)
    _atomic_write_text(
        destination / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    partial_path = destination / "results_partial.jsonl"

    rows: list[dict[str, Any]] = []
    if concurrency == 1:
        for score_input in plan.inputs:
            row = _score_one(score_input, destination, grader)
            rows.append(row)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = [
                pool.submit(_score_one, score_input, destination, grader)
                for score_input in plan.inputs
            ]
            for future in futures.as_completed(pending):
                row = future.result()
                rows.append(row)
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows.sort(key=lambda row: str(row.get("task_id")))
    results_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    failures = [
        row
        for row in rows
        if not (row.get("ok") and row.get("outcome_pass") and row.get("process_pass"))
    ]
    _atomic_write_text(destination / "results.jsonl", results_text)
    _atomic_write_text(
        destination / "failures.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
    )
    report = write_model_report(destination)
    _atomic_write_text(destination / "metrics.json", json.dumps(report["overall"], indent=2) + "\n")
    return rows, report


def _first_value(
    cli_value: str | None,
    environment_names: Sequence[str],
    env_file: Mapping[str, str],
    fallback: Any = None,
) -> Any:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()
    for name in environment_names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    for name in environment_names:
        value = env_file.get(name)
        if value and value.strip():
            return value.strip()
    return fallback


@contextmanager
def _judge_environment(
    args: argparse.Namespace,
    plan: ScorePlan,
) -> Iterator[None]:
    values = _load_env_file(Path(args.env_file).expanduser().resolve())
    source = plan.source_manifest
    api_key = _first_value(args.api_key, ("CXBENCH_API_KEY", "OPENAI_API_KEY"), values)
    if not api_key or api_key == "replace_me":
        raise ScoreError("API key is required via --api-key, CXBENCH_API_KEY, or OPENAI_API_KEY")
    base_url = _first_value(
        args.base_url,
        ("CXBENCH_BASE_URL", "OPENAI_BASE_URL"),
        values,
        source.get("base_url") or "https://api.openai.com/v1",
    )
    wire_api = _first_value(
        args.wire_api,
        ("CXBENCH_WIRE_API", "OPENAI_WIRE_API"),
        values,
        source.get("wire_api") or "chat",
    )
    if wire_api not in {"chat", "responses"}:
        raise ScoreError("wire API must be 'chat' or 'responses'")
    max_output_tokens = _first_value(
        str(args.max_output_tokens) if args.max_output_tokens is not None else None,
        ("CXBENCH_MAX_OUTPUT_TOKENS", "OPENAI_MAX_OUTPUT_TOKENS"),
        values,
        source.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS,
    )
    try:
        if int(max_output_tokens) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ScoreError("max output tokens must be a positive integer") from exc

    updates = {
        "OPENAI_API_KEY": str(api_key),
        "OPENAI_BASE_URL": str(base_url).rstrip("/"),
        "OPENAI_WIRE_API": str(wire_api),
        "OPENAI_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if wire_api == "responses":
        reasoning_effort = _first_value(
            None,
            ("CXBENCH_REASONING_EFFORT", "OPENAI_REASONING_EFFORT"),
            values,
            source.get("reasoning_effort") or "low",
        )
        updates["OPENAI_REASONING_EFFORT"] = str(reasoning_effort)

    old = {name: os.environ.get(name) for name in _MANAGED_ENV_KEYS}
    os.environ.pop("OPENAI_REASONING_EFFORT", None)
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score existing cxbench traces without rerunning the agent or user simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        help="trace.jsonl, one task-run directory, or a model-eval batch directory",
    )
    parser.add_argument(
        "--output",
        help="new score directory; defaults to SOURCE/scores/<timestamp>-<grader>",
    )
    parser.add_argument("--task", help="task YAML override for one task run")
    parser.add_argument("--final-state", help="final_state.json override for one task run")
    parser.add_argument("--mode", choices=["least_privilege", "direct", "high_permission"])
    parser.add_argument(
        "--grader",
        choices=["hybrid", "rule", "llm"],
        default="hybrid",
        help="hybrid is the formal cxbench verdict; rule and llm are diagnostic only",
    )
    parser.add_argument(
        "--judge-model",
        help="judge model; bare names use the OpenAI-compatible endpoint",
    )
    parser.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--env-file", default=str(ENV_FILE), help="dotenv-style provider settings")
    parser.add_argument("--api-key", help="API key; prefer OPENAI_API_KEY in the env file")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL used only by the judge")
    parser.add_argument("--wire-api", choices=["chat", "responses"], help="judge transport")
    parser.add_argument("--max-output-tokens", type=_positive_int, help="judge output limit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs without scoring or writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_values = _load_env_file(Path(args.env_file).expanduser().resolve())
    judge_model = args.judge_model or _first_value(
        None,
        ("CXBENCH_JUDGE_MODEL",),
        env_values,
    )
    try:
        plan = build_score_plan(
            args.source,
            task_path=args.task,
            final_state_path=args.final_state,
            mode=args.mode,
            judge_model=judge_model,
        )
        validate_score_plan(plan, grader=args.grader)
        output_dir = (
            Path(args.output).expanduser().resolve()
            if args.output
            else default_output_dir(plan, grader=args.grader)
        )
        if output_dir.exists():
            raise ScoreError(f"output directory already exists: {output_dir}")
    except (OSError, ScoreError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print("cxbench existing-record scorer")
    print(f"  source: {plan.source}")
    print(f"  source_kind: {plan.source_kind}")
    print(f"  records: {len(plan.inputs)}")
    input_errors = [item for item in plan.inputs if item.error]
    print(f"  scoreable_records: {len(plan.inputs) - len(input_errors)}")
    print(f"  input_errors: {len(input_errors)}")
    print(f"  grader: {args.grader}")
    if args.grader in {"llm", "hybrid"}:
        models = _common_value([str(item.judge_model) for item in plan.inputs if item.judge_model])
        print(f"  judge_model: {models}")
    print(f"  output: {output_dir}")
    print("  source_artifacts_modified: no")
    if args.dry_run:
        for item in input_errors:
            print(f"  input_error {item.task_id}: {item.error}")
        print("dry-run complete; no API request was sent and no file was written")
        return 1 if input_errors else 0

    try:
        if args.grader in {"llm", "hybrid"}:
            with _judge_environment(args, plan):
                rows, report = score_existing(
                    plan,
                    output_dir,
                    grader=args.grader,
                    concurrency=args.concurrency,
                )
        else:
            rows, report = score_existing(
                plan,
                output_dir,
                grader=args.grader,
                concurrency=args.concurrency,
            )
    except KeyboardInterrupt:
        print("scoring interrupted; completed rows remain in results_partial.jsonl")
        return 130
    except (OSError, ScoreError, ValueError, json.JSONDecodeError) as exc:
        print(f"scoring failed: {exc}")
        return 2

    overall = report["overall"]
    print(f"report: {output_dir / 'report.md'}")
    print(
        f"both_pass={overall['both_pass_rate']:.2%} "
        f"outcome_pass={overall['outcome_pass_rate']:.2%} "
        f"process_pass={overall['process_pass_rate']:.2%} "
        f"errors={overall['error_count']}"
    )
    return 1 if any(not row.get("ok") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
