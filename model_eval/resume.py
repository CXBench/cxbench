#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import signal
import shutil
import sys
from contextlib import contextmanager
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from cxbench.task_schema import TaskSpec
from scripts import run_eval_batch as batch

from .analysis import MODE_RUNTIME_NAMES, write_experiment_analysis
from .run import ENV_FILE, REPO_ROOT, _load_env_file, _resolve_value


DEFAULT_MODE_DIR = (
    REPO_ROOT / "runs" / "mode_ablation" / "20260903-113231-gpt-5.4-mini-direct-mediate" / "mediate"
)
DEFAULT_CONCURRENCY = 20
DEFAULT_TRANSPORT_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
MANAGED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_WIRE_API",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_MAX_OUTPUT_TOKENS",
    "OPENAI_MAX_RETRIES",
    "OPENAI_RETRY_BACKOFF_SECONDS",
    "PYTHONUTF8",
    "PYTHONUNBUFFERED",
)


class ResumeError(RuntimeError):
    """Raised when an existing batch cannot be resumed safely."""


@dataclass(frozen=True)
class ManifestTask:
    task_id: str
    path: Path


@dataclass(frozen=True)
class ResumeJob:
    task: ManifestTask
    previous: dict[str, Any] | None
    run_dir: Path | None

    @property
    def is_retry(self) -> bool:
        return self.previous is not None


@dataclass(frozen=True)
class ResumePlan:
    tasks: tuple[ManifestTask, ...]
    existing: dict[str, dict[str, Any]]
    retry_jobs: tuple[ResumeJob, ...]
    missing_jobs: tuple[ResumeJob, ...]
    skipped_task_ids: tuple[str, ...]

    @property
    def jobs(self) -> tuple[ResumeJob, ...]:
        return self.retry_jobs + self.missing_jobs


@dataclass(frozen=True)
class ModeState:
    mode_dir: Path
    manifest: dict[str, Any]
    tasks: tuple[ManifestTask, ...]
    partial_path: Path
    task_runs_dir: Path
    plan: ResumePlan
    source_signature: tuple[int, int] | None


def _positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _nonnegative_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResumeError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResumeError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeError(f"expected a JSON object in {path}")
    return value


def _portable_name(raw_path: str) -> str:
    return raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_manifest_task(raw_path: str) -> Path:
    supplied = Path(raw_path)
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.append(REPO_ROOT / supplied)
    candidates.append(
        REPO_ROOT / "benchmarks" / "cxbench-120-v20260708" / "tasks" / _portable_name(raw_path)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ResumeError(f"manifest task cannot be found on this platform: {raw_path}")


def load_manifest_tasks(manifest: dict[str, Any]) -> tuple[ManifestTask, ...]:
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ResumeError("manifest contains no tasks")

    tasks: list[ManifestTask] = []
    seen: set[str] = set()
    for entry in raw_tasks:
        if isinstance(entry, str):
            raw_path = entry
            expected_hash = None
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            raw_path = entry["path"]
            expected_hash = entry.get("sha256")
        else:
            raise ResumeError("manifest task entries must be paths or objects containing a path")
        task_path = _resolve_manifest_task(raw_path)
        if expected_hash and _sha256(task_path) != expected_hash:
            raise ResumeError(f"task changed since the original run: {task_path}")
        task_id = TaskSpec.load(task_path).id
        if task_id in seen:
            raise ResumeError(f"manifest contains duplicate task id: {task_id}")
        seen.add(task_id)
        tasks.append(ManifestTask(task_id=task_id, path=task_path))

    expected_count = manifest.get("task_count")
    if isinstance(expected_count, int) and expected_count != len(tasks):
        raise ResumeError(
            f"manifest task_count is {expected_count}, but {len(tasks)} tasks were loaded"
        )
    return tuple(tasks)


def load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ResumeError(f"partial results do not exist: {path}") from exc

    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResumeError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ResumeError(f"expected a JSON object on {path}:{line_number}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ResumeError(f"missing task_id on {path}:{line_number}")
        latest[task_id] = row
    return latest


def _retry_run_dir(task_id: str, result: dict[str, Any], task_runs_dir: Path) -> Path:
    raw_run_dir = result.get("run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
        raise ResumeError(f"error result for {task_id} has no reusable run_dir")
    leaf = _portable_name(raw_run_dir)
    if not leaf.startswith(f"{task_id}-"):
        raise ResumeError(f"run_dir for {task_id} has an unexpected name: {leaf}")

    parent = task_runs_dir.resolve()
    target = (task_runs_dir / leaf).resolve()
    if target.parent != parent:
        raise ResumeError(f"run_dir for {task_id} is outside task_runs: {raw_run_dir}")
    if target.is_symlink():
        raise ResumeError(f"refusing to clean symlinked run directory: {target}")
    if target.exists() and not target.is_dir():
        raise ResumeError(f"run_dir is not a directory: {target}")
    return target


def build_resume_plan(
    tasks: Sequence[ManifestTask],
    existing: dict[str, dict[str, Any]],
    task_runs_dir: Path,
) -> ResumePlan:
    task_ids = {task.task_id for task in tasks}
    unknown = sorted(set(existing) - task_ids)
    if unknown:
        raise ResumeError(
            f"partial results contain tasks absent from the manifest: {', '.join(unknown)}"
        )

    retry_jobs: list[ResumeJob] = []
    missing_jobs: list[ResumeJob] = []
    skipped: list[str] = []
    for task in tasks:
        previous = existing.get(task.task_id)
        if previous is None:
            missing_jobs.append(ResumeJob(task=task, previous=None, run_dir=None))
        elif previous.get("verdict") == "error":
            retry_jobs.append(
                ResumeJob(
                    task=task,
                    previous=previous,
                    run_dir=_retry_run_dir(task.task_id, previous, task_runs_dir),
                )
            )
        else:
            skipped.append(task.task_id)

    retry_dirs = [job.run_dir for job in retry_jobs]
    if len(set(retry_dirs)) != len(retry_dirs):
        raise ResumeError("multiple error records point to the same run directory")
    return ResumePlan(
        tasks=tuple(tasks),
        existing=dict(existing),
        retry_jobs=tuple(retry_jobs),
        missing_jobs=tuple(missing_jobs),
        skipped_task_ids=tuple(skipped),
    )


def clean_retry_directories(jobs: Sequence[ResumeJob]) -> None:
    targets = [job.run_dir for job in jobs]
    if any(target is None for target in targets):
        raise ResumeError("an error retry is missing its original run directory")

    for target in targets:
        assert target is not None
        if target.is_symlink():
            raise ResumeError(f"refusing to clean symlinked run directory: {target}")
        if target.exists() and not target.is_dir():
            raise ResumeError(f"run_dir is not a directory: {target}")
        target.mkdir(parents=False, exist_ok=True)
        if target.is_symlink():
            raise ResumeError(f"refusing to clean symlinked run directory: {target}")
        for child in target.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                raise ResumeError(f"unsupported artifact type in {target}: {child}")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_partial_results(
    path: Path,
    tasks: Sequence[ManifestTask],
    results: dict[str, dict[str, Any]],
) -> None:
    task_ids = {task.task_id for task in tasks}
    unknown = sorted(set(results) - task_ids)
    if unknown:
        raise ResumeError(
            f"cannot write results for tasks absent from the manifest: {', '.join(unknown)}"
        )
    rows = [results[task.task_id] for task in tasks if task.task_id in results]
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, content)


def _batch_args(manifest: dict[str, Any]) -> dict[str, Any]:
    required = (
        "mode",
        "model",
        "judge_model",
        "user_sim",
        "user_model",
        "task_timeout",
    )
    missing = [name for name in required if manifest.get(name) is None]
    if missing:
        raise ResumeError(f"manifest is missing runtime settings: {', '.join(missing)}")
    return {
        "mode": manifest["mode"],
        "model": manifest["model"],
        "judge_model": manifest["judge_model"],
        "user_sim": manifest["user_sim"],
        "user_model": manifest["user_model"],
        "user_max_attempts": int(manifest.get("user_max_attempts", 3)),
        "task_timeout": int(manifest["task_timeout"]),
    }


def _run_worker(
    task_path: str,
    args_values: dict[str, Any],
    task_runs_dir: str,
    runtime_grader: str,
    run_dir: str | None,
) -> dict[str, Any]:
    return batch._run_one(
        Path(task_path),
        argparse.Namespace(**args_values),
        Path(task_runs_dir),
        runtime_grader,
        run_dir=Path(run_dir) if run_dir else None,
    )


def _process_worker_initializer() -> None:
    if os.name != "nt":
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _next_attempt(previous: dict[str, Any]) -> int:
    raw_attempt = previous.get("attempt", 1)
    try:
        return max(2, int(raw_attempt) + 1)
    except (TypeError, ValueError):
        return 2


def _worker_error(job: ResumeJob, exc: BaseException) -> dict[str, Any]:
    return {
        "task_id": job.task.task_id,
        "domain": job.previous.get("domain") if job.previous else None,
        "scenario": job.previous.get("scenario") if job.previous else None,
        "task_path": str(job.task.path),
        "run_dir": str(job.run_dir) if job.run_dir else None,
        "ok": False,
        "outcome_pass": False,
        "process_pass": False,
        "verdict": "error",
        "false_pass_demo": False,
        "rule_failures": [],
        "llm_judge_failures": [],
        "elapsed_seconds": 0,
        "error": f"worker {type(exc).__name__}: {exc}",
    }


def run_jobs(
    plan: ResumePlan,
    *,
    partial_path: Path,
    task_runs_dir: Path,
    manifest: dict[str, Any],
    concurrency: int,
    executor_name: str,
) -> dict[str, dict[str, Any]]:
    results = dict(plan.existing)
    jobs = plan.jobs
    if not jobs:
        write_partial_results(partial_path, plan.tasks, results)
        return results

    runtime_grader = str(manifest.get("execution_grader") or manifest.get("grader"))
    final_grader = str(manifest.get("final_grader") or manifest.get("grader"))
    if runtime_grader != final_grader:
        raise ResumeError("resume requires execution_grader and final_grader to match")
    args_values = _batch_args(manifest)
    executor_type = (
        futures.ProcessPoolExecutor if executor_name == "process" else futures.ThreadPoolExecutor
    )
    worker_count = min(concurrency, len(jobs))

    executor_kwargs: dict[str, Any] = {"max_workers": worker_count}
    if executor_name == "process":
        executor_kwargs["initializer"] = _process_worker_initializer
    pool = executor_type(**executor_kwargs)
    pending: dict[futures.Future[dict[str, Any]], ResumeJob] = {}
    try:
        pending = {
            pool.submit(
                _run_worker,
                str(job.task.path),
                args_values,
                str(task_runs_dir),
                runtime_grader,
                str(job.run_dir) if job.run_dir else None,
            ): job
            for job in jobs
        }
        completed = 0
        for future in futures.as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
                if result.get("task_id") != job.task.task_id:
                    raise ResumeError(
                        f"worker returned task_id {result.get('task_id')!r} for {job.task.task_id}"
                    )
            except BrokenProcessPool:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve a resumable row for worker failures.
                result = _worker_error(job, exc)

            if job.previous is None:
                result = batch._with_attempt_metadata(result, attempt=1)
            else:
                result = batch._with_attempt_metadata(
                    result,
                    attempt=_next_attempt(job.previous),
                    initial_result=job.previous,
                )
            results[job.task.task_id] = result
            write_partial_results(partial_path, plan.tasks, results)
            completed += 1
            error_lines = str(result.get("error") or "").splitlines()
            error = error_lines[0][:240] if error_lines else ""
            print(
                f"resume_done {completed}/{len(jobs)} {job.task.task_id} "
                f"{result.get('verdict')} {error}",
                flush=True,
            )
    except BaseException:
        for future in pending:
            future.cancel()
        if executor_name == "process":
            processes = getattr(pool, "_processes", {}) or {}
            for process in processes.values():
                process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results


def _ordered_results(
    tasks: Sequence[ManifestTask],
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [task.task_id for task in tasks if task.task_id not in results]
    if missing:
        raise ResumeError(f"resume finished without results for: {', '.join(missing)}")
    return [results[task.task_id] for task in tasks]


def write_final_outputs(
    mode_dir: Path,
    manifest: dict[str, Any],
    tasks: Sequence[ManifestTask],
    results: dict[str, dict[str, Any]],
    *,
    concurrency: int,
    executor_name: str,
    retried_count: int,
    missing_count: int,
) -> dict[str, Any]:
    rows = sorted(_ordered_results(tasks, results), key=lambda row: str(row.get("task_id")))
    metrics = batch._rates(rows)
    failures = [
        row
        for row in rows
        if not (row.get("ok") and row.get("outcome_pass") and row.get("process_pass"))
    ]
    _atomic_write_text(
        mode_dir / "results.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    _atomic_write_text(
        mode_dir / "failures.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
    )
    _atomic_write_text(
        mode_dir / "metrics.json", json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )

    updated_manifest = dict(manifest)
    updated_manifest.setdefault("initial_concurrency", manifest.get("concurrency"))
    updated_manifest["concurrency"] = concurrency
    updated_manifest["resume_executor"] = executor_name
    history = list(updated_manifest.get("resume_history") or [])
    history.append(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "concurrency": concurrency,
            "executor": executor_name,
            "retried_error_count": retried_count,
            "previously_missing_count": missing_count,
        }
    )
    updated_manifest["resume_history"] = history
    _atomic_write_text(
        mode_dir / "manifest.json",
        json.dumps(updated_manifest, indent=2, ensure_ascii=False) + "\n",
    )
    batch._write_summary(mode_dir, updated_manifest, metrics, rows)
    return metrics


@contextmanager
def _provider_environment(
    launcher: dict[str, Any],
    *,
    env_file: Path,
    api_key: str | None,
    transport_retries: int,
    retry_backoff_seconds: float,
) -> Iterator[None]:
    values = _load_env_file(env_file.expanduser().resolve())
    resolved_key = _resolve_value(api_key, ("CXBENCH_API_KEY", "OPENAI_API_KEY"), values)
    if not resolved_key or resolved_key == "replace_me":
        raise ResumeError("API key is required via --api-key or CXBENCH_API_KEY")

    required = ["base_url", "wire_api", "max_output_tokens"]
    if launcher.get("wire_api") == "responses":
        required.append("reasoning_effort")
    missing = [name for name in required if launcher.get(name) in {None, ""}]
    if missing:
        raise ResumeError(f"launcher.json is missing endpoint settings: {', '.join(missing)}")
    updates = {
        "OPENAI_API_KEY": resolved_key,
        "OPENAI_BASE_URL": str(launcher["base_url"]),
        "OPENAI_WIRE_API": str(launcher["wire_api"]),
        "OPENAI_MAX_OUTPUT_TOKENS": str(launcher["max_output_tokens"]),
        "OPENAI_MAX_RETRIES": str(transport_retries),
        "OPENAI_RETRY_BACKOFF_SECONDS": str(retry_backoff_seconds),
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if launcher["wire_api"] == "responses":
        updates["OPENAI_REASONING_EFFORT"] = str(launcher["reasoning_effort"])
    old = {name: os.environ.get(name) for name in MANAGED_ENV_KEYS}
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


@contextmanager
def _resume_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
    except OSError as exc:
        handle.close()
        raise ResumeError(f"another resume process holds {path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        handle.close()


def _validate_mode_dir(raw_mode_dir: str) -> Path:
    mode_dir = Path(raw_mode_dir).expanduser().resolve()
    runs_root = (REPO_ROOT / "runs").resolve()
    try:
        mode_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ResumeError(f"mode directory must be inside {runs_root}") from exc
    if not mode_dir.is_dir():
        raise ResumeError(f"mode directory does not exist: {mode_dir}")
    return mode_dir


def _validate_manifests(mode_dir: Path, manifest: dict[str, Any], launcher: dict[str, Any]) -> None:
    if manifest.get("run_id") != mode_dir.name:
        raise ResumeError("mode directory name does not match manifest run_id")
    if manifest.get("agent_provider") != "openai" or manifest.get("judge_provider") not in {
        "same",
        "openai",
    }:
        raise ResumeError(
            "this resume entrypoint currently requires the model_eval OpenAI-compatible provider setup"
        )
    for key in ("model", "grader"):
        if launcher.get(key) != manifest.get(key):
            raise ResumeError(f"launcher and mode manifest disagree on {key}")


def _new_mode_manifest(
    template: dict[str, Any],
    *,
    mode_label: str,
    concurrency: int,
) -> dict[str, Any]:
    if mode_label not in MODE_RUNTIME_NAMES:
        raise ResumeError(f"cannot initialize unknown mode: {mode_label}")
    manifest = json.loads(json.dumps(template))
    for key in ("initial_concurrency", "resume_executor", "resume_history"):
        manifest.pop(key, None)
    manifest.update(
        {
            "run_id": mode_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": MODE_RUNTIME_NAMES[mode_label],
            "concurrency": concurrency,
            "retry_concurrency": concurrency,
        }
    )
    return manifest


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def prepare_mode_state(
    mode_dir: Path,
    launcher: dict[str, Any],
    *,
    template_manifest: dict[str, Any] | None = None,
    concurrency: int,
    create: bool,
) -> ModeState:
    manifest_path = mode_dir / "manifest.json"
    partial_path = mode_dir / "results_partial.jsonl"
    task_runs_dir = mode_dir / "task_runs"

    if mode_dir.exists():
        if not mode_dir.is_dir():
            raise ResumeError(f"mode path is not a directory: {mode_dir}")
        manifest = _read_json(manifest_path)
        if partial_path.is_file():
            existing = load_latest_results(partial_path)
        elif (mode_dir / "results.jsonl").is_file():
            existing = load_latest_results(mode_dir / "results.jsonl")
            if create:
                write_partial_results(partial_path, load_manifest_tasks(manifest), existing)
        else:
            existing = {}
            if create:
                _atomic_write_text(partial_path, "")
        if create:
            task_runs_dir.mkdir(parents=True, exist_ok=True)
    else:
        if template_manifest is None:
            raise ResumeError(f"mode directory does not exist: {mode_dir}")
        manifest = _new_mode_manifest(
            template_manifest,
            mode_label=mode_dir.name,
            concurrency=concurrency,
        )
        existing = {}
        if create:
            mode_dir.mkdir(parents=False, exist_ok=False)
            task_runs_dir.mkdir()
            _atomic_write_text(
                manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
            _atomic_write_text(partial_path, "")

    _validate_manifests(mode_dir, manifest, launcher)
    tasks = load_manifest_tasks(manifest)
    plan = build_resume_plan(tasks, existing, task_runs_dir)
    signature = _file_signature(partial_path) if partial_path.is_file() else None
    return ModeState(
        mode_dir=mode_dir,
        manifest=manifest,
        tasks=tasks,
        partial_path=partial_path,
        task_runs_dir=task_runs_dir,
        plan=plan,
        source_signature=signature,
    )


def execute_mode_state(
    state: ModeState,
    *,
    concurrency: int,
    executor_name: str,
) -> dict[str, Any]:
    if state.source_signature is None:
        raise ResumeError(f"partial results were not initialized for {state.mode_dir}")
    with _resume_lock(state.mode_dir / ".resume.lock"):
        if _file_signature(state.partial_path) != state.source_signature:
            raise ResumeError(
                f"{state.partial_path} changed while the resume plan was being prepared; retry"
            )
        clean_retry_directories(state.plan.retry_jobs)
        results = run_jobs(
            state.plan,
            partial_path=state.partial_path,
            task_runs_dir=state.task_runs_dir,
            manifest=state.manifest,
            concurrency=concurrency,
            executor_name=executor_name,
        )
        return write_final_outputs(
            state.mode_dir,
            state.manifest,
            state.tasks,
            results,
            concurrency=concurrency,
            executor_name=executor_name,
            retried_count=len(state.plan.retry_jobs),
            missing_count=len(state.plan.missing_jobs),
        )


def write_paired_analysis(
    experiment_dir: Path,
    launcher: dict[str, Any],
    tasks: Sequence[ManifestTask],
    mode_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exit_codes = {
        mode: 0 if metrics["error_count"] == 0 else 1 for mode, metrics in mode_metrics.items()
    }
    analysis = write_experiment_analysis(
        experiment_dir=experiment_dir,
        task_paths=[task.path for task in tasks],
        mode_dirs={mode: experiment_dir / mode for mode in MODE_RUNTIME_NAMES},
        model=str(launcher["model"]),
        grader=str(launcher["grader"]),
        benchmark=str(launcher["benchmark"]),
        exit_codes=exit_codes,
    )
    updated_launcher = dict(launcher)
    updated_launcher["exit_codes"] = exit_codes
    updated_launcher["analysis_complete"] = True
    updated_launcher["resume_completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_text(
        experiment_dir / "launcher.json",
        json.dumps(updated_launcher, indent=2, ensure_ascii=False) + "\n",
    )
    return analysis


def _print_plan(
    mode_dir: Path, plan: ResumePlan, concurrency: int, executor_name: str, dry_run: bool
) -> None:
    print(f"cxbench mode-ablation resume: {mode_dir.name}")
    print(f"  mode_dir: {mode_dir}")
    print(f"  manifest_tasks: {len(plan.tasks)}")
    print(f"  existing_unique_results: {len(plan.existing)}")
    print(f"  skipped_existing_non_errors: {len(plan.skipped_task_ids)}")
    print(f"  retried_errors: {len(plan.retry_jobs)}")
    print(f"  previously_missing: {len(plan.missing_jobs)}")
    print(f"  scheduled: {len(plan.jobs)}")
    print(f"  executor: {executor_name}")
    print(f"  concurrency: {min(concurrency, len(plan.jobs)) if plan.jobs else 0}")
    for job in plan.retry_jobs:
        print(f"  reuse_error_dir: {job.run_dir}")
    if dry_run:
        print("dry-run complete; no directory was cleaned and no API request was sent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the interrupted mediate ablation in place. Existing non-error verdicts are skipped; "
            "error rows are replaced after their original run directories are cleaned."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode-dir", default=str(DEFAULT_MODE_DIR), help="existing mode output directory"
    )
    parser.add_argument("--env-file", default=str(ENV_FILE), help="file containing CXBENCH_API_KEY")
    parser.add_argument("--api-key", help="API key; prefer CXBENCH_API_KEY in model_eval/.env")
    parser.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument(
        "--continue-direct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after mediate, create or resume direct in the same timestamped experiment directory",
    )
    parser.add_argument(
        "--transport-retries",
        type=_positive_int,
        default=DEFAULT_TRANSPORT_RETRIES,
        help="retries per truncated, disconnected, or retryable HTTP model request",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=_nonnegative_float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="base exponential backoff between transport retries",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show exactly what would run without changing files"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode_metrics: dict[str, dict[str, Any]] = {}
    try:
        mode_dir = _validate_mode_dir(args.mode_dir)
        experiment_dir = mode_dir.parent
        launcher = _read_json(experiment_dir / "launcher.json")
        state = prepare_mode_state(
            mode_dir,
            launcher,
            concurrency=args.concurrency,
            create=False,
        )
        launcher_modes = {
            item.get("label") for item in launcher.get("modes", []) if isinstance(item, dict)
        }
        continue_direct = args.continue_direct and mode_dir.name == "mediate"
        if continue_direct and "direct" not in launcher_modes:
            raise ResumeError("launcher.json does not declare a direct phase")

        _print_plan(state.mode_dir, state.plan, args.concurrency, args.executor, args.dry_run)
        if args.dry_run:
            if continue_direct:
                direct_state = prepare_mode_state(
                    experiment_dir / "direct",
                    launcher,
                    template_manifest=state.manifest,
                    concurrency=args.concurrency,
                    create=False,
                )
                _print_plan(
                    direct_state.mode_dir,
                    direct_state.plan,
                    args.concurrency,
                    args.executor,
                    True,
                )
            return 0

        with _provider_environment(
            launcher,
            env_file=Path(args.env_file),
            api_key=args.api_key,
            transport_retries=args.transport_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        ):
            mode_metrics[state.mode_dir.name] = execute_mode_state(
                state,
                concurrency=args.concurrency,
                executor_name=args.executor,
            )
            if continue_direct:
                direct_state = prepare_mode_state(
                    experiment_dir / "direct",
                    launcher,
                    template_manifest=state.manifest,
                    concurrency=args.concurrency,
                    create=True,
                )
                _print_plan(
                    direct_state.mode_dir,
                    direct_state.plan,
                    args.concurrency,
                    args.executor,
                    False,
                )
                mode_metrics["direct"] = execute_mode_state(
                    direct_state,
                    concurrency=args.concurrency,
                    executor_name=args.executor,
                )
        if continue_direct:
            analysis = write_paired_analysis(experiment_dir, launcher, state.tasks, mode_metrics)
            print(f"paired_summary: {experiment_dir / 'summary.md'}")
            print(f"paired_ok_count: {analysis['paired_ok_count']}/{analysis['task_count']}")
    except (OSError, ResumeError, ValueError) as exc:
        print(f"resume error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "resume interrupted; completed replacements remain in results_partial.jsonl",
            file=sys.stderr,
        )
        return 130

    for mode, metrics in mode_metrics.items():
        print(f"{mode}_results: {experiment_dir / mode / 'results.jsonl'}")
        print(f"{mode}_ok_count: {metrics['ok_count']}/{metrics['n']}")
        print(f"{mode}_error_count: {metrics['error_count']}")
    return 0 if all(metrics["error_count"] == 0 for metrics in mode_metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
