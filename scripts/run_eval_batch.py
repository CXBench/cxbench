#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxbench.eval_config import model_string  # noqa: E402
from cxbench.grader import write_grade  # noqa: E402
from cxbench.openai_compat import ProviderTransportError, apply_provider_env, restore_provider_env  # noqa: E402
from cxbench.report import write_report  # noqa: E402
from cxbench.runtime import Runtime, RuntimeTimeoutError  # noqa: E402
from cxbench.task_schema import TaskSpec  # noqa: E402


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = (
            sorted(ROOT.glob(pattern))
            if not Path(pattern).is_absolute()
            else sorted(Path("/").glob(pattern[1:]))
        )
        if not matched:
            raise FileNotFoundError(f"task pattern matched nothing: {pattern}")
        paths.extend(path for path in matched if path.is_file())
    dedup: dict[str, Path] = {}
    for path in paths:
        dedup[str(path.resolve())] = path.resolve()
    return list(dedup.values())


def _finish_attempt(runtime: Runtime | None, runs_dir: Path, failure: Exception | None) -> None:
    if runtime is None or not getattr(runtime, "run_dir", None):
        return
    run_dir = Path(runtime.run_dir)
    if run_dir.is_symlink() or run_dir.parent.resolve() != runs_dir.resolve():
        return
    if failure is None or (run_dir / "report.md").is_file():
        try:
            (run_dir / "drafts").rmdir()
        except OSError:
            pass
        return
    try:
        if getattr(runtime, "trace", None):
            runtime._write_trace()
        (run_dir / "error.json").write_text(
            json.dumps({"type": type(failure).__name__, "error": str(failure)}, indent=2) + "\n",
            encoding="utf-8",
        )
        archive_dir = runs_dir.parent / "failed_attempts"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{run_dir.name}.tar.gz"
        with tarfile.open(archive_path, "x:gz") as archive:
            archive.add(run_dir, arcname=run_dir.name)
        # Verify the archive before removing this worker's abandoned artifacts.
        with tarfile.open(archive_path, "r:gz") as archive:
            for path in run_dir.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    with archive.extractfile(
                        f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}"
                    ) as saved:
                        if hashlib.sha256(saved.read()).hexdigest() != _sha256(path):
                            raise OSError(f"archive verification failed: {path}")
        shutil.rmtree(run_dir)
    except (OSError, tarfile.TarError) as exc:
        print(f"failed attempt retained at {run_dir}: {exc}", file=sys.stderr, flush=True)


def _run_one(
    task_path: Path, args: argparse.Namespace, runs_dir: Path, runtime_grader: str
) -> dict[str, Any]:
    started = time.perf_counter()
    task: TaskSpec | None = None
    runtime: Runtime | None = None
    failure: Exception | None = None
    try:
        task = TaskSpec.load(task_path)
        runtime = Runtime(
            task=task,
            mode=args.mode,
            model=f"openai:{args.model}",
            runs_dir=runs_dir,
            grader=runtime_grader,
            judge_model=model_string(args.judge_model)
            if runtime_grader in {"llm", "hybrid"}
            else None,
            user_sim=args.user_sim,
            user_model=model_string(args.user_model),
            user_max_attempts=args.user_max_attempts,
            max_elapsed_seconds=args.task_timeout,
        )
        result = runtime.run()
        grade = result["grade"]
        trace_path = Path(result["run_dir"]) / "trace.jsonl"
        user_interactions = 0
        tool_calls = 0
        provider_calls = 0
        tool_budget_reached_count = 0
        interaction_budget_reached = 0
        provider_step_budget_reached = 0
        tool_call_blocked_by_budget = 0
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                event_type = event.get("event_type")
                if event_type == "assistant_message":
                    user_interactions += 1
                elif event_type == "tool_call":
                    tool_calls += 1
                elif event_type == "provider_call":
                    provider_calls += 1
                elif event_type == "tool_budget_reached":
                    tool_budget_reached_count += 1
                elif event_type == "interaction_budget_reached":
                    interaction_budget_reached += 1
                elif event_type == "provider_step_budget_reached":
                    provider_step_budget_reached += 1
                elif event_type == "tool_call_blocked_by_budget":
                    tool_call_blocked_by_budget += 1
        return {
            "task_id": task.id,
            "domain": task.domain,
            "scenario": task.scenario,
            "task_path": str(task_path),
            "run_dir": result["run_dir"],
            "ok": True,
            "execution_grader": runtime_grader,
            "final_grader": runtime_grader,
            "outcome_pass": bool(grade.get("outcome_pass")),
            "process_pass": bool(grade.get("process_pass")),
            "verdict": grade.get("verdict"),
            "false_pass_demo": bool(grade.get("false_pass_demo")),
            "rule_failures": [r for r in grade.get("rules", []) if not r.get("passed")],
            "llm_judge_process_pass": (grade.get("llm_judge") or {}).get("process_pass"),
            "turns": user_interactions,
            "user_interactions": user_interactions,
            "tool_calls": tool_calls,
            "provider_calls": provider_calls,
            "provider_steps": provider_calls,
            "forced_response_count": 0,
            "tool_budget_reached_count": tool_budget_reached_count,
            "interaction_budget_reached": bool(interaction_budget_reached),
            "provider_step_budget_reached": bool(provider_step_budget_reached),
            "tool_call_blocked_by_budget": tool_call_blocked_by_budget,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": None,
        }
    except ProviderTransportError as exc:
        failure = exc
        raise
    except RuntimeTimeoutError as exc:
        failure = exc
        return {
            "task_id": task.id if task else task_path.stem,
            "domain": task.domain if task else None,
            "scenario": task.scenario if task else None,
            "task_path": str(task_path),
            "run_dir": None,
            "ok": False,
            "execution_grader": runtime_grader,
            "final_grader": runtime_grader,
            "outcome_pass": False,
            "process_pass": False,
            "verdict": "timeout",
            "false_pass_demo": False,
            "rule_failures": [],
            "llm_judge_process_pass": None,
            "turns": 0,
            "user_interactions": 0,
            "tool_calls": 0,
            "provider_calls": 0,
            "provider_steps": 0,
            "forced_response_count": 0,
            "tool_budget_reached_count": 0,
            "interaction_budget_reached": False,
            "provider_step_budget_reached": False,
            "tool_call_blocked_by_budget": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"RuntimeTimeoutError: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 - batch runner must preserve failures as artifacts.
        failure = exc
        return {
            "task_id": task.id if task else task_path.stem,
            "domain": task.domain if task else None,
            "scenario": task.scenario if task else None,
            "task_path": str(task_path),
            "run_dir": None,
            "ok": False,
            "execution_grader": runtime_grader,
            "final_grader": runtime_grader,
            "outcome_pass": False,
            "process_pass": False,
            "verdict": "error",
            "false_pass_demo": False,
            "rule_failures": [],
            "llm_judge_process_pass": None,
            "turns": 0,
            "user_interactions": 0,
            "tool_calls": 0,
            "provider_calls": 0,
            "provider_steps": 0,
            "forced_response_count": 0,
            "tool_budget_reached_count": 0,
            "interaction_budget_reached": False,
            "provider_step_budget_reached": False,
            "tool_call_blocked_by_budget": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    finally:
        _finish_attempt(runtime, runs_dir, failure)


def _graded_result(result: dict[str, Any], config: dict, grade: dict) -> dict[str, Any]:
    return {
        **result,
        "execution_grader": config["execution_grader"],
        "final_grader": config["final_grader"],
        "outcome_pass": bool(grade.get("outcome_pass")),
        "process_pass": bool(grade.get("process_pass")),
        "verdict": grade.get("verdict"),
        "false_pass_demo": bool(grade.get("false_pass_demo")),
        "rule_failures": [r for r in grade.get("rules", []) if not r.get("passed")],
        "llm_judge_process_pass": (grade.get("llm_judge") or {}).get("process_pass"),
    }


def _regrade_one(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not result.get("ok") or not result.get("run_dir"):
        return result
    started = time.perf_counter()
    try:
        run_dir = Path(result["run_dir"])
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        task = TaskSpec.load(config["task_path"])
        grade = write_grade(
            run_dir,
            task,
            config["mode"],
            grader=args.grader,
            judge_model=model_string(args.judge_model),
        )
        config = {
            **config,
            "execution_grader": config.get("execution_grader", config.get("grader")),
            "final_grader": args.grader,
            "grader": args.grader,
            "judge_model": model_string(args.judge_model),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_report(run_dir, grade)
        return {
            **_graded_result(result, config, grade),
            "regrade_elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except ProviderTransportError:
        raise
    except Exception as exc:  # noqa: BLE001
        result = dict(result)
        result.update(
            {
                "ok": False,
                "verdict": "regrade_error",
                "process_pass": False,
                "error": f"regrade {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "regrade_elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return result


def _delayed_attempt(
    delay: float, worker: Callable[..., dict[str, Any]], arguments: tuple[Any, ...]
) -> dict[str, Any]:
    time.sleep(delay)
    return worker(*arguments)


def _completed_results(
    pool: futures.Executor,
    worker: Callable[..., dict[str, Any]],
    arguments: list[tuple[Any, ...]],
    *,
    retry_delay: float,
    stage: str,
) -> Iterator[dict[str, Any]]:
    pending = {pool.submit(worker, *values): (values, 1) for values in arguments}
    while pending:
        done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
        for future in done:
            values, attempt = pending.pop(future)
            try:
                result = future.result()
            except ProviderTransportError:
                for waiting in pending:
                    waiting.cancel()
                raise
            except Exception as exc:  # KeyboardInterrupt/SystemExit must still stop the batch.
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if result.get("ok") and result.get("verdict") not in {
                "error",
                "timeout",
                "regrade_error",
            }:
                yield {**result, "attempts" if stage == "agent" else "regrade_attempts": attempt}
                continue
            task_label = result.get("task_id") or (
                values[0].get("task_id") if isinstance(values[0], dict) else values[0]
            )
            print(
                f"{stage}_retry {task_label} attempt={attempt} {result.get('error') or result.get('verdict')}",
                flush=True,
            )
            # Reuse the original input so judge retries keep the completed trace.
            pending[pool.submit(_delayed_attempt, retry_delay, worker, values)] = (
                values,
                attempt + 1,
            )


def _rates(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    oks = [r for r in results if r.get("ok")]

    def rate(key: str) -> float:
        return round(sum(1 for r in results if r.get(key)) / n, 4) if n else 0.0

    both = sum(1 for r in results if r.get("outcome_pass") and r.get("process_pass"))
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted({str(r.get("domain") or "unknown") for r in results}):
        subset = [r for r in results if str(r.get("domain") or "unknown") == domain]
        by_domain[domain] = _rates(subset) if subset != results else {}
    elapsed = [float(r.get("elapsed_seconds") or 0) for r in oks]
    return {
        "n": n,
        "ok_count": len(oks),
        "error_count": n - len(oks),
        "outcome_pass_rate": rate("outcome_pass"),
        "process_pass_rate": rate("process_pass"),
        "both_pass_rate": round(both / n, 4) if n else 0.0,
        "false_pass_demo_rate": rate("false_pass_demo"),
        "avg_turns": round(statistics.mean([r.get("turns", 0) for r in oks]), 3) if oks else 0,
        "avg_user_interactions": round(
            statistics.mean([r.get("user_interactions", 0) for r in oks]), 3
        )
        if oks
        else 0,
        "avg_tool_calls": round(statistics.mean([r.get("tool_calls", 0) for r in oks]), 3)
        if oks
        else 0,
        "avg_provider_calls": round(statistics.mean([r.get("provider_calls", 0) for r in oks]), 3)
        if oks
        else 0,
        "avg_provider_steps": round(statistics.mean([r.get("provider_steps", 0) for r in oks]), 3)
        if oks
        else 0,
        "forced_response_count": sum(int(r.get("forced_response_count", 0)) for r in oks),
        "tool_budget_reached_count": sum(int(r.get("tool_budget_reached_count", 0)) for r in oks),
        "tool_call_blocked_by_budget_count": sum(
            int(r.get("tool_call_blocked_by_budget", 0)) for r in oks
        ),
        "interaction_budget_reached_count": sum(
            1 for r in oks if r.get("interaction_budget_reached")
        ),
        "provider_step_budget_reached_count": sum(
            1 for r in oks if r.get("provider_step_budget_reached")
        ),
        "elapsed_seconds_total_sum": round(sum(elapsed), 3),
        "elapsed_seconds_mean_per_task": round(statistics.mean(elapsed), 3) if elapsed else 0,
        "by_domain": by_domain,
    }


def _write_summary(
    out_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    failures = [
        r for r in results if not (r.get("ok") and r.get("outcome_pass") and r.get("process_pass"))
    ]
    lines = [
        f"# cxbench eval batch: {manifest['run_id']}",
        "",
        f"- model: `{manifest['model']}`",
        f"- agent_provider: `{manifest['agent_provider']}`",
        f"- grader: `{manifest['grader']}`",
        f"- judge_model: `{manifest['judge_model']}` via `{manifest['judge_provider']}`",
        f"- tasks: {metrics['n']}",
        f"- concurrency: {manifest['concurrency']}",
        f"- executor: `{manifest.get('executor', 'thread')}`",
        f"- reasoning_effort: `{manifest.get('reasoning_effort') or 'not set'}`",
        "",
        "## Metrics",
        "",
        f"- ok_count: {metrics['ok_count']}/{metrics['n']}",
        f"- outcome_pass_rate: {metrics['outcome_pass_rate']}",
        f"- process_pass_rate: {metrics['process_pass_rate']}",
        f"- both_pass_rate: {metrics['both_pass_rate']}",
        f"- false_pass_demo_rate: {metrics['false_pass_demo_rate']}",
        f"- avg_turns: {metrics['avg_turns']}",
        f"- avg_user_interactions: {metrics['avg_user_interactions']}",
        f"- avg_tool_calls: {metrics['avg_tool_calls']}",
        f"- avg_provider_steps: {metrics['avg_provider_steps']}",
        f"- tool_budget_reached_count: {metrics['tool_budget_reached_count']}",
        f"- interaction_budget_reached_count: {metrics['interaction_budget_reached_count']}",
        "",
        "## Failures",
        "",
    ]
    if not failures:
        lines.append("- none")
    else:
        for item in failures[:50]:
            lines.append(
                f"- {item.get('task_id')}: verdict={item.get('verdict')} outcome={item.get('outcome_pass')} "
                f"process={item.get('process_pass')} error={item.get('error') or ''} run={item.get('run_dir')}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser(
    *, default_task: str | None = None, default_runs_dir: str = "runs/eval_batches"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a cxbench task batch with isolated provider env."
    )
    parser.set_defaults(default_task=default_task)
    parser.add_argument(
        "--task",
        action="append",
        required=default_task is None,
        help="task glob; can be passed multiple times",
    )
    parser.add_argument("--model", required=True, help="agent model name without openai: prefix")
    parser.add_argument(
        "--agent-provider",
        choices=["openai", "deepseek", "openrouter", "siliconflow", "shubiaobiao"],
        default="openai",
    )
    parser.add_argument(
        "--mode",
        choices=["least_privilege", "direct", "high_permission"],
        default="least_privilege",
    )
    parser.add_argument("--grader", choices=["rule", "llm", "hybrid"], default="hybrid")
    parser.add_argument(
        "--judge-model",
        default="gpt-5.4-mini",
        help="judge model name, optionally prefixed with openai:/deepseek:/openrouter:/siliconflow:/shubiaobiao:",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["same", "openai", "deepseek", "openrouter", "siliconflow", "shubiaobiao"],
        default="openai",
    )
    parser.add_argument(
        "--user-sim",
        choices=["llm", "react", "verify", "reflection"],
        default="llm",
        help="LLM customer simulator strategy; deterministic is test-only and unavailable for eval batches.",
    )
    parser.add_argument(
        "--user-model",
        required=True,
        help="LLM user simulator model name, optionally prefixed with openai:/deepseek:/openrouter:/siliconflow:/shubiaobiao:",
    )
    parser.add_argument("--user-max-attempts", type=int, default=3)
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=600,
        help="per-task wall-clock timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="seconds between unlimited task/judge retries (default: 5)",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="parallel worker type; process workers isolate task runtimes and model clients",
    )
    parser.add_argument("--runs-dir", default=default_runs_dir)
    parser.add_argument("--run-id")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        raise ValueError("--retry-delay must be a finite non-negative number")


def run_batch(args: argparse.Namespace) -> int:
    validate_args(args)
    dotenv = _load_dotenv(ROOT / ".env")
    tasks = _task_paths(args.task or [args.default_task])
    for task_path in tasks:
        TaskSpec.load(task_path)
    run_id = (
        args.run_id
        or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.model.replace('/', '_').replace(':', '_')}"
    )
    out_dir = (ROOT / args.runs_dir / run_id).resolve()
    task_runs_dir = out_dir / "task_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    task_runs_dir.mkdir(parents=True, exist_ok=True)
    # Semantic judging retries the completed trace without rerunning the agent.
    runtime_grader = "rule"

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "agent_provider": args.agent_provider,
        "mode": args.mode,
        "grader": args.grader,
        "execution_grader": runtime_grader,
        "final_grader": args.grader,
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "user_sim": args.user_sim,
        "user_model": args.user_model,
        "task_timeout": args.task_timeout,
        "retry_delay": args.retry_delay,
        "concurrency": args.concurrency,
        "executor": args.executor,
        "wire_api": os.environ.get("OPENAI_WIRE_API"),
        "reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT"),
        "max_output_tokens": os.environ.get("OPENAI_MAX_OUTPUT_TOKENS"),
        "api_key_stored": False,
        "task_count": len(tasks),
        "tasks": [{"path": str(path), "sha256": _sha256(path)} for path in tasks],
    }
    partial_path = out_dir / "results_partial.jsonl"
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    old = apply_provider_env(args.agent_provider, dotenv)
    try:
        executor_type = (
            futures.ProcessPoolExecutor
            if args.executor == "process"
            else futures.ThreadPoolExecutor
        )
        with executor_type(max_workers=args.concurrency) as pool:
            results = []
            arguments = [(path, args, task_runs_dir, runtime_grader) for path in tasks]
            for item in _completed_results(
                pool, _run_one, arguments, retry_delay=args.retry_delay, stage="agent"
            ):
                results.append(item)
                # Flush each result immediately so partial progress survives interruptions.
                with partial_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(
                    f"agent_done {len(results)}/{len(tasks)} {item.get('task_id')} {item.get('verdict')} {item.get('error') or ''}",
                    flush=True,
                )
    finally:
        restore_provider_env(old)

    if args.grader in {"llm", "hybrid"} and runtime_grader != args.grader:
        judge_provider = (
            args.agent_provider if args.judge_provider == "same" else args.judge_provider
        )
        old = apply_provider_env(judge_provider, dotenv)
        try:
            executor_type = (
                futures.ProcessPoolExecutor
                if args.executor == "process"
                else futures.ThreadPoolExecutor
            )
            with executor_type(max_workers=args.concurrency) as pool:
                regraded = []
                arguments = [(item, args) for item in results]
                for item in _completed_results(
                    pool, _regrade_one, arguments, retry_delay=args.retry_delay, stage="judge"
                ):
                    regraded.append(item)
                    print(
                        f"judge_done {len(regraded)}/{len(results)} {item.get('task_id')} {item.get('verdict')} {item.get('error') or ''}",
                        flush=True,
                    )
                results = regraded
        finally:
            restore_provider_env(old)

    results = sorted(results, key=lambda r: str(r.get("task_id")))
    metrics = _rates(results)
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    failures = [
        r for r in results if not (r.get("ok") and r.get("outcome_pass") and r.get("process_pass"))
    ]
    (out_dir / "failures.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failures), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_summary(out_dir, manifest, metrics, results)
    print(f"out_dir: {out_dir}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["error_count"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return run_batch(args)
    except ProviderTransportError as exc:
        print(f"batch paused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
