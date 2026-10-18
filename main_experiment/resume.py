"""Run or resume a main experiment with the same options as main_experiment.run.

Use the original --runs-dir and --run-id to resume that batch. A task_runs
directory containing report.md is reused, including a failing verdict. Missing
final grading is completed on its recorded trace; completed judges are reused.
Without an existing run ID, execution delegates to main_experiment.run.
As in that runner, rule is only an intermediate execution grade; formal paper
scores still require hybrid. Keep the original task selection and model options.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from cxbench.eval_config import RuntimeConfig, model_string, normalize_tool_mode
from cxbench.task_schema import TaskSpec

from . import run, score

batch = run.batch


def build_parser() -> argparse.ArgumentParser:
    parser = run.build_parser()
    parser.prog = "python -m main_experiment.resume"
    parser.description = __doc__
    return parser


def _write(path: Path, text: str) -> None:
    # Replace snapshots atomically so an interrupted resume remains recoverable.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_results(path: Path, results: dict[str, dict]) -> None:
    _write(
        path,
        "".join(json.dumps(results[key], ensure_ascii=False) + "\n" for key in sorted(results)),
    )


def _recorded_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # An interrupted append can leave an unfinished last record.
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise ValueError(f"invalid result record: {path}:{index + 1}") from None
        if not isinstance(value, dict):
            raise ValueError(f"expected a result object: {path}:{index + 1}")
        rows.append(value)
    return rows


def _validate(args: argparse.Namespace, out_dir: Path) -> tuple[dict, dict[str, TaskSpec], Path]:
    batch.validate_args(args)
    for name in ("concurrency", "user_max_attempts", "task_timeout", "max_output_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.base_url and args.agent_provider != "openai":
        raise ValueError("--base-url requires --agent-provider openai")
    RuntimeConfig.parse(
        mode=args.mode,
        model=model_string(args.model),
        runs_dir=args.runs_dir,
        grader=args.grader,
        judge_model=model_string(args.judge_model),
        user_sim=args.user_sim,
        user_model=model_string(args.user_model),
        user_max_attempts=args.user_max_attempts,
        max_elapsed_seconds=args.task_timeout,
    )
    paths = batch._task_paths(args.task or [args.default_task])
    tasks = {task.id: task for task in (TaskSpec.load(path) for path in paths)}
    if not tasks or len(tasks) != len(paths):
        raise ValueError("select nonempty task files with unique task IDs")
    manifest = score._json(out_dir / "manifest.json")
    for name in ("model", "agent_provider", "user_sim"):
        if manifest.get(name) != getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} differs from the existing batch")
    if normalize_tool_mode(manifest.get("mode", "")) != normalize_tool_mode(args.mode):
        raise ValueError("--mode differs from the existing batch")
    if manifest.get("final_grader", manifest.get("grader")) != args.grader:
        raise ValueError("--grader differs from the existing batch")
    for name in ("user_model", "judge_model"):
        if model_string(manifest.get(name, "")) != model_string(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} differs from the existing batch")
    old_judge_provider = manifest.get("judge_provider")
    if old_judge_provider == "same":
        old_judge_provider = manifest["agent_provider"]
    judge_provider = args.agent_provider if args.judge_provider == "same" else args.judge_provider
    if old_judge_provider != judge_provider:
        raise ValueError("--judge-provider differs from the existing batch")
    if manifest.get("max_output_tokens") is not None:
        if int(manifest["max_output_tokens"]) != args.max_output_tokens:
            raise ValueError("--max-output-tokens differs from the existing batch")
    if args.wire_api and manifest.get("wire_api") not in (None, args.wire_api):
        raise ValueError("--wire-api differs from the existing batch")
    planned = manifest.get("tasks", [])
    if not planned or manifest.get("task_count") != len(planned):
        raise ValueError("manifest tasks/task_count is missing or inconsistent")
    recorded = {score._task_path(item["path"]): item["sha256"] for item in planned}
    selected = {path.resolve(): batch._sha256(path) for path in paths}
    if len(recorded) != len(planned) or recorded != selected:
        raise ValueError("selected tasks or task sha256 differ from the existing batch")
    score_dir = Path(args.score_output).resolve() if args.score_output else out_dir / "paper_scores"
    if args.grader == "hybrid":
        if (
            score_dir == out_dir
            or out_dir.is_relative_to(score_dir)
            or score_dir.is_relative_to(out_dir / "task_runs")
        ):
            raise ValueError("score directory must be separate from the batch and its task runs")
        if score_dir.exists() and not score_dir.is_dir():
            raise ValueError(f"score output is not a directory: {score_dir}")
    return manifest, tasks, score_dir


def _restore(
    directory: Path,
    task: TaskSpec,
    args: argparse.Namespace,
    previous: dict,
) -> tuple[dict, dict]:
    config = score._json(directory / "config.json")
    grade = score._json(directory / "grade.json")
    trace = score._jsonl(directory / "trace.jsonl")
    score._json(directory / "final_state.json")
    expected = {
        "task_id": task.id,
        "model": model_string(args.model),
        "user_model": model_string(args.user_model),
        "user_sim": args.user_sim,
        "user_max_attempts": args.user_max_attempts,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"recorded {name} differs from requested configuration: {directory}")
    if (
        score._task_path(config.get("task_path", "")) != task.path.resolve()
        or normalize_tool_mode(config.get("mode", "")) != normalize_tool_mode(args.mode)
        or grade.get("task_id") != task.id
    ):
        raise ValueError(f"recorded task or mode differs: {directory}")
    if grade.get("verdict") not in {"pass", "fail"}:
        raise ValueError(f"report has no completed grade: {directory}")
    final_grader = grade.get("grader")
    if final_grader not in {"rule", args.grader}:
        raise ValueError(f"incompatible recorded grader: {directory}")
    if final_grader in {"hybrid", "llm"}:
        judge = grade.get("llm_judge")
        if (
            not isinstance(judge, dict)
            or type(judge.get("process_pass")) is not bool
            or not isinstance(judge.get("findings"), list)
            or any(
                not isinstance(item, dict)
                or type(item.get("passed")) is not bool
                or not isinstance(item.get("rule_id"), str)
                or not isinstance(item.get("evidence"), str)
                for item in judge["findings"]
            )
        ):
            raise ValueError(f"report has malformed judge evidence: {directory}")
        if config.get("judge_model") not in (None, model_string(args.judge_model)):
            raise ValueError(f"recorded judge model differs: {directory}")
    counts = Counter(event.get("event_type") for event in trace)
    result = {
        **previous,
        "task_id": task.id,
        "domain": task.domain,
        "scenario": task.scenario,
        "task_path": str(task.path.resolve()),
        "run_dir": str(directory),
        "ok": True,
        "execution_grader": config.get("execution_grader", "rule"),
        "final_grader": final_grader,
        "verdict": grade["verdict"],
        "outcome_pass": bool(grade.get("outcome_pass")),
        "process_pass": bool(grade.get("process_pass")),
        "false_pass_demo": bool(grade.get("false_pass_demo")),
        "rule_failures": [item for item in grade.get("rules", []) if not item.get("passed")],
        "llm_judge_process_pass": (grade.get("llm_judge") or {}).get("process_pass"),
        "turns": counts["assistant_message"],
        "user_interactions": counts["assistant_message"],
        "tool_calls": counts["tool_call"],
        "provider_calls": counts["provider_call"],
        "provider_steps": counts["provider_call"],
        "forced_response_count": 0,
        "tool_budget_reached_count": counts["tool_budget_reached"],
        "interaction_budget_reached": bool(counts["interaction_budget_reached"]),
        "provider_step_budget_reached": bool(counts["provider_step_budget_reached"]),
        "tool_call_blocked_by_budget": counts["tool_call_blocked_by_budget"],
        "elapsed_seconds": previous.get("elapsed_seconds", 0),
        "error": None,
    }
    result.pop("traceback", None)
    config["task_path"] = result["task_path"]
    # A judge may have written grade.json just before an interruption. Repair its
    # metadata offline instead of paying to judge that same trace again.
    config.update(
        execution_grader=result["execution_grader"], final_grader=final_grader, grader=final_grader
    )
    if final_grader in {"hybrid", "llm"}:
        config["judge_model"] = model_string(args.judge_model)
    return result, config


def _recover(
    out_dir: Path, tasks: dict[str, TaskSpec], args: argparse.Namespace
) -> tuple[dict, dict]:
    task_runs = (out_dir / "task_runs").resolve()
    candidates: dict[str, list[tuple[Path, dict]]] = {key: [] for key in tasks}
    seen = set()
    for filename in ("results.jsonl", "results_partial.jsonl"):
        for row in _recorded_rows(out_dir / filename):
            task_id = row.get("task_id")
            if task_id not in tasks or not isinstance(row.get("run_dir"), str):
                continue
            # Always use this batch's copy, even when the recorded absolute path
            # still exists in a different checkout or under an old batch name.
            name = row["run_dir"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            directory = (task_runs / name).resolve()
            if not directory.is_relative_to(task_runs):
                raise ValueError(f"recorded run is outside task_runs: {directory}")
            if (directory / "report.md").is_file():
                candidates[task_id].append((directory, row))
                seen.add(directory)
    for report in sorted(task_runs.glob("*/report.md")):
        if not report.is_file():
            continue
        directory = report.parent.resolve()
        if directory in seen:
            continue
        if not directory.is_relative_to(task_runs):
            raise ValueError(f"recorded run is outside task_runs: {directory}")
        config = score._json(directory / "config.json")
        task_id = config.get("task_id")
        if task_id not in tasks:
            raise ValueError(f"report belongs to an unplanned task: {directory}")
        candidates[task_id].append((directory, {}))
    results, configs = {}, {}
    for task_id, options in candidates.items():
        if options:
            directory, previous = options[0]
            results[task_id], configs[directory] = _restore(
                directory, tasks[task_id], args, previous
            )
    return results, configs


def _stage(
    args: argparse.Namespace, out_dir: Path, results: dict, arguments: list, *, stage: str
) -> None:
    if not arguments:
        return
    provider = args.agent_provider
    worker = batch._run_one
    if stage == "judge":
        provider = provider if args.judge_provider == "same" else args.judge_provider
        worker = batch._regrade_one
    old = batch.apply_provider_env(provider, batch._load_dotenv(batch.ROOT / ".env"))
    try:
        executor = (
            batch.futures.ProcessPoolExecutor
            if args.executor == "process"
            else batch.futures.ThreadPoolExecutor
        )
        with executor(max_workers=args.concurrency) as pool:
            for index, item in enumerate(
                batch._completed_results(
                    pool,
                    worker,
                    arguments,
                    retry_delay=args.retry_delay,
                    stage=stage,
                ),
                1,
            ):
                results[item["task_id"]] = item
                _write_results(out_dir / "results_partial.jsonl", results)
                print(
                    f"{stage}_done {index}/{len(arguments)} {item['task_id']} {item['verdict']}",
                    flush=True,
                )
    finally:
        batch.restore_provider_env(old)


def _finish(
    args: argparse.Namespace, out_dir: Path, score_dir: Path, manifest: dict, results: dict
) -> None:
    ordered = [results[key] for key in sorted(results)]
    metrics = batch._rates(ordered)
    _write_results(out_dir / "results.jsonl", results)
    _write_results(
        out_dir / "failures.jsonl",
        {
            key: item
            for key, item in results.items()
            if not (item.get("ok") and item.get("outcome_pass") and item.get("process_pass"))
        },
    )
    _write_json(out_dir / "metrics.json", metrics)
    batch._write_summary(out_dir, manifest, metrics, ordered)
    if args.grader == "hybrid":
        # The original scorer requires a fresh destination. Stage its complete
        # offline output, then replace only score files; never delete old runs.
        score_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".resume-scores-", dir=score_dir.parent
        ) as temporary:
            staged = Path(temporary) / "scores"
            run.score_batches({args.model: out_dir}, staged)
            score_dir.mkdir(exist_ok=True)
            for path in sorted(staged.iterdir(), key=lambda item: item.name == "report.md"):
                path.replace(score_dir / path.name)
        print(f"paper_scores: {score_dir}")
    else:
        print("diagnostic grader only; no formal paper scores generated")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = (batch.ROOT / args.runs_dir / args.run_id).resolve() if args.run_id else None
    if out_dir is None or not out_dir.exists():
        return run.main(argv)
    try:
        manifest, tasks, score_dir = _validate(args, out_dir)
        results, configs = _recover(out_dir, tasks, args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    missing = [task.path for key, task in tasks.items() if key not in results]
    judging = sum(item["final_grader"] != args.grader for item in results.values())
    to_grade = judging + len(missing) if args.grader != "rule" else 0
    print(
        f"tasks: {len(tasks)}; reused: {len(results)}; to_run: {len(missing)}; to_grade: {to_grade}"
    )
    print(f"out_dir: {out_dir}")
    if args.dry_run:
        print("dry-run complete; no API calls or files written")
        return 0
    try:
        for directory, config in configs.items():
            if config != score._json(directory / "config.json"):
                _write_json(directory / "config.json", config)
            batch.write_report(directory, score._json(directory / "grade.json"))
        with run._provider_options(args):
            _stage(
                args,
                out_dir,
                results,
                [(path, args, out_dir / "task_runs", "rule") for path in missing],
                stage="agent",
            )
            _stage(
                args,
                out_dir,
                results,
                [(item, args) for item in results.values() if item["final_grader"] != args.grader],
                stage="judge",
            )
        _finish(args, out_dir, score_dir, manifest, results)
    except KeyboardInterrupt:
        print("interrupted; completed task records remain in the run directory")
        return 130
    except (OSError, ValueError, KeyError, TypeError, batch.ProviderTransportError) as exc:
        print(f"experiment failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
