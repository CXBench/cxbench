"""Rejudge recorded main-experiment traces and write a new hybrid batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cxbench.eval_config import PROVIDER_PREFIXES, model_string
from cxbench.grader import LLMTraceJudge
from cxbench.task_schema import TaskSpec

from . import resume, run, score

batch = run.batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path, help="completed source batch")
    parser.add_argument("--output", required=True, type=Path, help="output batch directory")
    parser.add_argument("--resume", action="store_true", help="continue the existing output batch")
    parser.add_argument("--judge-model", help="judge model, optionally provider-prefixed")
    parser.add_argument(
        "--inherit-source-config",
        action="store_true",
        help="inherit judge model and recorded provider options from the source manifest",
    )
    parser.add_argument(
        "--judge-provider", choices=["same", *(prefix[:-1] for prefix in PROVIDER_PREFIXES)]
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--executor", choices=["thread", "process"])
    parser.add_argument(
        "--retry-delay", type=float, help="delay between judge retries, as in main_experiment.run"
    )
    parser.add_argument("--base-url", help="endpoint override for --judge-provider openai")
    parser.add_argument("--wire-api", choices=["chat", "responses"])
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    parser.add_argument(
        "--task-release", type=Path, help="replacement task release containing tasks/*.yaml"
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--dry-run", action="store_true", help="validate without API calls or writes"
    )
    parser.set_defaults(grader="hybrid")
    return parser


def _inherit_source_config(args: argparse.Namespace, manifest: dict) -> None:
    """Fill judge options from a completed source batch when requested."""
    if not args.judge_model:
        args.judge_model = manifest.get("judge_model")
    if args.judge_provider is None:
        args.judge_provider = manifest.get("judge_provider", args.judge_provider)
    if args.wire_api is None:
        args.wire_api = manifest.get("wire_api")
    if args.max_output_tokens is None:
        recorded = manifest.get("max_output_tokens")
        if recorded is None:
            raise ValueError("source manifest is missing required configuration: max_output_tokens")
        args.max_output_tokens = int(recorded)
    for name in ("concurrency", "executor", "retry_delay"):
        if getattr(args, name) is None:
            if manifest.get(name) is None:
                raise ValueError(f"source manifest is missing required configuration: {name}")
            setattr(args, name, manifest[name])
    if args.reasoning_effort is None:
        args.reasoning_effort = manifest.get("reasoning_effort")
    if args.base_url is None:
        args.base_url = manifest.get("base_url")
    missing = [
        name
        for name, value in (
            ("judge_model", args.judge_model),
            ("wire_api", args.wire_api),
            ("judge_provider", args.judge_provider),
        )
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise ValueError(f"source manifest is missing required configuration: {', '.join(missing)}")


def _task_without_rules(task: TaskSpec) -> dict[str, Any]:
    """Return a task's business definition with only the semantic rubric removed."""
    values = yaml.safe_load(task.path.read_text(encoding="utf-8"))
    values.get("policy", {}).pop("llm_judge", None)
    return values


def _replacement_tasks(release: Path, source_selected: dict, task_ids: list[str]) -> dict:
    task_dir = release / "tasks"
    if not task_dir.is_dir():
        raise ValueError(f"task release has no tasks directory: {task_dir}")
    replacements = {}
    for path in sorted(task_dir.glob("*.yaml")):
        task = TaskSpec.load(path)
        if task.id in replacements:
            raise ValueError(f"duplicate task ID in task release: {task.id}")
        replacements[task.id] = task
    requested = set(task_ids)
    unknown = requested - set(source_selected)
    if unknown:
        raise ValueError(f"unknown task ID: {', '.join(sorted(unknown))}")
    selected_ids = requested or (set(source_selected) & set(replacements))
    if not selected_ids:
        raise ValueError("task release has no matching source task IDs")
    missing = selected_ids - set(replacements)
    if missing:
        raise ValueError(f"task release is missing task ID: {', '.join(sorted(missing))}")
    result = {}
    for task_id in selected_ids:
        replacement = replacements[task_id]
        original = source_selected[task_id][1]
        if _task_without_rules(original) != _task_without_rules(replacement):
            raise ValueError(f"task release changes non-rule fields for task {task_id}")
        result[task_id] = replacement
    return result


def _prompt_metadata() -> dict[str, str]:
    # The fixed prompt needs no provider client or credentials for dry-run validation.
    judge = LLMTraceJudge.__new__(LLMTraceJudge)
    return {
        "judge_system_prompt_version": LLMTraceJudge.SYSTEM_PROMPT_VERSION,
        "judge_system_prompt_sha256": hashlib.sha256(judge._system_prompt().encode()).hexdigest(),
    }


def _resume_manifest(
    args: argparse.Namespace,
    source: Path,
    output: Path,
    original: dict,
    selected: dict,
) -> dict:
    saved = score._json(output / "manifest.json")
    if Path(saved.get("source_batch", "")).resolve() != source:
        raise ValueError("--batch differs from the original rejudge source")
    for name in ("model", "agent_provider"):
        if saved.get(name) != original.get(name):
            raise ValueError(f"source {name} differs from the existing rejudge")
    if saved.get("grader") != "hybrid" or saved.get("final_grader") != "hybrid":
        raise ValueError("resume requires an existing hybrid rejudge")
    if model_string(saved.get("judge_model", "")) != model_string(args.judge_model):
        raise ValueError("--judge-model differs from the existing rejudge; choose a new output")
    provider = saved.get("judge_provider")
    if provider == "same":
        provider = saved["agent_provider"]
    if provider != args.judge_provider:
        raise ValueError("--judge-provider differs from the existing rejudge")
    if int(saved.get("max_output_tokens", 8192)) != args.max_output_tokens:
        raise ValueError("--max-output-tokens differs from the existing rejudge")
    if args.wire_api and saved.get("wire_api") not in (None, args.wire_api):
        raise ValueError("--wire-api differs from the existing rejudge")
    for name, value in _prompt_metadata().items():
        if name in saved and saved[name] != value:
            raise ValueError("judge system prompt differs from the existing rejudge")
    if args.inherit_source_config and saved.get("reasoning_effort") != args.reasoning_effort:
        raise ValueError("reasoning effort differs from the existing rejudge")
    args.wire_api = args.wire_api or saved.get("wire_api")
    planned = saved.get("tasks", [])
    expected = {
        task.path.resolve(): batch._sha256(task.path) for _, task, _, _ in selected.values()
    }
    recorded = {score._task_path(item["path"]): item["sha256"] for item in planned}
    if (
        saved.get("task_count") != len(planned)
        or len(recorded) != len(planned)
        or recorded != expected
    ):
        raise ValueError("tasks or task hashes differ from the existing rejudge")
    return saved


def _recover(args: argparse.Namespace, output: Path, selected: dict) -> tuple[dict, dict]:
    previous = {}
    for filename in ("results_partial.jsonl", "results.jsonl"):
        for row in resume._recorded_rows(output / filename):
            if row.get("task_id") not in selected:
                raise ValueError(f"unplanned recorded task: {row.get('task_id')}")
            previous[row["task_id"]] = row
    results, configs = {}, {}
    for task_id, (original, task, source_run, source_config) in selected.items():
        directory = output / "task_runs" / source_run.name
        if not directory.resolve().is_relative_to(output):
            raise ValueError(f"rejudge run is outside its output batch: {directory}")
        for filename in ("trace.jsonl", "final_state.json"):
            path = directory / filename
            if not path.is_file() or batch._sha256(path) != batch._sha256(source_run / filename):
                raise ValueError(f"rejudge input missing or changed: {path}")
        grade_path = directory / "grade.json"
        if not grade_path.is_file():
            continue
        try:
            grade = score._json(grade_path)
        except json.JSONDecodeError:
            # A process interruption can leave an unfinished grade write.
            if task_id in previous or (directory / "report.md").exists():
                raise
            continue
        if (
            grade.get("task_id") != task_id
            or grade.get("mode") != source_config["mode"]
            or grade.get("grader") != "hybrid"
            or grade.get("verdict") not in {"pass", "fail"}
            or any(
                type(grade.get(key)) is not bool
                for key in (
                    "outcome_pass",
                    "process_pass",
                    "hard_rule_process_pass",
                    "false_pass_demo",
                )
            )
            or not isinstance(grade.get("rules"), list)
        ):
            raise ValueError(f"invalid recorded hybrid grade: {directory}")
        score._judge_pass(grade, task_id)
        try:
            config = score._json(directory / "config.json")
        except (FileNotFoundError, json.JSONDecodeError):
            config = dict(source_config)
        mutable = {"task_path", "grader", "final_grader", "execution_grader", "judge_model"}
        if {k: v for k, v in config.items() if k not in mutable} != {
            k: v for k, v in source_config.items() if k not in mutable
        }:
            raise ValueError(f"recorded execution configuration changed: {directory}")
        if config.get("judge_model") not in (
            source_config.get("judge_model"),
            model_string(args.judge_model),
        ):
            raise ValueError(f"recorded judge model differs: {directory}")
        # Legacy rejudges may have saved grade.json before updating config/report
        # or the parent process's results snapshot. The grade is already complete.
        config.update(
            task_path=str(task.path.resolve()),
            execution_grader=source_config.get("execution_grader", source_config.get("grader")),
            grader="hybrid",
            final_grader="hybrid",
            judge_model=model_string(args.judge_model),
        )
        row = {
            **original,
            **previous.get(task_id, {}),
            "task_id": task_id,
            "task_path": str(task.path.resolve()),
            "run_dir": str(directory),
            "source_run_dir": str(source_run),
            "ok": True,
            "error": None,
        }
        row.pop("traceback", None)
        results[task_id] = batch._graded_result(row, config, grade)
        configs[directory] = config
    return results, configs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source, output = args.batch.resolve(), args.output.resolve()
    try:
        if args.resume and not output.is_dir():
            raise ValueError(f"--resume requires an existing output directory: {output}")
        if not args.resume and output.exists():
            raise ValueError(
                f"output already exists; use --resume or choose a new directory: {output}"
            )
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must be separate from the source batch")
        manifest, selected, _ = score._load_batch_runs(source, require_hybrid=False)
        if args.inherit_source_config:
            _inherit_source_config(args, manifest)
            source_model = model_string(manifest["judge_model"])
            for task_id, (_, _, _, config) in selected.items():
                if model_string(config.get("judge_model") or "") != source_model:
                    raise ValueError(f"source judge model differs from manifest: {task_id}")
        for name, default in {
            "max_output_tokens": 8192,
            "concurrency": 20,
            "executor": "thread",
            "retry_delay": 5.0,
            "judge_provider": "openai",
        }.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
        batch.validate_args(args)
        if args.concurrency < 1:
            raise ValueError("--concurrency must be positive")
        if args.executor not in {"thread", "process"}:
            raise ValueError("executor must be thread or process")
        if args.wire_api is not None and args.wire_api not in {"chat", "responses"}:
            raise ValueError("wire API must be chat or responses")
        if (
            not args.judge_model
            or not args.judge_model.strip()
            or not model_string(args.judge_model).split(":", 1)[1].strip()
        ):
            raise ValueError("--judge-model must name a model (or use --inherit-source-config)")
        if args.max_output_tokens < 1:
            raise ValueError("--max-output-tokens must be positive")
        if args.task_id:
            unknown = set(args.task_id) - set(selected)
            if unknown:
                raise ValueError(f"unknown task ID: {', '.join(sorted(unknown))}")
            selected = {task_id: selected[task_id] for task_id in dict.fromkeys(args.task_id)}
        if args.task_release:
            replacements = _replacement_tasks(args.task_release.resolve(), selected, args.task_id)
            selected = {
                task_id: (result, replacements[task_id], source_run, config)
                for task_id, (result, _, source_run, config) in selected.items()
                if task_id in replacements
            }
        args.model = manifest["model"]
        args.agent_provider = manifest["agent_provider"]
        if not isinstance(args.model, str) or not args.model.strip():
            raise ValueError("source manifest must name the agent model")
        if args.judge_provider == "same":
            args.judge_provider = args.agent_provider
        if args.judge_provider not in {prefix[:-1] for prefix in PROVIDER_PREFIXES}:
            raise ValueError("invalid judge provider")
        if args.base_url and args.judge_provider != "openai":
            raise ValueError("--base-url requires --judge-provider openai")
        results, configs = {}, {}
        if args.resume:
            manifest = _resume_manifest(args, source, output, manifest, selected)
            results, configs = _recover(args, output, selected)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    print(f"tasks: {len(selected)}; judge_model: {args.judge_model}; grader: hybrid")
    print(f"wire_api: {args.wire_api}; max_output_tokens: {args.max_output_tokens}")
    if args.inherit_source_config and not args.base_url:
        print("source has no recorded base_url; using current provider environment")
    print(f"reused: {len(results)}; to_grade: {len(selected) - len(results)}")
    print(f"out_dir: {output}")
    if args.dry_run:
        print("dry-run complete; no API calls or files written")
        return 0

    try:
        output.mkdir(parents=True, exist_ok=args.resume)
        if not args.resume:
            manifest = {
                **manifest,
                "run_id": output.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_batch": str(source),
                "source_judge_model": manifest.get("judge_model"),
                "grader": "hybrid",
                "final_grader": "hybrid",
                "judge_model": args.judge_model,
                "judge_provider": args.judge_provider,
                "concurrency": args.concurrency,
                "executor": args.executor,
                "retry_delay": args.retry_delay,
                "wire_api": args.wire_api,
                "max_output_tokens": args.max_output_tokens,
                "reasoning_effort": args.reasoning_effort,
                "base_url": args.base_url,
                **_prompt_metadata(),
                "task_count": len(selected),
                "tasks": [
                    {"path": str(task.path.resolve()), "sha256": batch._sha256(task.path)}
                    for _, task, _, _ in selected.values()
                ],
            }
            resume._write_json(output / "manifest.json", manifest)
        arguments = []
        for task_id, (result, task, directory, config) in selected.items():
            destination = output / "task_runs" / directory.name
            if task_id in results:
                config = configs[destination]
                resume._write_json(destination / "config.json", config)
                if not (destination / "report.md").is_file():
                    batch.write_report(destination, score._json(destination / "grade.json"))
                continue
            destination.mkdir(parents=True, exist_ok=args.resume)
            # Copy only grading inputs; the original batch retains all execution artifacts.
            for filename in ("trace.jsonl", "final_state.json"):
                shutil.copyfile(directory / filename, destination / filename)
            resume._write_json(
                destination / "config.json",
                {
                    **config,
                    "task_path": str(task.path.resolve()),
                    "judge_model": model_string(args.judge_model),
                },
            )
            arguments.append(
                (
                    {
                        **result,
                        "task_id": task_id,
                        "task_path": str(task.path.resolve()),
                        "run_dir": str(destination),
                        "source_run_dir": str(directory),
                    },
                    args,
                )
            )
        resume._write_results(output / "results_partial.jsonl", results)
        with run._provider_options(args):
            resume._stage(args, output, results, arguments, stage="judge")
        resume._finish(args, output, output / "paper_scores", manifest, results)
    except KeyboardInterrupt:
        print("interrupted; rerun the same command with --resume to reuse completed judges")
        return 130
    except (OSError, ValueError, KeyError, TypeError, batch.ProviderTransportError) as exc:
        print(f"rejudge failed: {exc}")
        print("rerun the same command with --resume after resolving the error")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
