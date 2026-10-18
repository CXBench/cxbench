#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from cxbench.task_schema import TaskSpec
from model_eval import benchmark as model_eval
from model_eval.reporting import write_model_report
from model_eval.run import BENCHMARK_ID, REPO_ROOT, ConfigError, _validate_run_id


ABLATION_ID = "no_agent_direct_submit"
RUNS_DIR = Path("runs/no_agent_direct_submit_ablation")
DIRECT_MODE = "direct"
RUN_ID_SUFFIX = "no-agent-direct-submit-ablation"
GRADING_POLICY_LISTS = (
    "deterministic_hard_rules",
    "semantic_judge_rules",
    "hybrid_hint_rules",
)


def build_parser() -> argparse.ArgumentParser:
    parser = model_eval.build_parser()
    parser.description = (
        "Run only direct-mode cxbench records and grade them without the "
        "no_agent_direct_submit rule."
    )
    mode_action = next(action for action in parser._actions if action.dest == "mode")
    mode_action.choices = (DIRECT_MODE,)
    mode_action.default = DIRECT_MODE
    mode_action.help = "fixed to direct; mediate/least-privilege is not rerun"
    return parser


def resolve_config(
    args: argparse.Namespace,
    *,
    require_api_key: bool = True,
) -> model_eval.BenchmarkConfig:
    requested_run_id = args.run_id
    base_args = argparse.Namespace(**vars(args))
    base_args.mode = DIRECT_MODE
    base_args.run_id = None
    config = model_eval.resolve_config(base_args, require_api_key=require_api_key)

    if requested_run_id:
        run_id = _validate_run_id(requested_run_id)
    else:
        run_id = _validate_run_id(f"{config.run_id}-{RUN_ID_SUFFIX}")
    config = replace(config, mode=DIRECT_MODE, run_id=run_id)
    if output_dir(config).exists():
        raise ConfigError(f"output directory already exists for --run-id {run_id!r}")
    return config


def output_dir(config: model_eval.BenchmarkConfig) -> Path:
    return REPO_ROOT / RUNS_DIR / config.run_id


def ablate_task_mapping(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a task mapping with only the target rule removed."""
    task = copy.deepcopy(dict(raw))
    removed_from: list[str] = []

    rules = task.get("rules")
    if isinstance(rules, list):
        filtered, removed = _without_target_rule(rules)
        task["rules"] = filtered
        if removed:
            removed_from.append("rules")

    policy = task.get("policy")
    if isinstance(policy, dict):
        grading_policy = policy.get("grading_policy")
        if isinstance(grading_policy, dict):
            for key in GRADING_POLICY_LISTS:
                values = grading_policy.get(key)
                if not isinstance(values, list):
                    continue
                filtered, removed = _without_target_rule(values)
                grading_policy[key] = filtered
                if removed:
                    removed_from.append(f"policy.grading_policy.{key}")

    remaining_paths = _exact_rule_paths(task)
    if remaining_paths:
        paths = ", ".join(remaining_paths)
        raise ConfigError(f"{ABLATION_ID} remains in unsupported task locations: {paths}")
    return task, removed_from


def write_ablated_task_specs(
    source_tasks: Sequence[Path],
    destination: Path,
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    destination.mkdir(parents=True, exist_ok=False)
    derived_paths: list[Path] = []
    provenance: list[dict[str, Any]] = []

    for source in source_tasks:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ConfigError(f"task YAML must be a mapping: {source}")
        derived, removed_from = ablate_task_mapping(raw)
        target = destination / source.name
        target.write_text(
            yaml.safe_dump(derived, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        loaded = TaskSpec.load(target)
        if ABLATION_ID in loaded.rules:
            raise ConfigError(f"failed to remove {ABLATION_ID} from {target}")
        derived_paths.append(target)
        provenance.append(
            {
                "task_id": loaded.id,
                "source_path": _repo_path(source),
                "source_sha256": _sha256(source),
                "derived_path": _repo_path(target),
                "derived_sha256": _sha256(target),
                "removed_from": removed_from,
            }
        )

    return tuple(derived_paths), provenance


def build_batch_command(
    config: model_eval.BenchmarkConfig,
    task_specs: Sequence[Path],
) -> list[str]:
    derived_config = replace(config, tasks=tuple(task_specs), mode=DIRECT_MODE)
    command = model_eval.build_batch_command(derived_config)
    command[command.index("--runs-dir") + 1] = RUNS_DIR.as_posix()
    return command


def _without_target_rule(values: list[Any]) -> tuple[list[Any], bool]:
    filtered = [value for value in values if value != ABLATION_ID]
    return filtered, len(filtered) != len(values)


def _exact_rule_paths(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            paths.extend(_exact_rule_paths(item, child))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_exact_rule_paths(item, f"{path}[{index}]"))
        return paths
    return [path] if value == ABLATION_ID else []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _enrich_manifest(
    config: model_eval.BenchmarkConfig,
    provenance: Sequence[Mapping[str, Any]],
) -> None:
    path = output_dir(config) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "benchmark": BENCHMARK_ID,
            "base_url": config.endpoint.base_url,
            "wire_api": config.endpoint.wire_api,
            "max_output_tokens": config.max_output_tokens,
            "ablation": {
                "id": ABLATION_ID,
                "disabled_rule_ids": [ABLATION_ID],
                "mode": DIRECT_MODE,
                "mediate_rerun": False,
                "grader": config.grader,
                "preserved_submit_rules": [
                    "confirmation_before_submit",
                    "submit_matches_confirmed_draft",
                ],
                "task_specs_are_derived": True,
                "source_tasks": list(provenance),
            },
        }
    )
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_plan(config: model_eval.BenchmarkConfig, dry_run: bool) -> None:
    print("cxbench no_agent_direct_submit ablation")
    print(f"  benchmark: {BENCHMARK_ID}")
    print(f"  model: {config.model}")
    print(f"  user_model: {config.user_model}")
    print(f"  judge_model: {config.judge_model}")
    print(f"  grader: {config.grader}")
    print(f"  mode: {DIRECT_MODE} (fixed; no mediate rerun)")
    print(f"  disabled_rule: {ABLATION_ID}")
    print(f"  tasks: {len(config.tasks)}")
    print(f"  executor: {config.executor}")
    print(f"  concurrency: {config.concurrency}")
    print(f"  output: {output_dir(config)}")
    print(
        f"  api_key: {'provided' if config.api_key else 'not required for dry-run'} (never written)"
    )
    if dry_run:
        print("dry-run complete; no task specs, run directory, or API requests were created")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args, require_api_key=not args.dry_run)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))

    _print_plan(config, args.dry_run)
    if args.dry_run:
        return 0

    experiment_dir = output_dir(config)
    try:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        task_specs, provenance = write_ablated_task_specs(
            config.tasks, experiment_dir / "task_specs"
        )
        completed = subprocess.run(
            build_batch_command(config, task_specs),
            cwd=REPO_ROOT,
            env=model_eval.build_child_environment(config),
            check=False,
        )
        if not (experiment_dir / "results.jsonl").is_file():
            raise ConfigError(
                "batch stopped before producing results.jsonl; inspect results_partial.jsonl"
            )
        _enrich_manifest(config, provenance)
        report = write_model_report(experiment_dir)
    except KeyboardInterrupt:
        print(
            "evaluation interrupted; completed records remain in the output directory",
            file=sys.stderr,
        )
        return 130
    except (ConfigError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    overall = report["overall"]
    print(f"report: {experiment_dir / 'report.md'}")
    print(f"domain metrics: {experiment_dir / 'domain_metrics.csv'}")
    print(
        f"both_pass={overall['both_pass_rate']:.2%} "
        f"outcome_pass={overall['outcome_pass_rate']:.2%} "
        f"process_pass={overall['process_pass_rate']:.2%} "
        f"errors={overall['error_count']}"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
