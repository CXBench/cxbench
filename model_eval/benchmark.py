#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .reporting import write_model_report
from .run import (
    BENCHMARK_ID,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_WIRE_API,
    DOMAINS,
    ENV_FILE,
    REPO_ROOT,
    ConfigError,
    Endpoint,
    _load_env_file,
    _positive_int,
    _resolve_value,
    _validate_run_id,
    discover_tasks,
    normalize_endpoint,
    normalize_model,
)


RUNS_DIR = Path("runs/model_eval")
DEFAULT_CONCURRENCY = 20
DEFAULT_EXECUTOR = "process"
DEFAULT_MODE = "least_privilege"


@dataclass(frozen=True)
class BenchmarkConfig:
    api_key: str
    model: str
    judge_model: str
    user_model: str
    endpoint: Endpoint
    max_output_tokens: int
    tasks: tuple[Path, ...]
    mode: str
    grader: str
    concurrency: int
    executor: str
    task_timeout: int
    run_id: str

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / RUNS_DIR / self.run_id


def _safe_run_id(model: str) -> str:
    model_part = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_part[:64]}"
    candidate = base
    suffix = 2
    while (REPO_ROOT / RUNS_DIR / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _positive_setting(
    cli_value: int | None,
    env_name: str,
    env_file: dict[str, str],
    default: int,
) -> int:
    raw_value = _resolve_value(
        str(cli_value) if cli_value is not None else None, (env_name,), env_file
    )
    if not raw_value:
        return default
    try:
        return _positive_int(raw_value)
    except argparse.ArgumentTypeError as exc:
        raise ConfigError(f"{env_name} {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one OpenAI model on the released cxbench benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key", help="API key; prefer OPENAI_API_KEY in model_eval/.env")
    parser.add_argument("--model", help=f"bare OpenAI model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--judge-model", help="hybrid judge model; defaults to the evaluated model")
    parser.add_argument(
        "--user-model", help="LLM user simulator model; defaults to the evaluated model"
    )
    parser.add_argument("--url", "--base-url", dest="base_url", help="OpenAI-compatible base URL")
    parser.add_argument("--env-file", default=str(ENV_FILE), help="dotenv-style launcher settings")
    parser.add_argument("--max-output-tokens", type=_positive_int)
    parser.add_argument(
        "--domain", action="append", choices=DOMAINS, help="select a domain; repeat as needed"
    )
    parser.add_argument("--limit", type=_positive_int, help="balanced task subset size")
    parser.add_argument(
        "--mode", choices=["least_privilege", "direct"], help="runtime permission mode"
    )
    parser.add_argument(
        "--grader", choices=["hybrid", "rule", "llm"], help="hybrid is required for formal results"
    )
    parser.add_argument("--concurrency", type=_positive_int, help="parallel task processes")
    parser.add_argument(
        "--executor", choices=["process", "thread"], help="process is the WSL-first default"
    )
    parser.add_argument(
        "--task-timeout", type=_positive_int, help="per-task wall-clock timeout in seconds"
    )
    parser.add_argument("--run-id", help="stable output directory name")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print the plan without API calls"
    )
    return parser


def resolve_config(args: argparse.Namespace, *, require_api_key: bool = True) -> BenchmarkConfig:
    env_file = _load_env_file(Path(args.env_file).expanduser().resolve())
    api_key = _resolve_value(args.api_key, ("CXBENCH_API_KEY", "OPENAI_API_KEY"), env_file)
    if require_api_key and (not api_key or api_key == "replace_me"):
        raise ConfigError("API key is required via --api-key or OPENAI_API_KEY")

    model = normalize_model(
        _resolve_value(args.model, ("CXBENCH_MODEL",), env_file) or DEFAULT_MODEL
    )
    judge_model = normalize_model(
        _resolve_value(args.judge_model, ("CXBENCH_JUDGE_MODEL",), env_file) or model
    )
    user_model = normalize_model(
        _resolve_value(args.user_model, ("CXBENCH_USER_MODEL",), env_file) or model
    )
    raw_url = (
        _resolve_value(args.base_url, ("CXBENCH_BASE_URL", "OPENAI_BASE_URL"), env_file)
        or DEFAULT_BASE_URL
    )
    endpoint = normalize_endpoint(raw_url, DEFAULT_WIRE_API)
    max_output_tokens = _positive_setting(
        args.max_output_tokens,
        "CXBENCH_MAX_OUTPUT_TOKENS",
        env_file,
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    concurrency = _positive_setting(
        args.concurrency,
        "CXBENCH_CONCURRENCY",
        env_file,
        DEFAULT_CONCURRENCY,
    )
    task_timeout = _positive_setting(args.task_timeout, "CXBENCH_TASK_TIMEOUT", env_file, 600)
    executor = _resolve_value(args.executor, ("CXBENCH_EXECUTOR",), env_file) or DEFAULT_EXECUTOR
    if executor not in {"process", "thread"}:
        raise ConfigError("CXBENCH_EXECUTOR must be 'process' or 'thread'")
    mode = _resolve_value(args.mode, ("CXBENCH_MODE",), env_file) or DEFAULT_MODE
    if mode not in {"least_privilege", "direct"}:
        raise ConfigError("CXBENCH_MODE must be 'least_privilege' or 'direct'")
    grader = _resolve_value(args.grader, ("CXBENCH_GRADER",), env_file) or "hybrid"
    if grader not in {"hybrid", "rule", "llm"}:
        raise ConfigError("CXBENCH_GRADER must be 'hybrid', 'rule', or 'llm'")

    tasks = discover_tasks(args.domain, args.limit)
    run_id = _validate_run_id(args.run_id) if args.run_id else _safe_run_id(model)
    if (REPO_ROOT / RUNS_DIR / run_id).exists():
        raise ConfigError(f"output directory already exists for --run-id {run_id!r}")
    return BenchmarkConfig(
        api_key=api_key,
        model=model,
        judge_model=judge_model,
        user_model=user_model,
        endpoint=endpoint,
        max_output_tokens=max_output_tokens,
        tasks=tasks,
        mode=mode,
        grader=grader,
        concurrency=concurrency,
        executor=executor,
        task_timeout=task_timeout,
        run_id=run_id,
    )


def build_batch_command(config: BenchmarkConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "model_eval.batch_entrypoint",
        "--model",
        config.model,
        "--agent-provider",
        "openai",
        "--mode",
        config.mode,
        "--grader",
        config.grader,
        "--judge-model",
        config.judge_model,
        "--judge-provider",
        "same",
        "--user-sim",
        "llm",
        "--user-model",
        config.user_model,
        "--task-timeout",
        str(config.task_timeout),
        "--concurrency",
        str(config.concurrency),
        "--executor",
        config.executor,
        "--runs-dir",
        RUNS_DIR.as_posix(),
        "--run-id",
        config.run_id,
    ]
    for task in config.tasks:
        command.extend(["--task", task.relative_to(REPO_ROOT).as_posix()])
    return command


def build_child_environment(config: BenchmarkConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_REASONING_EFFORT", None)
    environment.pop("CXBENCH_REASONING_EFFORT", None)
    environment.update(
        {
            "OPENAI_API_KEY": config.api_key,
            "OPENAI_BASE_URL": config.endpoint.base_url,
            "OPENAI_WIRE_API": config.endpoint.wire_api,
            "OPENAI_MAX_OUTPUT_TOKENS": str(config.max_output_tokens),
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _print_plan(config: BenchmarkConfig, dry_run: bool) -> None:
    print("cxbench model evaluation")
    print(f"  benchmark: {BENCHMARK_ID}")
    print(f"  model: {config.model}")
    print(f"  user_model: {config.user_model}")
    print(f"  judge_model: {config.judge_model}")
    print(f"  base_url: {config.endpoint.base_url}")
    print(f"  wire_api: {config.endpoint.wire_api}")
    print(f"  grader: {config.grader}")
    print(f"  mode: {config.mode}")
    print(f"  tasks: {len(config.tasks)}")
    print(f"  executor: {config.executor}")
    print(f"  concurrency: {config.concurrency}")
    print(f"  output: {config.output_dir}")
    print(
        f"  api_key: {'provided' if config.api_key else 'not required for dry-run'} (never written)"
    )
    if dry_run:
        print("dry-run complete; no API request was sent")


def _enrich_manifest(config: BenchmarkConfig) -> None:
    path = config.output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "benchmark": BENCHMARK_ID,
            "base_url": config.endpoint.base_url,
            "wire_api": config.endpoint.wire_api,
            "max_output_tokens": config.max_output_tokens,
            "api_key_stored": False,
        }
    )
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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

    try:
        completed = subprocess.run(
            build_batch_command(config),
            cwd=REPO_ROOT,
            env=build_child_environment(config),
            check=False,
        )
        if not (config.output_dir / "results.jsonl").is_file():
            raise ConfigError(
                "batch stopped before producing results.jsonl; inspect results_partial.jsonl"
            )
        _enrich_manifest(config)
        report = write_model_report(config.output_dir)
    except KeyboardInterrupt:
        print(
            "evaluation interrupted; completed rows remain in results_partial.jsonl",
            file=sys.stderr,
        )
        return 130
    except (ConfigError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    overall = report["overall"]
    print(f"report: {config.output_dir / 'report.md'}")
    print(f"domain metrics: {config.output_dir / 'domain_metrics.csv'}")
    print(
        f"both_pass={overall['both_pass_rate']:.2%} "
        f"outcome_pass={overall['outcome_pass_rate']:.2%} "
        f"process_pass={overall['process_pass_rate']:.2%} "
        f"errors={overall['error_count']}"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
