#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

from .analysis import MODE_RUNTIME_NAMES, write_experiment_analysis


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
ENV_FILE = PROJECT_DIR / ".env"
BENCHMARK_ID = "cxbench-banking-judge-v20260917"
TASKS_DIR = REPO_ROOT / "benchmarks" / BENCHMARK_ID / "tasks"
BATCH_RUNNER = REPO_ROOT / "scripts" / "run_eval_batch.py"
RUNS_DIR = Path("runs/mode_ablation")
DOMAINS = (
    "banking",
    "government_benefits",
    "healthcare",
    "insurance",
    "telecom",
    "utilities",
)
PROVIDER_PREFIXES = (
    "crs:",
    "deepseek:",
    "openrouter:",
    "siliconflow:",
    "shubiaobiao:",
)
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_WIRE_API = "chat"
DEFAULT_EXECUTOR = "process"


class ConfigError(ValueError):
    """Raised when launcher configuration is missing or invalid."""


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    wire_api: str


@dataclass(frozen=True)
class LaunchConfig:
    api_key: str
    model: str
    endpoint: Endpoint
    tasks: tuple[Path, ...]
    grader: str
    concurrency: int
    task_timeout: int
    run_id: str
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    executor: str = DEFAULT_EXECUTOR
    modes: tuple[str, ...] = tuple(MODE_RUNTIME_NAMES)

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / RUNS_DIR / self.run_id


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolve_value(cli_value: str | None, names: Sequence[str], env_file: dict[str, str]) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    for name in names:
        value = env_file.get(name)
        if value and value.strip():
            return value.strip()
    return ""


def normalize_model(raw_model: str) -> str:
    model = raw_model.strip()
    if model.startswith("openai:"):
        model = model.split(":", 1)[1].strip()
    elif model.startswith(PROVIDER_PREFIXES):
        raise ConfigError(
            "MODEL must be a bare model name when a custom URL is supplied; remove the provider prefix"
        )
    if not model or model in {"your-model-name", "replace_me"}:
        raise ConfigError("MODEL is required")
    if any(character.isspace() for character in model):
        raise ConfigError("MODEL must not contain whitespace")
    return model


def normalize_endpoint(raw_url: str, requested_wire_api: str = "auto") -> Endpoint:
    url = raw_url.strip().rstrip("/")
    if not url or url in {"https://api.example.com/v1", "replace_me"}:
        raise ConfigError("BASE_URL is required")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("BASE_URL must be an absolute http:// or https:// URL")
    if any(character.isspace() for character in url):
        raise ConfigError("BASE_URL must not contain whitespace")
    if parsed.username or parsed.password:
        raise ConfigError("BASE_URL must not contain credentials; use the API key setting")
    if parsed.query or parsed.fragment:
        raise ConfigError("BASE_URL must not contain a query string or fragment")

    path = parsed.path.rstrip("/")
    lowered_path = path.lower()
    inferred_wire_api: str | None = None
    if lowered_path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
        inferred_wire_api = "chat"
    elif lowered_path.endswith("/responses"):
        path = path[: -len("/responses")]
        inferred_wire_api = "responses"

    wire_api = requested_wire_api if requested_wire_api != "auto" else inferred_wire_api or "chat"
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "")).rstrip("/")
    return Endpoint(base_url=base_url, wire_api=wire_api)


def _task_domain(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("domain:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise ConfigError(f"benchmark task has no domain: {path}")


def _round_robin_limit(
    tasks: Iterable[tuple[Path, str]], domains: Sequence[str], limit: int
) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path, domain in tasks:
        grouped[domain].append(path)
    for paths in grouped.values():
        paths.sort(key=lambda item: item.name)

    selected: list[Path] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for domain in domains:
            paths = grouped.get(domain, [])
            if offset < len(paths):
                selected.append(paths[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def discover_tasks(
    domains: Sequence[str] | None = None, limit: int | None = None
) -> tuple[Path, ...]:
    if not TASKS_DIR.is_dir():
        raise ConfigError(f"benchmark task directory does not exist: {TASKS_DIR}")
    task_paths = sorted(TASKS_DIR.glob("*.yaml"), key=lambda item: item.name)
    if not task_paths:
        raise ConfigError(f"benchmark contains no YAML tasks: {TASKS_DIR}")

    requested_domains = tuple(dict.fromkeys(domains or DOMAINS))
    unknown_domains = sorted(set(requested_domains) - set(DOMAINS))
    if unknown_domains:
        raise ConfigError(f"unknown benchmark domain(s): {', '.join(unknown_domains)}")
    tasks_with_domains = [(path, _task_domain(path)) for path in task_paths]
    selected = [
        (path, domain) for path, domain in tasks_with_domains if domain in requested_domains
    ]
    if not selected:
        raise ConfigError("the selected domains contain no benchmark tasks")
    if limit is not None and limit > len(selected):
        raise ConfigError(f"--limit {limit} exceeds the {len(selected)} selected benchmark tasks")
    if limit is not None:
        return tuple(_round_robin_limit(selected, requested_domains, limit))
    return tuple(path for path, _domain in selected)


def _safe_run_id(model: str) -> str:
    model_part = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    model_part = model_part[:64]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{timestamp}-{model_part}-direct-mediate"
    candidate = base
    suffix = 2
    while (REPO_ROOT / RUNS_DIR / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ConfigError(
            "--run-id may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return run_id


def build_batch_command(config: LaunchConfig, mode: str) -> list[str]:
    if mode not in MODE_RUNTIME_NAMES:
        raise ConfigError(f"unknown experiment mode: {mode}")
    command = [
        sys.executable,
        "-m",
        "model_eval.batch_entrypoint",
        "--model",
        config.model,
        "--agent-provider",
        "openai",
        "--mode",
        MODE_RUNTIME_NAMES[mode],
        "--grader",
        config.grader,
        "--judge-model",
        config.model,
        "--judge-provider",
        "same",
        "--user-sim",
        "llm",
        "--user-model",
        config.model,
        "--task-timeout",
        str(config.task_timeout),
        "--concurrency",
        str(config.concurrency),
        "--executor",
        config.executor,
        "--runs-dir",
        (RUNS_DIR / config.run_id).as_posix(),
        "--run-id",
        mode,
    ]
    for task in config.tasks:
        command.extend(["--task", task.relative_to(REPO_ROOT).as_posix()])
    return command


def build_child_environment(config: LaunchConfig) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.pop("OPENAI_REASONING_EFFORT", None)
    child_env.pop("CXBENCH_REASONING_EFFORT", None)
    child_env.update(
        {
            "OPENAI_API_KEY": config.api_key,
            "OPENAI_BASE_URL": config.endpoint.base_url,
            "OPENAI_WIRE_API": config.endpoint.wire_api,
            "OPENAI_MAX_OUTPUT_TOKENS": str(config.max_output_tokens),
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return child_env


def _write_launcher_manifest(config: LaunchConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "benchmark": BENCHMARK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "base_url": config.endpoint.base_url,
        "wire_api": config.endpoint.wire_api,
        "max_output_tokens": config.max_output_tokens,
        "grader": config.grader,
        "executor": config.executor,
        "agent_model": config.model,
        "user_sim_model": config.model,
        "judge_model": config.model,
        "task_count": len(config.tasks),
        "total_task_runs": len(config.tasks) * len(config.modes),
        "modes": [
            {"label": mode, "runtime_mode": MODE_RUNTIME_NAMES[mode]} for mode in config.modes
        ],
        "ablation": {
            "disabled_rule_ids": ["no_agent_direct_submit"],
            "preserved_submit_rules": [
                "confirmation_before_submit",
                "submit_matches_confirmed_draft",
            ],
        },
        "retry_errors_once": True,
        "retry_timeouts": False,
        "retry_concurrency": 1,
        "tasks": [task.relative_to(REPO_ROOT).as_posix() for task in config.tasks],
        "api_key_stored": False,
    }
    (config.output_dir / "launcher.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
        description="Run paired direct and mediate evaluations for the submit-permission ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key", help="API key; prefer CXBENCH_API_KEY in model_eval/.env")
    parser.add_argument("--model", help=f"bare model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--url",
        "--base-url",
        dest="base_url",
        help=f"OpenAI-compatible base URL or full endpoint URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--env-file", default=str(ENV_FILE), help="file containing CXBENCH_* launcher settings"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        help=(
            "per-provider-call output cap; may also be set with "
            f"CXBENCH_MAX_OUTPUT_TOKENS (default: {DEFAULT_MAX_OUTPUT_TOKENS})"
        ),
    )
    parser.add_argument(
        "--domain", action="append", choices=DOMAINS, help="run only this domain; repeat as needed"
    )
    parser.add_argument("--limit", type=_positive_int, help="run a balanced subset of N tasks")
    parser.add_argument(
        "--concurrency", type=_positive_int, default=4, help="number of tasks evaluated in parallel"
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default=DEFAULT_EXECUTOR,
        help="parallel worker type; process is the WSL-first default",
    )
    parser.add_argument(
        "--task-timeout", type=_positive_int, default=600, help="per-task timeout in seconds"
    )
    parser.add_argument(
        "--grader",
        choices=["hybrid", "rule", "llm"],
        default="hybrid",
        help="hybrid is the formal cxbench verdict; rule and llm are diagnostic only",
    )
    parser.add_argument("--run-id", help="optional stable output directory name")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate configuration without calling the API"
    )
    return parser


def resolve_config(args: argparse.Namespace) -> LaunchConfig:
    env_path = Path(args.env_file).expanduser().resolve()
    env_file = _load_env_file(env_path)
    api_key = _resolve_value(
        args.api_key,
        ("CXBENCH_API_KEY", "OPENAI_API_KEY"),
        env_file,
    )
    if not api_key or api_key == "replace_me":
        raise ConfigError("API key is required via --api-key or CXBENCH_API_KEY")
    raw_model = _resolve_value(args.model, ("CXBENCH_MODEL",), env_file) or DEFAULT_MODEL
    raw_url = (
        _resolve_value(
            args.base_url,
            ("CXBENCH_BASE_URL",),
            env_file,
        )
        or DEFAULT_BASE_URL
    )
    model = normalize_model(raw_model)
    endpoint = normalize_endpoint(raw_url, DEFAULT_WIRE_API)
    raw_max_output_tokens = _resolve_value(
        str(args.max_output_tokens) if args.max_output_tokens is not None else None,
        ("CXBENCH_MAX_OUTPUT_TOKENS",),
        env_file,
    )
    try:
        max_output_tokens = (
            _positive_int(raw_max_output_tokens)
            if raw_max_output_tokens
            else DEFAULT_MAX_OUTPUT_TOKENS
        )
    except argparse.ArgumentTypeError as exc:
        raise ConfigError(f"max output tokens {exc}") from exc
    tasks = discover_tasks(args.domain, args.limit)
    run_id = _validate_run_id(args.run_id) if args.run_id else _safe_run_id(model)
    if (REPO_ROOT / RUNS_DIR / run_id).exists():
        raise ConfigError(f"output directory already exists for --run-id {run_id!r}")
    return LaunchConfig(
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        tasks=tasks,
        grader=args.grader,
        concurrency=args.concurrency,
        task_timeout=args.task_timeout,
        run_id=run_id,
        max_output_tokens=max_output_tokens,
        executor=args.executor,
    )


def _print_plan(config: LaunchConfig, dry_run: bool) -> None:
    print("cxbench direct / mediate ablation")
    print(f"  benchmark: {BENCHMARK_ID}")
    print(f"  model: {config.model}")
    print(f"  base_url: {config.endpoint.base_url}")
    print(f"  wire_api: {config.endpoint.wire_api}")
    print(f"  max_output_tokens: {config.max_output_tokens}")
    print(f"  grader: {config.grader}")
    print(f"  tasks: {len(config.tasks)}")
    print(f"  modes: {', '.join(config.modes)} ({len(config.tasks) * len(config.modes)} task runs)")
    print("  ablated_rule: no_agent_direct_submit")
    print(f"  concurrency: {config.concurrency}")
    print(f"  executor: {config.executor}")
    print("  error_retry: once, serial (timeouts excluded)")
    print(f"  output: {config.output_dir}")
    print("  api_key: provided (never written to results)")
    if dry_run:
        print("dry-run complete; no API request was sent")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
        if not BATCH_RUNNER.is_file():
            raise ConfigError(f"batch runner does not exist: {BATCH_RUNNER}")
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))

    _print_plan(config, args.dry_run)
    if args.dry_run:
        return 0

    exit_codes: dict[str, int] = {}
    try:
        _write_launcher_manifest(config)
        child_env = build_child_environment(config)
        for mode in config.modes:
            print(f"\nstarting mode: {mode} (runtime={MODE_RUNTIME_NAMES[mode]})", flush=True)
            completed = subprocess.run(
                build_batch_command(config, mode),
                cwd=REPO_ROOT,
                env=child_env,
                check=False,
            )
            exit_codes[mode] = completed.returncode
    except KeyboardInterrupt:
        print(
            "evaluation interrupted; completed task artifacts remain in the output directory",
            file=sys.stderr,
        )
        return 130
    except OSError as exc:
        print(f"failed to start evaluation: {exc}", file=sys.stderr)
        return 2

    analysis = write_experiment_analysis(
        experiment_dir=config.output_dir,
        task_paths=config.tasks,
        mode_dirs={mode: config.output_dir / mode for mode in config.modes},
        model=config.model,
        grader=config.grader,
        benchmark=BENCHMARK_ID,
        exit_codes=exit_codes,
    )
    launcher_path = config.output_dir / "launcher.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["exit_codes"] = exit_codes
    launcher["analysis_complete"] = True
    launcher_path.write_text(
        json.dumps(launcher, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nresults: {config.output_dir}")
    print(f"summary: {config.output_dir / 'summary.md'}")
    print(f"per-task JSONL: {config.output_dir / 'task_results.jsonl'}")
    print(f"domain metrics: {config.output_dir / 'domain_metrics.csv'}")
    for mode in config.modes:
        metrics = analysis["modes"][mode]
        print(
            f"{mode}: both_pass={metrics['both_pass_rate']:.2%} "
            f"outcome_pass={metrics['outcome_pass_rate']:.2%} "
            f"process_pass={metrics['process_pass_rate']:.2%} "
            f"errors={metrics['error_count']}",
        )
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
