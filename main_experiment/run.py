"""Run the main benchmark and summarize its completed hybrid grades."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from cxbench.eval_config import RuntimeConfig, model_string
from cxbench.task_schema import TaskSpec
from scripts import run_eval_batch as batch

from .score import score_batches

DEFAULT_TASK = "benchmarks/cxbench-120/tasks/*.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = batch.build_parser(default_task=DEFAULT_TASK, default_runs_dir="runs/main_experiment")
    parser.prog = "python -m main_experiment.run"
    parser.description = (
        "Run the revised 120-task benchmark and produce hybrid scores. "
        "Explicit --task patterns replace the default corpus."
    )
    parser.add_argument("--score-output", help="score directory; default: RUN_DIR/paper_scores")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint for --agent-provider openai")
    parser.add_argument(
        "--wire-api", choices=["chat", "responses"], help="OpenAI-compatible transport override"
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=8192, help="provider output limit (default: 8192)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate tasks and options without API calls or writes",
    )
    return parser


@contextmanager
def _provider_options(args: argparse.Namespace) -> Iterator[None]:
    overrides = {"OPENAI_MAX_OUTPUT_TOKENS": str(args.max_output_tokens)}
    if args.base_url:
        overrides["OPENAI_BASE_URL"] = args.base_url
    if args.wire_api:
        overrides["OPENAI_WIRE_API"] = args.wire_api
    if getattr(args, "reasoning_effort", None):
        overrides["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        batch.validate_args(args)
        for name in ("concurrency", "user_max_attempts", "task_timeout", "max_output_tokens"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.base_url and args.agent_provider != "openai":
            raise ValueError(
                "--base-url requires --agent-provider openai; other providers use their environment settings"
            )
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
        task_paths = batch._task_paths(args.task or [args.default_task])
        tasks = [TaskSpec.load(path) for path in task_paths]
        if not tasks:
            raise ValueError("no task files selected")
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("selected task files contain duplicate task IDs")
        args.run_id = args.run_id or (
            datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            + "-"
            + args.model.replace("/", "_").replace(":", "_")
        )
        out_dir = (batch.ROOT / args.runs_dir / args.run_id).resolve()
        score_dir = (
            Path(args.score_output).resolve() if args.score_output else out_dir / "paper_scores"
        )
        if out_dir.exists():
            raise ValueError(f"run directory already exists: {out_dir}; choose a new --run-id")
        if args.grader == "hybrid":
            if (
                score_dir == out_dir
                or out_dir.is_relative_to(score_dir)
                or score_dir.is_relative_to(out_dir / "task_runs")
            ):
                raise ValueError(
                    "score directory must be separate from the batch and its task runs"
                )
            if score_dir.exists():
                raise ValueError(f"score directory already exists: {score_dir}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"tasks: {len(tasks)}")
    print(f"model: {args.model}; user_model: {args.user_model}; judge_model: {args.judge_model}")
    print(
        f"mode: {args.mode}; grader: {args.grader}; executor: {args.executor}; concurrency: {args.concurrency}"
    )
    print(f"out_dir: {out_dir}")
    if args.dry_run:
        print("dry-run complete; no API calls or files written")
        return 0

    try:
        with _provider_options(args):
            exit_code = batch.run_batch(args)
        if exit_code:
            return exit_code
        if args.grader == "hybrid":
            score_batches({args.model: out_dir}, score_dir)
            print(f"paper_scores: {score_dir}")
        else:
            print("diagnostic grader only; no formal paper scores generated")
    except KeyboardInterrupt:
        print("interrupted; completed task records remain in the run directory")
        return 130
    except (OSError, ValueError, batch.ProviderTransportError) as exc:
        print(f"experiment failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
