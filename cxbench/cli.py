from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_generation import (
    accept_candidates,
    generate_benchmark,
    repair_candidates_with_feedback,
)
from .eval_config import GRADERS, RuntimeConfig
from .grader import write_grade
from .openai_compat import OpenAICompatError
from .report import write_report
from .retry import retry_until_completed
from .runtime import Runtime
from .task_schema import TaskSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cxbench", description="Minimal cxbench process-compliance runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one task")
    run_parser.add_argument("--task", required=True, help="path to task YAML")
    run_parser.add_argument(
        "--mode",
        choices=["least_privilege", "direct", "high_permission"],
        default="least_privilege",
        help="tool permission mode; high_permission is an alias for direct",
    )
    run_parser.add_argument(
        "--model",
        default="auto",
        help="agent adapter: auto, scripted:compliant, scripted:direct_bad, openai:<model>",
    )
    run_parser.add_argument(
        "--grader",
        choices=["rule", "llm", "hybrid"],
        default="hybrid",
        help="rule and llm graders are each incomplete; hybrid is the only mode that yields the accurate cxbench verdict and is the default",
    )
    run_parser.add_argument(
        "--judge-model", help="LLM judge model, e.g. openai:gpt-5-mini or crs:gpt-5.4-mini"
    )
    run_parser.add_argument(
        "--user-sim",
        choices=["llm", "react", "verify", "reflection"],
        default="llm",
        help="LLM customer simulator strategy; deterministic is test-only",
    )
    run_parser.add_argument(
        "--user-model",
        required=True,
        help="LLM user simulator model, e.g. openai:gpt-5.4-mini or crs:gpt-5.4-mini",
    )
    run_parser.add_argument(
        "--user-max-attempts", type=int, default=3, help="max verify/reflection attempts"
    )
    run_parser.add_argument("--runs-dir", default="runs", help="directory for run artifacts")

    grade_parser = subparsers.add_parser("grade", help="re-grade an existing run")
    grade_parser.add_argument("--run", required=True, help="path to run directory")
    grade_parser.add_argument(
        "--grader", choices=["rule", "llm", "hybrid"], help="override grader from run config"
    )
    grade_parser.add_argument("--judge-model", help="override judge model from run config")

    generate_parser = subparsers.add_parser("generate", help="generate benchmark candidate tasks")
    generate_parser.add_argument("--config", required=True, help="generation config YAML")
    generate_parser.add_argument("--out", required=True, help="candidate run directory")
    generate_parser.add_argument(
        "--workers", type=int, default=1, help="number of nested Codex prompt shards"
    )
    generate_parser.add_argument("--executor", choices=["local", "codex"], default="local")
    generate_parser.add_argument(
        "--reuse-rule-selection", help="selected_rule_cards.jsonl from a prior run"
    )
    generate_parser.add_argument(
        "--rule-registry", help="manual rule registry YAML for rule-seeded generation"
    )
    generate_parser.add_argument(
        "--no-merge-grading-specs",
        action="store_true",
        help="leave grading_specs/*.json separate instead of embedding them into task policy.llm_judge",
    )
    generate_parser.add_argument(
        "--review-model",
        help="optional LLM quality gate model, e.g. openai:gpt-5.4-mini; failing reviews block candidate acceptance",
    )
    generate_parser.add_argument(
        "--feedback-rounds",
        type=int,
        default=0,
        help="number of nested Codex repair rounds to run after the quality gate",
    )

    repair_parser = subparsers.add_parser(
        "repair", help="repair generated candidates using quality-gate feedback"
    )
    repair_parser.add_argument("--run", required=True, help="candidate run directory")
    repair_parser.add_argument(
        "--review-model", required=True, help="LLM quality gate model, e.g. openai:gpt-5.4-mini"
    )
    repair_parser.add_argument(
        "--rounds", type=int, default=1, help="number of feedback repair rounds"
    )

    accept_parser = subparsers.add_parser("accept", help="copy validated candidates into tasks")
    accept_parser.add_argument("--run", required=True, help="candidate run directory")
    accept_parser.add_argument(
        "--dest", default="tasks", help="destination directory for accepted task YAML"
    )
    accept_parser.add_argument(
        "--candidate", help="candidate relative path, e.g. candidates/BK-01.yaml"
    )
    accept_parser.add_argument(
        "--force", action="store_true", help="overwrite existing accepted task"
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "grade":
        return _grade(args)
    if args.command == "generate":
        return _generate(args)
    if args.command == "repair":
        return _repair(args)
    if args.command == "accept":
        return _accept(args)
    parser.error("unknown command")
    return 2


def _run(args: Any) -> int:
    task = TaskSpec.load(args.task)
    try:
        runtime_options = dict(
            mode=args.mode,
            model=args.model,
            runs_dir=args.runs_dir,
            grader=args.grader,
            judge_model=args.judge_model,
            user_sim=args.user_sim,
            user_model=args.user_model,
            user_max_attempts=args.user_max_attempts,
        )
        RuntimeConfig.parse(**runtime_options)
        result = retry_until_completed(
            lambda: Runtime(task=task, **runtime_options).run(),
            label=f"task {task.id}",
        )
    except ValueError as exc:
        print(f"run_error: {exc}")
        return 2
    grade = result["grade"]
    print(f"run_dir: {result['run_dir']}")
    print(f"report: {result['report']}")
    print(f"outcome_pass: {grade['outcome_pass']}")
    print(f"process_pass: {grade['process_pass']}")
    print(f"false_pass_demo: {grade['false_pass_demo']}")
    print(f"grader: {grade['grader']}")
    return 0


def _grade(args: Any) -> int:
    run_dir = Path(args.run)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json in run dir: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = TaskSpec.load(config["task_path"])
    grader = args.grader or config.get("grader", "hybrid")
    judge_model = args.judge_model or config.get("judge_model")
    try:
        if grader not in GRADERS:
            raise ValueError("grader must be 'rule', 'llm', or 'hybrid'")
        if grader in {"llm", "hybrid"} and not judge_model:
            raise ValueError("LLM grader requires --judge-model openai:<model>")
        for name in ("trace.jsonl", "final_state.json"):
            if not (run_dir / name).is_file():
                raise ValueError(f"missing {name} in run dir: {run_dir}")
        grade = retry_until_completed(
            lambda: write_grade(
                run_dir, task, config["mode"], grader=grader, judge_model=judge_model
            ),
            label=f"task {task.id} grading",
        )
    except ValueError as exc:
        print(f"grade_error: {exc}")
        return 2
    report = write_report(run_dir, grade)
    print(f"grade: {run_dir / 'grade.json'}")
    print(f"report: {report}")
    print(f"outcome_pass: {grade['outcome_pass']}")
    print(f"process_pass: {grade['process_pass']}")
    print(f"false_pass_demo: {grade['false_pass_demo']}")
    print(f"grader: {grade['grader']}")
    return 0


def _generate(args: Any) -> int:
    if args.feedback_rounds and not args.review_model:
        print("generate_error: --feedback-rounds requires --review-model")
        return 2
    try:
        validation = generate_benchmark(
            args.config,
            args.out,
            workers=args.workers,
            executor=args.executor,
            reuse_rule_selection=args.reuse_rule_selection,
            rule_registry=args.rule_registry,
            merge_grading_specs=not args.no_merge_grading_specs,
            review_model=args.review_model,
            feedback_rounds=args.feedback_rounds,
        )
    except (OSError, ValueError, OpenAICompatError) as exc:
        print(f"generate_error: {exc}")
        return 2
    print(f"run_dir: {validation['run_dir']}")
    print(f"candidate_count: {validation['candidate_count']}")
    print(f"valid_count: {validation['valid_count']}")
    print(
        f"merged_grading_specs: {sum(1 for item in validation.get('merged_grading_specs', []) if item.get('merged'))}"
    )
    if validation.get("quality_gate"):
        gate = validation["quality_gate"]
        print(f"quality_gate: {gate.get('pass_count', 0)}/{gate.get('review_count', 0)} passed")
    print(f"validation: {Path(args.out) / 'validation.json'}")
    return 0 if validation["valid"] else 1


def _repair(args: Any) -> int:
    try:
        validation = repair_candidates_with_feedback(
            args.run, review_model=args.review_model, rounds=args.rounds
        )
    except (OSError, ValueError, OpenAICompatError) as exc:
        print(f"repair_error: {exc}")
        return 2
    print(f"run_dir: {validation['run_dir']}")
    print(f"candidate_count: {validation['candidate_count']}")
    print(f"valid_count: {validation['valid_count']}")
    if validation.get("quality_gate"):
        gate = validation["quality_gate"]
        print(f"quality_gate: {gate.get('pass_count', 0)}/{gate.get('review_count', 0)} passed")
    print(f"feedback_rounds: {len(validation.get('feedback_rounds', []))}")
    print(f"validation: {Path(args.run) / 'validation.json'}")
    return 0 if validation["valid"] else 1


def _accept(args: Any) -> int:
    try:
        manifest = accept_candidates(
            args.run, args.dest, candidate=args.candidate, force=args.force
        )
    except (OSError, ValueError) as exc:
        print(f"accept_error: {exc}")
        return 1
    print(f"accepted_count: {len(manifest['accepted'])}")
    for path in manifest["accepted"]:
        print(f"accepted: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
