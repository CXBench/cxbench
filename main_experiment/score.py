"""Score completed hybrid batches using the paper's offline aggregation rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from cxbench.grader import grade_run, normalize_llm_judge
from cxbench.task_schema import TaskSpec
from cxbench.statistics import summarize_records

ROOT = Path(__file__).resolve().parents[1]


class ScoreError(ValueError):
    """An incomplete or incompatible batch cannot produce formal scores."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScoreError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ScoreError(f"expected JSON objects: {path}")
    return rows


def _task_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and (ROOT / path).is_file():
        return (ROOT / path).resolve()
    # Archived batches may record the same checkout at a different location.
    normalized = str(value).replace("\\", "/")
    for anchor in ("benchmarks/", "tasks/"):
        index = normalized.find(anchor)
        if index >= 0 and (ROOT / normalized[index:]).is_file():
            return (ROOT / normalized[index:]).resolve()
    raise ScoreError(f"task file unavailable: {value}")


def _load_batch_runs(
    batch_dir: Path,
    *,
    require_hybrid: bool = True,
) -> tuple[dict, dict[str, tuple[dict, TaskSpec, Path, dict]], dict[str, str]]:
    manifest = _json(batch_dir / "manifest.json")
    if require_hybrid and manifest.get("final_grader", manifest.get("grader")) != "hybrid":
        raise ScoreError(f"formal scoring requires a hybrid batch: {batch_dir}")
    planned = manifest.get("tasks")
    count = manifest.get("task_count")
    if (
        not isinstance(planned, list)
        or not planned
        or type(count) is not int
        or count != len(planned)
    ):
        raise ScoreError(
            f"manifest tasks/task_count is missing, empty, or inconsistent: {batch_dir}"
        )
    tasks: dict[str, tuple[TaskSpec, Path]] = {}
    hashes = {}
    for item in planned:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ScoreError(f"invalid manifest task: {item!r}")
        path = _task_path(item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if item.get("sha256") != digest:
            raise ScoreError(f"task changed since the original run (sha256 mismatch): {path}")
        task = TaskSpec.load(path)
        if task.id in tasks:
            raise ScoreError(f"duplicate planned task: {task.id}")
        tasks[task.id] = (task, path)
        hashes[task.id] = digest

    selected: dict[str, dict] = {}
    for result in _jsonl(batch_dir / "results.jsonl"):
        task_id = result.get("task_id")
        if not isinstance(task_id, str) or task_id not in tasks:
            raise ScoreError(f"unplanned result task: {task_id!r}")
        if task_id in selected:
            raise ScoreError(f"duplicate result task: {task_id}")
        if (
            result.get("ok") is not True
            or result.get("verdict") not in {"pass", "fail"}
            or result.get("error")
        ):
            raise ScoreError(f"task execution did not complete successfully: {task_id}")
        if (
            require_hybrid
            and result.get("final_grader", manifest.get("final_grader", manifest.get("grader")))
            != "hybrid"
        ):
            raise ScoreError(f"result requires a hybrid final grader: {task_id}")
        selected[task_id] = result
    missing = tasks.keys() - selected.keys()
    if missing:
        raise ScoreError(f"batch is incomplete; missing result tasks: {', '.join(sorted(missing))}")

    runs = {}
    for task_id, result in selected.items():
        task, task_path = tasks[task_id]
        recorded_run = result.get("run_dir")
        if not isinstance(recorded_run, str) or not recorded_run:
            raise ScoreError(f"missing successful run_dir: {task_id}")
        run_dir = Path(recorded_run)
        if not run_dir.is_absolute():
            run_dir = batch_dir / run_dir
        if not run_dir.is_dir():
            run_dir = batch_dir / "task_runs" / recorded_run.replace("\\", "/").rsplit("/", 1)[-1]
        run_dir = run_dir.resolve()
        if not run_dir.is_relative_to((batch_dir / "task_runs").resolve()):
            raise ScoreError(f"selected run is outside the batch task_runs: {run_dir}")
        config, recorded_grade = _json(run_dir / "config.json"), _json(run_dir / "grade.json")
        if (
            _task_path(config.get("task_path", "")) != task_path
            or recorded_grade.get("task_id") != task_id
        ):
            raise ScoreError(f"selected run does not match its planned task: {task_id}")
        if result.get("task_path") and _task_path(result["task_path"]) != task_path:
            raise ScoreError(f"result task_path does not match its planned task: {task_id}")
        if require_hybrid and (
            config.get("final_grader", config.get("grader")) != "hybrid"
            or recorded_grade.get("grader") != "hybrid"
        ):
            raise ScoreError(f"selected run requires recorded hybrid grading: {task_id}")
        mode = config.get("mode")
        if mode not in {"least_privilege", "direct"}:
            raise ScoreError(f"missing or invalid run mode: {task_id}")
        _jsonl(run_dir / "trace.jsonl")
        _json(run_dir / "final_state.json")
        runs[task_id] = (result, task, run_dir, config)
    return manifest, runs, hashes


def _judge_pass(recorded_grade: dict, task_id: str) -> bool:
    judge = recorded_grade.get("llm_judge")
    if (
        not isinstance(judge, dict)
        or type(judge.get("process_pass")) is not bool
        or not isinstance(judge.get("findings"), list)
    ):
        raise ScoreError(f"missing or malformed recorded judge evidence: {task_id}")
    findings = judge["findings"]
    if any(
        not isinstance(finding, dict)
        or type(finding.get("passed")) is not bool
        or not isinstance(finding.get("rule_id"), str)
        or not isinstance(finding.get("evidence"), str)
        for finding in findings
    ):
        raise ScoreError(f"malformed recorded judge findings: {task_id}")
    return normalize_llm_judge(judge)["process_pass"]


def _load_batch(batch_dir: Path) -> tuple[dict[str, dict], dict[str, bool], dict[str, str]]:
    _, runs, hashes = _load_batch_runs(batch_dir)
    records, judge_pass = {}, {}
    for task_id, (_, task, run_dir, config) in runs.items():
        recorded_grade = _json(run_dir / "grade.json")
        judge_pass[task_id] = _judge_pass(recorded_grade, task_id)
        mode = config["mode"]
        trace = _jsonl(run_dir / "trace.jsonl")
        final_state = _json(run_dir / "final_state.json")
        # Rule-only execution is an offline component; the reported verdict is hybrid.
        grade = grade_run(task, trace, final_state, mode, grader="rule")
        policy = grade["grading_policy"]
        soft = set(policy.get("semantic_judge_rules", [])) | set(
            policy.get("hybrid_hint_rules", [])
        )
        records[task_id] = {
            "task_id": task_id,
            "outcome_pass": grade["outcome_pass"],
            "rules": {rule["rule_id"]: rule["passed"] for rule in grade["rules"]},
            "soft_rules": sorted(soft),
            "judge_pass": judge_pass[task_id],
            "run_dir": str(run_dir),
            "task_sha256": hashes[task_id],
        }
    return records, judge_pass, hashes


def score_batches(
    batches: Mapping[str, str | Path],
    output_dir: str | Path,
    reference_model: str | None = None,
) -> dict[str, Any]:
    """Validate all batches, then write fresh offline hybrid paper statistics."""
    if not batches:
        raise ScoreError("provide at least one --batch LABEL=PATH")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ScoreError(f"score output already exists; choose a new directory: {output}")
    filenames = {}
    for label in batches:
        if not isinstance(label, str) or not label.strip() or "\n" in label or "\r" in label:
            raise ScoreError("model labels must be nonempty and contain no newlines")
        filenames[label] = re.sub(r"[^A-Za-z0-9._-]", "_", label)
    if len(set(filenames.values())) != len(filenames):
        raise ScoreError("model labels collide after filename normalization; use distinct labels")
    records, judges = {}, {}
    expected_hashes = None
    for label, directory in batches.items():
        batch_dir = Path(directory).resolve()
        try:
            rows, decisions, hashes = _load_batch(batch_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ScoreError(f"{label}: {exc}") from exc
        if expected_hashes is not None and hashes != expected_hashes:
            raise ScoreError(
                "all compared models must contain the same planned tasks and task hashes"
            )
        expected_hashes = hashes
        records[label] = rows
        judges.update({(label, task_id): passed for task_id, passed in decisions.items()})
    if reference_model is not None and reference_model not in records:
        raise ScoreError(f"reference model is not among --batch labels: {reference_model}")
    summary = summarize_records(records, judges, reference_model=reference_model)
    summary["batches"] = {
        label: str(Path(directory).resolve()) for label, directory in batches.items()
    }
    summary["reference_model"] = reference_model
    output = Path(output_dir).resolve()
    for directory in batches.values():
        source = Path(directory).resolve()
        if (
            output == source
            or source.is_relative_to(output)
            or output.is_relative_to(source / "task_runs")
        ):
            raise ScoreError("score output must not overwrite a batch or its task runs")
    output.mkdir(parents=True, exist_ok=False)
    for label, rows in records.items():
        (output / f"rules-{filenames[label]}.jsonl").write_text(
            "".join(
                json.dumps(rows[task_id], ensure_ascii=False) + "\n" for task_id in sorted(rows)
            ),
            encoding="utf-8",
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Main experiment scores",
        "",
        "Hybrid pass = outcome AND current hard rules AND the normalized recorded judge decision.",
        "Outcome and rules are recomputed offline; semantic and hybrid-hint rules are excluded from hard gates.",
        "",
        "| Model | Completed | Outcome pass | Hybrid pass | False pass | Hybrid rate | 95% Wilson CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in summary["per_model"].items():
        low, high = row["ci95"]
        display_label = label.replace("|", "\\|")
        report.append(
            f"| {display_label} | {row['completed']} | {row['outcome_pass']} | {row['both_pass']} | {row['false_pass']} | {row['rate']:.1%} | {low:.0%} - {high:.0%} |"
        )
    report += [
        "",
        "Taxonomy, leave-one-rule-out sensitivity, and paired exact McNemar tests are in summary.json.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="completed hybrid batch; repeat for multiple models",
    )
    parser.add_argument("--output", required=True, type=Path, help="directory for score artifacts")
    parser.add_argument(
        "--reference-model", help="optional baseline label excluded from comparison-model ordering"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    batches = {}
    for value in args.batch:
        label, separator, path = value.partition("=")
        if not separator or not path or label in batches:
            parser.error("--batch must be LABEL=PATH with a unique label")
        batches[label] = path
    try:
        summary = score_batches(batches, args.output, reference_model=args.reference_model)
    except (ScoreError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary["per_model"], indent=2, ensure_ascii=False))
    print(f"scores: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
