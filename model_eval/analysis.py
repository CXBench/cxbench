from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cxbench.task_schema import TaskSpec


MODE_RUNTIME_NAMES = {
    "mediate": "least_privilege",
    "direct": "direct",
}

CLASSIFICATION_DEFINITIONS = {
    "both-pass": "outcome_pass=true and process_pass=true",
    "false-pass": "outcome_pass=true and process_pass=false",
    "both-fail": "outcome_pass=false and process_pass=false",
    "outcome-fail": "outcome_pass=false and process_pass=true",
    "timeout": "the task exceeded its wall-clock limit",
    "error": "the task did not produce a valid grade",
}


def classify_result(result: Mapping[str, Any] | None) -> str:
    if result is None:
        return "error"
    if not result.get("ok"):
        return "timeout" if result.get("verdict") == "timeout" else "error"
    outcome_pass = bool(result.get("outcome_pass"))
    process_pass = bool(result.get("process_pass"))
    if outcome_pass and process_pass:
        return "both-pass"
    if outcome_pass:
        return "false-pass"
    if not process_pass:
        return "both-fail"
    return "outcome-fail"


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _task_metadata(task_paths: Iterable[Path]) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    for path in task_paths:
        task = TaskSpec.load(path)
        metadata.append(
            {
                "task_id": task.id,
                "domain": task.domain,
                "task_path": str(path.resolve()),
            }
        )
    return sorted(metadata, key=lambda item: item["task_id"])


def _relative_path(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _failure_ids(result: Mapping[str, Any] | None, key: str) -> list[str]:
    if result is None:
        return []
    failures = result.get(key)
    if not isinstance(failures, list):
        return []
    return [
        str(item.get("rule_id", "unknown"))
        for item in failures
        if isinstance(item, dict) and not item.get("passed", False)
    ]


def _mode_task_result(
    result: Mapping[str, Any] | None,
    *,
    experiment_dir: Path,
) -> dict[str, Any]:
    classification = classify_result(result)
    run_dir_value = result.get("run_dir") if result else None
    run_dir = Path(str(run_dir_value)).resolve() if run_dir_value else None
    trace_path = run_dir / "trace.jsonl" if run_dir else None
    grade_path = run_dir / "grade.json" if run_dir else None
    rule_failure_ids = _failure_ids(result, "rule_failures")
    judge_failure_ids = _failure_ids(result, "llm_judge_failures")
    failure_reasons = [f"rule:{item}" for item in rule_failure_ids]
    failure_reasons.extend(f"judge:{item}" for item in judge_failure_ids)
    if result is None:
        failure_reasons.append("missing_result")
    elif not result.get("ok"):
        failure_reasons.append(
            str(result.get("error") or result.get("verdict") or "execution_error")
        )
    elif classification == "outcome-fail" and not failure_reasons:
        failure_reasons.append("outcome_not_achieved")
    elif classification == "both-fail" and not failure_reasons:
        failure_reasons.append("outcome_and_process_not_achieved")
    elif classification == "false-pass" and not failure_reasons:
        failure_reasons.append("process_not_compliant")

    return {
        "success": classification == "both-pass",
        "classification": classification,
        "ok": bool(result and result.get("ok")),
        "outcome_pass": bool(result and result.get("outcome_pass")),
        "process_pass": bool(result and result.get("process_pass")),
        "verdict": result.get("verdict") if result else "missing",
        "failure_reasons": failure_reasons,
        "ablated_rules": list(result.get("ablated_rules") or []) if result else [],
        "rule_failure_ids": rule_failure_ids,
        "llm_judge_failure_ids": judge_failure_ids,
        "error": result.get("error") if result else "result file or task row is missing",
        "run_dir": str(run_dir) if run_dir else None,
        "trace_path": str(trace_path) if trace_path else None,
        "trace_relative_path": _relative_path(trace_path, experiment_dir) if trace_path else None,
        "grade_path": str(grade_path) if grade_path else None,
        "grade_relative_path": _relative_path(grade_path, experiment_dir) if grade_path else None,
        "turns": int(result.get("turns") or 0) if result else 0,
        "tool_calls": int(result.get("tool_calls") or 0) if result else 0,
        "elapsed_seconds": float(result.get("elapsed_seconds") or 0) if result else 0.0,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = Counter(str(row["classification"]) for row in rows)
    completed = sum(1 for row in rows if row.get("ok"))
    outcome_pass = sum(1 for row in rows if row.get("outcome_pass"))
    process_pass = sum(1 for row in rows if row.get("process_pass"))
    both_pass = counts["both-pass"]

    def rate(count: int) -> float:
        return round(count / n, 4) if n else 0.0

    return {
        "n": n,
        "completed_count": completed,
        "error_count": n - completed,
        "outcome_pass_count": outcome_pass,
        "outcome_pass_rate": rate(outcome_pass),
        "process_pass_count": process_pass,
        "process_pass_rate": rate(process_pass),
        "both_pass_count": both_pass,
        "both_pass_rate": rate(both_pass),
        "false_pass_count": counts["false-pass"],
        "false_pass_rate": rate(counts["false-pass"]),
        "both_fail_count": counts["both-fail"],
        "both_fail_rate": rate(counts["both-fail"]),
        "outcome_fail_count": counts["outcome-fail"],
        "timeout_count": counts["timeout"],
        "classification_counts": dict(sorted(counts.items())),
        "rate_denominator": "all scheduled tasks; errors and timeouts count as non-passes",
    }


def build_analysis(
    *,
    experiment_dir: Path,
    tasks: Sequence[Mapping[str, str]],
    mode_results: Mapping[str, Sequence[Mapping[str, Any]]],
    model: str,
    grader: str,
    benchmark: str,
    exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    task_by_id = {str(item["task_id"]): dict(item) for item in tasks}
    results_by_mode = {
        mode: {str(row.get("task_id")): row for row in rows} for mode, rows in mode_results.items()
    }
    unexpected_ids = sorted(
        {
            task_id
            for rows in results_by_mode.values()
            for task_id in rows
            if task_id not in task_by_id
        }
    )
    for task_id in unexpected_ids:
        source = next(rows[task_id] for rows in results_by_mode.values() if task_id in rows)
        task_by_id[task_id] = {
            "task_id": task_id,
            "domain": str(source.get("domain") or "unknown"),
            "task_path": str(source.get("task_path") or ""),
        }

    per_task: list[dict[str, Any]] = []
    for task_id in sorted(task_by_id):
        task = task_by_id[task_id]
        item: dict[str, Any] = {
            "task_id": task_id,
            "domain": task["domain"],
            "task_path": task["task_path"],
        }
        for mode in MODE_RUNTIME_NAMES:
            item[mode] = _mode_task_result(
                results_by_mode.get(mode, {}).get(task_id),
                experiment_dir=experiment_dir,
            )
        per_task.append(item)

    mode_metrics: dict[str, Any] = {}
    domain_metrics: list[dict[str, Any]] = []
    domains = sorted({str(item["domain"]) for item in per_task})
    for mode, runtime_mode in MODE_RUNTIME_NAMES.items():
        rows = [item[mode] for item in per_task]
        mode_metrics[mode] = {
            "runtime_mode": runtime_mode,
            "exit_code": (exit_codes or {}).get(mode),
            **_aggregate(rows),
        }
        for domain in domains:
            domain_rows = [item[mode] for item in per_task if item["domain"] == domain]
            domain_metrics.append(
                {
                    "mode": mode,
                    "runtime_mode": runtime_mode,
                    "domain": domain,
                    **_aggregate(domain_rows),
                }
            )

    transition_counts = Counter(
        f"{item['mediate']['classification']} -> {item['direct']['classification']}"
        for item in per_task
    )
    mediate_both = mode_metrics["mediate"]["both_pass_rate"]
    direct_both = mode_metrics["direct"]["both_pass_rate"]
    mediate_outcome = mode_metrics["mediate"]["outcome_pass_rate"]
    direct_outcome = mode_metrics["direct"]["outcome_pass_rate"]
    paired_ok_count = sum(1 for item in per_task if item["mediate"]["ok"] and item["direct"]["ok"])

    return {
        "schema_version": 1,
        "experiment": "direct-submit-rule-ablation",
        "benchmark": benchmark,
        "model": model,
        "grader": grader,
        "ablated_rule_ids": ["no_agent_direct_submit"],
        "classification_definitions": CLASSIFICATION_DEFINITIONS,
        "task_count": len(per_task),
        "paired_ok_count": paired_ok_count,
        "unexpected_task_ids": unexpected_ids,
        "modes": mode_metrics,
        "domain_metrics": domain_metrics,
        "comparison": {
            "both_pass_rate_delta_direct_minus_mediate": round(direct_both - mediate_both, 4),
            "outcome_pass_rate_delta_direct_minus_mediate": round(
                direct_outcome - mediate_outcome, 4
            ),
            "transition_counts": dict(sorted(transition_counts.items())),
        },
        "tasks": per_task,
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_task_csv(path: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    fields = ["task_id", "domain", "task_path"]
    mode_fields = [
        "success",
        "classification",
        "ok",
        "outcome_pass",
        "process_pass",
        "verdict",
        "failure_reasons",
        "ablated_rules",
        "run_dir",
        "trace_path",
        "grade_path",
        "turns",
        "tool_calls",
        "elapsed_seconds",
    ]
    fieldnames = [
        *fields,
        *(f"{mode}_{field}" for mode in MODE_RUNTIME_NAMES for field in mode_fields),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            row = {field: task.get(field) for field in fields}
            for mode in MODE_RUNTIME_NAMES:
                mode_row = task[mode]
                for field in mode_fields:
                    value = mode_row.get(field)
                    row[f"{mode}_{field}"] = (
                        _json_cell(value) if isinstance(value, (list, dict)) else value
                    )
            writer.writerow(row)


def _write_domain_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "runtime_mode",
        "domain",
        "n",
        "completed_count",
        "error_count",
        "outcome_pass_count",
        "outcome_pass_rate",
        "process_pass_count",
        "process_pass_rate",
        "both_pass_count",
        "both_pass_rate",
        "false_pass_count",
        "false_pass_rate",
        "both_fail_count",
        "both_fail_rate",
        "outcome_fail_count",
        "timeout_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rate_cell(metrics: Mapping[str, Any], prefix: str) -> str:
    count = int(metrics[f"{prefix}_count"])
    n = int(metrics["n"])
    rate = float(metrics[f"{prefix}_rate"])
    return f"{count}/{n} ({rate:.2%})"


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _trace_cell(task: Mapping[str, Any], mode: str) -> str:
    result = task[mode]
    label = _markdown_text(result["classification"])
    relative = result.get("trace_relative_path")
    return f"[{label}]({relative})" if relative else label


def _failure_cell(task: Mapping[str, Any], mode: str) -> str:
    result = task[mode]
    if result["success"]:
        return "-"
    reasons = result.get("failure_reasons") or [result["classification"]]
    return _markdown_text(", ".join(str(reason) for reason in reasons))


def _write_summary(path: Path, analysis: Mapping[str, Any]) -> None:
    mediate = analysis["modes"]["mediate"]
    direct = analysis["modes"]["direct"]
    lines = [
        "# CXBench direct / mediate ablation",
        "",
        f"- Benchmark: `{analysis['benchmark']}`",
        f"- Model: `{analysis['model']}`",
        f"- Grader: `{analysis['grader']}`",
        "- Mediate runtime mode: `least_privilege`",
        "- Direct ablation: `no_agent_direct_submit` is excluded from grading; confirmation, field-match, and all other rules remain active.",
        "- Rate denominator: all scheduled tasks; errors and timeouts count as non-passes.",
        "",
        "## Overall",
        "",
        "| mode | completed | both-pass | outcome-pass | process-pass | false-pass | both-fail | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in (("mediate", mediate), ("direct", direct)):
        lines.append(
            f"| {mode} | {metrics['completed_count']}/{metrics['n']} | "
            f"{_rate_cell(metrics, 'both_pass')} | {_rate_cell(metrics, 'outcome_pass')} | "
            f"{_rate_cell(metrics, 'process_pass')} | "
            f"{_rate_cell(metrics, 'false_pass')} | {_rate_cell(metrics, 'both_fail')} | "
            f"{metrics['error_count']} |"
        )
    lines.extend(
        [
            "",
            f"- Direct minus mediate both-pass delta: {analysis['comparison']['both_pass_rate_delta_direct_minus_mediate']:+.2%}",
            f"- Direct minus mediate outcome-pass delta: {analysis['comparison']['outcome_pass_rate_delta_direct_minus_mediate']:+.2%}",
            f"- Successfully paired tasks: {analysis['paired_ok_count']}/{analysis['task_count']}",
            "",
            "## Domain metrics",
            "",
            "| mode | domain | completed | both-pass | outcome-pass | process-pass | false-pass | both-fail |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metrics in analysis["domain_metrics"]:
        lines.append(
            f"| {metrics['mode']} | {metrics['domain']} | {metrics['completed_count']}/{metrics['n']} | "
            f"{_rate_cell(metrics, 'both_pass')} | {_rate_cell(metrics, 'outcome_pass')} | "
            f"{_rate_cell(metrics, 'process_pass')} | "
            f"{_rate_cell(metrics, 'false_pass')} | {_rate_cell(metrics, 'both_fail')} |"
        )
    lines.extend(
        [
            "",
            "## Per-task results",
            "",
            "Classification links open the corresponding full `trace.jsonl`.",
            "",
            "| task | domain | mediate success | mediate classification | mediate failure | direct success | direct classification | direct failure |",
            "|---|---|---:|---|---|---:|---|---|",
        ]
    )
    for task in analysis["tasks"]:
        lines.append(
            f"| {_markdown_text(task['task_id'])} | {_markdown_text(task['domain'])} | "
            f"{'yes' if task['mediate']['success'] else 'no'} | {_trace_cell(task, 'mediate')} | "
            f"{_failure_cell(task, 'mediate')} | "
            f"{'yes' if task['direct']['success'] else 'no'} | {_trace_cell(task, 'direct')} | "
            f"{_failure_cell(task, 'direct')} |"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
        ]
    )
    for label, definition in CLASSIFICATION_DEFINITIONS.items():
        lines.append(f"- `{label}`: {definition}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_analysis(
    *,
    experiment_dir: Path,
    task_paths: Sequence[Path],
    mode_dirs: Mapping[str, Path],
    model: str,
    grader: str,
    benchmark: str,
    exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    tasks = _task_metadata(task_paths)
    mode_results = {
        mode: _load_results(mode_dirs[mode] / "results.jsonl") for mode in MODE_RUNTIME_NAMES
    }
    analysis = build_analysis(
        experiment_dir=experiment_dir,
        tasks=tasks,
        mode_results=mode_results,
        model=model,
        grader=grader,
        benchmark=benchmark,
        exit_codes=exit_codes,
    )
    (experiment_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "task_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in analysis["tasks"]),
        encoding="utf-8",
    )
    _write_task_csv(experiment_dir / "task_results.csv", analysis["tasks"])
    _write_domain_csv(experiment_dir / "domain_metrics.csv", analysis["domain_metrics"])
    _write_summary(experiment_dir / "summary.md", analysis)
    return analysis
