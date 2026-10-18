from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_METRICS = ("both_pass", "outcome_pass", "process_pass")


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    completed = sum(1 for row in rows if row.get("ok"))
    outcome_pass = sum(1 for row in rows if row.get("ok") and row.get("outcome_pass"))
    process_pass = sum(1 for row in rows if row.get("ok") and row.get("process_pass"))
    both_pass = sum(
        1 for row in rows if row.get("ok") and row.get("outcome_pass") and row.get("process_pass")
    )

    def rate(count: int) -> float:
        return round(count / n, 4) if n else 0.0

    return {
        "n": n,
        "completed_count": completed,
        "error_count": n - completed,
        "both_pass_count": both_pass,
        "both_pass_rate": rate(both_pass),
        "outcome_pass_count": outcome_pass,
        "outcome_pass_rate": rate(outcome_pass),
        "process_pass_count": process_pass,
        "process_pass_rate": rate(process_pass),
    }


def build_model_report(
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task_ids = [str(row.get("task_id") or "") for row in results]
    if any(not task_id for task_id in task_ids):
        raise ValueError("every result row must contain task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("results.jsonl contains duplicate task ids")
    expected_count = manifest.get("task_count")
    if isinstance(expected_count, int) and expected_count != len(results):
        raise ValueError(
            f"manifest declares {expected_count} tasks but results.jsonl contains {len(results)} rows"
        )

    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        by_domain[str(row.get("domain") or "unknown")].append(row)

    return {
        "schema_version": 1,
        "benchmark": manifest.get("benchmark"),
        "run_id": manifest.get("run_id"),
        "model": manifest.get("model"),
        "wire_api": manifest.get("wire_api"),
        "reasoning_effort": manifest.get("reasoning_effort"),
        "mode": manifest.get("mode"),
        "grader": manifest.get("grader"),
        "rate_denominator": "all scheduled tasks; errors and timeouts count as non-passes",
        "overall": aggregate_results(results),
        "domains": [
            {"domain": domain, **aggregate_results(rows)}
            for domain, rows in sorted(by_domain.items())
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object on {path}:{line_number}")
        rows.append(value)
    return rows


def _metric_cell(metrics: Mapping[str, Any], name: str) -> str:
    count = int(metrics[f"{name}_count"])
    n = int(metrics["n"])
    rate = float(metrics[f"{name}_rate"])
    return f"{count}/{n} ({rate:.2%})"


def _write_domain_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fieldnames = [
        "domain",
        "n",
        "completed_count",
        "error_count",
        "both_pass_count",
        "both_pass_rate",
        "outcome_pass_count",
        "outcome_pass_rate",
        "process_pass_count",
        "process_pass_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    overall = report["overall"]
    reasoning_effort = report.get("reasoning_effort") or "not set"
    lines = [
        "# CXBench model evaluation",
        "",
        f"- Benchmark: `{report.get('benchmark')}`",
        f"- Model: `{report.get('model')}`",
        f"- Wire API: `{report.get('wire_api')}`",
        f"- Reasoning effort: `{reasoning_effort}`",
        f"- Runtime mode: `{report.get('mode')}`",
        f"- Grader: `{report.get('grader')}`",
        f"- Executor: `{manifest.get('executor')}` with concurrency `{manifest.get('concurrency')}`",
        "- Rate denominator: all scheduled tasks; errors and timeouts count as non-passes.",
        "",
        "## Overall",
        "",
        "| completed | both-pass | outcome-pass | process-pass | errors |",
        "|---:|---:|---:|---:|---:|",
        f"| {overall['completed_count']}/{overall['n']} | {_metric_cell(overall, 'both_pass')} | "
        f"{_metric_cell(overall, 'outcome_pass')} | {_metric_cell(overall, 'process_pass')} | "
        f"{overall['error_count']} |",
        "",
        "## Domain metrics",
        "",
        "| domain | completed | both-pass | outcome-pass | process-pass | errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in report["domains"]:
        lines.append(
            f"| {domain['domain']} | {domain['completed_count']}/{domain['n']} | "
            f"{_metric_cell(domain, 'both_pass')} | {_metric_cell(domain, 'outcome_pass')} | "
            f"{_metric_cell(domain, 'process_pass')} | {domain['error_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_report(output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    results = _read_jsonl(output_dir / "results.jsonl")
    report = build_model_report(manifest, results)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_domain_csv(output_dir / "domain_metrics.csv", report["domains"])
    _write_markdown(output_dir / "report.md", report, manifest)
    return report
