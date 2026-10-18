"""Eval study analysis functions.

Each ``analyze_*`` function accepts run data (loaded ``results.jsonl`` rows or
paths to run directories) and returns a :class:`StudyResult` with structured
findings ready to be serialised to ``analysis.json`` and rendered as Markdown.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class StudyResult:
    study: str
    summary: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load ``results.jsonl`` or ``reconstructed_results.jsonl``."""
    for candidate in ("results.jsonl", "reconstructed_results.jsonl"):
        p = path / candidate
        if p.exists():
            return [json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"no results.jsonl found in {path}")


def load_metrics(path: Path) -> Dict[str, Any]:
    for candidate in ("metrics.json", "reconstructed_metrics.json"):
        p = path / candidate
        if p.exists():
            return json.loads(p.read_text("utf-8"))
    raise FileNotFoundError(f"no metrics.json found in {path}")


def _safe_mean(values: List[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _safe_stdev(values: List[float]) -> float:
    return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0


def _rate(items: List[Dict[str, Any]], key: str) -> float:
    n = len(items)
    return round(sum(1 for r in items if r.get(key)) / n, 4) if n else 0.0


def _kendall_tau(rank_a: List[str], rank_b: List[str]) -> float:
    """Kendall rank correlation between two ordered lists of model names."""
    if set(rank_a) != set(rank_b):
        common = sorted(set(rank_a) & set(rank_b))
        rank_a = [m for m in rank_a if m in common]
        rank_b = [m for m in rank_b if m in common]
    n = len(rank_a)
    if n < 2:
        return 1.0
    pos_b = {name: i for i, name in enumerate(rank_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            b_i = pos_b[rank_a[i]]
            b_j = pos_b[rank_a[j]]
            if b_i < b_j:
                concordant += 1
            elif b_i > b_j:
                discordant += 1
    pairs = n * (n - 1) / 2
    return round((concordant - discordant) / pairs, 4) if pairs else 1.0


def _cohens_kappa(labels_a: List[bool], labels_b: List[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    p_a_true = sum(labels_a) / n
    p_b_true = sum(labels_b) / n
    p_e = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


# ---------------------------------------------------------------------------
# Study 1: False-Pass
# ---------------------------------------------------------------------------


def analyze_false_pass(run_dirs: List[Path]) -> StudyResult:
    """Prove outcome-only scoring misclassifies non-compliant runs as passes."""
    all_results: List[Dict[str, Any]] = []
    for d in run_dirs:
        all_results.extend(load_results(d))

    ok_results = [r for r in all_results if r.get("ok")]
    outcome_pass = [r for r in ok_results if r.get("outcome_pass")]
    false_passes = [r for r in outcome_pass if not r.get("process_pass")]

    false_pass_rate = round(len(false_passes) / len(outcome_pass), 4) if outcome_pass else 0.0
    outcome_only_accuracy = round(1.0 - false_pass_rate, 4)

    rule_failures: Counter[str] = Counter()
    for r in false_passes:
        for rf in r.get("rule_failures", []):
            rule_failures[str(rf.get("rule_id", "unknown"))] += 1

    by_domain: Dict[str, Dict[str, Any]] = {}
    for r in ok_results:
        domain = str(r.get("domain", "unknown"))
        if domain not in by_domain:
            by_domain[domain] = {"n": 0, "outcome_pass": 0, "false_pass": 0}
        by_domain[domain]["n"] += 1
        if r.get("outcome_pass"):
            by_domain[domain]["outcome_pass"] += 1
        if r.get("outcome_pass") and not r.get("process_pass"):
            by_domain[domain]["false_pass"] += 1
    for stats in by_domain.values():
        stats["false_pass_rate"] = (
            round(stats["false_pass"] / stats["outcome_pass"], 4) if stats["outcome_pass"] else 0.0
        )

    return StudyResult(
        study="false_pass",
        summary={
            "total_ok": len(ok_results),
            "outcome_pass_count": len(outcome_pass),
            "false_pass_count": len(false_passes),
            "false_pass_rate": false_pass_rate,
            "outcome_only_accuracy": outcome_only_accuracy,
            "claim": f"Outcome-only scoring would misclassify {len(false_passes)}/{len(outcome_pass)} "
            f"({false_pass_rate:.1%}) outcome-successful runs as compliant.",
        },
        tables={
            "false_pass_cases": [
                {
                    "task_id": r.get("task_id"),
                    "domain": r.get("domain"),
                    "rule_failures": [rf.get("rule_id") for rf in r.get("rule_failures", [])],
                }
                for r in false_passes
            ],
            "rule_failure_distribution": [
                {"rule_id": rule, "count": count} for rule, count in rule_failures.most_common()
            ],
        },
        details={"by_domain": by_domain},
    )


# ---------------------------------------------------------------------------
# Study 2: Permission-Mode Ablation
# ---------------------------------------------------------------------------


def analyze_permission_ablation(
    lp_dir: Path,
    direct_dir: Path,
) -> StudyResult:
    """Compare least_privilege vs direct mode on paired tasks."""
    lp = {r["task_id"]: r for r in load_results(lp_dir) if r.get("ok")}
    direct = {r["task_id"]: r for r in load_results(direct_dir) if r.get("ok")}
    common = sorted(set(lp) & set(direct))

    paired: List[Dict[str, Any]] = []
    lp_pass = direct_pass = 0
    for tid in common:
        lp_r = lp[tid]
        d_r = direct[tid]
        lp_both = bool(lp_r.get("outcome_pass") and lp_r.get("process_pass"))
        d_both = bool(d_r.get("outcome_pass") and d_r.get("process_pass"))
        if lp_both:
            lp_pass += 1
        if d_both:
            direct_pass += 1
        paired.append(
            {
                "task_id": tid,
                "domain": lp_r.get("domain"),
                "lp_pass": lp_both,
                "direct_pass": d_both,
                "lp_turns": lp_r.get("turns", 0),
                "direct_turns": d_r.get("turns", 0),
                "difficulty_delta": int(d_both) - int(lp_both),
            }
        )

    n = len(common)
    return StudyResult(
        study="permission_mode_ablation",
        summary={
            "paired_count": n,
            "lp_pass_rate": round(lp_pass / n, 4) if n else 0.0,
            "direct_pass_rate": round(direct_pass / n, 4) if n else 0.0,
            "delta": round((direct_pass - lp_pass) / n, 4) if n else 0.0,
            "lp_harder_count": sum(1 for p in paired if p["difficulty_delta"] > 0),
        },
        tables={"paired_comparison": paired},
    )


# ---------------------------------------------------------------------------
# Study 3: Model Benchmark
# ---------------------------------------------------------------------------


def analyze_model_ranking(
    model_runs: Dict[str, Path],
) -> StudyResult:
    """Rank models by outcome, process, and combined pass rates."""
    rows: List[Dict[str, Any]] = []
    for model_name, run_dir in model_runs.items():
        results = load_results(run_dir)
        ok = [r for r in results if r.get("ok")]
        n = len(results)
        rows.append(
            {
                "model": model_name,
                "n": n,
                "ok_count": len(ok),
                "outcome_pass_rate": _rate(results, "outcome_pass"),
                "process_pass_rate": _rate(results, "process_pass"),
                "both_pass_rate": round(
                    sum(1 for r in results if r.get("outcome_pass") and r.get("process_pass")) / n,
                    4,
                )
                if n
                else 0.0,
                "false_pass_rate": _rate(results, "false_pass_demo"),
                "avg_turns": _safe_mean([r.get("turns", 0) for r in ok]),
                "avg_tool_calls": _safe_mean([r.get("tool_calls", 0) for r in ok]),
            }
        )

    rows.sort(key=lambda r: -r["both_pass_rate"])
    for i, row in enumerate(rows, 1):
        row["rank_both"] = i

    rows_by_outcome = sorted(rows, key=lambda r: -r["outcome_pass_rate"])
    for i, row in enumerate(rows_by_outcome, 1):
        row["rank_outcome"] = i

    return StudyResult(
        study="model_benchmark",
        summary={
            "model_count": len(rows),
            "top_model": rows[0]["model"] if rows else None,
            "top_both_pass_rate": rows[0]["both_pass_rate"] if rows else 0.0,
        },
        tables={"ranking": rows},
    )


# ---------------------------------------------------------------------------
# Study 4: Ranking Divergence
# ---------------------------------------------------------------------------


def analyze_ranking_divergence(
    model_runs: Dict[str, Path],
) -> StudyResult:
    """Show outcome-only vs process-aware rankings differ."""
    benchmark = analyze_model_ranking(model_runs)
    ranking_table = benchmark.tables["ranking"]

    rank_outcome = [
        r["model"] for r in sorted(ranking_table, key=lambda r: r.get("rank_outcome", 0))
    ]
    rank_both = [r["model"] for r in sorted(ranking_table, key=lambda r: r.get("rank_both", 0))]

    tau = _kendall_tau(rank_outcome, rank_both)

    rank_changes: List[Dict[str, Any]] = []
    for row in ranking_table:
        rank_changes.append(
            {
                "model": row["model"],
                "rank_outcome": row.get("rank_outcome"),
                "rank_both": row.get("rank_both"),
                "rank_change": (row.get("rank_outcome", 0) or 0) - (row.get("rank_both", 0) or 0),
            }
        )

    return StudyResult(
        study="ranking_divergence",
        summary={
            "kendall_tau": tau,
            "rankings_identical": tau == 1.0,
            "claim": f"Kendall tau = {tau}. "
            + (
                "Rankings are identical."
                if tau == 1.0
                else "Process-compliance scoring changes the model ranking."
            ),
        },
        tables={
            "rank_comparison": rank_changes,
            "rank_outcome_order": [{"rank": i + 1, "model": m} for i, m in enumerate(rank_outcome)],
            "rank_both_order": [{"rank": i + 1, "model": m} for i, m in enumerate(rank_both)],
        },
    )


# ---------------------------------------------------------------------------
# Study 5: Grader Validity (Codex as gold-standard)
# ---------------------------------------------------------------------------


def analyze_grader_validity(
    run_dir: Path,
    codex_grades_path: Path,
) -> StudyResult:
    """Compare LLM judge grades against Codex gold-standard grades."""
    results = load_results(run_dir)
    codex_grades = json.loads(codex_grades_path.read_text("utf-8"))

    agreements: List[Dict[str, Any]] = []
    llm_labels: List[bool] = []
    codex_labels: List[bool] = []

    per_rule_llm: Dict[str, List[bool]] = defaultdict(list)
    per_rule_codex: Dict[str, List[bool]] = defaultdict(list)

    for r in results:
        tid = r.get("task_id")
        if tid not in codex_grades:
            continue
        cg = codex_grades[tid]
        llm_pp = bool(r.get("process_pass"))
        codex_pp = bool(cg.get("process_pass"))
        llm_labels.append(llm_pp)
        codex_labels.append(codex_pp)
        agreements.append(
            {
                "task_id": tid,
                "llm_process_pass": llm_pp,
                "codex_process_pass": codex_pp,
                "agree": llm_pp == codex_pp,
            }
        )

        for cr in cg.get("rules", []):
            rid = cr.get("rule_id")
            if not rid:
                continue
            per_rule_codex[rid].append(bool(cr.get("passed")))
            matching_llm = next(
                (rf for rf in r.get("rule_failures", []) if rf.get("rule_id") == rid),
                None,
            )
            per_rule_llm[rid].append(not bool(matching_llm))

    n = len(agreements)
    agreement_rate = round(sum(1 for a in agreements if a["agree"]) / n, 4) if n else 0.0
    kappa = _cohens_kappa(llm_labels, codex_labels)

    per_rule_kappa: List[Dict[str, Any]] = []
    for rid in sorted(set(per_rule_llm) & set(per_rule_codex)):
        k = _cohens_kappa(per_rule_llm[rid], per_rule_codex[rid])
        per_rule_kappa.append({"rule_id": rid, "kappa": k, "n": len(per_rule_llm[rid])})

    return StudyResult(
        study="grader_validity",
        summary={
            "n": n,
            "agreement_rate": agreement_rate,
            "cohens_kappa": kappa,
        },
        tables={
            "per_task_agreement": agreements,
            "per_rule_kappa": per_rule_kappa,
        },
    )


# ---------------------------------------------------------------------------
# Study 6: Task Compliance Validity (Codex audit)
# ---------------------------------------------------------------------------


def analyze_task_validity(
    codex_audit_path: Path,
) -> StudyResult:
    """Aggregate Codex-produced task quality audits."""
    audits = json.loads(codex_audit_path.read_text("utf-8"))

    rows: List[Dict[str, Any]] = []
    valid_count = 0
    issue_counter: Counter[str] = Counter()

    for tid, audit in audits.items():
        is_valid = bool(audit.get("valid", True))
        issues = audit.get("issues", [])
        if is_valid:
            valid_count += 1
        for issue in issues:
            issue_counter[str(issue.get("category", "unknown"))] += 1
        rows.append(
            {
                "task_id": tid,
                "valid": is_valid,
                "issue_count": len(issues),
                "issues": issues,
            }
        )

    n = len(rows)
    return StudyResult(
        study="task_compliance_validity",
        summary={
            "n": n,
            "valid_count": valid_count,
            "valid_rate": round(valid_count / n, 4) if n else 0.0,
        },
        tables={
            "per_task": rows,
            "issue_distribution": [
                {"category": cat, "count": cnt} for cat, cnt in issue_counter.most_common()
            ],
        },
    )


# ---------------------------------------------------------------------------
# Study 7: Domain Transfer
# ---------------------------------------------------------------------------


def analyze_domain_transfer(
    model_runs: Dict[str, Path],
) -> StudyResult:
    """Cross-tabulate model × domain performance."""
    matrix: List[Dict[str, Any]] = []
    all_domains: set[str] = set()

    for model_name, run_dir in model_runs.items():
        results = load_results(run_dir)
        by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            by_domain[str(r.get("domain", "unknown"))].append(r)
        all_domains.update(by_domain)
        for domain, items in sorted(by_domain.items()):
            n = len(items)
            matrix.append(
                {
                    "model": model_name,
                    "domain": domain,
                    "n": n,
                    "outcome_pass_rate": _rate(items, "outcome_pass"),
                    "process_pass_rate": _rate(items, "process_pass"),
                    "both_pass_rate": round(
                        sum(1 for r in items if r.get("outcome_pass") and r.get("process_pass"))
                        / n,
                        4,
                    )
                    if n
                    else 0.0,
                }
            )

    return StudyResult(
        study="domain_transfer",
        summary={"domains": sorted(all_domains), "models": sorted(model_runs.keys())},
        tables={"matrix": matrix},
    )


# ---------------------------------------------------------------------------
# Study 8: Benchmark Robustness
# ---------------------------------------------------------------------------


def analyze_robustness(
    repeat_dirs: List[Path],
) -> StudyResult:
    """Measure cross-run variance for the same (model, tasks) configuration."""
    per_task: Dict[str, List[bool]] = defaultdict(list)
    run_rates: List[float] = []

    for d in repeat_dirs:
        results = load_results(d)
        n = len(results)
        both = sum(1 for r in results if r.get("outcome_pass") and r.get("process_pass"))
        run_rates.append(round(both / n, 4) if n else 0.0)
        for r in results:
            tid = r.get("task_id", "")
            per_task[tid].append(bool(r.get("outcome_pass") and r.get("process_pass")))

    flip_rate_per_task: List[Dict[str, Any]] = []
    for tid, verdicts in sorted(per_task.items()):
        if len(verdicts) < 2:
            continue
        flip = 1 - (max(sum(verdicts), len(verdicts) - sum(verdicts)) / len(verdicts))
        flip_rate_per_task.append(
            {"task_id": tid, "flip_rate": round(flip, 4), "runs": len(verdicts)}
        )

    cv = (
        round(_safe_stdev(run_rates) / _safe_mean(run_rates), 4)
        if _safe_mean(run_rates) > 0
        else float("inf")
    )

    return StudyResult(
        study="benchmark_robustness",
        summary={
            "repeat_count": len(repeat_dirs),
            "mean_both_pass_rate": _safe_mean(run_rates),
            "stdev_both_pass_rate": _safe_stdev(run_rates),
            "cv": cv,
            "high_variance_tasks": sum(1 for t in flip_rate_per_task if t["flip_rate"] > 0.3),
        },
        tables={
            "per_run_rates": [
                {"run": str(d.name), "both_pass_rate": r} for d, r in zip(repeat_dirs, run_rates)
            ],
            "per_task_flip_rate": flip_rate_per_task,
        },
    )


# ---------------------------------------------------------------------------
# Study 9: Tool/Turn Budget
# ---------------------------------------------------------------------------


def analyze_budget(
    run_dirs: List[Path],
) -> StudyResult:
    """Analyse tool-call and user-interaction budget consumption."""
    all_results: List[Dict[str, Any]] = []
    for d in run_dirs:
        all_results.extend(load_results(d))

    ok = [r for r in all_results if r.get("ok")]
    passing = [r for r in ok if r.get("outcome_pass") and r.get("process_pass")]
    failing = [r for r in ok if not (r.get("outcome_pass") and r.get("process_pass"))]

    def _dist(items: List[Dict[str, Any]], key: str) -> Dict[str, float]:
        vals = [r.get(key, 0) for r in items]
        if not vals:
            return {"mean": 0, "median": 0, "p90": 0, "max": 0}
        sorted_vals = sorted(vals)
        p90_idx = int(len(sorted_vals) * 0.9)
        return {
            "mean": _safe_mean(vals),
            "median": sorted_vals[len(sorted_vals) // 2],
            "p90": sorted_vals[min(p90_idx, len(sorted_vals) - 1)],
            "max": max(vals),
        }

    budget_exhaustion = sum(
        1
        for r in ok
        if r.get("interaction_budget_reached") or r.get("tool_budget_reached_count", 0) > 0
    )

    by_domain: Dict[str, Dict[str, Any]] = {}
    for r in ok:
        domain = str(r.get("domain", "unknown"))
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(r)
    domain_stats = {
        domain: {
            "n": len(items),
            "tool_calls": _dist(items, "tool_calls"),
            "turns": _dist(items, "turns"),
        }
        for domain, items in sorted(by_domain.items())
    }

    return StudyResult(
        study="tool_turn_budget",
        summary={
            "total": len(ok),
            "budget_exhaustion_count": budget_exhaustion,
            "budget_exhaustion_rate": round(budget_exhaustion / len(ok), 4) if ok else 0.0,
        },
        tables={
            "overall": [
                {"group": "all", **_dist(ok, "tool_calls"), "metric": "tool_calls"},
                {"group": "all", **_dist(ok, "turns"), "metric": "turns"},
                {"group": "pass", **_dist(passing, "tool_calls"), "metric": "tool_calls"},
                {"group": "pass", **_dist(passing, "turns"), "metric": "turns"},
                {"group": "fail", **_dist(failing, "tool_calls"), "metric": "tool_calls"},
                {"group": "fail", **_dist(failing, "turns"), "metric": "turns"},
            ],
        },
        details={"by_domain": domain_stats},
    )


# ---------------------------------------------------------------------------
# Study 10: LLM Grader Ablation
# ---------------------------------------------------------------------------


def analyze_grader_ablation(
    baseline_dir: Path,
    ablated_grades: Dict[str, Path],
) -> StudyResult:
    """Compare process_pass rate when rules are removed from grading."""
    baseline = load_results(baseline_dir)
    baseline_by_id = {r["task_id"]: r for r in baseline if r.get("ok")}
    baseline_rate = _rate(list(baseline_by_id.values()), "process_pass")

    ablation_rows: List[Dict[str, Any]] = []
    for ablation_name, grade_path in ablated_grades.items():
        ablated = json.loads(grade_path.read_text("utf-8"))
        ablated_pass = sum(1 for g in ablated.values() if g.get("process_pass"))
        n = len(ablated)
        ablated_rate = round(ablated_pass / n, 4) if n else 0.0
        ablation_rows.append(
            {
                "ablation": ablation_name,
                "n": n,
                "process_pass_rate": ablated_rate,
                "baseline_rate": baseline_rate,
                "delta": round(ablated_rate - baseline_rate, 4),
            }
        )

    return StudyResult(
        study="grader_ablation",
        summary={"baseline_rate": baseline_rate, "ablation_count": len(ablation_rows)},
        tables={"ablations": ablation_rows},
    )


# ---------------------------------------------------------------------------
# Study 11-13: Compliance Process Verification
# ---------------------------------------------------------------------------

COMPLIANCE_GROUPS = {
    "requirement_verification": {
        "description": "确认用户需求后再提供建议",
        "rules": [
            "session_context_before_read",
            "read_before_check",
            "check_before_draft",
        ],
    },
    "misleading_promise": {
        "description": "误导性胡乱承诺的检查",
        "rules": [
            "no_coverage_promise",
            "no_insurance_outcome_promise",
            "no_benefits_outcome_promise",
            "no_promise_disclosure_before_confirmation",
            "no_agent_direct_submit",
        ],
    },
    "incomplete_disclosure": {
        "description": "披露不完全",
        "rules": [
            "disclosure_before_confirmation",
            "rights_before_confirmation",
            "fair_hearing_rights_before_confirmation",
            "disconnection_rights_before_confirmation",
        ],
    },
}


def analyze_compliance_group(
    run_dirs: List[Path],
    group_name: str,
) -> StudyResult:
    """Analyse pass/fail rates for a specific compliance rule group."""
    group = COMPLIANCE_GROUPS.get(group_name)
    if not group:
        raise ValueError(f"unknown compliance group: {group_name}")

    target_rules = set(group["rules"])
    all_results: List[Dict[str, Any]] = []
    for d in run_dirs:
        all_results.extend(load_results(d))

    ok = [r for r in all_results if r.get("ok")]

    per_rule: Dict[str, Dict[str, int]] = {
        rule: {"pass": 0, "fail": 0, "not_applicable": 0} for rule in target_rules
    }
    per_task: List[Dict[str, Any]] = []

    for r in ok:
        task_failures = {str(rf.get("rule_id")) for rf in r.get("rule_failures", [])}
        task_rule_status: Dict[str, str] = {}
        for rule in target_rules:
            if rule in task_failures:
                per_rule[rule]["fail"] += 1
                task_rule_status[rule] = "fail"
            else:
                per_rule[rule]["pass"] += 1
                task_rule_status[rule] = "pass"
        per_task.append(
            {
                "task_id": r.get("task_id"),
                "domain": r.get("domain"),
                **task_rule_status,
            }
        )

    rule_rates: List[Dict[str, Any]] = []
    for rule in sorted(target_rules):
        stats = per_rule[rule]
        total = stats["pass"] + stats["fail"]
        rule_rates.append(
            {
                "rule_id": rule,
                "pass_count": stats["pass"],
                "fail_count": stats["fail"],
                "pass_rate": round(stats["pass"] / total, 4) if total else 0.0,
            }
        )

    group_fail_count = sum(
        1
        for r in ok
        if any(str(rf.get("rule_id")) in target_rules for rf in r.get("rule_failures", []))
    )

    return StudyResult(
        study=f"compliance_{group_name}",
        summary={
            "group": group_name,
            "description": group["description"],
            "n": len(ok),
            "group_fail_count": group_fail_count,
            "group_fail_rate": round(group_fail_count / len(ok), 4) if ok else 0.0,
        },
        tables={
            "per_rule_rates": rule_rates,
            "per_task": per_task,
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ANALYSIS_REGISTRY = {
    "false_pass": analyze_false_pass,
    "permission_mode_ablation": analyze_permission_ablation,
    "model_benchmark": analyze_model_ranking,
    "ranking_divergence": analyze_ranking_divergence,
    "grader_validity": analyze_grader_validity,
    "task_compliance_validity": analyze_task_validity,
    "domain_transfer": analyze_domain_transfer,
    "benchmark_robustness": analyze_robustness,
    "tool_turn_budget": analyze_budget,
    "grader_ablation": analyze_grader_ablation,
    "compliance_requirement_verification": lambda dirs: analyze_compliance_group(
        dirs, "requirement_verification"
    ),
    "compliance_misleading_promise": lambda dirs: analyze_compliance_group(
        dirs, "misleading_promise"
    ),
    "compliance_incomplete_disclosure": lambda dirs: analyze_compliance_group(
        dirs, "incomplete_disclosure"
    ),
}
