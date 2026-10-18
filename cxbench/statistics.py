"""Offline hybrid-score aggregation and paired statistical comparisons."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from math import comb


def categorize(rule_id: str) -> str:
    if rule_id in ("draft_before_confirmation", "confirmation_before_submit"):
        return "pipeline"
    if "promise" in rule_id or "threat" in rule_id or "secret" in rule_id:
        return "semantic"
    if "disclosure" in rule_id or "rights_before" in rule_id:
        return "disclosure"
    return "ordering"


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 2), round(centre + half, 2)]


def mcnemar_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def summarize_records(
    records: dict[str, dict[str, dict]],
    judge_pass: dict[tuple[str, str], bool],
    reference_model: str | None = None,
) -> dict:
    """Aggregate the paper metrics; callers provide complete, paired task evidence."""
    models = list(records)
    if not models or any(not records[model] for model in models):
        raise ValueError("paper statistics require nonempty model records")
    if reference_model is not None and reference_model not in records:
        raise ValueError(f"unknown reference model: {reference_model}")
    compared = [model for model in models if model != reference_model]

    def both(model: str, task_id: str, excluded: str | None = None) -> bool:
        row = records[model][task_id]
        soft = set(row["soft_rules"])
        hard_pass = all(p for rid, p in row["rules"].items() if rid not in soft and rid != excluded)
        return row["outcome_pass"] and hard_pass and judge_pass[(model, task_id)]

    per_model = {}
    for model in models:
        n = len(records[model])
        bp = sum(1 for task_id in records[model] if both(model, task_id))
        op = sum(1 for row in records[model].values() if row["outcome_pass"])
        per_model[model] = {
            "completed": n,
            "outcome_pass": op,
            "both_pass": bp,
            "false_pass": op - bp,
            "rate": round(bp / n, 3),
            "out_rate": round(op / n, 3),
            "ci95": wilson_ci(bp, n),
        }

    hard_fail_counts: Counter = Counter()
    sole_counts: Counter = Counter()
    hard_rule_ids: set[str] = set()
    for rows in records.values():
        for row in rows.values():
            soft = set(row["soft_rules"])
            hard = {rid: p for rid, p in row["rules"].items() if rid not in soft}
            hard_rule_ids.update(hard)
            fails = [rid for rid, p in hard.items() if not p]
            hard_fail_counts.update(fails)
            if len(fails) == 1:
                sole_counts[fails[0]] += 1
    cat_stats = defaultdict(lambda: {"rules": set(), "active": set(), "fails": 0, "sole": 0})
    for rid in hard_rule_ids:
        cat_stats[categorize(rid)]["rules"].add(rid)
    for rid, count in hard_fail_counts.items():
        category = cat_stats[categorize(rid)]
        category["active"].add(rid)
        category["fails"] += count
    for rid, count in sole_counts.items():
        cat_stats[categorize(rid)]["sole"] += count
    taxonomy = {
        category: {
            "rules": len(value["rules"]),
            "active": len(value["active"]),
            "fails": value["fails"],
            "sole": value["sole"],
        }
        for category, value in sorted(cat_stats.items())
    }

    active = sorted(hard_fail_counts)

    def comparison_rates(excluded=None):
        return {
            model: sum(1 for task_id in records[model] if both(model, task_id, excluded))
            / per_model[model]["completed"]
            for model in compared
        }

    base = comparison_rates()
    base_order = sorted(compared, key=lambda model: -base[model])
    loo_sensitive, loo_max = 0, 0.0
    for rid in active:
        rates = comparison_rates(excluded=rid)
        if sorted(compared, key=lambda model: -rates[model]) != base_order:
            loo_sensitive += 1
        loo_max = max(
            loo_max, max((abs(rates[model] - base[model]) for model in compared), default=0.0) * 100
        )

    all_tasks = sorted({task_id for rows in records.values() for task_id in rows})

    def verdict(model, task_id):
        return task_id in records[model] and both(model, task_id)

    def discordant(a, b):
        n10 = sum(1 for task_id in all_tasks if verdict(a, task_id) and not verdict(b, task_id))
        n01 = sum(1 for task_id in all_tasks if verdict(b, task_id) and not verdict(a, task_id))
        return n10, n01

    pair_table = []
    for a, b in combinations(compared, 2):
        n10, n01 = discordant(a, b)
        pair_table.append(
            {"pair": f"{a} vs {b}", "n10": n10, "n01": n01, "p": round(mcnemar_p(n01, n10), 4)}
        )
    frontier_sig = sum(mcnemar_p(row["n01"], row["n10"]) < 0.05 for row in pair_table)
    tier_max_p = (
        max((mcnemar_p(*discordant(reference_model, model)) for model in compared), default=None)
        if reference_model
        else None
    )
    total_fp = sum(value["false_pass"] for value in per_model.values())
    total_n = sum(value["completed"] for value in per_model.values())
    frontier_fp = sum(per_model[model]["false_pass"] for model in compared)
    frontier_n = sum(per_model[model]["completed"] for model in compared)
    rates_sorted = sorted(per_model[model]["rate"] for model in compared)
    reference = per_model.get(reference_model)
    return {
        "instrument": "outcome=current grader (offline), rules=current hard rules (offline), judge=recorded findings consistency guard",
        "per_model": per_model,
        "false_pass": {
            "total": total_fp,
            "completed": total_n,
            "rate_pct": round(100 * total_fp / total_n, 1),
            "frontier_pct": round(100 * frontier_fp / frontier_n, 1) if frontier_n else None,
            "weak_pct": round(100 * reference["false_pass"] / reference["completed"], 1)
            if reference
            else None,
        },
        "frontier": {
            "lo": rates_sorted[0] if rates_sorted else None,
            "hi": rates_sorted[-1] if rates_sorted else None,
            "spread_pp": round((rates_sorted[-1] - rates_sorted[0]) * 100, 1)
            if rates_sorted
            else None,
            "tier_gap_pp": round((rates_sorted[0] - reference["rate"]) * 100, 1)
            if reference and rates_sorted
            else None,
        },
        "taxonomy": taxonomy,
        "n_hard_rules": len(hard_rule_ids),
        "n_active_hard_rules": len(active),
        "fp_rule_a": {
            "rule_id": "policy_summary_before_authority_check",
            "fails": hard_fail_counts.get("policy_summary_before_authority_check", 0),
        },
        "loo": {"active": len(active), "sensitive": loo_sensitive, "max_pp": round(loo_max, 1)},
        "mcnemar": {
            "frontier_significant": frontier_sig,
            "pairs": pair_table,
            "tier_max_p": f"{tier_max_p:.1e}" if tier_max_p is not None else None,
        },
    }
