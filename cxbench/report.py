from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def write_report(run_dir: Path, grade: Dict[str, Any]) -> Path:
    lines = [
        f"# cxbench run report: {grade['task_id']}",
        "",
        f"- Mode: `{grade['mode']}`",
        f"- Grader: `{grade.get('grader', 'rule')}`",
        f"- Outcome pass: `{grade['outcome_pass']}`",
        f"- Process pass: `{grade['process_pass']}`",
        f"- Verdict: `{grade['verdict']}`",
        f"- False-pass demo: `{grade['false_pass_demo']}`",
        "",
        "## Rule Results",
        "",
    ]
    for rule in grade["rules"]:
        mark = "PASS" if rule["passed"] else "FAIL"
        seq = f" seq={rule['seq']}" if rule.get("seq") is not None else ""
        lines.append(f"- `{mark}` `{rule['rule_id']}`{seq}: {rule['evidence']}")
    if "llm_judge" in grade:
        lines.extend(["", "## LLM Judge", ""])
        judge = grade["llm_judge"]
        lines.append(f"- Process pass: `{judge['process_pass']}`")
        provider_meta = judge.get("provider_meta") or {}
        if provider_meta:
            usage = provider_meta.get("usage") or {}
            lines.append(
                "- Provider: `{provider}` `{wire_api}` model=`{model}` latency_ms=`{latency}` cost_usd=`{cost}`".format(
                    provider=provider_meta.get("provider"),
                    wire_api=provider_meta.get("wire_api"),
                    model=provider_meta.get("model_returned"),
                    latency=provider_meta.get("latency_ms"),
                    cost=provider_meta.get("cost_usd"),
                )
            )
            if usage:
                lines.append(
                    "- Usage: input_tokens=`{input}` output_tokens=`{output}` total_tokens=`{total}`".format(
                        input=usage.get("input_tokens") or usage.get("prompt_tokens"),
                        output=usage.get("output_tokens") or usage.get("completion_tokens"),
                        total=usage.get("total_tokens"),
                    )
                )
        if judge.get("rationale"):
            lines.append(f"- Rationale: {judge['rationale']}")
        for finding in judge.get("findings", []):
            mark = "PASS" if finding["passed"] else "FAIL"
            seq = f" seq={finding['seq']}" if finding.get("seq") is not None else ""
            lines.append(
                f"- `{mark}` `{finding['rule_id']}` severity={finding['severity']}{seq}: {finding['evidence']}"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `trace.jsonl`: ordered runtime events",
            "- `final_state.json`: final in-memory domain state",
            "- `drafts/`: draft artifacts created by draft tools",
            "- `grade.json`: deterministic grader output",
        ]
    )
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
