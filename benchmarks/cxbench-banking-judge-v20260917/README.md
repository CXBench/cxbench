# Revised judge tasks: six-domain extension

This directory contains 120 tasks (20 per domain) derived from
[`cxbench-120-v20260708`](../cxbench-120-v20260708/README.md). It is a separate
semantic-rubric variant. The historical directory name is retained. The original
20 banking files are unchanged; the September 18 extension adds telecom,
insurance, healthcare, utilities, and government-benefits tasks.

## Scope of the revision

Only `policy.llm_judge` differs from the matching original task. Task IDs,
business facts, tools, deterministic rules, outcome definitions, confirmation
contracts, and the rest of the policy remain unchanged.

The banking rubric version is `banking-llm-judge-v20260917`; the added tasks use
`cross-domain-llm-judge-v20260918`. The fixed judge system prompt is
`trace-adjudication-v20260918`. The rubrics specify:

- One finding per declared checklist ID, with explicit applicability.
- Trace sequence numbers, event actors, and observable evidence.
- Separate treatment of requested arguments, saved fields, customer-visible
  messages, confirmations, and actual submissions.
- Permitted masked or descriptive representations when presenting a draft.
- Correction of unconfirmed drafts without an automatic first-attempt penalty.
- Explicit terminal-result requirements and task-specific disclosure checks.

The new checklists preserve all original checklist IDs and source provenance.
Insurance's legacy `high` and `medium` severities are normalized to `major`;
any failed item still fails semantic grading. Cross-domain adjudication methods
live in the fixed system prompt; objects, tool names, obligations, and branch
conditions remain in each task. The prompt change also applies when evaluating
the original release, so historical scores must identify the prompt version.

A semantic pass still requires every finding to pass. Official evaluation
continues to use `hybrid`, combining deterministic hard rules and the LLM judge.

## Reproduce

Follow the root [installation and model setup](../../README.md), then run:

```bash
uv run --frozen python -m main_experiment.run \
  --model "$AGENT_MODEL" \
  --user-model "$USER_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --task 'benchmarks/cxbench-banking-judge-v20260917/tasks/*.yaml' \
  --mode least_privilege --grader hybrid \
  --executor process --concurrency 4 \
  --run-id six-domain-judge-revised
```

Add `--dry-run` to validate without API calls or run artifacts. The public
experiment runners default to this revised 120-task release. For the banking-only variant,
use `tasks/BK-*.yaml` instead of `tasks/*.yaml`. Do not combine revised and
original task glob patterns: their task IDs intentionally overlap.

Report this variant's release, per-domain rubric versions, system prompt version,
and agent, user, and judge models. Scores from different rubrics are separate
evaluations.

## Validation

Run `uv run --frozen pytest tests/test_judge_variant.py tests/test_mvp.py` for
schema, provenance, confirmation-field coverage, and grader regression checks.
These checks run offline. Rule-only replay checks deterministic and outcome
invariance; it is not a formal score. See the public
[validation record](../../docs/validation.md) for checks on this source release.

## Provenance

`release_manifest.json` records current SHA-256 hashes and the corresponding
source-task hashes from the original release. The source comparison treats
`source_spec` and rule-card identifiers inside each task as provenance;
the revised inline checklist is the operative LLM rubric.
