# CXBench: 120 tasks across six domains

This is the single task release distributed with the code. It contains 20 tasks
per domain: banking, telecom, insurance, healthcare, utilities, and government
benefits. The task YAML retains the updated experimental settings and semantic
checklists. Consolidating the directory does not change task contents.

## Versions and integrity

- Corpus directory: `benchmarks/cxbench-120/`.
- Manifest release ID: `cxbench-120-v20260918`.
- Banking rubric: `banking-llm-judge-v20260917`.
- Other domain rubrics: `cross-domain-llm-judge-v20260918`.
- Fixed judge prompt: `trace-adjudication-v20260918`.
- Formal grading: `hybrid`.

`release_manifest.json` records each task's SHA-256 hash, identity, title, and
rubric version. Historical source hashes, checklist IDs, and normalized hashes
of business contracts and judge provenance allow regression checks without
shipping the older corpus. Historical source identifiers are metadata, not
paths that the runtime needs to resolve.

## Semantic checklists

The rubrics require a finding for each declared checklist ID, with applicability
and trace evidence. They distinguish requested arguments, saved draft fields,
customer-visible disclosures, confirmation, and actual submission. Task-specific
checks define permitted representations and terminal-result requirements.
A semantic pass requires every applicable finding to pass. Formal evaluation
combines that result with deterministic hard rules.

## Run

Follow the root [installation and API setup](../../README.md), then use:

```bash
./run.sh --dry-run
./run.sh --smoke-test --run-id openai-smoke
./run.sh --run-id openai-120
```

To select one domain with the general runner:

```bash
uv run --frozen python -m main_experiment.run \
  --model gpt-5.6-sol --agent-provider openai \
  --user-model openai:gpt-5.6-sol \
  --judge-model openai:gpt-5.6-sol --judge-provider openai \
  --task 'benchmarks/cxbench-120/tasks/BK-*.yaml' \
  --wire-api responses --mode least_privilege --grader hybrid \
  --executor process --concurrency 1 --run-id banking
```

Add `--dry-run` for schema/options validation without API calls or run artifacts.
Report the corpus, rubric, and system-prompt versions with the agent, user, and
judge model identifiers. Keep derived tasks separate from this release.

## Validation

`tests/test_judge_variant.py` verifies all task hashes, business-contract hashes,
checklist provenance, confirmation-field coverage, and judge prompt invariants.
See the [validation record](../../docs/validation.md) for the scope of tested
integration paths. Diagnostic rule-only tests are not formal compliance scores.
