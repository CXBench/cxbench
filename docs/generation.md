# Data synthesis and acceptance

## Pipeline

The pipeline takes a policy corpus or curated rule registry, a domain contract,
and a scenario matrix. It selects rule cards, emits a tool/interface manifest,
constructs candidate YAML tasks and JSON grading specifications, merges semantic
checklists into task policies, and validates the resulting candidates. An
optional LLM quality review checks source, state, tool, and outcome alignment.

The bundled corpus is a versioned snapshot. No network download is required to
evaluate released tasks or run local synthesis. The extractor uses deterministic
section splitting and compliance-keyword selection (up to 50 selected cards);
it is not a legal interpretation model. Review the selected evidence.

## Offline smoke fixture

```bash
uv run --frozen cxbench generate \
  --config configs/benchmark_generation/smoke.yaml \
  --executor local --out generated/local-smoke
uv run --frozen cxbench accept \
  --run generated/local-smoke --dest generated/accepted
```

The fixture embeds a complete banking task derived from `tasks/CX-01.us.yaml`.
It exercises extraction, grading-spec merging, validation, and acceptance with
no model calls. Its task is also executed with a scripted agent and deterministic
user in `tests/test_release_workflows.py`. Rule-only grading there is a diagnostic
integration check, not a formal benchmark score.

The local executor requires each selected scenario to contain a complete `task`
mapping. A matrix containing scenario hints alone is rejected rather than turned
into a task with nonexistent tools. The local fixture does not demonstrate novel
LLM data synthesis or reproduce the released 120-task generation history.

## Generate new candidates

Install and authenticate the standard [Codex CLI](https://github.com/openai/codex).
Verify `codex exec --help` works in your shell. Its authentication and model
configuration are separate from CXBench's `.env` provider profiles.

```bash
uv run --frozen cxbench generate \
  --config configs/benchmark_generation/banking.yaml \
  --executor codex --workers 1 --out generated/banking-new \
  --review-model openai:gpt-5.6-sol
```

This launches `codex exec` in per-worker directories with workspace-write
sandboxing. Prompts and inputs are kept under the generation output. `--workers`
controls scenario shards; the current implementation invokes workers
sequentially. Prompts restrict reading to worker inputs and writing to outputs;
they are instructions, not an additional OS-level read isolation boundary.

The Codex CLI produces tasks and grading specs. `--review-model` calls the
CXBench compatible client for an independent quality review; the example uses
OpenAI credentials. This review concerns candidate quality, not the evaluated
agent's performance. Inspect every candidate and its evidence before release.

Each run needs a fresh or empty `--out` directory. Existing nonempty runs are
preserved. Use `--reuse-rule-selection PREVIOUS/selected_rule_cards.jsonl` to
reuse a fixed selection in a new run. Use `--rule-registry registry.yaml` to seed
generation from curated, versioned rule IDs rather than keyword extraction.

## Configuration and task contract

Configuration paths are resolved relative to the configuration file:

```yaml
domain: banking
business_system: banking_customer_ops_v0
corpus_manifest: ../../data/corpus_manifests/lii_markdown.yaml
domain_rd: ../../docs/rd/banking-customer-ops-rd.md
scenario_matrix: ../../data/scenario_matrices/banking_reg_e_mvp.yaml
target_count: 3
user_sim_profiles: [llm, verify]
```

`target_count` must be positive and must not exceed the scenario count. A corpus
manifest contains `chunks` with `id` and `path`; chunk paths are relative to that
manifest. `rule_registry` may replace `corpus_manifest`. Other domain scenario
matrices are provided under `data/scenario_matrices/`; create a configuration
with the corresponding business system and domain contract before using them.

A task must provide runnable initial state, valid domain tools, reference facts,
confirmation fields, process rules, policy evidence, and outcome semantics. Keep
read/check/draft/notice tools separate from submission gateways. Prefer semantic
checklists for disclosures whose wording need not match an exact string.
The generated `worker_inputs/interface_manifest.json` lists the supported
interfaces. Its tool validation rejects invented tools. Domain contracts in
`docs/rd/` describe the original system design; actual runtime behavior and tests
take precedence when those design notes are less complete than the implementation.

`12 CFR 1005.17` is the overdraft opt-in source; written-confirmation requirements
for error resolution belong to `12 CFR 1005.11`. Source IDs should preserve this
distinction when authoring matrices and rule registries.

## Review, repair, and acceptance

Inspect `validation.json`, `selection_manifest.json`, `selected_rule_cards.jsonl`,
`grading_specs/`, and `quality_reviews/` when present. Validation can reconcile
some hidden facts with initial state and reports those changes. Schema validation
does not establish that every possible task branch is executable or legally
correct; replay candidates and review semantic requirements.

Repair a reviewed run with quality-gate feedback:

```bash
uv run --frozen cxbench repair --run generated/banking-new \
  --review-model openai:gpt-5.6-sol --rounds 1
```

Repair invokes the same optional Codex CLI and then revalidates/reviews outputs.
If only static validation was run, acceptance does not add an LLM review. If a
quality review was requested and failed, those candidates are not eligible.
Acceptance checks each candidate's validation-time hash; edited candidates need
fresh validation and review. Existing destination files are preserved unless
`--force` is explicitly supplied. Use a new destination for a new task release.

Original candidate-directory paths and rule-card/spec IDs retained in released
tasks are provenance labels. The inline policy and checklist are operative at
runtime; historical generation directories are not needed to execute the tasks.

## Corpus maintenance

The existing URL manifest is included. To refresh it deliberately, edit the
scoped manifest, then download and convert:

```bash
uv run --frozen python scripts/download_lii_corpus.py --delay 10
uv run --frozen python scripts/convert_lii_html_to_markdown.py \
  --output data/raw/lii_markdown --write-manifest
```

This is optional maintenance, not an installation step. Respect current source
terms and rate limits. Review the converted legal text, manifests, and
`domain_index.json`, then rerun domain-policy and task tests. Updating the corpus
changes model context and requires a new identified experiment configuration.
See [NOTICE.md](../NOTICE.md) for source attribution.
