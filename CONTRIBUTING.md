# Contributing

## Setup and checks

```bash
uv sync --frozen --python 3.12
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
./run.sh --dry-run
uv build
```

Keep changes focused and add regression tests for behavior changes. Tests use
mock providers or deterministic fixtures and must run without credentials.
Separate live experiments from offline CI. Describe the trigger, changed
behavior, validation, and any change in experimental comparability in a PR.

## Experimental integrity

Released task YAML and manifests form an identified benchmark version. Preserve
them when improving infrastructure. A changed task, policy source, judge rubric,
prompt, or grading rule needs an explicit provenance record and, when appropriate,
a new release identifier. Do not rewrite task hashes merely to hide a change.

Keep valid failing verdicts. Retry execution or provider failures according to
the documented protocol; do not retry a completed experiment until it passes.
Formal reports require hybrid grading and complete paired task sets. Identify
rule-only or LLM-only output as diagnostic, including tests and ablations.

## Runtime and prompts

Maintain clear boundaries between agent-visible tools, runtime submissions,
customer confirmation, hidden reference facts, and grading. Preserve complete
domain policy context and stable prompt prefixes. New domain tools need schema,
runtime, outcome, and grader tests that exercise meaningful failure conditions.

## Publication hygiene

Commit source, tests, versioned task manifests, configuration templates, and
documentation. Keep credentials, environments, caches, generated candidates,
raw run logs, and private results out of source distributions. `.gitignore`
helps with Git but does not sanitize a folder or archive; inspect exported
contents before sharing. Never paste keys into issues or logs.

Confirm licensing with the project owners before distributing contributions.
Third-party policy sources retain their own terms; see [NOTICE.md](NOTICE.md).
