# cxbench 120-task release v20260708

This release contains the accepted 120-task cxbench corpus across six regulated customer-experience domains.

## Contents

| Item | Value |
|---|---|
| Release ID | `cxbench-120-v20260708` |
| Release date | 2026-07-08 |
| Source candidates | `generated/benchmark_candidates/codex-6domain-20260625-v2` |
| Task directory | `tasks/` |
| Task count | 120 |
| Domains | 6 |

## Domain Counts

| Domain | Tasks |
|---|---:|
| `banking` | 20 |
| `government_benefits` | 20 |
| `healthcare` | 20 |
| `insurance` | 20 |
| `telecom` | 20 |
| `utilities` | 20 |

## Validation

The task YAML files were accepted through `uv run cxbench accept` from the six validated candidate runs. Each source domain run reported 20 valid candidates before acceptance.

The source candidate directory is a historical provenance label and is not
included or needed at runtime. For current execution and grading conventions,
see the [experiment protocol](../../docs/experiments.md). This is the original
task release; the public runners default to the revised semantic-rubric release.

## Files

- `tasks/*.yaml`: accepted task YAML corpus
- `release_manifest.json`: release metadata, domain counts, task IDs, and file hashes
