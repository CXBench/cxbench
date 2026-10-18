# cxbench Benchmark Releases

This directory contains accepted benchmark corpus releases.

## Releases

| Release | Tasks | Domains | Date | Notes |
|---|---:|---:|---|---|
| [`cxbench-120-v20260708`](./cxbench-120-v20260708/) | 120 | 6 | 2026-07-08 | Accepted 20 tasks per domain from `generated/benchmark_candidates/codex-6domain-20260625-v2`. |

See the [experiment protocol](../docs/experiments.md) for current grading and reporting conventions.

## Rubric variants

| Variant | Tasks | Scope |
| --- | ---: | --- |
| [`cxbench-banking-judge-v20260917`](./cxbench-banking-judge-v20260917/) | 120 | Default: revised six-domain LLM checklists; only `policy.llm_judge` changes. Original task IDs are reused. |

The revised directory retains its historical banking name but covers all six
domains. Each directory contains task YAML and a hash manifest. Select one
release at a time: task IDs intentionally overlap.
