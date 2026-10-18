# Provenance and release scope

## Source lineage

| Source | Identifier | Role |
| --- | --- | --- |
| Public upstream | `PacificStudio/cxbench`, `04891200f5e8e2ac41f0403645190a14989454df` | Original project lineage and synthesis reference |
| Updated experiment snapshot | Base `96cc4273b76bc9bb025f9b7fe9bb9e8a05c30141` plus local changes | Runtime, updated experimental settings, retries, revised semantic grading |
| Consolidated source snapshot | Local integration prepared before this release | Immediate source for the clean export |

The second row identifies a working-tree snapshot, not a claim that all its
content exists in the named commit. Source fingerprint inventories are kept in
the local verification artifacts. Personal directory paths and private results
are intentionally excluded from the public tree.

The task manifest retains byte-level provenance for the single distributed
120-task corpus in `benchmarks/cxbench-120/`. Its task YAML is unchanged from the
updated experiment snapshot. Historical source hashes, business-contract hashes,
and checklist provenance are retained as metadata without a second task folder.
Corpus snapshots retain their source metadata.

## Preserved experimental behavior

This export retains the updated runtime, domain tools, confirmation/submission
mechanics, task budgets, judge prompt and checklists, provider retry policy,
unlimited execution retries, same-trace judge retries, recovery checks, and
hybrid scoring. The public API entry points use standard OpenAI configuration.

The README uses outcome/process and trace-evidence terminology consistent with
the accompanying research design. Private manuscript drafts and numerical paper
tables are not part of this export. No conference acceptance claim is implied.

## Release-maintenance changes

- Default public experiment selection is the revised six-domain task release.
- The OpenAI wrapper supports smoke resume and external results directories, and
  leaves custom dotenv base URLs to the provider loader.
- Reusable statistical aggregation lives in `cxbench/statistics.py`; dated
  scripts tied to private historical result directories are omitted.
- A local synthesis fixture exercises a real task; hint-only matrices are
  rejected by the local executor. Runnable domain examples replace placeholder
  tools in generation prompts.
- Generation uses the standard Codex CLI instead of a private delegation skill.
  Existing nonempty output directories are protected, unsupported tools are
  rejected, and candidate hashes are checked at acceptance.
- The corpus manifest correctly identifies 12 CFR 1005.17 as overdraft opt-in.
  No released task or legal snapshot text is rewritten for this correction.
- Source packaging, documentation, formatting, offline CI, and regression
  coverage are prepared for external users.

See [CHANGELOG.md](../CHANGELOG.md) and [validation.md](validation.md). These
maintenance changes do not claim to reproduce a complete historical study.
