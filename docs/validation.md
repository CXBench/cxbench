# Release validation

The local validation environment uses macOS and CPython 3.12.14 outside the
publishable folder. The source suite checks runtime behavior, schema integrity,
provider request construction, retry/recovery, generation, and scoring without
credentials. This record distinguishes integration checks from benchmark results.

## Current distribution

The canonical corpus is `benchmarks/cxbench-120/`: 120 tasks across six domains.
Task YAML bytes match the updated experimental snapshot. Integrity tests verify
current task hashes and historical business/checklist provenance from the
manifest, without requiring another corpus directory.

The API interface uses `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and the official
OpenAI endpoint by default. Mocked transport tests cover Responses streaming and
non-streaming requests, reasoning settings, function-call lifecycle, Chat
Completions, output validation, and provider isolation. Launcher tests cover
OpenAI model roles, smoke selection, resume, and external output directories.

Local checks for this revision:

- Full offline suite: **564 tests passed** on Python 3.12.14.
- Ruff lint and formatting checks passed.
- Full and smoke dry runs selected 120 tasks and one task respectively.
- Local synthesis generated and accepted one fixture task without API calls.
- Source and wheel builds passed; the source archive contains one benchmark corpus.
- Local documentation links and corpus paths resolve.

CI runs the same offline suite on Python 3.12 and 3.13 on Linux.

## Earlier integration evidence

Before the API interface cleanup, a compatible endpoint completed one live run
of revised task `BK-RAW-001.us` using `gpt-5.6-sol` for all roles. It completed one
agent attempt and one judge attempt without execution errors. The business
outcome succeeded; hybrid process grading failed. The valid failing verdict was
preserved. Resume reused the task without further model calls.

That earlier run used a separately configured compatible service. It was not a
call to `api.openai.com`, and it does not validate official OpenAI account access
or the current default endpoint. The cleanup does not relabel that result as an
OpenAI-hosted test. Raw traces, credentials, and environments remain outside the
source tree.

## Limits

The current default official endpoint has not been exercised with a live OpenAI
API key in this revision. Full 120-task live evaluation and live Codex-based novel
synthesis/feedback repair have not been rerun. The local synthesis smoke is a
deterministic fixture, and optional Codex invocation is covered by mocked tests.
No paper-table reproduction or performance claim follows from these checks.

Licensing still requires owner confirmation; see [LICENSING.md](../LICENSING.md).
