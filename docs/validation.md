# Release validation

Verified on 2026-10-18 on macOS with CPython 3.12.14, using an isolated environment
outside the release directory. This is an integration record, not a benchmark
performance claim.

| Check | Result |
| --- | --- |
| Locked source installation | `uv sync --frozen --python 3.12` succeeded |
| Offline regression suite | 564 tests passed |
| Lint and formatting | Ruff checks passed |
| Task release integrity | Original and revised task hashes preserved; revised changes confined to semantic rubric fields |
| Domain policy corpus | All indexed sources resolve |
| Full and smoke dry runs | Correctly select 120 tasks and 1 task; no API calls or run writes |
| Local synthesis | Generated and accepted a complete fixture; executed its outcome and rule-only checks |
| Domain generation examples | All six examples replayed successfully with diagnostic rule-only grading |
| Source/wheel build | Both artifacts built successfully |
| Installed wheel | Imported outside the source tree; CLI, bundled policy index, and diagnostic runtime smoke succeeded |
| Private-credential scan | No source-profile credential values found in the release |
| Core runtime preservation | 15 runtime/protocol modules have identical Python syntax trees to the consolidated source before export |

## Live CRS integration

A fresh run executed revised task `BK-RAW-001.us` from this release. Agent, user
simulator, and semantic judge all used CRS `gpt-5.6-sol` through streaming
Responses. The run used least-privilege mode and final hybrid grading.

| Observation | Recorded value |
| --- | --- |
| Execution completed (`ok`) | true |
| Execution error | none |
| Agent attempts / judge attempts | 1 / 1 |
| Outcome success | true |
| Hybrid process success | false |
| Final verdict | fail |
| User interactions | 7 |
| Tool calls | 19 |
| Agent provider calls | 26 |
| Execution / judge elapsed time | approximately 248 s / 45 s |

The failing compliance verdict is a valid measurement and was preserved. The
batch produced formal score artifacts. A subsequent smoke resume reused the
completed task (`reused=1`, `to_run=0`, `to_grade=0`) without API credentials or
new model calls. Logs, credentials, environments, and raw live traces are kept
outside the publishable folder.

## Scope and remaining limits

The full 120-task live benchmark was not rerun for this export. No historical
paper table is claimed to be reproduced. Optional Codex-based novel synthesis
and feedback repair were validated through mocked invocation and offline tests,
not a live Codex generation run. The local generation fixture is deterministic.

CI is configured for Python 3.12 and 3.13 on Linux; this local record covers
Python 3.12 on macOS. External model availability, provider costs, and stochastic
judge decisions can change. Licensing still requires owner confirmation as
described in [LICENSING.md](../LICENSING.md).
