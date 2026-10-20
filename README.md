# CXBench

CXBench evaluates whether customer-service agents complete a request **and follow
the required process**. Tasks run against simulated business systems; traces
record tool calls, disclosures, drafts, customer confirmations, and submissions.
Evaluation combines state-based outcomes, deterministic process rules, and an
LLM judge with task-specific semantic checklists.

This source release includes data synthesis, six simulated domains, a revised
120-task benchmark, resumable experiments, trace rejudging, and offline scoring.
One task corpus is distributed under `benchmarks/cxbench-120/`, using the revised
experimental protocol. No historical results or credentials are included.

[中文使用说明](docs/quickstart.zh-CN.md) · [Experiment protocol](docs/experiments.md) ·
[Data synthesis](docs/generation.md) · [Provider configuration](docs/providers.md) ·
[Architecture](docs/architecture.md)

Paper: [arXiv:2610.25954](https://arxiv.org/abs/2610.25954)
([PDF](https://arxiv.org/pdf/2610.25954)).

## Requirements and installation

- Python 3.12 or newer; Python 3.12 is the reproducibility baseline.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for locked installs.
- macOS, Linux, or Windows through WSL for the shell launcher.
- A compatible API account for live agent, user-simulator, and judge calls.
  Offline tests and local synthesis need no API key or GPU.

From the root of this source checkout:

```bash
uv sync --frozen --python 3.12
uv run --frozen cxbench --help
uv run --frozen pytest -q
```

Without uv, `python -m pip install -e .` installs the runtime, but does not use the
locked development environment. Experiments are supported from a source checkout;
the wheel contains the `cxbench` runtime and policy corpus, not the repository's
experiment entry points, benchmark YAML files, or tests.

## Quick start

### 1. Validate the experiment without API calls

```bash
./run.sh --dry-run
./run.sh --smoke-test --dry-run
```

These select 120 revised tasks and one revised banking task respectively. They
validate options and task schemas without creating run artifacts.

### 2. Configure OpenAI privately

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Keep this file private. The default endpoint is
`https://api.openai.com/v1`. The launcher uses the OpenAI Responses API with
streaming enabled by default and `gpt-5.6-sol` for all three roles by
default. Model availability depends on the endpoint. To select another supported
model, set `CXBENCH_AGENT_MODEL`, `CXBENCH_USER_MODEL`, and
`CXBENCH_JUDGE_MODEL` in the **shell environment**; use the `openai:` prefix for
user and judge models. See [API configuration](docs/providers.md) for request
settings and explicitly configured compatible endpoints.

### 3. Run a live smoke test

```bash
./run.sh --smoke-test --run-id openai-smoke
```

This makes real, billable API calls for the agent, user simulator, and hybrid
judge. Artifacts appear under `runs/main_experiment/openai-smoke/`.
A successfully executed task may receive a failing benchmark verdict: execution
success and compliance success are different measurements.

To resume the same smoke test, retain the task selection and model settings:

```bash
./run.sh --smoke-test --resume --run-id openai-smoke
```

Completed verdicts, including failures, are reused. A judge retry uses the saved
trace. It does not rerun the agent to seek a passing verdict.

### 4. Run the full revised benchmark

```bash
./run.sh --run-id openai-120 --concurrency 1
./run.sh --resume --run-id openai-120 --concurrency 1
```

The launcher defaults to one process, a 1,200-second per-attempt task timeout,
least-privilege mode, LLM user simulation, and final hybrid grading. Change
concurrency to suit provider limits. Task/judge execution retries are unbounded;
the timeout is **per attempt**, not a total run limit. Interrupt a persistently
failing run and inspect its logs before resuming. See the
[retry and scoring protocol](docs/experiments.md).

Use `--runs-dir /path/to/results` to keep results outside this repository.

## Data synthesis

The offline fixture exercises rule-card extraction, task construction, grading
specification merging, validation, and acceptance:

```bash
uv run --frozen cxbench generate \
  --config configs/benchmark_generation/smoke.yaml \
  --executor local --out generated/local-smoke
uv run --frozen cxbench accept \
  --run generated/local-smoke --dest generated/accepted
```

This is a deterministic template-based smoke test, not novel LLM-generated data.
The local executor requires complete task mappings. New task synthesis uses the
optional Codex CLI executor, scenario matrices, domain interfaces, and rule cards:

```bash
uv run --frozen cxbench generate \
  --config configs/benchmark_generation/banking.yaml \
  --executor codex --workers 1 --out generated/banking-candidates
```

Install and authenticate the [Codex CLI](https://github.com/openai/codex) separately
for this command. The generator uses standard `codex exec`; no private delegation
skill is required. Static validation is not a substitute for semantic review and
runtime replay. Review candidates before acceptance and version them separately
from the released benchmark. [Full generation guide](docs/generation.md).

## Benchmark and grading

| Domain | Revised tasks |
| --- | ---: |
| Banking | 20 |
| Telecom | 20 |
| Insurance | 20 |
| Healthcare | 20 |
| Utilities | 20 |
| Government benefits | 20 |

The single release is `benchmarks/cxbench-120/`, covering all six domains.
Each task has a SHA-256 entry in its manifest. Business-contract hashes and
historical source metadata preserve provenance without shipping a duplicate
task corpus. The semantic rubrics and fixed judge prompt are also versioned.

**Formal results require `hybrid`.** Rule-only grading supports offline tests and
ablations. LLM-only grading is a diagnostic view. Neither alone is a formal
CXBench compliance result. The headline success rate requires both outcome and
hybrid process success; see [metric definitions](docs/experiments.md).

## Results and rejudging

A batch contains `manifest.json`, `results.jsonl`, `metrics.json`, `summary.md`,
and per-task artifacts under `task_runs/`. Task artifacts include `trace.jsonl`,
`config.json`, `final_state.json`, `grade.json`, and `report.md`. Completed hybrid
batches also receive `paper_scores/` with offline aggregate statistics.

```bash
uv run --frozen python -m main_experiment.score \
  --batch gpt-5.6-sol=runs/main_experiment/openai-120 \
  --output runs/scores/revised-120

uv run --frozen python -m main_experiment.rejudge \
  --batch runs/main_experiment/openai-smoke \
  --output runs/rejudged/openai-smoke \
  --judge-provider openai --judge-model openai:gpt-5.6-sol
```

Scoring rejects incomplete batches, duplicate task IDs, changed task hashes, and
incompatible task sets across models. Rejudging writes a separate output batch.
Keep derived tasks and ablation outputs separate from the canonical release.

## Repository map

| Path | Purpose |
| --- | --- |
| `cxbench/` | Runtime, providers, domain tools, graders, generation, statistics |
| `main_experiment/` | Run, resume, rejudge, and formal scoring |
| `model_eval/` | Advanced domain selection and compatible evaluation launchers |
| `no_agent_direct_submit_ablation/` | Explicit direct-submission rule ablation |
| `benchmarks/cxbench-120/` | The single 120-task release and integrity manifest |
| `tasks/` | Small runtime examples and regression fixtures |
| `data/` | Policy corpus, source metadata, and scenario matrices |
| `configs/` | Portable generation configurations |
| `scripts/` | Batch worker, OpenAI launcher, corpus downloader/converter |
| `tests/` | Offline regression tests, mocked provider tests, release checks |
| `docs/` | User guides, protocol, architecture, provenance, domain contracts |

## Development and reproducibility

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
uv build
```

CI runs offline; credentials are never needed for its checks. Record the code
revision, task hashes, rubric/prompt versions, model roles, provider settings,
permission mode, concurrency, and retry policy with any reported results. LLM
outputs remain stochastic even at temperature zero. A small smoke test verifies
integration and does not reproduce a full paper table.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing tasks or grading. The
[provenance note](docs/provenance.md) distinguishes preserved experimental behavior
from release-maintenance changes. [Validation](docs/validation.md) records what
was actually checked for this release.

## Licensing and sources

See [LICENSING.md](LICENSING.md) for code licensing status and
[NOTICE.md](NOTICE.md) for third-party policy sources. Legal source texts and
dependency licenses are separate from the project's own code. These simulations
are research artifacts and do not provide legal advice.

The original project is [PacificStudio/cxbench](https://github.com/PacificStudio/cxbench).

## Citation

If you use CXBench in your research, please cite our
[arXiv preprint](https://arxiv.org/abs/2610.25954):

```bibtex
@misc{yu2026cxbench,
  title = {CXBench: A Benchmark for Process-Compliant LLM Agents in Regulated Customer-Experience Workflows},
  author = {Boxi Yu and Yuzhong Zhang and Zhenfeng Su and Linfu Xi and Yinghan Yao and Shitao Chen and Yang Cao and Liting Lin and Lionel Briand and Emir Mu{\~n}oz},
  year = {2026},
  eprint = {2610.25954},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2610.25954}
}
```

[CITATION.cff](CITATION.cff) supplies GitHub's "Cite this repository" metadata and
sets this paper as the preferred citation. For reproducibility, also report the
exact code revision and task manifest used.
