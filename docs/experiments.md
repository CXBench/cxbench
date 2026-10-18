# Experiment protocol

## Evaluation object

A task specifies a customer goal, hidden reference facts, an initial business
state, available tools, a confirmation contract, deterministic rules, and a
semantic judge checklist. The agent receives task-relevant observations and the
complete, stable policy context for its domain. Hidden reference facts support
the user simulator and evaluation; they are not directly exposed to the agent.

The six simulated systems cover banking, telecom, insurance, healthcare,
utilities, and government benefits. They operate on local synthetic state and
do not connect to real customer accounts. A trace records observable events and
actors; evaluation must distinguish a tool request from its successful result,
a saved draft from customer-visible disclosure, and confirmation from submission.

## Released tasks

The single distributed corpus is `benchmarks/cxbench-120/`, identified in its
manifest as `cxbench-120-v20260918`: 120 tasks, 20 per domain.
Its banking rubric version is `banking-llm-judge-v20260917`; the other five domains
use `cross-domain-llm-judge-v20260918`. The fixed judge prompt version is
`trace-adjudication-v20260918`. Older task corpora are not distributed alongside it.

The manifest records task hashes, historical source hashes, and business-contract
hashes. Task content retains the updated semantic checklists. Preserve the task
and judge versions in reported results; do not pool scores from different rubrics.

## Execution modes

In `least_privilege` mode, the agent can read, check, prepare drafts, and issue
notices. Submission tools are withheld from its tool list. The runtime submits
a saved draft through the business-system gateway after explicit customer
confirmation. The domain wrapper also rejects direct agent submission.

In `direct` mode (`high_permission` is a compatibility alias), submission tools
are available to the agent. This mode changes execution authority, not the
grading requirements. An agent can reach the business outcome while failing a
process rule such as `no_agent_direct_submit`.

These modes implement selected safeguards. They do not guarantee every required
disclosure, ordering obligation, or semantic condition; the grader evaluates the
resulting trace.

## Model roles and defaults

| Setting | `./run.sh` default |
| --- | --- |
| Agent | OpenAI `gpt-5.6-sol` |
| User simulator | OpenAI `gpt-5.6-sol`, `llm` strategy |
| Semantic judge | OpenAI `gpt-5.6-sol` |
| Transport | Responses API with streaming |
| Mode | `least_privilege` |
| Final grader | `hybrid` |
| Workers | 1 process |
| Task timeout | 1,200 seconds per execution attempt |
| Output limit | 8,192 tokens per model response |
| Execution/judge retry delay | 5 seconds |

These launcher settings provide a conservative OpenAI reproduction path. The
general Python runner retains the broader batch defaults (20 thread workers,
600-second task timeout, provider/model options). Set all roles explicitly when
using it. For example:

```bash
uv run --frozen python -m main_experiment.run \
  --agent-provider openai --model gpt-5.6-sol \
  --user-model openai:gpt-5.6-sol \
  --judge-provider openai --judge-model openai:gpt-5.6-sol \
  --wire-api responses --mode least_privilege --grader hybrid \
  --executor process --concurrency 1 --task-timeout 1200 \
  --task 'benchmarks/cxbench-120/tasks/BK-*.yaml' \
  --run-id banking-revised
```

Task YAML defines interaction and tool budgets. The provider-step limit is the
larger of `max_turns` and `max_user_interactions + max_tool_calls + 5`.
Chat Completions requests specify temperature zero. Responses requests omit
temperature and use provider reasoning settings when configured. Exact model
outputs are not guaranteed to repeat.

## Retries and recovery

Provider request retries and experiment retries are separate:

1. The transport retries transient failures, with 5 additional retries by default
   and a 2-second base backoff. Non-retryable HTTP errors fail immediately;
   exhausted transport retries propagate and stop the batch.
2. The batch retries incomplete task execution, timeouts, and recoverable execution
   errors without an attempt-count cap. A new attempt gets a new runtime.
3. The agent stage first saves deterministic outcome/process evidence. Semantic
   grading runs separately, and a judge error retries the same completed trace.
4. A valid `pass` or `fail` verdict completes the task. A compliance failure is
   never a reason to rerun the agent.

The per-attempt deadline is checked by the runtime and does not replace HTTP
timeouts. An in-flight request may finish after that deadline. There is no total
batch time or cost cap. Cancellation remains the operator's responsibility.
Failed attempt artifacts are archived when possible; successful task records
are flushed incrementally for recovery.

For resume, repeat the original model roles, task selection, run ID, and results
directory using `python -m main_experiment.resume`, or `./run.sh --resume`.
Keep `--smoke-test` when resuming a smoke batch. Resume validates task hashes and
configuration, reuses completed reports (including failures), and finishes
missing grading. It refuses a changed task selection.

## Grading and reported metrics

For task i, let O_i denote outcome success, H_i the conjunction of applicable
deterministic hard rules, and J_i the semantic judge decision. Formal process
success is P_i = H_i AND J_i. Headline success requires O_i AND P_i.

| Metric | Definition |
| --- | --- |
| Outcome success rate | mean(O_i) |
| Hybrid process success rate | mean(P_i) |
| Joint success rate | mean(O_i AND P_i) |
| Outcome-only false pass | O_i AND NOT P_i |
| False-pass rate in formal summaries | count(outcome-only false passes) / completed tasks |

The semantic judge produces a finding for each required checklist item with
applicability and trace evidence. Its consistency guard rejects contradictory
top-level pass claims. Only hard deterministic rules contribute to H_i; semantic
or hint rules identified by the task's grading policy are not counted twice.

`rule` and `llm` are incomplete diagnostic views. Rule-only use in regression
tests, intermediate execution records, and offline hard-rule reconstruction is
intentional; formal scores require a completed hybrid result. Offline scoring
recomputes deterministic evidence with the current code and combines it with
the recorded judge findings. It does not make new model calls.

The scorer requires complete, paired task sets when comparing models, verifies
task hashes, and refuses duplicate/missing/error results. It reports Wilson 95%
intervals, paired exact McNemar tests, and leave-one-hard-rule-out sensitivity.
The McNemar p-values are unadjusted; no multiple-comparison correction is applied.
Keep the code revision fixed across compared batches, since deterministic rules
are reconstructed by the scorer. Inspect `paper_scores/summary.json` for formal
metrics; `metrics.json` describes the runtime batch.

## Advanced evaluation and ablations

`python -m model_eval.benchmark --help` offers domain selection and balanced
subsets, with the revised release as its default. These advanced launchers use
their own endpoint configuration; see [providers](providers.md).

`python -m no_agent_direct_submit_ablation.run --help` exposes the targeted
submission-rule ablation. Keep ablation tasks and outputs separate from the
canonical tasks, and identify the removed rule in reports. Removing a process
requirement changes what a pass means, even if the final grader remains hybrid.

## Minimum report metadata

Record the code revision, task release and hashes, system-prompt and rubric
versions, all model roles and returned model IDs where available, provider and
wire API, permission mode, output and interaction budgets, retry policy,
concurrency, task timeout, number of completed tasks, and final grader.
Report execution errors separately from valid failing verdicts. A one-task smoke
test supports an integration claim only, not a performance or paper-reproduction
claim.
