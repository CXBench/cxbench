# Architecture and task authoring

## Execution flow

```text
Task YAML + domain policy context
              |
              v
      Runtime / agent adapter <----> LLM user simulator
              |
        domain tools
   read -> check -> draft -> notice
              |
      customer confirmation
              |
      submission gateway
              |
     trace + final state
              |
   outcome + hard rules + LLM judge
              |
       hybrid grade / report
```

The exact ordering and optional branches depend on each task. The diagram shows
the components, not a universal regulatory workflow.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `task_schema.py` | Parse and validate task schema and budgets |
| `runtime.py` | Interaction loop, permission mode, confirmation, trace artifacts |
| `adapters.py` | Scripted regression agents and LLM action adapter |
| `user_sim.py` | LLM user strategies and deterministic test simulator |
| `domain_system.py`, `tools_*.py` | Domain state, tool semantics, submission wrapper |
| `domain_policy.py` | Resolve bundled policy snapshots and stable domain context |
| `openai_compat.py` | Provider profiles, HTTP transport, streaming, output validation |
| `grader.py` | State outcomes, deterministic rules, semantic judge, hybrid verdict |
| `benchmark_generation.py` | Source selection, candidate generation, review, acceptance |
| `statistics.py` | Offline metrics, confidence intervals, paired comparisons |
| `main_experiment/` | Batch orchestration, resume, rejudging, formal scoring |

## Task structure

Read a complete released task rather than building from a partial schema sample.
`data/generation_examples/` contains one runnable revised task per domain for
generator prompts. The examples are templates drawn from the benchmark, not an
independent evaluation split.

| Field | Meaning |
| --- | --- |
| `id`, `title`, `domain`, `scenario` | Identity and domain context |
| `business_system` | Simulated backend implementation |
| `customer_goal`, `seed_message` | Customer request and opening message |
| `hidden_facts` | Reference values and expected outcome branches |
| `initial_state` | Synthetic session and domain records |
| `tools` | `read`, `check`, `draft`, `notice`, `submit` tool lists |
| `confirmation_contract` | Fields that must be represented before confirmation |
| `rules` | Deterministic checks applicable to the task |
| `policy` | Sources, process requirements, grading policy, semantic checklist |
| Budget fields | Interaction/tool limits; see `TaskSpec` for defaults |

Submission, disclosure, and confirmation checks need observable evidence.
Avoid defining requirements that cannot be expressed by available tools, agent
messages, or state. Keep source-based obligations distinct from benchmark runtime
controls such as draft-confirmation gating and `no_agent_direct_submit`.

## Extending a domain

Add domain tool behavior and state semantics together with focused tests. Update
the domain factory, tool schemas exposed to the model, generation interface,
outcome checker, applicable rules, and policy index as needed. Add complete task
fixtures for success and meaningful failure branches. Do not weaken existing
graders to make a new task pass.

Use stable prompt prefixes: common instructions, then full domain context, then
task details and dynamic conversation/trace content. Do not move per-task IDs,
timestamps, or trace tails ahead of long fixed policy blocks. Domain policy
context must remain consistent across tasks in that domain.

## Testing boundaries

Offline tests use deterministic simulators and mocked HTTP/provider responses.
They validate execution mechanics, transport parsing, retry behavior, resume
integrity, and scoring. They cannot establish external model availability or
semantic judge quality. Live smoke tests cover provider integration; broader
evaluation and human auditing are needed for research claims.

The business systems are simulations. A benchmark process pass measures the
requirements encoded by its tasks and graders; it does not establish legal
compliance for a real deployment.
