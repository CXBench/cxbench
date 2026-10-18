# Provider configuration

## CRS quick start

Copy `.env.example` to `.env` at the repository root and set:

```dotenv
CRS_OAI_KEY=your_private_key
CRS_OAI_BASE_URL=https://crs.uuid.im/openai/v1
CRS_OAI_WIRE_API=responses
CRS_OAI_RESPONSES_STREAM=true
```

The CRS launcher explicitly enables Responses streaming. The tested endpoint
requires it; non-streaming requests have returned HTTP 400. The loader uses the
CRS base URL from the process environment or `.env`, falling back to the default
above. The shell launcher does not override a custom `.env` endpoint.

Keys are read at runtime. They are not required by offline tests, dry runs, or
local template generation. Never place credentials in source code or CLI
arguments, and never commit `.env` or run logs containing provider responses.

Model selection for `./run.sh` comes from shell variables, not `.env`:

```bash
export CXBENCH_AGENT_MODEL=gpt-5.6-sol
export CXBENCH_USER_MODEL=crs:gpt-5.6-sol
export CXBENCH_JUDGE_MODEL=crs:gpt-5.6-sol
./run.sh --smoke-test --run-id crs-smoke
```

Use models enabled for your account. An API error indicating an unavailable
model requires choosing another supported identifier; changing credentials or
retry counts does not establish model availability.

## Shared request settings

| Variable | Default | Meaning |
| --- | ---: | --- |
| `OPENAI_TIMEOUT_SECONDS` | 120 | Request timeout in seconds |
| `OPENAI_MAX_RETRIES` | 5 | Additional transient request attempts |
| `OPENAI_RETRY_BACKOFF_SECONDS` | 2 | Base transport retry delay |
| `OPENAI_VALIDATION_RETRIES` | 1 | Additional attempts for malformed model output |
| `OPENAI_MAX_OUTPUT_TOKENS` | 1,024 in the raw client | Per-response output cap; the experiment runner sets 8,192 |

The batch's `--max-output-tokens` controls its output cap. The launcher sets
Responses streaming for CRS even when the user-simulator and judge use explicit
`crs:` model strings. Shell environment values normally take precedence over
dotenv values; explicit command-line runner options override the corresponding
generic provider settings for that run.

## Other supported profiles

| Provider | Key variable | Base URL variable | Transport variable |
| --- | --- | --- | --- |
| OpenAI-compatible | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | `OPENAI_WIRE_API` |
| CRS | `CRS_OAI_KEY` | `CRS_OAI_BASE_URL` | `CRS_OAI_WIRE_API` |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` | `OPENROUTER_WIRE_API` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` | `DEEPSEEK_WIRE_API` |
| SiliconFlow | `SILICONFLOW_API_KEY` | `SILICONFLOW_BASE_URL` | `SILICONFLOW_WIRE_API` |
| Shubiaobiao | `SHUBIAOBIAO_API_KEY` | `SHUBIAOBIAO_BASE_URL` | `SHUBIAOBIAO_WIRE_API` |

Transport values are `chat` and `responses`. Set an explicit base URL for routed
providers. Prefix user/judge models with the intended provider to avoid reliance
on key auto-detection. The general batch runner takes a bare agent model name
and an explicit `--agent-provider`.

For a generic compatible endpoint:

```bash
uv run --frozen python -m main_experiment.run \
  --agent-provider openai --model MODEL_ID \
  --user-model openai:MODEL_ID \
  --judge-provider openai --judge-model openai:MODEL_ID \
  --base-url https://YOUR_ENDPOINT/v1 --wire-api chat \
  --executor process --concurrency 1 \
  --task 'benchmarks/cxbench-banking-judge-v20260917/tasks/BK-RAW-001.us.yaml' \
  --run-id compatible-smoke
```

Supply `OPENAI_API_KEY` privately. `--base-url` applies only to the generic
`openai` profile; other profiles use their named environment variables.

## Advanced launcher configuration

`model_eval` and `no_agent_direct_submit_ablation` accept `--env-file` (default
`model_eval/.env`) and `CXBENCH_API_KEY`, `CXBENCH_MODEL`, `CXBENCH_BASE_URL`, and
`CXBENCH_MAX_OUTPUT_TOKENS`. They route through the generic compatible client.
These are separate from the root CRS launcher settings; use each entry point's
`--help` and `--dry-run` to inspect the plan. Prefer the root launcher for CRS.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| 401/403 or missing API key | Active provider, credential variable, account access |
| 404/model unavailable | Exact model identifier supported by that endpoint |
| CRS rejects non-streaming requests | `CRS_OAI_RESPONSES_STREAM=true`; use `./run.sh` |
| 429, connection failure, timeout | Provider limits, network, and request timeout; reduce concurrency |
| Repeated task/judge retries | Inspect the failing attempt/trace; retries do not fix unsupported task interfaces |
| Resume rejects task hashes/options | Restore the original selection and configuration; use a new run for changes |
| Run succeeds but verdict is `fail` | Read `report.md` and judge findings; this can be a valid measurement |

The adapter uses the platform TLS trust store. Keep certificate validation
enabled and fix the local trust configuration if TLS verification fails.
