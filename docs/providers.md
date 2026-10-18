# OpenAI API configuration

## Standard setup

Copy `.env.example` to `.env` at the repository root and set your own key:

```dotenv
OPENAI_API_KEY=your_private_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_WIRE_API=responses
OPENAI_RESPONSES_STREAM=true
OPENAI_REASONING_EFFORT=low
OPENAI_MAX_OUTPUT_TOKENS=8192
```

The client sends standard OpenAI HTTP requests with bearer-token authentication.
The public launcher uses `POST /v1/responses`, supporting function calls and
streaming events. The key and base URL are supplied through the standard
`OPENAI_*` variables. `openai:<model>` never falls back to another provider's key.
No credentials are stored in task files or result manifests.

The default model is `gpt-5.6-sol` for all three roles. Model access depends on
your OpenAI account. See the official [model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
and [Responses API guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).
The root launcher uses Responses explicitly; the generic client also supports
Chat Completions when `OPENAI_WIRE_API=chat` is selected. Tool/reasoning support
can differ across model and API combinations, so prefer Responses for this setup.

## Select model roles

Model-selection variables belong in the shell environment:

```bash
export CXBENCH_AGENT_MODEL=gpt-5.6-sol
export CXBENCH_USER_MODEL=openai:gpt-5.6-sol
export CXBENCH_JUDGE_MODEL=openai:gpt-5.6-sol
./run.sh --smoke-test --run-id openai-smoke
```

The agent model is a bare identifier; user and judge model strings include
`openai:` to select the adapter. These model-role variables are distinct from
API credentials. Never place credentials in CLI arguments or commit `.env`.

## Request controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | required for live calls | Private API credential |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API root |
| `OPENAI_WIRE_API` | `responses` | Generic client transport: `responses` or `chat` |
| `OPENAI_RESPONSES_STREAM` | `true` | Enable streaming Responses; `false` is supported |
| `OPENAI_REASONING_EFFORT` | `low` | Responses reasoning effort; an empty value omits the field |
| `OPENAI_TIMEOUT_SECONDS` | 120 | Request timeout in seconds |
| `OPENAI_MAX_RETRIES` | 5 | Additional transient request attempts |
| `OPENAI_RETRY_BACKOFF_SECONDS` | 2 | Base request retry delay |
| `OPENAI_VALIDATION_RETRIES` | 1 | Additional attempts for malformed model output |
| `OPENAI_MAX_OUTPUT_TOKENS` | 1,024 in the raw client | The experiment runner and example environment use 8,192 |

Shell environment values take precedence over dotenv values. Runner CLI settings
such as `--wire-api` and `--max-output-tokens` override the corresponding generic
settings during that run. The root shell launcher fixes the wire API to Responses;
use the general Python runner to explicitly select Chat Completions.
Streaming remains configurable and is not forced by the launcher.

## Compatible endpoints

A compatible service can be configured explicitly through `OPENAI_BASE_URL` and
`OPENAI_API_KEY` while retaining the same request format and `openai:` model
prefix. This does not imply that the service is operated by OpenAI, supports all
OpenAI models, or implements every streaming/tool feature. Use model identifiers
and request settings supported by that endpoint. The official OpenAI endpoint
remains the default.

## Advanced launchers

`model_eval` and the submission-rule ablation accept `--env-file`; their default
is `model_eval/.env`. Pass `--env-file .env` to reuse the root configuration.
They accept `OPENAI_API_KEY` and `OPENAI_BASE_URL`; the existing `CXBENCH_API_KEY`
and `CXBENCH_BASE_URL` aliases remain available. These launchers retain their
Chat Completions default, while the recommended root launcher uses Responses.
Inspect `--help` and `--dry-run` when using the advanced paths.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing key, 401, or 403 | `OPENAI_API_KEY`, account access, and selected endpoint |
| Model unavailable | Exact identifier and account/model permissions |
| 429 or timeouts | Request limits, network, request timeout; reduce concurrency |
| Unsupported request parameter | Model/API combination; use the documented Responses configuration |
| Repeated task/judge errors | Inspect failed attempts and traces; execution retries do not fix incompatible interfaces |
| Resume rejects hashes/options | Repeat the original selection and configuration |
| Completed run has a `fail` verdict | Read the report and judge findings; execution success differs from compliance success |

Keep TLS verification enabled. Offline tests and dry runs do not require an API
key. An earlier compatible-endpoint smoke run is not evidence that the official
OpenAI endpoint was tested; see [validation](validation.md).
