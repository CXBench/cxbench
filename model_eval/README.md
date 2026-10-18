# Advanced launchers

These source-checkout entry points support domain filtering and balanced subsets.
They default to the revised six-domain task release. For the simplest CRS path,
use the root [launcher](../README.md).

```bash
uv run --frozen python -m model_eval.benchmark --help
uv run --frozen python -m model_eval.resume --help
uv run --frozen python -m model_eval.score --help
```

The `model_eval.run` entry point additionally supports permission-mode comparisons.
Settings come from CLI options, environment, and `--env-file` (default
`model_eval/.env`). Provide private `CXBENCH_API_KEY` and an appropriate
`CXBENCH_BASE_URL`; these launchers use the generic compatible client.
See [providers](../docs/providers.md). Validate with `--dry-run` before API calls.

Hybrid is the formal evaluation mode. Rule/LLM diagnostics and ablations must be
identified explicitly in reports. Use `main_experiment.score` for the documented
strict, complete-batch formal statistics.
