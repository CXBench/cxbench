# Main experiments

Use the root [quick start](../README.md) for OpenAI. This package provides:

- `python -m main_experiment.run`: validate and run a batch; default is the revised release.
- `python -m main_experiment.resume`: reuse completed tasks and finish missing execution/grading.
- `python -m main_experiment.rejudge`: evaluate saved traces in a separate output batch.
- `python -m main_experiment.score`: aggregate complete hybrid batches without model calls.

Each entry point supports `--help`. Run from the source repository root.
Keep model roles, task selection, and run directories stable when resuming.
See [protocol](../docs/experiments.md) and [provider setup](../docs/providers.md).
