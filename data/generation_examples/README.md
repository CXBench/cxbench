# Runnable domain templates

Each YAML file is an unmodified task from the revised benchmark release, selected
as a complete interface example for its domain. These are templates for generation
prompts, not new evaluation data or an independent split. The generator checks
the example's business-system identifier before using it.

The offline release tests replay all six examples with a scripted agent and
deterministic user. This checks runtime integration with rule-only grading; it
does not establish formal semantic compliance.
