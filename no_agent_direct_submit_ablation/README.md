# Direct-submission rule ablation

This optional launcher creates derived task specifications with the
`no_agent_direct_submit` requirement removed and runs the modified benchmark.
It shares advanced configuration with `model_eval`.

```bash
uv run --frozen python -m no_agent_direct_submit_ablation.run --help
```

Keep its derived tasks, provenance, and outputs separate from canonical runs.
Label the removed requirement in reported results. The resulting measurement
answers an ablation question and is not a canonical CXBench score.
