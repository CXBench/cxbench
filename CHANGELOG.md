# Changelog

## 0.1.0 — clean source consolidation

- Consolidate runtime, six-domain revised experiments, data synthesis, corpus,
  tests, and reusable analysis into a standalone source tree.
- Default to the revised 120-task release; preserve the original release for
  explicit comparisons and keep released task hashes unchanged.
- Repair CRS smoke resume and external output selection; preserve dotenv
  endpoint settings and streaming Responses transport.
- Replace private generation-worker tooling with standard Codex CLI invocation.
  Add executable templates, reject incomplete local inputs and unknown tools,
  protect existing generation runs, and detect edits after validation.
- Extract portable scoring statistics from historical paper scripts.
- Add English and Chinese usage guides, protocol and source documentation,
  contributor guidance, and offline CI.

Licensing remains subject to owner confirmation; see [LICENSING.md](LICENSING.md).
