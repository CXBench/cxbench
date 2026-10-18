# Source and third-party notices

## Project sources

The original public project is
[PacificStudio/cxbench](https://github.com/PacificStudio/cxbench). This release
consolidates that project's synthesis/runtime lineage with a later experiment
snapshot. Commit identifiers and release-maintenance changes are recorded in
[docs/provenance.md](docs/provenance.md). This notice is attribution, not a grant
of rights; project licensing status is described in [LICENSING.md](LICENSING.md).

## Legal and policy corpus

`data/raw/lii/` contains 34 scoped source snapshots from the
[Cornell Legal Information Institute](https://www.law.cornell.edu/).
`data/raw/lii_markdown/` contains extracted legal text. Per-source manifests,
metadata, and Markdown frontmatter retain the source URL, identity, and available
retrieval/conversion information. Domain context is mapped by
`domain_index.json`. Source reference paths to the historical repository are
provenance labels; that repository is not a runtime dependency.

The converter removes page navigation and normalizes formatting; it is not
intended to summarize or alter legal requirements. The raw pages may also contain
website formatting and notices. Consult the source site's current terms and
the applicable rights for each source before redistribution. No blanket license
over third-party text is asserted by this project. Federal legal text, state
materials, and website editorial content can have different rights conditions.

`data/compliance/insurance_claim_flow_rules.md` is a benchmark-specific process
specification. It must not be presented as statutory text. The simulated tools,
confirmation gates, and agent submission restrictions are benchmark controls;
not every such control is itself a legal requirement.

## Dependencies

Runtime dependencies are PyYAML and json-repair; development/build dependencies
are identified by `pyproject.toml` and `uv.lock`. Their licenses remain with their
respective authors. Install them through the package manager; no dependency
source trees or virtual environments are vendored in this release.
