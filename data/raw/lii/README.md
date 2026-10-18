# Cornell LII scoped raw corpus

This directory is for Cornell Legal Information Institute (LII) pages that are already cited by cxbench reference tasks/snapshots.

Important boundary: this is **not** a full-site LII mirror. `https://www.law.cornell.edu/robots.txt` declares `Crawl-delay: 10`, and cxbench should not aggressively crawl Cornell/LII. Use this corpus only for the scoped URLs in `manifest.jsonl`.

Files:

- `manifest.jsonl` - the versioned list of 34 URLs; historical reference paths describe discovery and are not runtime dependencies.
- `pages/*.html` - downloaded HTML pages.
- `meta/*.json` - source URL, final URL, fetch time, content type, byte count, SHA-256.

Commands:

```bash
uv run --frozen python scripts/download_lii_corpus.py --delay 10
uv run --frozen python scripts/convert_lii_html_to_markdown.py --output data/raw/lii_markdown --write-manifest
```

For audit-grade gold policy, prefer official machine-readable sources such as eCFR, govinfo, OLRC, and EUR-Lex when available. LII is useful as a readable mirror and as compatibility with old cxbench snapshots.

These are optional maintenance commands, run from the repository root. Evaluation
uses the bundled snapshot without downloads. See [source notices](../../../NOTICE.md)
and the [generation guide](../../../docs/generation.md).
