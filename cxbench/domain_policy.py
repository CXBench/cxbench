from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .task_schema import TaskSpec

_PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_ROOT.parent


def _resolve_data_root() -> Path:
    """Locate the data/ directory.

    Priority:
    1. ``cxbench/data/`` — when installed from a wheel that bundles data via
       ``force-include``.
    2. ``<repo_root>/data/`` — editable / source checkout.
    """
    packaged = _PACKAGE_ROOT / "data"
    if packaged.is_dir():
        return packaged
    return REPO_ROOT / "data"


DATA_ROOT = _resolve_data_root()
POLICY_ROOT = DATA_ROOT / "raw" / "lii_markdown"
DOMAIN_INDEX_PATH = POLICY_ROOT / "domain_index.json"
MARKDOWN_MANIFEST_PATH = POLICY_ROOT / "manifest.jsonl"


@dataclass(frozen=True)
class DomainPolicySource:
    id: str
    title: str
    url: str
    path: Path
    text: str


_BUSINESS_SYSTEM_DOMAIN_PREFIXES = {
    "banking_": "banking",
    "telecom_": "telecom",
    "insurance_": "insurance",
    "healthcare_": "healthcare",
    "utilities_": "utilities",
    "government_benefits_": "government_benefits",
}


def infer_domain(domain: str | None = None, business_system: str | None = None) -> str:
    """Resolve benchmark domain from explicit domain or business system."""

    if domain and domain not in {"", "None"}:
        return domain
    business_system = business_system or ""
    for prefix, resolved in _BUSINESS_SYSTEM_DOMAIN_PREFIXES.items():
        if business_system.startswith(prefix):
            return resolved
    return "banking"


def domain_for_task(task: TaskSpec) -> str:
    return infer_domain(task.domain, task.business_system)


@lru_cache(maxsize=1)
def load_domain_index() -> dict[str, Any]:
    if not DOMAIN_INDEX_PATH.exists():
        return {"version": "missing", "domains": {}}
    with DOMAIN_INDEX_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_markdown_manifest() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not MARKDOWN_MANIFEST_PATH.exists():
        return rows
    for line in MARKDOWN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def _resolve_source_path(rel_or_abs: str) -> Path:
    """Resolve a path from the manifest or extra_markdown_paths.

    Manifest entries look like ``data/raw/lii_markdown/pages/foo.md``.
    In packaged mode we strip the leading ``data/`` and resolve against
    ``DATA_ROOT``; in dev mode we resolve against ``REPO_ROOT``.
    """
    candidate = Path(rel_or_abs)
    if candidate.is_absolute():
        return candidate
    # Try packaged layout: cxbench/data/<rest>
    if str(candidate).startswith("data/"):
        packaged = DATA_ROOT / str(candidate)[5:]  # strip "data/"
        if packaged.exists():
            return packaged
    # Dev layout: repo root
    dev_path = REPO_ROOT / candidate
    if dev_path.exists():
        return dev_path
    # Fallback: return the dev-style path so callers see a sensible "missing" error
    return dev_path


def load_domain_policy_sources(domain: str) -> list[DomainPolicySource]:
    index = load_domain_index()
    manifest = load_markdown_manifest()
    entry = dict(index.get("domains", {}).get(domain, {}))
    sources: list[DomainPolicySource] = []

    for source_id in entry.get("source_ids", []) or []:
        source_id = str(source_id)
        row = manifest.get(source_id)
        if not row:
            continue
        path = _resolve_source_path(str(row["output_markdown"]))
        if not path.exists():
            continue
        sources.append(
            DomainPolicySource(
                id=source_id,
                title=str(row.get("title") or source_id),
                url=str(row.get("url") or ""),
                path=path,
                text=path.read_text(encoding="utf-8"),
            )
        )

    for extra in entry.get("extra_markdown_paths", []) or []:
        path = _resolve_source_path(str(extra))
        if not path.exists():
            continue
        sources.append(
            DomainPolicySource(
                id=path.stem,
                title=_title_from_markdown(path),
                url="",
                path=path,
                text=path.read_text(encoding="utf-8"),
            )
        )
    return sources


def domain_policy_context_for_task(task: TaskSpec) -> str:
    domain = domain_for_task(task)
    sources = load_domain_policy_sources(domain)
    if not sources:
        return ""
    return render_domain_policy_context(domain, sources)


def render_domain_policy_context(domain: str, sources: list[DomainPolicySource]) -> str:
    parts = [
        f"<CXBENCH_DOMAIN_POLICY_CONTEXT domain={json.dumps(domain)} source_count={len(sources)}>",
        "The following domain policy source materials are included wholesale for fairness across the finite benchmark task set.",
        "Use them as domain compliance context. Do not paraphrase them as legal advice, and do not ignore task-specific tools, state, or confirmation requirements.",
    ]
    for source in sources:
        try:
            rel = source.path.relative_to(DATA_ROOT)
            rel_str = f"data/{rel}"
        except ValueError:
            rel_str = str(source.path)
        parts.extend(
            [
                "",
                f"<DOMAIN_POLICY_SOURCE id={json.dumps(source.id)} title={json.dumps(source.title)} url={json.dumps(source.url)} path={json.dumps(rel_str)}>",
                source.text.strip(),
                "</DOMAIN_POLICY_SOURCE>",
            ]
        )
    parts.append("</CXBENCH_DOMAIN_POLICY_CONTEXT>")
    return "\n".join(parts)


def validate_domain_policy_index() -> list[str]:
    """Return validation errors for domain index and source files."""

    errors: list[str] = []
    index = load_domain_index()
    manifest = load_markdown_manifest()
    domains = index.get("domains", {}) if isinstance(index.get("domains"), Mapping) else {}
    used_ids: set[str] = set()
    for domain, entry in domains.items():
        if not isinstance(entry, Mapping):
            errors.append(f"domain entry must be object: {domain}")
            continue
        for source_id in entry.get("source_ids", []) or []:
            source_id = str(source_id)
            used_ids.add(source_id)
            row = manifest.get(source_id)
            if not row:
                errors.append(f"unknown source id for {domain}: {source_id}")
                continue
            path = _resolve_source_path(str(row.get("output_markdown", "")))
            if not path.exists():
                errors.append(f"missing markdown for {domain}:{source_id}: {path}")
        for extra in entry.get("extra_markdown_paths", []) or []:
            path = _resolve_source_path(str(extra))
            if not path.exists():
                errors.append(f"missing extra markdown for {domain}: {path}")
    missing = set(manifest) - used_ids
    if missing:
        errors.append(f"unclassified markdown source ids: {', '.join(sorted(missing))}")
    return errors


def _title_from_markdown(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem
