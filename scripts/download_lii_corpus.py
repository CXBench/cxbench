#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

USER_AGENT = "cxbench-lii-corpus/0.1 (research; respects law.cornell.edu robots crawl-delay)"


def read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def output_paths(out_dir: Path, entry: dict) -> tuple[Path, Path]:
    page = out_dir / "pages" / f"{entry['id']}.html"
    meta = out_dir / "meta" / f"{entry['id']}.json"
    return page, meta


def fetch(url: str) -> tuple[bytes, dict]:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        headers = {k.lower(): v for k, v in response.headers.items()}
        return body, {
            "status": response.status,
            "final_url": response.geturl(),
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
            "last_modified": response.headers.get("last-modified"),
            "date": response.headers.get("date"),
            "headers": headers,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a scoped Cornell LII corpus from manifest.jsonl."
    )
    parser.add_argument("--manifest", default="data/raw/lii/manifest.jsonl")
    parser.add_argument("--out-dir", default="data/raw/lii")
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="seconds between law.cornell.edu requests; robots.txt says 10",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all entries")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest))
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = Path(args.out_dir)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    last_host = ""
    for i, entry in enumerate(manifest, start=1):
        page_path, meta_path = output_paths(out_dir, entry)
        if page_path.exists() and meta_path.exists() and not args.force:
            skipped += 1
            print(f"[{i}/{len(manifest)}] skip {entry['url']}")
            continue
        host = urlparse(entry["url"]).netloc
        if last_host == host and args.delay > 0:
            time.sleep(args.delay)
        last_host = host
        print(f"[{i}/{len(manifest)}] fetch {entry['url']}")
        try:
            body, response_meta = fetch(entry["url"])
        except urllib.error.HTTPError as exc:
            failed += 1
            meta_path.write_text(
                json.dumps(
                    {
                        **entry,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "error": f"HTTP {exc.code}",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        except Exception as exc:
            failed += 1
            meta_path.write_text(
                json.dumps(
                    {
                        **entry,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "error": repr(exc),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        page_path.write_bytes(body)
        meta = {
            **entry,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "retrieved_http_date": response_meta.get("date"),
            "source_last_modified": response_meta.get("last_modified"),
            "final_url": response_meta.get("final_url"),
            "status": response_meta.get("status"),
            "content_type": response_meta.get("content_type"),
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "user_agent": USER_AGENT,
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        downloaded += 1
    print(
        json.dumps({"downloaded": downloaded, "skipped": skipped, "failed": failed}, sort_keys=True)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
