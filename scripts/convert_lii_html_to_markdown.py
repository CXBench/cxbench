#!/usr/bin/env python3
"""Convert scoped Cornell/LII HTML pages to wording-preserving Markdown.

This converter intentionally extracts only the legal text area (`div.section`)
and keeps the original statutory/regulatory wording. It does not summarize or
rewrite legal text; it only removes navigation/chrome and normalizes whitespace.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/raw/lii/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data/raw/lii_markdown"

BLOCK_TAGS = {"p", "div", "section", "article", "li", "tr", "table", "ul", "ol", "blockquote"}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
SKIP_TAGS = {"script", "style", "noscript", "svg", "form", "button", "iframe"}


def attrs_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {k: v or "" for k, v in attrs}


class CoreHtmlExtractor(HTMLParser):
    """Extract page title and the first LII legal-text content block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.in_h1 = False
        self.title_parts: list[str] = []
        self.capture = False
        self.depth = 0
        self.section_parts: list[str] = []
        self.done = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = attrs_dict(attrs)
        classes = set(a.get("class", "").split())
        if tag == "h1" and a.get("id") == "page_title":
            self.in_h1 = True
        is_legal_text = tag == "div" and (
            "section" in classes or "statereg-text" in classes or "appendix" in classes
        )
        if not self.capture and not self.done and is_legal_text:
            self.capture = True
            self.depth = 1
            self.section_parts.append(self.get_starttag_text() or "")
            return
        if self.capture:
            self.section_parts.append(self.get_starttag_text() or "")
            if tag not in {"br", "hr", "img", "meta", "link", "input"}:
                self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self.in_h1:
            self.in_h1 = False
        if self.capture:
            self.section_parts.append(f"</{tag}>")
            self.depth -= 1
            if self.depth <= 0:
                self.capture = False
                self.done = True

    def handle_data(self, data: str) -> None:
        if self.in_h1:
            self.title_parts.append(data)
        if self.capture:
            self.section_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        text = f"&{name};"
        if self.in_h1:
            self.title_parts.append(html.unescape(text))
        if self.capture:
            self.section_parts.append(text)

    def handle_charref(self, name: str) -> None:
        text = f"&#{name};"
        if self.in_h1:
            self.title_parts.append(html.unescape(text))
        if self.capture:
            self.section_parts.append(text)

    def handle_comment(self, data: str) -> None:
        if self.capture:
            self.section_parts.append(f"<!--{data}-->")


class MarkdownTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.heading_level: int | None = None
        self.list_stack: list[str] = []
        self.in_li = False
        self.cell_index = 0
        self.in_table = False
        self.span_classes: list[set[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in HEADING_TAGS:
            self._blankline()
            self.heading_level = HEADING_TAGS[tag]
            self.parts.append("#" * self.heading_level + " ")
        elif tag == "div":
            classes = set(attrs_dict(attrs).get("class", "").split())
            if "head" in classes:
                self._blankline()
                self.heading_level = 2
                self.parts.append("## ")
            elif classes & {"subsect", "subsection", "hd1", "fp-1", "fp1-2", "xref"}:
                self._blankline()
            elif self.parts and not self.parts[-1].endswith("\n"):
                self.parts.append("\n")
        elif tag == "span":
            classes = set(attrs_dict(attrs).get("class", "").split())
            self.span_classes.append(classes)
            if "num" in classes or "designator" in classes:
                self._blankline()
        elif tag == "p":
            self._blankline()
        elif tag in {"ul", "ol"}:
            self.list_stack.append(tag)
            self._newline()
        elif tag == "li":
            self._newline()
            marker = "1. " if self.list_stack and self.list_stack[-1] == "ol" else "- "
            self.parts.append(marker)
            self.in_li = True
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "table":
            self._blankline()
            self.in_table = True
        elif tag == "tr":
            self._newline()
            self.parts.append("| ")
            self.cell_index = 0
        elif tag in {"td", "th"}:
            if self.cell_index > 0:
                self.parts.append(" | ")
            self.cell_index += 1
        elif tag in {"sup", "sub"}:
            self.parts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in HEADING_TAGS:
            self.heading_level = None
            self._blankline()
        elif tag == "div" and self.heading_level == 2:
            self.heading_level = None
            self._blankline()
        elif tag == "span":
            classes = self.span_classes.pop() if self.span_classes else set()
            if "num" in classes or "designator" in classes:
                self._space()
            elif "heading" in classes:
                self._blankline()
        elif tag == "p":
            self._blankline()
        elif tag in {"ul", "ol"}:
            if self.list_stack:
                self.list_stack.pop()
            self._blankline()
        elif tag == "li":
            self.in_li = False
            self._newline()
        elif tag == "tr":
            self.parts.append(" |")
            self._newline()
        elif tag == "table":
            self.in_table = False
            self._blankline()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        # Preserve wording while collapsing layout whitespace introduced by HTML.
        text = re.sub(r"[ \t\r\f\v]+", " ", data.replace("\n", " "))
        if not text:
            return
        if text.isspace():
            self._space()
            return
        self.parts.append(text)

    def _space(self) -> None:
        if self.parts and not self.parts[-1].endswith((" ", "\n")):
            self.parts.append(" ")

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def _blankline(self) -> None:
        text = "".join(self.parts)
        if text and not text.endswith("\n\n"):
            if text.endswith("\n"):
                self.parts.append("\n")
            else:
                self.parts.append("\n\n")

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = html.unescape(text)
        lines = []
        for line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line.startswith("## §"):
                line = line.replace("   ", " ")
            lines.append(line)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text + "\n"


@dataclass
class ConvertedPage:
    id: str
    source_html: Path
    output_markdown: Path
    title: str
    source_url: str
    chars: int
    word_count: int
    status: str


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def convert_page(row: dict, output_dir: Path) -> ConvertedPage:
    page = ROOT / "data/raw/lii/pages" / f"{row['id']}.html"
    if not page.exists():
        raise FileNotFoundError(page)
    raw = page.read_text(encoding="utf-8", errors="replace")
    extractor = CoreHtmlExtractor()
    extractor.feed(raw)
    title = normalize_title("".join(extractor.title_parts))
    core = "".join(extractor.section_parts)
    status = "ok"
    if not core.strip():
        status = "missing_section"
        core = raw
    parser = MarkdownTextParser()
    parser.feed(core)
    body = parser.markdown()
    frontmatter = {
        "id": row["id"],
        "source": row.get("source", "cornell_lii"),
        "url": row.get("url", ""),
        "title": title,
        "conversion": "lii_core_html_to_markdown_v2",
        "wording_policy": "legal text extracted without paraphrase; whitespace and Markdown structure normalized",
    }
    out = output_dir / "pages" / f"{row['id']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    fm = (
        "---\n"
        + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in frontmatter.items())
        + "\n---\n\n"
    )
    heading = f"# {title}\n\n" if title and not body.lstrip().startswith("#") else ""
    text = fm + heading + body
    out.write_text(text, encoding="utf-8")
    return ConvertedPage(
        id=row["id"],
        source_html=page,
        output_markdown=out,
        title=title,
        source_url=row.get("url", ""),
        chars=len(text),
        word_count=len(re.findall(r"\w+", body)),
        status=status,
    )


def load_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_manifest(rows: Iterable[ConvertedPage], output_dir: Path) -> None:
    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda r: r.id):
            f.write(
                json.dumps(
                    {
                        "id": row.id,
                        "source_html": str(row.source_html.relative_to(ROOT)),
                        "output_markdown": str(row.output_markdown.resolve().relative_to(ROOT)),
                        "title": row.title,
                        "url": row.source_url,
                        "chars": row.chars,
                        "word_count": row.word_count,
                        "status": row.status,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    args.output = args.output.resolve()
    rows = load_manifest(args.manifest)
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["id"] in wanted]
        missing = wanted - {r["id"] for r in rows}
        if missing:
            raise SystemExit(f"unknown ids: {sorted(missing)}")
    converted = [convert_page(row, args.output) for row in rows]
    if args.write_manifest:
        # Regenerate manifest from current output metadata for this invocation.
        write_manifest(converted, args.output)
    for row in converted:
        print(
            json.dumps(
                {
                    "id": row.id,
                    "status": row.status,
                    "words": row.word_count,
                    "output": str(row.output_markdown),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
