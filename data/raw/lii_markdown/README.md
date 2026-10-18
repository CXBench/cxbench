# Cornell/LII 核心法规 Markdown 语料

本目录是 `data/raw/lii/pages/*.html` 的核心法规正文 Markdown 转换结果，用于后续构建 domain policy、policy brief、policy read/search 工具和 benchmark generation。

转换原则：

- 只抽取 LII 页面中的法规/法条正文区域，去除导航、广告、页脚、脚本等页面 chrome。
- 不总结、不改写、不翻译原合规流程措辞；只做 HTML 解码、Markdown 结构化和空白规范化。
- 每个 Markdown 文件保留 YAML frontmatter，包含 `id`、`url`、`title` 和转换版本。
- `manifest.jsonl` 记录每个源 HTML 与输出 Markdown 的映射、标题、URL、字符数和词数。
- `domain_index.json` 将 Markdown source 按 cxbench domain 分类；同一个 source 可以出现在多个 domain 中。

生成命令：

```bash
python3 scripts/convert_lii_html_to_markdown.py --output data/raw/lii_markdown --write-manifest
```

质量检查要点：

- Markdown 文件数量应与 `data/raw/lii/manifest.jsonl` 一致。
- 每个文件应包含 frontmatter、source URL、title 和正文。
- 不应包含 `Global site tag`、`Support Us`、`Skip to main content`、`googletag` 等页面 chrome。
- 后续如果新增 LII HTML，需要重新运行转换并检查 `manifest.jsonl`。
- 新增或删除 source 后，需要同步更新 `domain_index.json`，保证正式评测的 system prompt 能拿到完整 domain-level context。
