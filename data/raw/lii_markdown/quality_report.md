# LII HTML to Markdown 转换质量报告

## 结果

- 转换页数：34 / 34
- 输出目录：`data/raw/lii_markdown/pages/`
- 映射清单：`data/raw/lii_markdown/manifest.jsonl`
- 转换脚本：`scripts/convert_lii_html_to_markdown.py`
- nested Codex 分片运行：`.codex/experiments/lii_html_to_md/run-20260617-005359/`

## 覆盖范围

- CFR：28
- U.S. Code：3
- Texas regulations：3

## 父进程质量检查

已通过以下检查：

- Markdown 文件数量与 `data/raw/lii/manifest.jsonl` 完全一致。
- `manifest.jsonl` 有 34 行，所有 source HTML 都有对应 Markdown。
- 每个 Markdown 文件都有 YAML frontmatter、`id`、`url`、`title`。
- 未发现 LII 页面 chrome 关键词：`Global site tag`、`Support Us`、`Skip to main content`、`googletag`、`cookieconsent`。
- 未发现残留 `<script>`、`<html>`、`<body>`。
- 对可结构化抽取的页面，Markdown 与抽取核心 HTML 的 token 覆盖率为 100%。
- 已修复 state regulations / appendix 页面结构差异，避免 fallback 到整页 HTML。
- 已修复 U.S. Code heading/chapeau 粘连问题，例如 `reportsIf`、`GuidelinesThe`。

## 统计

- 最小词数：143
- 最大词数：10924
- 平均词数：1757.7
- 中位词数：1005.0

## 使用约束

这些 Markdown 是 source material 的结构化副本，不是 domain policy card。后续应从这些 Markdown 中再抽取：

- policy card
- agent-facing policy brief
- tool workflow mapping
- judge rubric
- citation/source reference

转换过程不应被视为法规解释或合规意见。
