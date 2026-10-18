# 中文使用说明

这份代码整合了数据合成、六个领域的模拟业务系统、新版 120 任务实验、失败重试、
断点恢复、同轨迹重评和离线统计。仅保留 `benchmarks/cxbench-120/` 这一套新版任务。

## 安装与离线检查

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录运行：

```bash
uv sync --frozen --python 3.12
uv run --frozen pytest -q
./run.sh --dry-run
./run.sh --smoke-test --dry-run
```

测试和 dry-run 不调用模型，不需要密钥。推荐从源码目录运行实验；wheel 只包含
`cxbench` 核心运行库及政策语料，不包含实验入口和 benchmark 任务文件。

## 配置与真实烟测

```bash
cp .env.example .env
```

在 `.env` 中填入自己的 `OPENAI_API_KEY`。不要把密钥、`.env` 或运行日志提交到公开仓库。
默认连接 `https://api.openai.com/v1`，使用 Responses API；流式响应默认开启，
可通过 `OPENAI_RESPONSES_STREAM` 配置。

```bash
./run.sh --smoke-test --run-id openai-smoke
```

默认 agent、user simulator、judge 都使用 OpenAI 的 `gpt-5.6-sol`。
这一步有真实 API 调用和费用，结果保存在 `runs/main_experiment/openai-smoke/`。
运行成功不等于合规评分通过；`ok=true` 且 verdict 为 `fail` 可以是有效实验结果。

如需替换模型，在 shell 环境中设置 `CXBENCH_AGENT_MODEL`、`CXBENCH_USER_MODEL`、
`CXBENCH_JUDGE_MODEL`。后两者使用 `openai:模型名`。模型必须在对应 OpenAI 账户上可用。

## 全量实验与恢复

```bash
./run.sh --run-id openai-120 --concurrency 1
./run.sh --resume --run-id openai-120 --concurrency 1
```

烟测恢复时保留 `--smoke-test`：

```bash
./run.sh --smoke-test --resume --run-id openai-smoke
```

恢复使用原任务选择、模型和运行目录，校验任务哈希。已完成的失败 verdict 也会复用，
不会为了取得 pass 而重跑；judge 的执行错误在同一份已完成 trace 上重试。
默认任务执行超时为每次尝试 1,200 秒，执行与判分错误的重试次数没有上限，
并非整个批次有 1,200 秒上限。持续错误时应中断并检查日志。

`--runs-dir /path/to/results` 可将结果写到仓库外。
唯一的 `benchmarks/cxbench-120/` 覆盖六个领域，每个领域 20 任务。
任务文件保持新版实验的内容，历史来源通过清单中的哈希记录。

## 数据合成

离线示例：

```bash
uv run --frozen cxbench generate \
  --config configs/benchmark_generation/smoke.yaml \
  --executor local --out generated/local-smoke
uv run --frozen cxbench accept \
  --run generated/local-smoke --dest generated/accepted
```

该示例基于完整模板，验证提取规则、构造任务、合并评分规范、校验和接受的流程，
不应宣称是模型合成的新数据。只有情景提示、没有完整 `task` 的矩阵不能使用 local 模式。

真正的新任务合成需要另外安装并登录 Codex CLI：

```bash
uv run --frozen cxbench generate \
  --config configs/benchmark_generation/banking.yaml \
  --executor codex --workers 1 --out generated/banking-new \
  --review-model openai:gpt-5.6-sol
```

Codex CLI 的认证独立于本项目 `.env`；示例中的质量审查使用 OpenAI。
每次生成使用新的输出目录，避免覆盖旧候选。静态校验通过之后仍需审查语义与来源，
并实际执行候选任务。[完整合成说明](generation.md)包含配置格式、反馈修复与语料更新方法。

## 结果解释

正式结果必须使用 `hybrid`：同时满足确定性的硬规则与 LLM judge 语义检查。
`rule` 只适用于烟测、CI 和消融；`llm` 单独使用只是诊断视角。
最终任务成功还要求业务 outcome 正确。运行目录中：

- `manifest.json`：任务哈希、模型与运行配置。
- `results.jsonl`、`metrics.json`、`summary.md`：批次结果。
- `task_runs/`：每项任务的 trace、状态、评分和报告。
- `paper_scores/`：完整 hybrid 批次的离线统计。

[实验协议](experiments.md)说明正式统计口径、重试和消融；
[配置指南](providers.md)说明其他服务商与常见错误；
[验证记录](validation.md)区分本次实际验证和尚未运行的内容。

## 发布与维护

依赖锁文件、测试、CI、任务清单和来源说明都应保留。不要发布本地密钥、虚拟环境、
缓存、私有实验结果或论文草稿。修改任务或评分协议时使用新版本并重新验证。

代码许可状态见 [LICENSING.md](../LICENSING.md)，第三方政策语料说明见
[NOTICE.md](../NOTICE.md)。在发布前由代码权利人确认许可证；第三方文本的权利归属
不能直接用项目代码许可证代替。
