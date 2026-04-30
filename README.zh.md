# SeekLink

[English](README.md) · [中文](README.zh.md)

[![PyPI](https://img.shields.io/pypi/v/seeklink)](https://pypi.org/project/seeklink/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Tests](https://github.com/simonsysun/seeklink/actions/workflows/test.yml/badge.svg)](https://github.com/simonsysun/seeklink/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SeekLink 是一个本地运行的语义搜索命令行工具，专为 Markdown 笔记库设计。
它索引一个文件夹里的 `.md` 文件，用关键词 + 向量混合检索找到相关内容，
返回带行号的结果——无论人还是 AI agent，都能用一句简单的 shell 命令精确定位到原文。

它的设计场景是：个人知识库、Obsidian 兼容的笔记 vault、中英文混合笔记、
以及本地 agent 工作流。它也很适合配合 [Andrej Karpathy 的 llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
这类 Markdown wiki 模式使用：agent 可以先搜索已有页面，读取精确的行窗口，
再更新 wiki——整个过程不需要把笔记库上传到任何云端服务。

**一切都在本地运行。** 不需要 API key，不需要云搜索服务，不需要安装 Obsidian 插件。

## 安装

```bash
uv tool install seeklink
# 或者
pip install seeklink
```

## 快速开始

```bash
# 1. 先建索引。
seeklink index --vault /path/to/vault

# 2. 再搜索。
seeklink search "机器学习" --vault /path/to/vault
```

日常使用的话，设置一个默认笔记库会更方便：

```bash
export SEEKLINK_VAULT=/path/to/vault
seeklink index
seeklink search "agent 记忆系统"
seeklink get notes/agent-memory-patterns.md:1 -C 20
```

当设置了 `SEEKLINK_VAULT` 且不传 `--vault` 时，`seeklink search` 和 `seeklink index`
会自动使用一个常驻守护进程（daemon），它把嵌入模型和可选的 reranker 保持在内存里，
避免每次调用都重新加载。`seeklink status` 和 `seeklink get` 始终走冷启动路径：
status 只读 SQLite 元数据，get 直接从磁盘读文件。

## 输出格式

文本搜索输出是稳定的：

```text
  SCORE  PATH[:LINE]  TITLE
           <内容预览，一行，最多 120 字符>
```

- `PATH` 是相对于笔记库根目录的路径。
- `LINE` 是 1-indexed 的行号，指向当前文件中最佳匹配 chunk 的第一行。
- 退出码：成功（包括无结果）为 `0`，笔记库/配置错误或文件缺失为 `1`。
- 分数在同一查询内可用来比较排序，但不要在开启 reranker 和关闭 reranker 的运行之间比较分数。

如果 agent 需要结构化输出，用 JSON 模式：

```bash
seeklink search "agent 记忆系统" --vault PATH --json
seeklink status --vault PATH --json
```

## 常用命令

### 搜索

```bash
seeklink search "查询内容" --vault PATH [选项]
```

选项：

```text
--top-k N          返回结果数量，默认 10
--json             输出一个机器可读的 JSON 对象
--tags TAG [TAG]   按标签筛选（AND 语义）
--folder PREFIX    按笔记库相对路径的文件夹前缀筛选
--rerank-k N|auto  Reranker 候选预算，默认 auto
--no-rerank        本次查询跳过交叉编码器重排序
--title-weight F   覆盖标题/别名/标题通道的权重，默认 1.5
```

### 精确读取

不依赖数据库或守护进程，直接从文件系统读取指定窗口：

```bash
seeklink get notes/spaced-repetition.md
seeklink get notes/spaced-repetition.md:12
seeklink get notes/spaced-repetition.md:12 -l 40
seeklink get notes/spaced-repetition.md:12 -C 20
```

`-l/--lines` 从指定行开始打印 N 行。`-C/--context` 打印指定行前后各 N 行（类似 `grep -C`）。
`../..` 等路径转义会被拒绝。

### 状态

```bash
seeklink status --vault PATH
seeklink status --vault PATH --json
```

Status 显示索引数量、模型名称、SQLite WAL 状态以及文件新鲜度警告。
它不会加载嵌入或重排序模型。

### 索引

```bash
seeklink index --vault PATH
seeklink index path/to/file.md --vault PATH
```

全库索引通过内容哈希跳过未修改的文件。单文件索引则更新指定的一个 Markdown 文件。

### 守护进程

```bash
seeklink daemon --vault PATH
```

通常不需要手动运行。`search` 和 `index` 在合适的时候会自动启动和重启守护进程。
给 `search` 或 `index` 传 `--vault` 会强制走一次性冷启动路径，因为守护进程在启动时就绑定到了一个笔记库。

## 搜索原理

SeekLink 用倒数排名融合（Reciprocal Rank Fusion）将四个通道合并：

| 通道 | 用途 |
|---|---|
| BM25 / FTS5 | 精确词汇、代码术语、缩写、中日文词汇匹配 |
| 向量搜索 | 跨不同措辞的语义匹配 |
| 标题 / 别名 / 章节标题 | 精确的笔记和章节查找 |
| Wikilink 入度 | 基于已有 `[[链接]]` 的图谱质量信号 |

默认嵌入模型是 `jinaai/jina-embeddings-v2-base-zh`，通过 `fastembed` 运行。
中日文全文搜索使用 jieba 分词器注册为 FTS5 自定义分词器；如果本地 Python/SQLite
环境无法安全注册（例如静态编译的 SQLite），SeekLink 会自动降级为 SQLite 内置的
trigram 分词器，而不是崩溃。

在 Apple Silicon 上，SeekLink 可以用 `mlx-community/Qwen3-Reranker-0.6B-mxfp8`
对候选结果进行重排序。Reranking 是本地且可选的——用 `--no-rerank` 跳过单次查询，
或设置 `SEEKLINK_RERANKER_MODEL=""` 全局禁用。

## Frontmatter

Markdown 的 YAML frontmatter 是可选的。如果有，SeekLink 会用它来做标签和别名：

```yaml
---
tags: [ai, memory]
aliases: [LLM memory, agent memory]
---
```

- `tags` 支持筛选搜索：`seeklink search "memory" --tags ai`
- `aliases` 在搜索中被索引，同时用于解析 wikilink 引用

## 存储

SeekLink 在笔记库内写入一个 SQLite 数据库：

```text
/path/to/vault/.seeklink/seeklink.db
```

数据库包含源文件元数据、chunk 文本、FTS5 表、sqlite-vec 向量以及 wikilink 图谱。
删除 `.seeklink/` 目录后重新运行 `seeklink index` 即可重建。

## 支持矩阵

| 维度 | 状态 |
|---|---|
| Python | 3.11、3.12、3.13、3.14 |
| 操作系统 | macOS 和 Linux |
| Windows | 不作为一等路径支持 |
| 文件格式 | Markdown `.md` |
| 笔记库类型 | 普通文件夹或 Obsidian 兼容 vault |
| 中日文 | 原生支持（jieba），静态 SQLite 环境下自动降级为 trigram |
| Reranker | Apple Silicon 上通过 MLX 可用；其他平台自动禁用 |
| 守护进程 | 一台机器一个笔记库 |

## 不适用的场景

- 托管式或多用户同步搜索。
- 未经转换的非 Markdown 来源。
- GUI 或 Obsidian 插件。
- 百万级笔记的亚毫秒搜索。
- 云端嵌入或云端重排序 API。

## 给 Agent 的说明

Agent 可以通过普通的子进程调用来使用 SeekLink：

```bash
seeklink status --vault PATH
seeklink index --vault PATH
seeklink search "查询" --vault PATH --json
seeklink get PATH:LINE -C 20 --vault PATH
```

对于高频调用场景，守护进程在 Unix socket（`~/.rhizome/seeklink.sock`）上暴露了一个
length-prefixed JSON 协议。大多数 agent 应该优先使用 CLI 的 JSON 输出，除非确实需要
socket 级别的延迟优势。

详见 [llms.txt](llms.txt) —— 一份精简的 agent 契约文档。

## 搜索质量评估

搜索质量测试位于 `tests/blind/`，评估方法见 [docs/blind-test.md](docs/blind-test.md)。
公开发布的质量声明应该有配套的 fixture 查询结果支撑，或者清晰标明为私有笔记库的测量数据。

## 贡献

```bash
git clone https://github.com/simonsysun/seeklink
cd seeklink
uv sync --dev
uv run python -m pytest tests/ -q
```

保持运行时依赖精简，保持公开文档面向用户，对用户可见的变更在 `CHANGELOG.md` 中添加条目。

## 许可证

MIT
