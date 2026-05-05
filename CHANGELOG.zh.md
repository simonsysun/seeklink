# 更新日志

[English](CHANGELOG.md) · [中文](CHANGELOG.zh.md)

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.6.0] - 2026-05-05

### 变更
- `seeklink search --rerank-k auto` 现在会对一般中文技术查询使用中等
  reranker 预算，只把最深预算保留给过滤搜索和 chunk / 向量索引类中文查询，
  从而在保持捆绑盲测质量门槛的同时降低可选 MLX reranker 延迟。
- 可选的 MLX Qwen3 reranker 现在会在可用时尝试使用两个 token 的
  embedding-head 评分路径，在支持的 MLX 模型对象上避免生成完整词表
  logits；如有需要仍可通过 `SEEKLINK_RERANK_SCORING=legacy` 回退到旧路径。
- 在 README 中新增可直接复制给 agent 的配置说明，并在 `llms.txt` 中强化本地 Markdown 笔记库检索的发现提示。
- 扩展 PyPI 关键词，覆盖 agent、本地搜索、Markdown 搜索和 llms.txt 发现路径。
- `seeklink search` 和单文件 `seeklink index` 现在支持 `--no-daemon`；
  脚本也可以用 `SEEKLINK_NO_DAEMON=1` 禁用守护进程，获得确定的冷启动行为。
- 新增 `seeklink doctor` / `seeklink doctor --json`，用于轻量检查运行环境
  和索引兼容性，不会下载或加载模型；如果本地 SeekLink 数据库/表结构不存在，
  可能会初始化它们。

### 修复
- 带 folder / tag 过滤的语义搜索现在会为窄范围检索请求足够多的向量候选，避免相关笔记因为未过滤的干扰项占满全局向量 top 200 而被漏掉。

### 开发
- 盲测 runner 现在支持 source 级 folder/tag 过滤、filtered-vector 诊断，以及可选的 answerability 标签，用于检查 top-10 命中是否真的包含 agent 需要的答案文本。
- 发布验证现在包含一个 8 条查询的过滤检索 fixture。在捆绑 fixture vault 上关闭 reranker 时，它的 Recall@10 为 1.000，Answerable@10 为 1.000；常规 22 条查询 fixture 在启用可选 MLX reranker 时仍为 Recall@10 0.985、MRR 0.977、nDCG@10 0.902。
- 将 `tests/blind/results/` 刷新为 v0.6 发布质量快照：v0.5 baseline、v0.6 shipping run、v0.6 filtered fixture，以及 v0.6 expansion reference。

## [0.5.0] - 2026-05-04

### 变更
- 全库 `seeklink index` 现在把进度输出到 stderr，最终的 `Done:` 摘要保留在 stdout（当设置了 `SEEKLINK_VAULT` 时也一样）。
- 全库建索引用更小的批次处理文本嵌入，减少在真实 Markdown 笔记库上出现的长尾卡顿。
- 索引现在记录生成向量时所使用的 embedder、向量维度、距离度量和 chunker 版本；当这些配置发生变化时，全库建索引会重建派生的索引内容。
- 当配置的嵌入维度改变时，全库建索引现在可以重建 sqlite-vec 表，支持在不改变默认 768 维模型的情况下进行自定义 embedder 实验。

### 修复
- 全库建索引不再硬性跳过 `todo/` 或 `archive/` 目录；这些是常见的 PKM 文件夹，除非被隐藏或移除，否则应该被索引。
- 中文疑问式查询现在会在 FTS5 匹配前去除常见的疑问词（如"哪些"、"吗"），让 `卵生动物有哪些？` 这类查询能走 BM25 通道，而不是降级为纯向量检索。
- 走规范化 BM25 路径的中文疑问式查询现在将 BM25 通道作为更轻的排序信号，减少无 reranker 时相邻关键词密集段落的过度提升。
- 在 Python 使用 SQLite 内置 trigram 分词器（而非可选的 jieba FTS 分词器）的环境下，中文疑问式查询现在也能正确保留 BM25 回退路径。
- 跟在带标题段落后的长文本现在按句子边界切分，而不是变成一个超大 chunk，减少生成式/列表密集型 Markdown 中的病态 chunk，同时保留代码块的完整性。
- 抑制了 jieba 导入和词典加载的杂讯输出，让 CLI stderr 专注于 SeekLink 的进度和警告信息。
- `seeklink search` 现在拒绝查询其存储的 embedder/chunker 元数据与当前活跃配置不匹配的向量索引，而不是悄悄将查询向量与不兼容的文档向量混合在一起。
- Reranker 评分改用数值稳定的二分类 softmax，避免极端模型 logits 时产生溢出。

### 开发
- Apple Silicon MLX 重排序现在通过可选的 `seeklink[mlx]` extra 暴露；基础安装不使用 MLX 仍可正常使用。
- `numpy` 现在声明为直接运行时依赖，因为 SeekLink 直接 import 了它。
- PyPI 发布工作流现在会先运行测试套件、检查构建产物，并在验证手动触发的 release tag 后再发布。
- 盲测结果 JSON 现在包含每个查询的 `failure_bucket` 标签和聚合的 bucket 计数，便于在搜索质量工作中区分候选生成失败、rerank 预算失败和 reranker 排序失败。
- 源码 checkout 现在声明了构建后端，`uv sync --dev` 会安装工作树中的 `seeklink` 命令行脚本，而不是在本地验证时回退到过时的全局安装命令。
- 将 `tests/blind/results/` 刷新为仅保留 v0.5 发布质量快照。在 22 条查询的捆绑 fixture 上，启用可选 MLX reranker 时，config A 的 mean Recall@10 为 0.985，MRR 为 0.977，nDCG@10 为 0.901；延迟测量保留在 JSON 结果文件中，因为它们因硬件和负载而异。

## [0.4.0] - 2026-04-29

### 新增
- `seeklink search --json` 和 `seeklink status --json` 为 agent 输出稳定的机器可读 stdout，取代人工阅读的文本格式。
- `seeklink get PATH:LINE -C N` 打印搜索命中行周围的上下文窗口（类似 `grep -C`），同时保留直接文件系统读取和路径转义保护。
- `seeklink search --rerank-k N`、`--rerank-k auto` 和 `--no-rerank` 允许调用方按需在精度和延迟之间做权衡，无需修改全局 reranker 配置。
- 源级元数据搜索现在将 Markdown 标题与笔记标题和 frontmatter 别名一起纳入索引，改进了按章节名搜索的效果，输出格式保持不变。

### 变更
- 全库建索引用按长度排序的批次处理嵌入，取代逐文件处理，提升实际 Markdown 笔记库上的首次建索引吞吐量，同时保持单文件索引行为和现有 SQLite schema 不变。
- MLX reranker 现在将每个段落截断为前 200 token 再评分，降低长 chunk 上的热查询延迟，保留完整的结果预览和 `seeklink get` 输出。
- `seeklink search` 现在默认使用 `--rerank-k auto`，对普通查找使用较小的 reranker 预算，对筛选类和技术性中文查询保留更深的 reranking。
- README、`llms.txt` 和搜索评估文档现在专注于简洁的使用说明、agent 契约和发布质量度量指导，而非产品定位或内部实验笔记。
- 现有索引迁移到 schema v3 并将源文件标记为未处理，以便下一次 `seeklink index` 可以填充标题元数据。

### 修复
- 将 `_sqlite3` 编译为内置模块（SQLite 符号不可见）的 Python 构建，现在回退到 SQLite 内置的 trigram FTS 分词器，而不是让 `sqlitefts` 跨界调用导致段错误。
- 筛选搜索现在在指定的标签/文件夹范围内对 BM25 和源元数据候选结果进行排序，避免本来相关的筛选笔记因为全局第一阶段被未筛选笔记填满而被丢弃。
- 精确的标题、别名和章节标题命中现在在 reranking 后将源元数据胜者保留在第 1 位，而更宽泛的标题匹配仍允许内容 reranker 重新排序。
- `seeklink search --rerank-k N` 现在即使 `N` 小于 `--top-k`，也会正确限制传给交叉编码器的候选数量；剩余结果保持第一阶段 RRF 顺序。
- `seeklink search` 和 `seeklink index` 现在当守护进程的笔记库、embedder 或 reranker 配置不再匹配调用方时会自动重启过时守护进程，避免切换笔记库或模型设置后反复进入冷启动回退。

### 开发
- 新增 CLI 契约冒烟测试，在捆绑的 `tests/corpus` 笔记库上运行文档中描述的 status、index、search、JSON 和 get 工作流。
- 盲测运行器现在记录 nDCG@10、Precision@5、MAP@10、reranker 预算元数据以及 config A 的第一阶段通道诊断信息。
- 将 `tests/blind/results/` 刷新为仅保留 v0.4 发布质量快照。在 22 条查询的捆绑 fixture 上，config A 的 mean Recall@10 为 0.985，MRR 为 0.977，nDCG@10 为 0.901，p95 延迟在本机 Apple Silicon 上为 2124 ms。

## [0.3.2] - 2026-04-23

仓库清理版本。相对于 0.3.1，不存在影响运行时行为的代码变更；此版本从公共仓库中移除了内部研发产物，使仓库读起来像一个已发布工具而非工作日志。

### 变更
- 将 0.3.0 / 0.3.1 的内容合并为单一发布条目（即本条）。早前的条目用流程细节描述了同一份代码，这些细节不属于公开的发布说明。
- 精简 `tests/blind/results/`，仅保留发布质量的 baseline、shipped 和 expansion-reference 测量结果。移除了中间迭代结果。
- 收紧了内部代码注释和测试文档字符串，使其描述当前行为而非产生它的迭代历史。
- README 中的度量数据声明显式标注为"pilot"并附样本量。

### 移除
- `docs/v0.3-plan.md` —— 不应出现在公开仓库中的内部设计草稿。已发布的设计在 README 的"搜索原理"章节和代码注释中有记录。

## [0.3.1] - 2026-04-23

### 新增
- **标题门控的 rerank 混合。** 当标题/别名通道在 rerank 候选池中产生高置信度匹配时，SeekLink 将归一化的第一阶段分数与 reranker 输出混合，保证精确的标题或别名命中（`Zettelkasten`、`RRF`、`遗忘曲线`、`[[别名]]`）保持在排名第 1 位，而不是被内容导向的 reranker 降级。无标题信号时，reranker 完全接管——与 v0.2.x 行为一致——因此差的排序仍然可以恢复。在 22 条查询的捆绑 pilot 上（见 `tests/blind/`）：mean MRR 从 0.932 提升至 0.977，mean Recall@10 不变，无查询质量回归。样本量有意设为 pilot 规模；欢迎更大规模的标注语料。
- **行范围检索。** `SearchResult` 现在携带 1-indexed 的 inclusive `line_start` / `line_end` 字段，通过索引器的 frontmatter 剥离映射回磁盘行号。CLI `search` 打印 `SCORE  PATH:LINE  TITLE`，使 agent 可以将命中行用于精确窗口读取。新增 `seeklink get PATH[:LINE] [-l N]` 命令直接从文件系统执行该窗口读取——无需访问数据库、不依赖守护进程、统一换行符处理、拒绝路径转义。
- **冷启动 `search` 的 reranker 对等。** `seeklink search --vault PATH`（冷启动路径）现在构建 reranker 并将其传递给搜索管道，与守护进程路径一致。在此变更之前，同一个查询根据守护进程是否在运行而返回不同的排序。
- **面向 Agent 的文档。** README 中新增"给 Agent 的说明"章节（最小工作流、输出契约、退出码、查询形状提示、守护进程 JSON 回退）。重写了 `llms.txt` 作为显式的契约文档。
- **盲测框架** `tests/blind/`：32 文件双语（中文/英文）fixture 语料库（`tests/corpus/`），22 条带标注的查询（`tests/blind/queries.yaml`），每次调用冷启动一次的运行器，测量均值和 p95 的 `recall_at_10` / `mrr` / `latency_ms`。三种配置：`A`（当前基线）、`B`（计划中的查询扩展——尚未发布）、`C`（手工构造的扩展，RRF 融合；理论上限）。用于审查本次发布。

### 修复
- **`seeklink get` 末尾换行符的计数问题。** 对以换行符结尾的文件执行 `get FILE:LINE` 时，不再将末尾 `\n` 计为额外的逻辑行。在 5 行文件上执行 `get FILE:6` 会正确输出超出文件末尾的警告，而不是返回空行。
- **纯标题匹配且文件缺失的问题。** 如果搜索结果中出现一个纯标题匹配但对应文件已从磁盘删除，`SearchResult.line_start` / `line_end` 现在保持 `0`，而不是返回一个误导性的 `path:1`。

### 变更
- **`SearchResult` 新增 `line_start` 和 `line_end` 字段（默认 `0`）。** 对现有调用方保持向后兼容；当 `search()` 被调用且传入 `vault_root` 时填充。
- **`FRONTMATTER_RE`** 现在是 `seeklink.ingest` 的公开导出（原为 `_FRONTMATTER_RE`），以便搜索层复用它做偏移映射。带下划线前缀的旧名称仍作为别名保留，以保持向后兼容。

### 开发
- 新增 PyYAML 为开发依赖（盲测运行器需要）。
- 测试套件：185 → 204 个测试（新增 19 个）。

### 延期
- `SEEKLINK_DEBUG=1` 混合分数日志。
- 每个结果的 `mtime > indexed_at` 偏差警告（守护进程路径；冷启动路径已通过 `check_freshness` 全局警告）。
- 基于 llama.cpp / GGUF 的 Linux reranker。

### 取代
- 本次发布取代同日发布的 `0.3.0` 标签，后者代码相同但 README 内容有误（快速开始顺序、延迟数字、`seeklink status` 描述不准确）。如果你固定版本号，请使用 `0.3.1`。

## [0.2.2] - 2026-04-19

### 修复
- PyPI 构建失败，原因是 `pyproject.toml` 同时携带了 SPDX 表达式 `license = "MIT"` 和旧式 `License :: OSI Approved :: MIT License` 分类器，这在 PEP 639 下被现代 setuptools 拒绝。v0.2.1 已在 GitHub 打标签但从未发布到 PyPI。v0.2.2 是该系列中第一个下游用户可以实际 `pip install` 的版本。相对 v0.2.1 无功能性变更——同样的守护进程优先调度、同样的笔记库/模型守卫、同样的元数据。

## [0.2.1] - 2026-04-18 —— **仅标记，未在 PyPI 上发布**

> 此标签已在 GitHub 上发布但从未到达 PyPI：重复的许可证声明导致构建失败。下述所有内容在 **[0.2.2]** 中发布。请勿固定到 `seeklink==0.2.1`。

### 新增
- 守护进程优先的 CLI 调度：`seeklink search` 和 `seeklink index` 在未传 `--vault` 时首次调用自动启动守护进程，之后约 10ms 响应后续调用。传 `--vault` 可强制冷启动。
- `cli_client.call()` 在重用守护进程前检查其笔记库和模型配置（embedder + reranker），防止已绑定到不同 `SEEKLINK_VAULT` / `SEEKLINK_EMBEDDER_MODEL` / `SEEKLINK_RERANKER_MODEL` 的过期守护进程悄无声息地服务或修改错误的数据库。
- `seeklink status` 现在打印配置的 embedder 和 reranker 名称（从环境变量计算，不加载重型模块）。
- PyPI `keywords` 和更丰富的分类器，提升可发现性。
- `Issues` 和 `Changelog` 项目链接。
- README：新增"适用场景"标语、"何时使用/何时不使用"章节、"对比"表格、"支持与限制"矩阵。
- `llms.txt` 位于仓库根目录，便于 LLM 辅助发现。
- `CHANGELOG.md`（即本文件）。

### 变更
- CI 矩阵扩展至 Python 3.11、3.12、3.13、3.14，与声明的分类器支持一致。
- CI "Verify install" 步骤不再用 `|| true` 遮蔽失败；改为对临时笔记库执行 `seeklink --help` 和 `seeklink status`。
- `seeklink status` 现在始终使用冷启动路径。它只读取 SQLite 统计信息和新鲜度，因此通过守护进程路由意味着要完整预热 embedder + reranker（首次使用可能高达 700MB 的下载量），仅仅为了打印几个数字。
- PyPI `description` 重写，显式提及 Obsidian 兼容性。

### 修复
- README 声称守护进程自动启动，但 CLI 实际每次调用都直接走冷启动。行为现在与文档一致。
- 阻止了绑定到不同笔记库或以不同 embedder/reranker 启动的过期守护进程在用户切换 `SEEKLINK_VAULT` / `SEEKLINK_EMBEDDER_MODEL` / `SEEKLINK_RERANKER_MODEL` 后悄无声息地返回错误结果。不匹配时 CLI 回退到冷启动；自动重启的后续跟进已记录在 TODOS.md。

## [0.2.0] - 2026-04-16

### 新增
- Unix socket 守护进程模式，带预先加载的 embedder 和可选的 MLX reranker（`seeklink daemon`）。模型在查询之间常驻内存，实现约 10ms 往返。
- 通过 Qwen3-Reranker-0.6B on MLX（Apple Silicon）的可选交叉编码器重排序。默认启用，通过 `SEEKLINK_RERANKER_MODEL=""` 禁用。
- 新鲜度检查：双向 mtime 扫描，在冷启动的 `status` / `search` 时报告过期、新增和删除的文件。
- 每个查询可通过 `--title-weight` 标志配置标题通道的 RRF 权重。

### 变更
- **CLI 优先架构。** 移除了 MCP 服务器（`seeklink serve`）。所有交互通过 CLI 子命令或 Unix socket 守护进程完成。
- 标题通道默认权重从 3.0 降至 1.5，使无标题内容（日记、日志笔记）能与有标题文章公平竞争。
- 运行时依赖从 6 个精简至 4 个（移除 `mcp`、`watchfiles`）。

### 移除
- MCP 服务器传输。曾经通过 MCP 的 agent 应通过 `subprocess` 调用 CLI，或通过 `seeklink.cli_client` 连接守护进程 socket。

## [0.1.0] - 2026-04-04

### 新增
- 首次公开发布。
- 四通道混合搜索：BM25（FTS5 + jieba）+ 向量（jina-embeddings-v2-base-zh）+ 知识图谱入度 + 标题/别名 FTS，通过倒数排名融合（Reciprocal Rank Fusion）合并。
- SQLite 存储（`.seeklink/seeklink.db`），使用 sqlite-vec 存储 768 维向量，FTS5 用于关键词和标题搜索。
- Wikilink 解析器，处理 Obsidian 风格的 `[[note]]` 和 `[[别名]]` 图谱边关系。
- 原生 CJK 分词，通过 jieba 注册为自定义 FTS5 分词器。
- MCP 服务器传输（`seeklink serve`）—— 于 v0.2.0 移除。

[Unreleased]: https://github.com/simonsysun/seeklink/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/simonsysun/seeklink/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/simonsysun/seeklink/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/simonsysun/seeklink/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/simonsysun/seeklink/releases/tag/v0.3.2
[0.3.1]: https://github.com/simonsysun/seeklink/releases/tag/v0.3.1
[0.3.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.3.0
[0.2.2]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.2
[0.2.1]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.1
[0.2.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.0
[0.1.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.1.0
