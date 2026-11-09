# Changelog

本文件记录面向使用者的可见变更。格式遵循 Keep a Changelog。

## [0.4.0] - 2026-11-09

### Added
- **自定义归因标签（M4-1）**：`attribution/labels.py` 提供
  `AttributionLabelSpec`、`LabelRegistry` 与 `@attribution_label` 装饰器。标签
  自带 `facet` + `field_path` 默认落点，并可选注册分类器：分类器按 `priority`
  作用于既有失败信号（例如把状态毫无变化的 `RESULT_MISMATCH` 重命名为
  `NOOP_ACTION`），首个命中者胜出，异常只记 warning。内置 7 个标签自动预注册，
  自定义标签不能与内置同名。profile 新增 `plugins.labels`。
- **opik_optimizer 原生多轮适配（M4-2）**：`optimize/opik_native.py` 通过
  `MetaPromptOptimizer` 的 **agent + metric** 扩展点优化 system prompt——
  agent 用优化器给出的 prompt 重建 `HarnessState` 并跑完整多轮 executor +
  checker，metric 读取通过/失败返回 1.0/0.0，因此搜索对象是真实端到端任务结果，
  而不是单次 LLM 调用。`opt.opik.backend: auto | native | metaprompt`（默认
  `auto`，装了 `opik_optimizer` 即用原生），预算由 `max_trials` / `sample_size`
  控制；无 Opik 服务端时使用进程内数据集。任一步失败按
  native → MetaPrompt → 规则模板 依次回退。
- **示例与测试**：`examples/plugins/labels.py`（`NOOP_ACTION` 标签）、
  `tests/test_opik_native.py`（真实 `opik_optimizer` 评测集成 + 契约/回退测试）、
  ADR-0006。

### Changed
- `FailureSignal` / `Attribution` / `AttributionGroup` / `ToolIssue` 的 `label`
  由枚举改为字符串；传入 `AttributionLabel` 成员时自动归一化为其值，向后兼容。
  自定义标签因此可直接进入归因结果、报告与假设生成 prompt。
- `Attributor` 接受 `LabelRegistry`，内置标签映射改由标签注册表提供；显式归因
  规则仍然先于标签默认落点执行。
- `harnessloop plugins` 现在同时列出内置/自定义标签。
- `optimizers` extra 增加 `opik_optimizer>=3.2`。

## [0.3.0] - 2026-11-09

### Added
- **归因规则插件（M3-1）**：`attribution/plugins.py` 提供 `AttributionRule`
  注册表与 `@attribution_rule` 装饰器，规则按 `priority` 先于内置标签映射执行，
  返回 `(facet, field_path)` 或 `None`；不可解析路径与异常只会被跳过，不会中断闭环。
- **EditOp 插件（M3-2）**：`optimize/plugins.py` 提供 `EditOpPlugin` 注册表与
  `@edit_op` 装饰器；`EditOp.kind` 放宽为字符串并新增 `payload` 字段承载结构化
  参数，自定义操作经 `HarnessState.apply_edit(edit, registry=...)` 分发。假设
  生成器会把已注册 op 名并入 `direction` 枚举。
- **插件加载器**：`plugins.py` 支持 `pkg.module[:attr]` 与 `file.py[:attr]` 两种
  spec，带扩展 API 主版本校验（`PluginVersionError`）；profile 新增
  `plugins.attribution` / `plugins.edit_ops`，相对路径按 profile 目录解析。
- **Trace 存储后端（M3-3）**：`store/trace_store.py` 定义 `TraceStore` 协议，
  提供 JSONL（默认）与 SQLite 实现；`store.backend` / `store.sqlite_name` 配置。
  评估结论见 `experiments/trace_storage/RESULTS.md`（≤1k 条 JSONL 无感，
  >5k 条建议 SQLite）与 ADR-0005。
- **MCP 执行器（M3-4）**：`runtime/mcp_executor.py` 通过官方 MCP SDK 的 stdio
  传输执行工具调用（`executor.kind: mcp`，`executor.mcp.command/cwd/env/timeout_s`），
  每个任务一个会话、与任务在同一协程内关闭；工具名需与 harness 一致。
- **文档与示例（M3-5）**：`docs/plugins.md` 插件开发指南、`docs/README.md` 文档
  索引、`examples/plugins/` 可直接运行的插件 demo（自定义归因规则 + `set_enum`
  编辑操作 + profile）。
- **CLI**：`harnessloop plugins --config <yaml>` 打印当前注册的规则与编辑操作。

### Changed
- `EditOp.kind` 由 `Literal` 改为 `str`（内置仍是 `add/remove/rewrite`，
  自定义值需先注册）；新增 `EditOp.payload`。
- `RuleTemplateMutator` 接受可选 `EditOpRegistry`；`Attributor` 接受可选
  `AttributionRegistry`。
- `LoopRunner` 的产物写入改经 `TraceStore`；`meta.json` 记录 executor / store /
  plugins 配置。
- 新增可选 extra `mcp`（`mcp>=2.0`）。

## [0.2.0] - 2026-11-09

### Added
- **API 线（M2-1/2/3）**：`kind: api` 任务格式（指令 + 初始状态 S0 + 状态断言）、
  内置 todo 域 Mock API 工具运行时（`runtime/api_runtime.py`）、`APIChecker`
  （断言求值 + 可选 `response_schema` JSON Schema 校验）、结构化状态差分
  （`eval/state_diff.py`，S0→S1 diff 进入归因证据与报告）。示例
  `examples/api_state/`（12 条任务）。断言路径支持 `tasks[0]`、
  `tasks[title=写周报].status` 等形式，ops：eq/ne/exists/not_exists/contains/
  length_eq/length_ge/length_le。
- **可插拔优化后端（M2-4/5）**：`opt.mutator: rule | gepa | opik`。
  `gepa` 用真实 gepa 引擎做有预算上限的反射变异搜索（`opt.gepa.*`）；
  `opik` 为 MetaPrompt 风格候选生成 + 抽样迷你评测（`opt.opik.*`）；
  不可用或失败时回退规则模板（ADR-0004）。
- **PydanticAIExecutor（M2-8）**：`runtime/pydantic_ai_executor.py`，以
  pydantic-ai 实现 `run(task, harness) -> Trajectory` 契约，验证执行器契约
  可框架无关迁移（工具 schema 原样注册、同一 validate_tool_call 规则集）。
- **只读迭代报告页（M2-7）**：`harnessloop ui --runs-dir <dir>`（Gradio），
  展示 pass@1 曲线、指标/决策表、逐轮归因分布、Harness Diff 与 report.md。
- **对比实验（M2-6）**：`experiments/rule_vs_reflective/`，规则模板 vs GEPA
  反射变异的确定性量化对比（好/差假设两场景，区分闭环可见与全局 tokens）。

### Changed
- `Task` 新增 `initial_state`；`Trajectory` 新增 `final_state`（API 线 S1）；
  `CheckResult` 新增 `state_diff`。
- LoopRunner/CLI 支持 `dataset.kind: api`；API 任务要求 `expected.assertions`。
- 新增核心依赖 `jsonschema`（response_schema 校验）；新增 extras：
  `optimizers`（gepa）、`executors`（pydantic-ai）、`ui`（gradio）。

## [0.1.0] - 2026-11-08

### Added
- M0/M1 全量：数据模型与配置、LLM 客户端抽象（LiteLLM + 自封装 asyncio
  工具调用循环）、SQL 线评测（sqlglot AST diff + aiosqlite 只读执行 + 结果集
  比对）、归因器（7 类标签 + facet/field_path）、假设生成（JSON Schema 强约束
  + 重试）、规则模板变异器、Threshold/Pareto Gate、存储布局与不可变快照、
  Markdown 迭代报告、CLI（run/eval/report/diff/init）。
- 示例 `examples/sql_golden/`（12 题 SQL 金标准 + SQLite 环境）。
- 模型价格表与 `.env` 覆盖（`HARNESSLOOP_INPUT_PRICE_PER_1M` /
  `HARNESSLOOP_OUTPUT_PRICE_PER_1M`），预算门控基于真实成本。
- 真实闭环验证：deepseek-flash 3 轮 66.7% → 100%，总成本 ≈ $0.04。
