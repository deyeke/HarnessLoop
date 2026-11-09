# HarnessLoop / 驭环 —— 项目计划与技术选型说明书

| 项目 | 内容 |
| --- | --- |
| 项目代号 | HarnessLoop（中文名：驭环） |
| 文档类型 | 项目说明书（项目计划 + 技术选型） |
| 版本 | v0.4（M0/M1/M2/M3/M4 已交付，2026-11-09） |
| 状态 | v0.1 / v0.2 / v0.3 / v0.4 均已实现 |
| 适用范围 | v0.1 至 v0.4 全阶段 |
| 文档来源 | `leader.txt`（定位、选型、需求、风险） |

---

## 1. 项目概述

### 1.1 一句话定义

HarnessLoop 是一个**库优先、可嵌入、以“失败归因驱动修改”为差异化**的轻量 Agent 闭环优化工具：冻结模型，只自动调优 Harness 层（Prompt 与 Tool Schema），用“评测 → 归因 → 假设 → 修改 → 复测”的闭环持续提升 Agent 任务成功率。

它不是又一个全功能 Agent 平台，也不是训练框架。

### 1.2 问题背景

当前 Agent 效果调优存在三个断层：

1. **调优动作不可解释**：工程师看到的是“成功率 62%”，但不知道失败集中在 Prompt 的哪一句、Tool Schema 的哪个字段，只能靠经验盲改。
2. **评测与修改脱节**：评测框架（pytest、Opik、LangSmith 等）能给出分数，但不产出“下一步该改什么”的可执行假设。
3. **自改进方案过重**：已有的自改进工作（SIA、ClawLoop）同时动模型权重或做任务合成，依赖 GPU 训练设施，复现成本和参与门槛高。

HarnessLoop 的判断是：**Agent 的大量失败并不需要改模型，而是 Harness 描述得不够好**。把这部分交给一个可解释、可复现、无 GPU 依赖的闭环，就能覆盖相当一部分真实收益。

### 1.3 核心闭环

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 1. 评测   │ → │ 2. 归因   │ → │ 3. 假设   │ → │ 4. 修改   │ → │ 5. 复测   │
│ Eval     │   │ Attribute│   │ Hypothesis│  │ Edit     │   │ Re-eval  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘
      ↑                                                           │
      └──────────────── 接受 / 回滚，保留版本链 ←──────────────────┘
```

闭环的动作空间被严格限制在 Harness 层，这是项目“轻”的根本原因。

### 1.4 定位与边界（核心约束）

| 类别 | 允许 | 说明 |
| --- | --- | --- |
| ✅ 可修改 | `system prompt` | 指令、约束、输出格式强调 |
| ✅ 可修改 | `few-shot 示例` | 增删改示例 |
| ✅ 可修改 | `tool description` | 工具用途、调用时机描述 |
| ✅ 可修改 | `tool JSON Schema` | 参数名、参数描述、枚举值、必填项 |
| ✅ 可修改 | `输出格式约束` | 结构化输出要求 |
| ❌ 不修改 | 模型权重 | 不做 SFT / RL / LoRA |
| ❌ 不修改 | Agent 运行时拓扑 | 不自动改写执行图、不增删节点 |
| ❌ 不引入 | 外部教师模型做任务合成 | 不做数据合成放大 |
| ❌ 不引入 | GPU 训练设施 | 全流程仅需 LLM API + CPU |

> **范围红线**：一旦允许“自动改写 Agent 架构”或“自动微调”，项目立刻变重，失去轻量与可复现的立身之本。该约束写入 `docs/adr/0001-scope-boundary.md`，任何突破边界的 PR 需架构评审。

### 1.5 与相关工作对比

| 项目 | 优化对象 | 是否需要训练 | 与 HarnessLoop 的关系 |
| --- | --- | --- | --- |
| **SIA** | Harness + 模型权重 | 是（含权重更新） | 同时更新 harness 与 weights；HarnessLoop 只取其 harness 快照目录组织方式作为参考 |
| **ClawLoop** | 任务合成 + SFT/RL | 是 | 走训练循环；HarnessLoop 明确不进入 |
| **Opik Agent Optimizer** | MetaPrompt / GEPA / HRPO，支持 function-calling schema | 否 | **作为可插拔优化后端复用，不重复造算法** |
| **LangGraph / Agent 平台** | 编排与运行时 | 否 | 不做平台、不做图编排；仅作为可选执行适配器 |
| **HarnessLoop** | Prompt + Tool Schema | 否 | 差异化在 **归因层**：把失败轨迹映射到具体的 harness 可编辑面 |

**核心差异化**：不发明优化算法，把优化器当后端；自己的核心贡献是**可解释的失败归因链**——从失败轨迹指向“该改 prompt 的哪一句 / 该改 schema 的哪个字段”。

---

## 2. 技术选型

### 2.1 选型原则

1. **库优先**：核心循环是一个可 `pip install` 的 Python 包，CLI 与 Web 都是薄壳。
2. **最薄的抽象**：执行层不绑定任何 Agent 框架，循环语义必须对使用者透明可读。
3. **复用成熟信号源**：SQL 解析用事实标准 sqlglot，评测底座用 pytest，优化算法用 GEPA/Opik，不自研。
4. **可解释优先于自动化程度**：宁可选规则模板 + 结构化假设，也不选不可解释的端到端黑盒优化。
5. **v0.1 零基础设施**：不引入数据库、不引入消息队列、不引入向量库，文件系统足够。

### 2.2 执行层：Agent 运行与工具调用

| 选项 | 适合场景 | 注意事项 |
| --- | --- | --- |
| HarnessX | 需要 provider-agnostic、异步、typed tool registration、内置评估钩子 | 库优先、无图编排，MCP 原生支持，适合作为后续适配器 |
| Pydantic AI | 类型安全、原生 async、最小依赖 | 适合作为 SQL/API Agent 的轻量运行时 |
| Agno | 全异步、轻量运行时 | async-first，比 LangGraph 更贴合“单 Agent 多轮”场景 |
| 自封装 LiteLLM | 追求极致可控、不想引入框架 | 工作量更大，但循环语义最透明 |

**决策（v0.1）**：`LiteLLM + 自封装 asyncio 执行器`。

**理由**：
- HarnessLoop 的循环逻辑不能被任何 Agent 框架的抽象绑架；
- LiteLLM 已经解决多 provider 归一化（OpenAI / Anthropic / 本地模型），这是唯一必须复用的部分；
- 执行器接口只有一行，后续接 HarnessX / Pydantic AI 只是写适配器。

**统一执行契约**：

```python
from typing import Protocol

class Executor(Protocol):
    async def run(self, task: Task, harness: HarnessState) -> Trajectory:
        """在给定 harness 下执行单个任务，产出完整轨迹。"""
        ...
```

| 阶段 | 实现 | 说明 |
| --- | --- | --- |
| v0.1 | `LiteLLMExecutor` | 原生 asyncio + 自写 tool-call 循环 |
| v0.2 | `PydanticAIExecutor` | 作为可选适配器，验证契约稳定性 |
| v0.3 | `MCPExecutor` | MCP（stdio）原生执行工具调用，HarnessX 等框架经 MCP 接入 |

### 2.3 评测层：SQL 与 API 两条线

#### 2.3.1 SQL 线

| 能力 | 选型 | 用途 |
| --- | --- | --- |
| 解析与验证 | **sqlglot** | 多方言解析、AST diff、transpile；做语法校验与结构差异归因（FROM 变更、JOIN 数量变化、WHERE 条件增删） |
| 执行验证 | **sqlite3 + aiosqlite** | v0.1 覆盖大部分评测场景，零外部依赖 |
| 多方言执行（可选） | **sqlspec** | 基于 sqlglot 的执行层，支持 PostgreSQL / SQLite / DuckDB，按需引入 |
| 结果比对 | 自研 `result_compare` | **优先执行结果集比对**：考虑行顺序、NULL 语义、浮点容差；不做字符串比对 |
| 语义等价补充 | **LLM-as-judge** | 仅用于“结果等价但写法不同”的兜底；必须校准后启用（见 §9 风险二） |

**结果比对规则（默认）**：

| 维度 | 默认策略 | 可配置 |
| --- | --- | --- |
| 行顺序 | 视为无关（ORDER BY 任务除外） | `order_sensitive` |
| NULL | NULL 一律按三值逻辑等价比较 | 固定 |
| 浮点 | 相对误差 `1e-6` | `float_tolerance` |
| 列顺序 | 视为无关 | `column_order_sensitive` |
| 列名 | 视为无关（只比列数 + 值） | `compare_column_names` |
| 多余行 | 严格失败 | 固定 |

#### 2.3.2 API 线

API Agent 没有统一的“黄金结果集”，因此采用**状态差分评测**：

```
初始环境状态 S0  →  Agent 执行工具调用  →  最终状态 S1
判定：assert(S1, expected_state_assertions)
```

| 阶段 | 能力 | 参考 |
| --- | --- | --- |
| v0.1 | JSON Schema 校验 + 关键字段断言 | 最简可用 |
| v0.2 | 状态差分（记录 S0/S1 并做结构化 diff） | Agent-Diff、VAKRA 思路 |
| v0.3 | 可插拔状态比较器 | 插件接口 |

#### 2.3.3 评测框架底座

**决策**：`pytest + pytest-asyncio` 作为评测底座，**不自造评测 DSL**。

- `parametrize` 天然表达“数据集 × harness 变体”的评测矩阵；
- `checkagent` 等插件已提供 `@pytest.mark.agent_eval` 与 LLM-judged 层叠测试模式；
- 每次“假设验证”直接生成/运行一组 pytest 用例，评测结果即测试报告；
- 好处：用户已有的 pytest 生态、CI 集成、失败重跑能力全部复用。

```python
# 评测矩阵的典型形态
@pytest.mark.parametrize("task", load_dataset("golden_sql.jsonl"))
@pytest.mark.parametrize("harness", load_harness_variants(["iter_0", "iter_1"]))
async def test_sql_task(task, harness):
    result = await evaluate_one(task, harness)
    assert result.outcome == "pass", result.attribution.evidence
```

### 2.4 优化层：假设生成与修改

这是 HarnessLoop 的核心差异化，也是最需要克制的地方。**不从零实现通用 prompt 优化器**，采用四层分工：

| 层 | 做什么 | 技术方案 | 归属 |
| --- | --- | --- | --- |
| **归因器** | 从失败轨迹提取结构化信号：解析失败 / 执行报错 / 结果不匹配 / 参数缺失 / 选错工具 | sqlglot AST diff、执行异常分类、tool-call 参数校验日志 | **自研（核心资产）** |
| **假设生成器** | 根据归因聚合，生成最小可证伪假设 | 单次 LLM 调用 + JSON Schema 结构化输出 | 自研逻辑 + LLM |
| **修改执行器** | 把假设翻译成 harness 的具体修改 | 优先 **GEPA 反射式变异**（读执行轨迹 → 诊断 → 定向修改）；也可接 Opik MetaPrompt / HRPO | 可插拔后端 |
| **接受准则** | 新 harness 是否保留 | 简单门控：全量分数提升超阈值；或 **Pareto 保留**（分数不降且修改量更小） | 自研策略 |

**为什么是 GEPA**：GEPA 的“反射 + Pareto 选择”消费的正是执行轨迹与错误信息，而非黑盒搜索。HarnessLoop 的归因器把原始轨迹压缩成结构化“失败摘要”再喂给 GEPA，形成**归因 → 定向变异**的紧耦合，样本效率显著高于随机变异。

**修改执行器接口**：

```python
class Mutator(Protocol):
    async def propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        failures: list[Attribution],
    ) -> HarnessState:
        """根据假设与失败证据，产出新的 harness 版本（不落盘）。"""
        ...
```

| 阶段 | 实现 | 说明 |
| --- | --- | --- |
| v0.1 | `RuleTemplateMutator` | 规则模板，例如“RESULT_MISMATCH 且 AST diff 显示 WHERE 缺失 → 在 prompt 中追加 WHERE 子句强调” |
| v0.2 | `GEPAMutator` | 接入 GEPA 反射式变异 |
| v0.2 | `OpikMutator` | 通过 Opik 优化器 SDK 接入 MetaPrompt / HRPO |
| v0.3 | 第三方 `Mutator` 插件 | 开放插件接口 |

### 2.5 存储、配置与可观测性

| 关注点 | v0.1 方案 | 演进 |
| --- | --- | --- |
| 轨迹存储 | JSONL（一行一条 Trajectory） | v0.3 评估 SQLite / DuckDB |
| 版本快照 | 文件系统，目录结构参考 SIA 的 `runs/run_{id}/gen_{n}/` | 保持不可变 |
| 配置 | Pydantic Settings + JSON/YAML profile，按 provider / model / harness 版本切分 | 不变 |
| 可复现 | 固定随机种子；记录每次 LLM 调用的完整请求/响应（或哈希） | 不变 |
| 可视化 | **Gradio 只读迭代报告页** | v0.2；后续按需评估 Streamlit 或 FastAPI + 轻前端 |

**目录结构（约定）**：

```
runs/
└── run_20261109_1030/
    ├── meta.json                 # run 级配置、种子、模型、预算
    ├── iter_0/
    │   ├── harness.json          # 不可变 harness 快照
    │   ├── trajectory.jsonl      # 本迭代全量轨迹
    │   ├── attribution.jsonl     # 归因结果
    │   ├── hypotheses.json       # 本轮假设
    │   ├── metrics.json          # 分数、token、费用、耗时
    │   └── report.md             # 迭代报告
    ├── iter_1/
    │   └── ...
    └── latest -> iter_n          # 当前接受版本指针
```

### 2.6 依赖清单（v0.1 基线）

| 层 | 依赖 | 版本策略 | 是否必需 |
| --- | --- | --- | --- |
| LLM 调用 | `litellm` | `>=1.40,<2` | 必需 |
| 数据模型/配置 | `pydantic`, `pydantic-settings` | `>=2.6` | 必需 |
| SQL 解析 | `sqlglot` | `>=25,<30` | 必需 |
| SQL 执行 | `aiosqlite`（标准库 `sqlite3`） | `>=0.20` | 必需 |
| CLI | `typer` 或 `click` | `>=0.12` | 必需（薄壳） |
| 评测 | `pytest`, `pytest-asyncio` | `>=8.0` / `>=0.24` | 必需 |
| 报告 | 标准库 `json` / 手写 Markdown 模板 | — | 必需 |
| UI（v0.2） | `gradio` | `>=4` | 可选 extra |
| 多方言执行（可选） | `sqlspec` | 按需 | 可选 extra |
| 优化后端（v0.2） | GEPA / Opik SDK | 按需 | 可选 extra |

依赖分组：`pip install harnessloop`（核心）/ `harnessloop[ui]` / `harnessloop[optimizers]` / `harnessloop[all]`。

### 2.7 明确“不选”的技术及理由

| 技术 | 不选理由 |
| --- | --- |
| LangGraph / 图编排框架 | 引入执行图抽象与项目“不改拓扑”的边界冲突，且增加学习成本 |
| 向量数据库 / RAG 基础设施 | v0.1 无此需求；示例检索用内存列表足够 |
| 自研 prompt 优化算法 | 已有 GEPA / Opik 等成熟实现，重复造轮子违背“复用成熟信号源” |
| SFT / RL / 权重微调 | 直接违反范围红线，且带来 GPU 依赖 |
| 自建评测 DSL | pytest 已足够，自建 DSL 会割裂生态 |
| 数据库 / 消息队列 | v0.1 数据量级（几十到几百条 × 若干迭代）用文件系统完全够用 |
| 前端工程（React 等） | Gradio 是“最快出可分享 demo”的路径 |

### 2.8 一句话选型立场

> 执行层用最薄的异步抽象（LiteLLM/asyncio）保持循环透明；评测层用 sqlglot + 执行验证 + pytest 矩阵建立可信信号；优化层用归因驱动的 GEPA/反射变异做定向修改；存储与 UI 用文件系统 + Gradio 撑过 v0.1。**壁垒不在算法，而在“从失败轨迹到 harness 可编辑面”的归因链条是否足够锐利。**

---

## 3. 系统架构

### 3.1 分层架构

```
┌───────────────────────────────────────────────────────────────┐
│  接口层    CLI (typer)     Python SDK     Gradio 只读报告页    │
├───────────────────────────────────────────────────────────────┤
│  编排层    LoopRunner（迭代循环 / 接受回滚 / 预算与并发控制）     │
├───────────────────────────────────────────────────────────────┤
│  核心层    Evaluator → Attributor → HypothesisGenerator →      │
│            Mutator → Gate（接受准则）                           │
├───────────────────────────────────────────────────────────────┤
│  适配层    Executor(LiteLLM/PydanticAI/HarnessX)               │
│            Checker(SQL/API)   Mutator(Rule/GEPA/Opik)          │
├───────────────────────────────────────────────────────────────┤
│  基础层    数据模型(Pydantic)  存储(FS/JSONL)  配置  日志/追踪   │
└───────────────────────────────────────────────────────────────┘
```

**依赖方向**：上层依赖下层，核心层只依赖接口（Protocol），不依赖具体实现——保证执行器、检查器、变异器三者均可替换。

### 3.2 包结构

```
harnessloop/
├── __init__.py            # 公共 API：LoopRunner, HarnessState, run_loop
├── models.py              # 数据模型（见 §4）
├── config.py              # Pydantic Settings + profile 加载
├── runtime/
│   ├── executor.py        # Executor Protocol
│   ├── litellm_executor.py# v0.1 实现
│   └── toolcall.py        # tool-call 循环与参数校验日志
├── eval/
│   ├── dataset.py         # SQL / API 任务加载
│   ├── sql_checker.py     # 解析 + 执行 + 结果比对
│   ├── api_checker.py     # JSON Schema + 字段断言（v0.2 状态差分）
│   └── compare.py         # 结果集比较规则
├── attribution/
│   ├── attributor.py      # 归因主流程
│   ├── sql_rules.py       # sqlglot AST diff 归因
│   └── api_rules.py       # 参数/工具选择归因
├── optimize/
│   ├── hypothesis.py      # 假设生成（LLM + 结构化输出）
│   ├── mutator.py         # Mutator Protocol
│   ├── rule_mutator.py    # v0.1 规则模板
│   └── gate.py            # 接受准则（阈值 / Pareto）
├── store/
│   ├── workspace.py       # runs/ 目录与快照管理
│   └── jsonl.py           # 轨迹/归因读写
├── report/
│   └── markdown.py        # 迭代报告生成
├── cli.py                 # typer 薄壳
└── ui/
    └── gradio_app.py      # v0.2 只读报告页

tests/
├── fixtures/              # 20–50 条自建 SQL 金标准任务
└── ...
```

### 3.3 核心接口

```python
class LoopRunner:
    def __init__(
        self,
        executor: Executor,
        checker: Checker,          # SQLChecker | APIChecker
        attributor: Attributor,
        hypothesis_gen: HypothesisGenerator,
        mutator: Mutator,
        gate: Gate,
        workspace: Workspace,
        config: LoopConfig,        # 并发、预算、轮数、种子
    ) -> None: ...

    async def run(self) -> RunSummary:
        """执行完整闭环，返回汇总；每次迭代落盘不可变快照。"""
        ...
```

| 组件 | 输入 | 输出 | 可替换 |
| --- | --- | --- | --- |
| `Executor` | `Task`, `HarnessState` | `Trajectory` | ✅ |
| `Checker` | `Task`, `Trajectory` | `Pass/Fail` + 中间信号 | ✅ |
| `Attributor` | `Task`, `Trajectory`, `CheckResult` | `Attribution` | ✅ |
| `HypothesisGenerator` | `list[Attribution]` | `list[Hypothesis]` | ✅ |
| `Mutator` | `HarnessState`, `list[Hypothesis]` | `HarnessState` | ✅ |
| `Gate` | 旧/新 `metrics` | `accept: bool`, `reason: str` | ✅ |

### 3.4 单次迭代数据流

```
1. 加载 dataset（SQL/API 任务列表）
2. 读取当前 harness 快照（iter_n/harness.json）
3. asyncio + Semaphore 并发执行 → trajectory.jsonl
4. 逐条 Checker 判定 → pass/fail
5. 失败样本进入 Attributor → attribution.jsonl（含 evidence + harness_facet）
6. 聚合归因分布 → HypothesisGenerator → hypotheses.json（1–3 条）
7. Mutator 生成候选 harness（内存态）
8. 在候选 harness 上复测（复用同一 dataset，固定种子）
9. Gate 比较指标 → 接受 / 回滚；写 iter_{n+1}/ 与 report.md
10. 检查预算/轮数终止条件 → 继续或结束
```

**失败回滚语义**：候选 harness 一律先写 `iter_{n+1}/candidate/`，Gate 通过后才提升为 `iter_{n+1}/harness.json`；被拒绝的候选保留在 `rejected/` 下供分析，**绝不覆盖已接受版本**。

---

## 4. 数据模型（最小核心）

```python
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema

class Example(BaseModel):
    input: str
    output: str

class HarnessState(BaseModel):
    system_prompt: str
    tools: list[ToolSchema]
    few_shot_examples: list[Example] = Field(default_factory=list)
    output_format: str | None = None

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class Step(BaseModel):
    model_output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observation: Any = None
    latency_ms: int | None = None

class Trajectory(BaseModel):
    task_id: str
    steps: list[Step]
    final_answer: Any = None
    outcome: Literal["pass", "fail"] | None = None
    tokens: dict[str, int] = Field(default_factory=dict)

class AttributionLabel(str, Enum):
    PARSE_FAIL      = "PARSE_FAIL"        # SQL 无法解析
    EXEC_ERROR      = "EXEC_ERROR"        # 执行报错
    RESULT_MISMATCH = "RESULT_MISMATCH"   # 结果集不一致
    WRONG_TOOL      = "WRONG_TOOL"        # 选了错误工具
    MISSING_PARAM   = "MISSING_PARAM"     # 缺失必填参数
    EXTRA_PARAM     = "EXTRA_PARAM"       # 多余/非法参数
    FORMAT_VIOLATION= "FORMAT_VIOLATION"  # 输出格式不合约束

# v0.4：信号/归因的 label 是字符串（内置成员自动归一化），
# 自定义标签经 harnessloop.attribution.labels 注册。
LabelName = str

class HarnessFacet(str, Enum):
    SYSTEM_PROMPT   = "system_prompt"
    TOOL_DESCRIPTION= "tool_description"
    SCHEMA_FIELD    = "schema_field"
    FEW_SHOT        = "few_shot"
    OUTPUT_FORMAT   = "output_format"

class Attribution(BaseModel):
    task_id: str
    label: LabelName
    evidence: str                 # 轨迹中的具体片段引用，如 "step 3: tool_call missing date_range"
    harness_facet: HarnessFacet  # 指向具体可编辑面
    field_path: str | None = None # 如 "tools[1].parameters.required"

class EditOp(BaseModel):
    kind: Literal["add", "remove", "rewrite"]
    target: HarnessFacet
    field_path: str
    content: str

class Hypothesis(BaseModel):
    target: HarnessFacet
    edit: EditOp
    rationale: list[Attribution]  # 支撑本假设的失败样本
    falsifiable_check: str        # 可证伪条件，如 "ITER 类任务通过率 +10pp"
```

### 4.1 关键设计说明

`Attribution.harness_facet` 是整个数据模型的枢纽：它**强制归因指向 harness 的具体可编辑面**，而不是停在“模型不行”这种不可操作的结论上。

```python
# ❌ 不可操作的归因（普通评测工具止步于此）
{"task_id": "t17", "label": "fail", "reason": "SQL 结果不对"}

# ✅ HarnessLoop 的归因（直接映射到编辑动作）
{
  "task_id": "t17",
  "label": "RESULT_MISMATCH",
  "evidence": "step 2: FROM users listed_tasks 构成隐式内连接，导致无任务用户被过滤",
  "harness_facet": "system_prompt",
  "field_path": "system_prompt",
}
```

**约束**：`field_path` 必须能在当前 `HarnessState` 上解析成功，Attributor 与 Mutator 均需校验；解析失败的归因视为无效并计入噪声统计。

---

## 5. 功能需求

### F1. 数据集加载与任务执行

| 项 | 要求 |
| --- | --- |
| 任务格式 A（SQL） | 问题 + 数据库 schema + 可选 gold SQL |
| 任务格式 B（API） | 自然语言指令 + 可用 tool 列表 + 预期最终状态断言 |
| 执行产出 | 完整 Trajectory：每步输入、模型输出、tool call 参数、tool 返回、最终答案、耗时、token 消耗 |
| 并发 | asyncio + Semaphore，并发度可配置 |
| 验收 | 20–50 条 SQL 任务可全量跑通，轨迹字段无缺失 |

### F2. 评测与归因

| 项 | 要求 |
| --- | --- |
| 二元判定 | 每条轨迹输出 `pass` / `fail` |
| 归因标签 | 内置 7 类：PARSE_FAIL、EXEC_ERROR、RESULT_MISMATCH、WRONG_TOOL、MISSING_PARAM、EXTRA_PARAM、FORMAT_VIOLATION；v0.4 起支持注册自定义标签（含可选分类器） |
| 可追溯 | 归因必须引用轨迹的具体片段（如“第 3 步 tool call 缺少 date_range 参数”） |
| 可编辑面 | 每条归因必须给出 `harness_facet` + `field_path` |
| 验收 | 随机抽 10 条失败样本，人工核对归因标签与 evidence 一致率 ≥ 80% |

### F3. 假设生成

| 项 | 要求 |
| --- | --- |
| 数量 | 每轮 1–3 条 |
| 结构 | 每条含 `target`（system_prompt / tool_description / schema_field）、`field_path`、`direction`（增/删/改写）、`rationale`（来自哪些失败样本） |
| 约束 | 结构化输出（JSON Schema 强约束），解析失败重试 ≤ 2 次 |
| 可证伪性 | 每条假设附带明确的验证条件 |
| 验收 | 假设中的 `field_path` 100% 可在当前 harness 解析；人工评估“可操作”比例 ≥ 70% |
| 后端 | v0.1 规则模板；v0.2 GEPA / MetaPrompt；v0.4 新增 opik_optimizer 原生多轮适配（agent + metric 注入） |

### F4. Harness 修改与复测

| 项 | 要求 |
| --- | --- |
| 修改 | 依据假设产出新 harness 版本快照（不可变） |
| 复测 | 在同一评测集上重跑，输出总分对比 + 逐样本变化（失败→通过 / 通过→失败） |
| 决策 | 接受或回滚，给出理由 |
| 版本链 | 保留完整链路，可从任意版本复现 |
| 验收 | 3 轮迭代后 `runs/` 下版本链完整，任一轮可单独复现分数 |

### F5. 报告输出

每个迭代周期输出一份 Markdown 报告，必含 5 个板块：

1. 分数曲线（历次迭代总分 / 各标签通过率）
2. 归因分布（标签占比 + 样例）
3. 本轮假设（含 rationale 引用的样本 ID）
4. Harness diff（unified diff 形式）
5. 接受 / 拒绝理由（Gate 判定依据）

**报告模板骨架**：

````markdown
# Iteration 3 Report — run_20261109_1030

## 1. 分数
| Iter | Pass@1 | Δ | Tokens | Cost |
| --- | --- | --- | --- | --- |

## 2. 归因分布
| Label | Count | 占比 | 典型样本 |

## 3. 本轮假设
- H1: target=system_prompt, direction=add, ...

## 4. Harness Diff
```diff
+ 涉及多表时优先考虑 LEFT JOIN...
```

## 5. 决策
ACCEPT — 总分 +12pp，修改量 1 处，满足 Pareto 保留条件。
````

---

## 6. 非功能需求

| 类别 | 要求 | 实现要点 |
| --- | --- | --- |
| **可复现** | 固定随机种子；记录每次 LLM 调用的完整请求/响应（或哈希）；harness 快照不可变 | 快照写入后置只读；`meta.json` 记录 seed 与依赖版本 |
| **异步并发** | 评测集几十到几百条，必须 asyncio + Semaphore 控制并发 | 避免触发 rate limit 拖慢循环；支持指数退避重试 |
| **成本可控** | 每轮记录 token 与费用估算；支持单轮预算上限 | 超预算立即中止本轮并保留已完成结果 |
| **库优先** | 核心循环以 Python 包提供（`pip install harnessloop`），CLI/Web 为薄壳 | 公共 API 稳定在 `harnessloop/__init__.py` |
| **可替换性** | 执行器 / 检查器 / 变异器 / 接受准则均为 Protocol | 新增后端不改核心层 |
| **可观测** | 结构化日志 + 每步耗时；失败可定位到具体任务与步骤 | JSONL 事件流，便于后续接 OpenTelemetry |
| **安全** | API Key 仅从环境变量/密钥文件读取，不写入 runs/ | 日志脱敏；`meta.json` 不含明文密钥 |

---

## 7. CLI 与配置设计

### 7.1 CLI（薄壳）

```bash
# 初始化一个 run 配置
harnessloop init --template sql-basic

# 执行闭环（默认 3 轮）
harnessloop run --config configs/sql_golden.yaml --rounds 3

# 只评测，不优化（用于基线）
harnessloop eval --config configs/sql_golden.yaml --harness runs/latest

# 生成/查看报告
harnessloop report --run runs/run_20261109_1030 --open

# 对比两个版本
harnessloop diff runs/run_x/iter_1/harness.json runs/run_x/iter_2/harness.json
```

### 7.2 配置示例

```yaml
# configs/sql_golden.yaml
provider: openai
model: gpt-4o-mini
seed: 42

dataset:
  path: tests/fixtures/golden_sql.jsonl
  type: sql
  db_path: tests/fixtures/chinook.sqlite

harness:
  path: harness/initial.json

loop:
  max_rounds: 3
  concurrency: 8
  budget_usd_per_round: 1.50

opt:
  hypothesis_model: gpt-4o            # 假设生成可单独指定强模型
  mutator: rule                        # rule | gepa | opik
  opik:
    backend: auto                      # auto | native | metaprompt（v0.4）
    max_trials: 4                      # 原生适配的搜索预算
    sample_size: 8                     # 每轮参与评测的失败任务数
  gate:
    strategy: pareto                   # threshold | pareto
    min_gain_pp: 2.0

plugins:                               # v0.3 / v0.4 扩展点
  attribution: [rules.py]              # 归因规则
  labels: [labels.py]                  # 自定义归因标签
  edit_ops: [edit_ops.py]              # 自定义编辑操作

report:
  format: markdown
  out_dir: runs
```

配置分层：`默认值（代码） < profile 文件 < 环境变量（HARNESSLOOP_*） < CLI 参数`。

---

## 8. 项目计划

> 说明：以下按**里程碑**组织，不绑定日历周期。每个里程碑以“可验证的交付物 + 退出标准”定义完成，而非以时间定义。

### 8.1 里程碑总览

| 里程碑 | 主题 | 核心范围 | 状态 |
| --- | --- | --- | --- |
| **M0** | 工程基线 | 仓库骨架、数据模型、CI、配置系统 | 已完成（v0.1） |
| **M1 (v0.1)** | 验证闭环成立 | SQL-only 全链路跑通，3 轮迭代产出报告 | 已完成（v0.1，真实 LLM 12/12 验证） |
| **M2 (v0.2)** | 扩展与对比 | API 线 + GEPA 后端 + Gradio 报告页 | 已完成（v0.2，76 项测试全绿） |
| **M3 (v0.3)** | 可组合性 | 归因规则与 EditOp 插件接口、存储评估 | 已完成（v0.3） |
| **M4 (v0.4)** | 语义与后端深化 | 自定义归因标签、opik_optimizer 原生多轮适配 | 已完成（v0.4） |

### 8.2 v0.1 / M1 —— 验证闭环（最高优先级）

**目标**：证明“归因驱动的 harness 修改”能在小规模金标准集上带来可测量的提升。

| 编号 | 任务 | 交付物 | 依赖 |
| --- | --- | --- | --- |
| M0-1 | 仓库骨架 + `pyproject.toml` + 依赖分组 + lint/format/type 检查 | 可安装的空包 | — |
| M0-2 | §4 数据模型全部落地为 Pydantic 模型 + 单元测试 | `models.py` | M0-1 |
| M0-3 | `Workspace` 目录管理与不可变快照 | `store/workspace.py` | M0-2 |
| M0-4 | Pydantic Settings + profile 加载 | `config.py` | M0-1 |
| M0-5 | CI：pytest + ruff + mypy（可选）+ 覆盖率门槛 | `.github/workflows/ci.yml` | M0-1 |
| M1-1 | `LiteLLMExecutor`：asyncio tool-call 循环、参数校验日志 | `runtime/` | M0-2 |
| M1-2 | SQL 任务格式与数据集加载 | `eval/dataset.py` | M0-2 |
| M1-3 | `SQLChecker`：sqlglot 解析 + aiosqlite 执行 + 结果集比对 | `eval/sql_checker.py` | M1-1 |
| M1-4 | `Attributor`：7 标签 + AST diff 归因 + facet 映射 | `attribution/` | M1-3 |
| M1-5 | `HypothesisGenerator`：单次 LLM + JSON Schema 结构化输出 | `optimize/hypothesis.py` | M1-4 |
| M1-6 | `RuleTemplateMutator`：≥ 3 条规则模板 | `optimize/rule_mutator.py` | M1-5 |
| M1-7 | `Gate`：阈值与 Pareto 两种策略 | `optimize/gate.py` | M0-2 |
| M1-8 | `LoopRunner` 编排：并发、预算、接受回滚 | 核心层 | M1-1~7 |
| M1-9 | Markdown 报告生成（5 板块） | `report/markdown.py` | M1-8 |
| M1-10 | 20–50 条自建 SQL 金标准任务集 | `tests/fixtures/` | — |
| M1-11 | 端到端演示：3 轮迭代，报告可读 | `examples/` + README | 全部 |

**v0.1 退出标准（Definition of Done）**：

- [x] `pip install -e .` 后 `harnessloop run` 可一条命令跑完 3 轮
- [x] 3 轮中出现至少 1 次 Gate 接受，总分相对 iter_0 有可测量提升
- [x] 每条失败样本都有标签 + evidence + facet，人工抽检一致率 ≥ 80%
- [x] 同一配置两次运行（固定 seed）分数完全一致
- [x] 单轮 token 与费用在报告中可见，预算超限能正确中止
- [x] 报告包含分数曲线、归因分布、假设、diff、决策 5 个板块
- [x] 核心层对 LiteLLM 之外无框架依赖（可替换性自检）

> 验证记录：真实 LLM（deepseek-flash）run `examples/sql_golden/runs/real_loop_priced/`，12/12 通过，总成本 ≈ $0.0397，成本可见。

### 8.3 v0.2 / M2 —— API 线与优化器后端

| 编号 | 任务 | 交付物 | 状态 |
| --- | --- | --- | --- |
| M2-1 | API 任务格式（指令 + tool 列表 + 状态断言） | `eval/dataset.py` 扩展 | 已完成（含 `ApiToolRuntime`，`examples/api_state/` 12 条任务） |
| M2-2 | `APIChecker`：JSON Schema 校验 + 关键字段断言 | `eval/api_checker.py` | 已完成 |
| M2-3 | 状态差分最小实现（S0/S1 + 结构化 diff） | `eval/state_diff.py` | 已完成（diff 进入失败证据与 CheckResult） |
| M2-4 | `GEPAMutator` 适配器 | `optimize/gepa_mutator.py` | 已完成（真实 gepa 引擎，ADR-0004） |
| M2-5 | `OpikMutator`（MetaPrompt / HRPO 可选） | `optimize/opik_mutator.py` | 已完成（MetaPrompt 风格；v0.4 增补 opik_optimizer 原生多轮适配，见 M4-2） |
| M2-6 | 实验：规则模板 vs 反射变异的样本效率对比 | `experiments/` + 结论文档 | 已完成（[RESULTS.md](../experiments/rule_vs_reflective/RESULTS.md)） |
| M2-7 | Gradio 只读迭代报告页 | `ui/gradio_app.py` | 已完成（`harnessloop ui`） |
| M2-8 | `PydanticAIExecutor` 适配器（验证契约稳定性） | `runtime/` | 已完成（TestModel 契约测试） |

**v0.2 退出标准**：

- [x] API 线端到端跑通，至少 10 条 API 任务（12 条，mock 闭环 e2e 通过）
- [x] GEPA 后端可插拔切换（配置项 `mutator: gepa`）
- [x] 输出"规则模板 vs GEPA"的量化对比（同样本预算下的提升幅度）
- [x] Gradio 页面可展示历史 run 的分数曲线与 diff

### 8.4 v0.3 / M3 —— 可组合性

| 编号 | 任务 | 交付物 | 状态 |
| --- | --- | --- | --- |
| M3-1 | 自定义归因规则插件接口（注册表 + 版本约束） | `attribution/plugins.py` | 已完成（`examples/plugins/rules.py`） |
| M3-2 | 自定义 `EditOp` 插件接口 | `optimize/plugins.py` | 已完成（`EditOp.payload` + 注册表） |
| M3-3 | Trace 存储方案评估并落地 | `store/trace_store.py` + [评估结论](../experiments/trace_storage/RESULTS.md) | 已完成（JSONL 默认，SQLite 可选，ADR-0005） |
| M3-4 | `HarnessXExecutor` 适配（MCP 场景） | `runtime/mcp_executor.py` | 已完成（MCP 原生 stdio 实现；HarnessX 经 MCP 接入） |
| M3-5 | 文档站与示例集扩充 | `docs/` + `examples/plugins/` | 已完成（[插件指南](plugins.md) + 文档索引） |

**v0.3 退出标准**：

- [x] 第三方不修改核心代码即可新增一个归因规则与一个 EditOp（`tests/test_plugins.py` 用 profile 加载示例插件跑通闭环）
- [x] 有存储方案的量化评估结论（文件系统在何规模下失效：[RESULTS.md](../experiments/trace_storage/RESULTS.md)）

### 8.5 v0.4 / M4 —— 语义标签与原生优化器

| 编号 | 任务 | 交付物 | 状态 |
| --- | --- | --- | --- |
| M4-1 | 自定义归因标签（注册表 + 版本约束 + 可选分类器） | `attribution/labels.py` | 已完成（`examples/plugins/labels.py`、`plugins.labels`） |
| M4-2 | opik_optimizer 原生多轮适配（agent + metric 注入） | `optimize/opik_native.py` | 已完成（ADR-0006；`tests/test_opik_native.py` 含真实 `evaluate_prompt` 集成） |
| M4-3 | 配置、CLI 与文档同步 | `config.py` / `cli.py` / `docs/` | 已完成（`opt.opik.backend`、`harnessloop plugins` 展示标签） |

**v0.4 退出标准**：

- [x] 第三方不修改核心代码即可注册一个自定义标签，并让其在归因结果与报告中生效（`tests/test_plugins.py` 的 `NOOP_ACTION`）
- [x] 标签注册表阻止覆盖内置标签、拒绝版本不匹配与重复注册，分类器异常被隔离
- [x] `opik_optimizer` 以真实多轮 executor + checker 为评测单元优化 system prompt（`tests/test_opik_native.py` 调用真实 `evaluate_prompt`）
- [x] 原生适配失败/未安装时按 MetaPrompt → 规则模板回退，不中断闭环
- [x] 全量测试与 `ruff` 全绿，版本号 0.4.0

### 8.6 任务依赖图

```
M0-1 ─┬─ M0-2 ─┬─ M0-3 ─┐
      │        ├─ M1-1 ─ M1-2 ─ M1-3 ─ M1-4 ─ M1-5 ─ M1-6 ─┐
      ├─ M0-4 ─┤        └─ M1-10 ───────────────────────────┤
      └─ M0-5  └─ M1-7 ──────────────────────────────────── ─┴─ M1-8 ─ M1-9 ─ M1-11
                                                                  │
                                              M2-1 ─ M2-2 ─ M2-3 ─┤
                                              M2-4 ─ M2-5 ─ M2-6 ─┤
                                              M2-7 ───────────────┘
```

### 8.7 测试策略

| 层级 | 范围 | 工具 |
| --- | --- | --- |
| 单元测试 | 数据模型校验、结果比对、AST diff 归因规则 | pytest |
| 契约测试 | 各 Protocol 实现的接口一致性（Executor/Checker/Mutator） | pytest + 参数化 fake |
| 端到端 | mock LLM 下的完整闭环（不产生真实费用） | pytest + `FakeLLM` |
| 真实评测 | 20–50 条金标准集，小模型 + 预算上限 | 手动/定时运行 |
| 回归 | 归因规则变更后，历史失败样本的标签稳定性 | 快照测试 |

**原则**：CI 中全部使用 mock LLM，不产生 API 费用；真实调用只在受控的 nightly / 手动任务中执行。

### 8.8 发布策略

| 版本 | 语义 | 稳定性承诺 |
| --- | --- | --- |
| 0.x | SemVer 预发布 | 公共 API 可能 breaking，需在 CHANGELOG 标注 |
| 1.0 | 首个稳定版 | 核心 Protocol 与 `LoopRunner` 接口冻结 |

发布物：PyPI 包、GitHub Release、README 中的 3 分钟 quickstart。

---

## 9. 风险登记册

| 编号 | 风险 | 表现 | 影响 | 应对措施 |
| --- | --- | --- | --- | --- |
| R1 | **归因信噪比失衡** | 标签太粗 → 假设退化为盲目 Prompt 改写；标签太细 → 规则维护成本爆炸 | 核心差异化失效 | v0.1 收敛到 **5–7 个标签**（已定 7 个）；每个标签必须有明确 `harness_facet` 指向；新增标签需附失败样本证据 |
| R2 | **LLM-as-judge 不可靠** | 未校准的 judge 把错误结果判为等价 | 评测信号失真，优化方向错误 | 仅作兜底；启用前用人工标注小样本计算与人类的 **Cohen's kappa**，低于阈值则禁用；judge 调用与结果写入轨迹以便审计 |
| R3 | **公开基准标注噪声** | BIRD Mini-Dev 错误率约 52.8%、Spider 2.0-Snow 约 62.8% | “提升”结论不可信 | **以自建小规模金标准集为主要证据**；公开基准仅作参考并显式标注噪声影响；报告注明数据集与标注来源 |
| R4 | **范围蔓延** | 逐渐滑向“自动改写 Agent 架构”“自动微调” | 项目变重、失去轻量优势 | 写入范围红线的 ADR；PR 检查清单中增加“是否触碰权重/拓扑”一项；代码层面 Executor 不暴露拓扑编辑能力 |
| R5 | **Provider / 限流不稳定** | 并发高时大量 429，评测不完整 | 迭代结论偏差 | Semaphore + 指数退避 + 失败重试上限；记录每次调用错误并计入报告“执行异常”统计 |
| R6 | **成本失控** | 每轮全量复测 × 多轮迭代 | 用户放弃使用 | 单轮预算上限（`budget_usd_per_round`）；支持采样评测与全量评测两档；费用实时累计，超限中止 |
| R7 | **过度依赖单一优化器** | GEPA 接口变动导致后端不可用 | v0.2 交付受阻 | Mutator 为 Protocol，规则模板作为永久兜底；GEPA 与 Opik 各自独立适配器 |

**风险复核节奏**：每个里程碑结束做一次登记册复核，新增风险须指定负责人与验证方式。

---

## 10. 度量指标

### 10.1 项目自身效能指标（每轮迭代记录）

| 指标 | 定义 | 目标方向 |
| --- | --- | --- |
| `pass_rate` | 通过样本 / 总样本 | ↑ |
| `delta_pp` | 相对上一接受版本的百分点变化 | ↑ |
| `attribution_coverage` | 失败样本中带有效 facet 的比例 | → 100% |
| `facet_resolvable_rate` | `field_path` 可在 harness 解析成功的比例 | → 100% |
| `hypothesis_actionable_rate` | 人工判定“可操作”的假设占比 | ≥ 70% |
| `token_per_round` | 单轮 token 消耗 | 可控 |
| `cost_per_round` | 单轮费用估算（USD） | 受预算约束 |
| `rollback_count` | 被 Gate 拒绝的候选数 | 反映信噪比 |
| `reproducibility` | 同配置两次运行的分数差 | = 0 |

### 10.2 差异化验证指标（对外证明价值）

| 指标 | 对比对象 | 说明 |
| --- | --- | --- |
| 样本效率 | 规则模板 vs GEPA 反射变异 | 同等 token 预算下的提升幅度 |
| 归因锐利度 | 普通评测（只有 fail）vs HarnessLoop（facet + evidence） | 人工可执行性评分 |
| 提升真实性 | 自建金标准集 vs 公开基准 | 双轨报告，避免噪声误导 |

---

## 11. 开源治理

| 项 | 约定 |
| --- | --- |
| 许可证 | Apache-2.0（专利授权友好，适合工具类库） |
| 目录约定 | `harnessloop/` 核心包、`docs/` 文档、`examples/` 示例、`experiments/` 实验记录、`runs/`（gitignore） |
| 提交规范 | Conventional Commits；CHANGELOG 由人工维护（0.x 阶段不引入自动化） |
| PR 检查清单 | 数据模型变更需迁移说明；新增依赖需说明为何不在“不选清单”；**是否触碰权重/拓扑（R4）** |
| 架构决策 | `docs/adr/` 记录关键决策（范围边界、执行器抽象、优化器后端、存储方案） |
| 贡献门槛 | 单元测试 + 类型注解 + 文档更新；核心层新增 Protocol 实现需附契约测试 |

### 11.1 建议的首批 ADR

| 编号 | 主题 |
| --- | --- |
| 0001 | 范围边界：只改 Prompt 与 Tool Schema |
| 0002 | 执行层抽象：LiteLLM + asyncio，不引入图编排 |
| 0003 | 评测底座：pytest 而非自研 DSL |
| 0004 | 优化后端：GEPA 优先，规则模板兜底 |
| 0005 | 存储：v0.1 文件系统，何时引入 SQLite/DuckDB |

---

## 12. 附录

### 12.1 初始 harness 示例

```json
{
  "system_prompt": "你是 SQL 分析助手。根据数据库 schema 回答用户问题，只输出一条可执行的 SQL。",
  "tools": [
    {
      "name": "run_sql",
      "description": "在指定数据库上执行一条只读 SQL 并返回结果集。",
      "parameters": {
        "type": "object",
        "properties": {
          "sql": { "type": "string", "description": "单条 SELECT 语句" }
        },
        "required": ["sql"]
      }
    }
  ],
  "few_shot_examples": [],
  "output_format": "```sql\n<query>\n```"
}
```

### 12.2 一条完整归因的示例

```json
{
  "task_id": "sql_017",
  "label": "RESULT_MISMATCH",
  "evidence": "step 1: 生成 `SELECT name FROM users, tasks WHERE tasks.user_id = users.id`，隐式内连接过滤掉了无任务的用户；gold 为 LEFT JOIN。AST diff: join_type INNER -> LEFT。",
  "harness_facet": "system_prompt",
  "field_path": "system_prompt"
}
```

### 12.3 对应的假设与修改

```json
{
  "target": "system_prompt",
  "edit": {
    "kind": "add",
    "target": "system_prompt",
    "field_path": "system_prompt",
    "content": "当需要保留主表全部记录时，使用 LEFT JOIN 而不是逗号连接或 INNER JOIN；即使用户未显式要求“全部”。"
  },
  "rationale": ["sql_017", "sql_023", "sql_041"],
  "falsifiable_check": "LEFT JOIN 相关子集（12 条）通过率提升 ≥ 15pp，且其他子集通过率不下降"
}
```

### 12.4 术语表

| 术语 | 含义 |
| --- | --- |
| Harness | Agent 的外部可配置层：system prompt、few-shot、tool 描述与 JSON Schema、输出格式约束 |
| Harness State | 某一时刻 Harness 的完整、不可变快照 |
| Trajectory | 一次任务执行的完整记录（每步输入、模型输出、工具调用、观察、最终答案、耗时、token） |
| Attribution | 失败归因：标签 + 证据片段 + 指向的 Harness 可编辑面 |
| Hypothesis | 可证伪的修改假设：目标 + 编辑操作 + 支撑失败样本 + 验证条件 |
| Mutator | 把假设翻译为 harness 修改的执行器（规则模板 / GEPA / Opik） |
| Gate | 接受准则：新 harness 是否保留（阈值 / Pareto） |

---

**文档结束**。变更请提交 PR 并更新本文件顶部的版本号与状态字段。
