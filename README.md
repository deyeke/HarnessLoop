# HarnessLoop / 驭环

库优先、可嵌入的 Agent 闭环优化工具：**冻结模型，只调 Harness**（system prompt、few-shot、tool description、tool JSON Schema、输出格式约束）。

核心闭环：**评测 → 归因 → 假设 → 修改 → 复测 → 接受/回滚**。

差异化不在优化算法，而在**归因链条**：每条失败都必须指向 harness 的具体可编辑面（`harness_facet` + `field_path`），而不是停在“模型不行”。

---

## 边界（硬约束）

| 可修改 | 不修改 |
| --- | --- |
| `system prompt` | 模型权重（不做 SFT / RL） |
| `few-shot 示例` | Agent 运行时拓扑（不改执行图） |
| `tool description` | 不引入外部教师模型做任务合成 |
| `tool JSON Schema`（参数名/描述/枚举/必填项） | 不依赖 GPU 训练设施 |
| `输出格式约束` | |

---

## 安装

```bash
pip install -e ".[dev]"
```

要求 Python 3.10+。LLM 通过 [LiteLLM](https://github.com/BerriAI/litellm) 调用。

### 配置 LLM 连接（.env）

复制 `.env.example` 为 `.env`，填入 model / url / key：

```bash
cp .env.example .env
```

```ini
HARNESSLOOP_MODEL=gpt-4o-mini
HARNESSLOOP_PROVIDER=openai
HARNESSLOOP_API_BASE=https://api.openai.com/v1
HARNESSLOOP_API_KEY=sk-your-key-here
HARNESSLOOP_TEMPERATURE=0.0
```

`.env` 查找顺序：仓库根目录 → profile 所在目录（后者优先）。真实环境变量优先于 `.env`，CLI 参数优先于两者。`.env` 已被 `.gitignore` 忽略，API Key 只保存在内存中，不会写入 YAML、`runs/`、`meta.json` 或报告。

优先级：`代码默认值 < YAML profile < .env / 环境变量 < CLI 参数`。

---

## 快速开始（SQL 金标准集，12 条任务）

```bash
# 1. 生成本地演示数据库
python examples/sql_golden/build_db.py

# 2. 配置 LLM 连接
cp examples/sql_golden/.env.example examples/sql_golden/.env
#    编辑 .env，填入 HARNESSLOOP_API_KEY（以及需要时的 API_BASE / MODEL）

# 3. 跑 3 轮闭环
harnessloop run --config examples/sql_golden/config.yaml --rounds 3

# 4. 查看迭代报告
harnessloop report --run examples/sql_golden/runs/<run_id>
```

只想评测、不优化：

```bash
harnessloop eval --config examples/sql_golden/config.yaml \
  --harness examples/sql_golden/harness.json
```

对比两个 harness 快照：

```bash
harnessloop diff runs/<run_id>/iter_0/harness.json runs/<run_id>/iter_1/harness.json
```

---

## 快速开始（API 线，12 条状态断言任务，v0.2）

```bash
# 1. 配置 LLM 连接（同上）
cp examples/api_state/.env.example examples/api_state/.env 2>/dev/null || true

# 2. 跑 3 轮闭环（内置 todo 域 Mock API，无需任何外部服务）
harnessloop run --config examples/api_state/config.yaml --rounds 3
```

API 任务没有金标准结果集，取而代之的是**状态断言**：任务声明初始状态 S0，
Agent 通过工具调用把环境推进到 S1，`APIChecker` 对 S1 求值，例如：

```json
{
  "task_id": "api_close_weekly_report",
  "kind": "api",
  "question": "把标题为「写周报」的任务状态更新为 done。",
  "initial_state": { "tasks": [ ... ], "next_task_id": 104 },
  "expected": {
    "assertions": [
      { "op": "eq", "path": "tasks[title=写周报].status", "value": "done" },
      { "op": "ne", "path": "tasks[title=修复登录bug].status", "value": "done" }
    ]
  }
}
```

路径支持 `tasks[0]`（下标）与 `tasks[title=写周报]`（过滤）；
断言 ops：`eq / ne / exists / not_exists / contains / length_eq / length_ge / length_le`。
每轮失败证据会附带 S0→S1 的结构化 diff（`eval/state_diff.py`）。

---

## 可插拔优化后端（v0.2）

```yaml
opt:
  mutator: rule        # rule（默认）| gepa | opik
  gepa:
    max_metric_calls: 40          # GEPA 搜索预算（每轮额外任务执行次数）
    reflection_minibatch_size: 3
```

- `rule`：规则模板变异器，直接应用 LLM 假设（最省 token，假设质量差时可能停滞）；
- `gepa`：真实 [GEPA](https://github.com/gepa-ai/gepa) 引擎对文本组件做有预算上限的
  反射变异搜索（工具 JSON Schema 不参与文本优化），候选仍由闭环 Gate 全量复测
  （`pip install harnessloop[optimizers]`）；
- `opik`：默认走 v0.4 的 **opik_optimizer 原生多轮适配**（见下节）；也可用
  `opt.opik.backend: metaprompt` 固定为 MetaPrompt 风格候选生成 + 抽样迷你评测。

规则模板是永久兜底：后端未安装或搜索失败时自动回退。
量化对比见 [实验结论](experiments/rule_vs_reflective/RESULTS.md)（好假设时 rule
全局 tokens ≈ GEPA 的 1/4；差假设时 rule 不收敛、GEPA 一轮救回）。

## 只读报告页（v0.2）

```bash
harnessloop ui --runs-dir examples/sql_golden/runs --port 7860
```

展示每个 run 的 pass@1 曲线、指标/决策表、逐轮归因分布、Harness Diff 与
report.md。纯只读，不能触发新的闭环（`pip install harnessloop[ui]`）。

## 扩展点：归因标签、归因规则与 EditOp 插件（v0.3 / v0.4）

第三方无需修改核心代码即可接入三个扩展点，在 profile 里声明即可：

```yaml
plugins:
  labels:
    - labels.py               # 自定义归因标签（v0.4）
  attribution:
    - rules.py                # 自定义归因规则（相对 profile 目录）
  edit_ops:
    - edit_ops.py             # 自定义编辑操作
```

- **归因标签（v0.4）**：新增失败类别并声明默认落点，可选分类器把一条既有失败
  （例如状态毫无变化的 `RESULT_MISMATCH`）重命名为业务标签（如 `NOOP_ACTION`）。
  标签会直接出现在归因分布与报告中。
- **归因规则**：把失败信号映射到具体的 `harness_facet` + `field_path`，按
  `priority` 先于标签默认落点执行；返回不可解析的路径或抛异常都会被跳过。
- **EditOp**：在 `add / remove / rewrite` 之外实现结构化修改（如收紧参数 enum），
  通过 `EditOp.payload` 传参，名称经注册表校验。

插件声明扩展 API 主版本，不匹配时直接报 `PluginVersionError`。
`harnessloop plugins --config <yaml>` 可列出当前注册的标签、规则与操作。
完整写法与示例见 [插件开发指南](docs/plugins.md)、`examples/plugins/`。

## MCP 执行器（v0.3）

工具调用可以交给任意 MCP（stdio）服务端执行，harness 仍然只负责模型看到的
工具 schema：

```yaml
executor:
  kind: mcp
  mcp:
    command: ["python", "server.py"]
    cwd: .
    timeout_s: 30
```

模型产出的工具名必须与服务端一致；每个任务一个会话，与任务在同一协程内关闭
（`pip install harnessloop[mcp]`）。

## Trace 存储（v0.3）

默认仍是 `iter_<n>/*.jsonl`（可 diff、零依赖）。单 run 轨迹超过约 5,000 条时，
报告页每次刷新都要全量解析，可切换 SQLite：

```yaml
store:
  backend: sqlite        # jsonl（默认）| sqlite
  sqlite_name: traces.db
```

两种后端实现同一个 `TraceStore` 协议，报告与 UI 不受影响。实测数据与阈值见
[Trace 存储基准](experiments/trace_storage/RESULTS.md)。

## opik_optimizer 原生多轮适配（v0.4）

`opik_optimizer` 默认只驱动“单次 LLM 调用”，无法承载工具调用型 Agent。HarnessLoop
用它的 **agent + metric** 扩展点做了原生适配：优化器选出候选 system prompt 后，
交给 HarnessLoop 跑完整的多轮 executor + checker，得分再回传给优化器。

```yaml
opt:
  mutator: opik
  opik:
    backend: auto        # auto（默认，装了就用原生）| native | metaprompt
    model: gpt-4o-mini   # 优化器内部反思用的模型，默认同主模型
    max_trials: 4        # 搜索预算
    sample_size: 8       # 每轮参与评测的失败任务数
```

`auto`/`native` 需要 `pip install harnessloop[optimizers]`（含 `opik_optimizer>=3.2`）；
未安装、无 API Key 或优化器抛错时依次回退到 MetaPrompt 变异器和规则模板。
无 Opik 服务端时评测使用进程内数据集，不依赖外部实验后端。

---

## 产物结构

```
runs/run_<id>/
├── meta.json                 # 模型、seed、数据集、版本
├── latest.json               # 当前接受的迭代与 harness
├── iter_0/                   # 基线
│   ├── harness.json          # 不可变快照
│   ├── trajectory.jsonl      # 全量轨迹
│   ├── check_results.jsonl   # 逐条判定与失败信号
│   ├── attribution.jsonl     # 归因（label + evidence + facet）
│   ├── metrics.json
│   └── report.md             # 5 板块报告
└── iter_1/                   # 候选 + 决策
    ├── candidate_harness.json
    ├── harness.json          # 本轮结束后“接受的”版本（被拒绝时等于上一版）
    ├── hypotheses.json
    ├── decision.json
    ├── trajectory.jsonl
    ├── metrics.json
    └── report.md
```

报告固定 5 个板块：分数曲线、归因分布、本轮假设、harness diff、接受/拒绝理由。

---

## 架构

```
CLI / Python SDK
      │
LoopRunner（并发、预算、接受/回滚、版本链）
      │
Evaluator → Attributor → HypothesisGenerator → Mutator → Gate
      │             │                │              │        │
 SQLChecker    sqlglot AST diff   单次 LLM     规则模板/   阈值 /
 (执行+比对)    + 7 类标签       + JSON 校验   GEPA(v0.2) Pareto
      │
LiteLLMExecutor（asyncio + tool-call 循环）→ SQLToolRuntime（只读）
```

所有可变组件均为 Protocol：`Executor` / `Checker` / `Mutator` / `Gate`，新增后端无需改核心层。

7 个内置归因标签：`PARSE_FAIL`、`EXEC_ERROR`、`RESULT_MISMATCH`、`WRONG_TOOL`、`MISSING_PARAM`、`EXTRA_PARAM`、`FORMAT_VIOLATION`；v0.4 起可通过 `plugins.labels` 注册自定义标签。

---

## CLI

| 命令 | 作用 |
| --- | --- |
| `harnessloop init --dir <path>` | 生成起始 profile 与 harness |
| `harnessloop run --config <yaml>` | 执行完整闭环 |
| `harnessloop eval --config <yaml> --harness <json>` | 只评测一个 harness |
| `harnessloop report --run <dir>` | 列出迭代报告与 latest 指针 |
| `harnessloop diff <a.json> <b.json>` | 对比两个 harness 快照 |
| `harnessloop ui --runs-dir <dir>` | 打开只读 Gradio 迭代报告页（v0.2） |
| `harnessloop plugins --config <yaml>` | 列出已加载的归因标签、规则与编辑操作（v0.3 / v0.4） |

环境变量覆盖（也可写在 `.env`）：`HARNESSLOOP_MODEL`、`HARNESSLOOP_PROVIDER`、`HARNESSLOOP_API_BASE`、`HARNESSLOOP_API_KEY`、`HARNESSLOOP_TEMPERATURE`、`HARNESSLOOP_DATASET`、`HARNESSLOOP_HARNESS`、`HARNESSLOOP_OUT_DIR`、`HARNESSLOOP_MAX_ROUNDS`、`HARNESSLOOP_CONCURRENCY`。

---

## Python API

```python
import asyncio
from harnessloop.config import load_config
from harnessloop.loop import LoopRunner

config = load_config("examples/sql_golden/config.yaml")
runner = LoopRunner.build(config)
summary = asyncio.run(runner.run())
print(summary.final_pass_rate, summary.accepted_iterations)
```

---

## 测试

```bash
python -m pytest -q
```

CI 中的测试全部使用脚本化 Fake LLM，不产生 API 费用；端到端用例覆盖“初始失败 → 归因 → 假设 → 修改 → 复测通过 → Gate 接受”的完整闭环。

---

## 路线图

| 版本 | 范围 |
| --- | --- |
| v0.1 | SQL-only；LiteLLM + asyncio；sqlglot + sqlite 执行验证；规则模板修改器；Markdown 报告 |
| v0.2 | API 线 + 状态断言/状态差分；GEPA / Opik 可插拔优化后端；PydanticAIExecutor 契约验证；Gradio 只读报告页 |
| v0.3 | 归因规则与 EditOp 插件接口；JSONL/SQLite trace 存储；MCP 执行器；插件指南与示例 |
| v0.4（当前） | 自定义归因标签；opik_optimizer 原生多轮适配 |
| v0.5 | HarnessX 直连（非 MCP） |

设计与需求细节见 [项目说明书](docs/HarnessLoop-项目说明书.md)、[插件开发指南](docs/plugins.md)与[文档索引](docs/README.md)。

## License

Apache-2.0
