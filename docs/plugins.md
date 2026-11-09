# 插件开发指南（v0.4）

HarnessLoop 对外暴露三个扩展点，第三方**无需修改核心代码**即可接入：

| 扩展点 | 作用 | 注册方式 | 代码位置 |
| --- | --- | --- | --- |
| 归因标签 `AttributionLabelSpec` | 新增失败类别，并声明它默认落到哪个 harness 字段 | `@attribution_label` 装饰器 | `harnessloop/attribution/labels.py` |
| 归因规则 `AttributionRule` | 决定一条具体失败信号落到哪个 harness 字段 | `@attribution_rule` 装饰器 | `harnessloop/attribution/plugins.py` |
| 编辑操作 `EditOpPlugin` | 在三种内置 `add/remove/rewrite` 之外实现结构化修改 | `@edit_op` 装饰器 | `harnessloop/optimize/plugins.py` |

三者共用 `harnessloop/plugins.py` 的加载器与版本检查。

---

## 1. 在 profile 中声明

```yaml
plugins:
  attribution:
    - rules.py                 # 相对 profile 目录
    - mypkg.rules:my_rule      # 可导入模块 + 属性
    - mypkg.rules              # 仅导入模块（装饰器自注册）
  labels:
    - labels.py
  edit_ops:
    - edit_ops.py
```

路径写法：

- `file.py[:attr]`：独立文件，无需打包，相对路径按 profile 所在目录解析；
- `pkg.module[:attr]`：已安装模块，`attr` 可以是规则/插件对象，也可以是模块（此时依靠导入副作用注册）。

用 `harnessloop plugins --config <yaml>` 可以打印当前 profile 实际注册了哪些标签、规则与操作。

---

## 2. 版本约束

三个注册表都声明 `API_VERSION = "1"`。插件可以写 `api_version="1.2"`，但**主版本必须匹配**，否则注册时抛 `PluginVersionError`：

```
plugin 'x' targets extension API v2, this build supports v1 (major versions must match)
```

这样插件 API 发生 breaking 变更时，旧插件会立刻失败而不是静默错配。

---

## 3. 归因规则

```python
from harnessloop.attribution.plugins import attribution_rule
from harnessloop.models import AttributionLabel, FailureSignal, HarnessFacet, HarnessState


@attribution_rule("status_assertion_to_update_tool", priority=10)
def status_assertion_to_update_tool(
    signal: FailureSignal, harness: HarnessState
) -> tuple[HarnessFacet, str] | None:
    if signal.label != AttributionLabel.RESULT_MISMATCH:
        return None                      # 不认领，交给后续规则
    if ".status" not in signal.evidence:
        return None
    return HarnessFacet.TOOL_DESCRIPTION, "tools.update_task.description"
```

约束与语义：

- **优先级**：数值小的先执行；都返回 `None` 时回退到标签注册表的默认落点（内置 7 类 + 自定义标签，保证永远有归因）。
- **路径必须可解析**：返回的 `field_path` 会经过 `harness.resolve_path` 校验，解析失败时该规则被跳过并打 warning，避免把失败归因到不存在的字段。
- **异常隔离**：规则抛异常只记录 warning，不会中断闭环。
- 返回 `None` 表示「不认领」，不是「归因不到」。

完整示例见 `examples/plugins/rules.py`。

---

## 4. 自定义归因标签

内置 7 个标签（`PARSE_FAIL` / `EXEC_ERROR` / `RESULT_MISMATCH` / `WRONG_TOOL` /
`MISSING_PARAM` / `EXTRA_PARAM` / `FORMAT_VIOLATION`）回答「失败属于哪一类」。
当业务语义需要更细的类别时，注册一个标签即可：

```python
from harnessloop.attribution.labels import attribution_label
from harnessloop.models import AttributionLabel, FailureSignal, HarnessFacet, HarnessState


@attribution_label(
    "NOOP_ACTION",
    facet=HarnessFacet.SYSTEM_PROMPT,
    field_path="system_prompt",
    priority=10,
    description="执行后环境状态完全没变：提示模型真正调用修改类工具",
)
def noop_action(signal: FailureSignal, harness: HarnessState) -> bool:
    return (
        signal.label == AttributionLabel.RESULT_MISMATCH
        and "状态差异：无变化" in signal.evidence
    )
```

语义与约束：

- **默认落点**：`facet` + `field_path` 是这个标签的归因默认值；路径同样要能被
  `harness.resolve_path` 解析，否则退回 `system_prompt`。
- **分类器可选**：示例中的函数把一条既有的 `RESULT_MISMATCH` **重命名为**
  `NOOP_ACTION`；不写分类器则只能由自定义 Checker 直接产出该标签。
- **优先级**：分类器按 `priority` 升序执行，第一个命中者胜出；异常只记 warning。
- **与规则的先后**：显式归因规则仍然最先执行（规则是对单条信号的精确判断），
  标签默认落点只在规则不认领时使用。
- **不能覆盖内置标签**：与内置 7 个名字冲突会抛 `PluginError`。
- **报告可见**：新的标签会出现在 `attribution.jsonl`、归因分布表和
  `harnessloop plugins` 输出中，无需改渲染代码。

完整示例见 `examples/plugins/labels.py`；单测见 `tests/test_plugins.py`。

---

## 5. 自定义 EditOp

内置 `add / remove / rewrite` 只能追加或替换标量。结构化修改（收紧 enum、新增 schema 属性、插入 few-shot 示例）用自定义 op：

```python
from harnessloop.models import EditOp, HarnessState
from harnessloop.optimize.plugins import edit_op


@edit_op("set_enum", description="设置某个 tool 参数的 enum 与描述")
def set_enum(harness: HarnessState, edit: EditOp) -> HarnessState:
    ...                                   # 读取 edit.payload 构造新状态
    return HarnessState.model_validate(data)
```

调用方通过 `EditOp` 传参：

```python
EditOp(
    kind="set_enum",                       # = 注册名，不能与内置 kind 冲突
    target="schema_field",
    field_path="tools.update_task.parameters.properties.status",
    payload={"tool": "update_task", "param": "status", "enum": ["open", "done"]},
)
```

约束：

- 必须返回**新的** `HarnessState`（不允许原地修改）；
- 名称不能是 `add/remove/rewrite`（会抛 `PluginError`）；
- 当 profile 注册了自定义 op 时，假设生成器的 JSON schema 会自动把该名字加入 `direction` 的取值集合，并把 `payload` 透传给 `EditOp`；
- 自定义 op 由规则模板变异器执行；`gepa`/`opik` 后端只编辑文本组件，不会产生自定义 op。

完整示例见 `examples/plugins/edit_ops.py`；端到端用例见 `tests/test_plugins.py`。

---

## 6. 本地验证

```bash
# 1. 列出 profile 注册的扩展点
harnessloop plugins --config examples/plugins/config.yaml

# 2. 用示例 profile 跑闭环（需先配好 .env）
harnessloop run --config examples/plugins/config.yaml --rounds 2

# 3. 跑测试（脚本化 LLM，无 API 费用）
python -m pytest tests/test_plugins.py -q
```

编写自己的插件时建议：

1. 先在单测里直接构造 `FailureSignal` / `HarnessState` 验证规则与操作；
2. 再接到 profile 上用 `harnessloop eval` 只评测一轮，确认归因落到预期字段；
3. 最后跑完整闭环。

---

## 7. 已知边界

- 注册表是**进程级**的：`LoopRunner.build` 会把 profile 里的插件注册进全局注册表，同一进程内多次构建会累积（名称冲突会报错）。
- 自定义标签只能重命名/新增类别，不会凭空产生失败信号——信号仍由 Checker 或分类器给出。
- 归因规则只影响「落到哪个字段」，不改变标签；标签只影响默认落点，不改变规则。
- 插件在进程内执行，等同本机代码权限，不要从不可信来源加载。
