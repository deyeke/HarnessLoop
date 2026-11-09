# ADR-0006：用 agent + metric 注入做 opik_optimizer 多轮适配

- 状态：已接受（v0.4）
- 日期：2026-11-09
- 相关：[可插拔优化后端](0004-optimizer-backends-as-proposal-engines.md)、`harnessloop/optimize/opik_native.py`

## 背景

`opik_optimizer` 的评测模型是「数据集条目 → 一次 LLM 输出 → metric 打分」。
而 HarnessLoop 要优化的是**多轮工具调用 Agent**：一次任务包含若干轮
tool-call / observation，最终由 Checker 对执行结果（SQL 结果集或环境状态差分）
判定通过与否。直接把任务文本塞进单次调用会丢失工具轨迹，得分也不代表真实任务
成功率。

## 决策

不重写优化器，也不改它的评测循环，而是使用它公开的扩展点：

1. **自定义 agent**：`_HarnessAgent.invoke_agent(prompts, dataset_item)` 从优化器
   给出的 `ChatPrompt` 取出候选 `system`，重建 `HarnessState`，调用 HarnessLoop
   的完整 `Executor.run` + `Checker.check`；
2. **自定义 metric**：agent 返回 `{task_id, passed, answer, error}` JSON，
   `harness_metric` 解析 `passed` 返回 1.0 / 0.0；
3. **进程内数据集**：无 `OPIK_API_KEY` / `OPIK_URL_OVERRIDE` 时使用内存中的
   `opik.Dataset` 子类，并把 `opik_optimizer.constants.ENABLE_EVALUATE_ON_DICT_ITEMS`
   临时置为 `True`，避免依赖 Opik 服务端；有服务端时改用真实 dataset；
4. **预算与回退**：`opt.opik.max_trials` / `sample_size` 限制搜索；`backend: auto`
   在包可用时启用原生适配，任何异常按 native → MetaPrompt → 规则模板回退。

## 备选方案

| 方案 | 不选原因 |
| --- | --- |
| 把多轮任务压成单轮 prompt | 丢失工具轨迹，metric 与真实任务成功率脱节 |
| fork / 改造 opik_optimizer | 紧跟上游成本高，违背「优化器当后端」的定位 |
| 只用 MetaPrompt 候选生成 + 自建迷你评测 | v0.2 已实现，但没用到优化器自身的搜索/反思能力 |
| 强制要求 Opik 服务端 | 提高上手门槛，离线/CI 无法运行 |

## 影响

- 正向：优化器直接以真实任务通过率为目标，样本效率来自端到端反馈；离线可运行。
- 代价：每轮候选评测的成本 = 候选数 × 抽样任务数 × 单任务多轮成本，需用
  `max_trials` / `sample_size` 控预算；优化器自身的反思调用使用
  `opt.opik.model`（默认同主模型）。
- 无 Opik 服务端时 opik SDK 会异步尝试上报 trace 并打印 401 警告，不影响评测与
  闭环结果；配置 `OPIK_API_KEY` / `OPIK_URL_OVERRIDE` 后自动切换到真实 dataset。
- 契约测试用 stub `opik_optimizer` 模块验证数据集/metric/最优 prompt 提取；
  另有一个真实 `opik_optimizer` 评测集成测试覆盖多轮 agent 路径。
