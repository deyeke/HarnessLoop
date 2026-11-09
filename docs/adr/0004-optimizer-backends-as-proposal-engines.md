# ADR-0004: 优化器后端作为提案引擎（而非外层循环）

- 状态：Accepted
- 日期：2026-11-09
- 关联：v0.2 / M2-4、M2-5（GEPAMutator、OpikMutator）；R7（规则模板永久兜底）

## 背景

v0.2 引入可插拔优化后端。GEPA（`gepa` 包）与 Opik（`opik_optimizer`）都自带
"外层循环"：内部自行组织 候选生成 → 评测 → 接受 的完整搜索。而 HarnessLoop
的闭环（评测 → 归因 → 假设 → 变异 → 复测 → Gate）本身就是外层循环，且承担着
存储、预算、报告与不可变快照等横切职责。

## 决策

**HarnessLoop 保持唯一外层循环；GEPA / Opik 以后端方式参与"变异"这一步。**

1. `opt.mutator: rule | gepa | opik` 选择每轮的变异器：
   - `rule`（默认）：RuleTemplateMutator，直接应用 LLM 假设的 edit；
   - `gepa`：GEPAMutator 把当前 harness 序列化为文本组件
     （`system_prompt` / `output_format` / `tools.<name>.description`，工具 JSON
     Schema 不参与文本优化——R4），用本轮失败样本构造 trainset、全量任务为
     valset，调用真实 `gepa.optimize` 做一次**有预算上限的**反射变异搜索
     （`opt.gepa.max_metric_calls`），返回其最优候选；随后仍由 LoopRunner 的
     Gate 全量复测并接受/回滚；
   - `opik`：OpikMutator 以 MetaPrompt 风格生成 K 个 system_prompt 候选，
     用同一执行器/评测器在失败样本上做抽样迷你评测后择优提案（详见其
     docstring 的适配性说明）。
2. 任一后端不可用（未安装）或搜索失败时，回退到规则模板变异（R7 兜底）。
3. gepa 的适配器实现其 `GEPAAdapter` 协议（`evaluate` / `make_reflective_dataset`
   / `propose_new_texts=None` 走默认提案），评测与闭环共用同一 executor+checker，
   保证分数可比；引擎调用放在工作线程，适配器内用独立事件循环桥接异步执行。

## 理由

- 双重外层循环会绕过预算门控、Pareto Gate 与不可变快照等闭环契约，让
  "Harness 修改的接受/回滚"失去唯一裁决者。
- 反射变异的价值在**提案质量**（证据→修复），不在接管流程；实验
  （`experiments/rule_vs_reflective/RESULTS.md`）表明：假设质量差时规则模板
  永不收敛而 GEPA 一轮救回；假设质量好时规则模板全局 tokens 约为 GEPA 的 1/4。
- `opik_optimizer` 的 MetaPrompt/HRPO 面向单次调用型 LLM 程序，无法原生存放
  多轮工具调用 Agent，v0.2 以自包含 MetaPrompt 策略先行，原生适配列为 v0.3 扩展点。

## 影响

- 每轮启用 GEPA 时存在**双重评测**（搜索内部 + 闭环复测）；由
  `max_metric_calls` 控制总开销，且实验口径区分"闭环可见 tokens"与
  "全局 tokens（含搜索）"。
- 优化器为可选依赖：`pip install harnessloop[optimizers]`；未安装时 `rule`
  模式不受影响。
- 未来引入其他优化器（DSPy/MIPRO 等）沿用同一提案后端模式。
