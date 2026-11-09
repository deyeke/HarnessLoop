# 实验：规则模板变异 vs 反射变异（GEPA 后端）

确定性 mock 场景，零 API 成本；同一数据集 / 执行器 / 评测器 / 预算。

| 场景 | 变异器 | 假设质量 | 最终 pass@1 | 首次 100% | 被接受迭代 | 全局 tokens（含优化器搜索） | 闭环可见 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rule_good | rule | good | 100.0% | iter_1 | [1] | 917 | 542 |
| gepa_good | gepa | good | 100.0% | iter_1 | [1] | 3689 | 542 |
| rule_vague | rule | vague | 50.0% | 未达成 | [] | 2205 | 1080 |
| gepa_vague | gepa | vague | 100.0% | iter_1 | [1] | 3689 | 542 |

## 逐轮明细

### rule_good（mutator=rule, 假设=good）

| 迭代 | pass@1 | prompt | completion |
| --- | --- | --- | --- |
| iter_0 | 50.0% | 220 | 50 |
| iter_1 | 100.0% | 225 | 47 |

停机原因：全部任务通过

### gepa_good（mutator=gepa, 假设=good）

| 迭代 | pass@1 | prompt | completion |
| --- | --- | --- | --- |
| iter_0 | 50.0% | 220 | 50 |
| iter_1 | 100.0% | 225 | 47 |

停机原因：全部任务通过

### rule_vague（mutator=rule, 假设=vague）

| 迭代 | pass@1 | prompt | completion |
| --- | --- | --- | --- |
| iter_0 | 50.0% | 220 | 50 |
| iter_1 | 50.0% | 220 | 50 |
| iter_2 | 50.0% | 220 | 50 |
| iter_3 | 50.0% | 220 | 50 |

停机原因：None

### gepa_vague（mutator=gepa, 假设=vague）

| 迭代 | pass@1 | prompt | completion |
| --- | --- | --- | --- |
| iter_0 | 50.0% | 220 | 50 |
| iter_1 | 100.0% | 225 | 47 |

停机原因：全部任务通过


## 结论

1. **假设质量好时，规则模板最省**：`rule_good` 用 917 全局 tokens（约 GEPA 的 1/4）一轮收敛；GEPA 的额外成本来自搜索期内部候选评测（3689 tokens），其中只有 542 tokens 计入闭环可见指标 —— 优化器搜索开销必须用全局口径看。
2. **假设质量差时，只有反射变异能收敛**：`rule_vague` 反复应用无效建议，3 轮后仍为 50%，且烧掉 2205 tokens；`gepa_vague` 从轨迹证据中提取出精确修复，iter_1 即 100%。这正是 R7 把规则模板定位为兜底、把 GEPA 定位为可插拔后端的量化依据。
3. **样本效率的含义是鲁棒性**：反射变异不依赖假设生成器的命中质量，为「LLM 假设失灵」提供了证据驱动的冗余路径；代价是每轮固定增加约 3.7k tokens 的搜索开销（mock 口径）。

## 方法与局限

- 全部行为由脚本化 LLM 决定（确定性、可复现），用于隔离「变异器策略」这一变量；不代表真实 LLM 下 GEPA 的绝对开销。
- GEPA 后端通过 `opt.mutator: gepa` 一键切换；`opt.gepa.max_metric_calls` 约束其搜索预算，避免优化器开销失控。
- 复现：`python experiments/rule_vs_reflective/run_experiment.py`（写入本文件）。
