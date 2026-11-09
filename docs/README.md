# 文档索引

| 文档 | 内容 |
| --- | --- |
| [项目说明书](HarnessLoop-项目说明书.md) | 定位、边界、架构、里程碑（M0–M4）、测试与发布策略 |
| [插件开发指南](plugins.md) | 归因标签/规则与 EditOp 扩展点、版本约束、配置写法、调试方法 |
| [ADR-0004](adr/0004-optimizer-backends-as-proposal-engines.md) | 优化器作为「提案引擎」而非外层循环 |
| [ADR-0005](adr/0005-trace-storage-jsonl-vs-sqlite.md) | Trace 存储：JSONL 默认、SQLite 可选的量化依据 |
| [ADR-0006](adr/0006-opik-native-multiturn-adapter.md) | opik_optimizer 原生多轮适配：agent + metric 注入 |
| [规则模板 vs 反射变异](../experiments/rule_vs_reflective/RESULTS.md) | 实验：好/差假设下两种变异器的收敛与 token 消耗 |
| [Trace 存储基准](../experiments/trace_storage/RESULTS.md) | 实验：JSONL 与 SQLite 在 1k–50k 条轨迹下的读写对比 |

快速上手见根目录 [README](../README.md)。
