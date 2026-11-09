# ADR-0005：Trace 存储默认 JSONL，超过规模阈值切换 SQLite

- 状态：已接受（v0.3）
- 日期：2026-11-09
- 相关：[Trace 存储基准](../experiments/trace_storage/RESULTS.md)

## 背景

v0.1/v0.2 把每轮轨迹写成 `iter_<n>/trajectory.jsonl` 与 `check_results.jsonl`。这种布局可 diff、可 grep、零依赖，但每次读取都要全量解析，且单任务查询必须线性扫描。

M3-3 要求给出「文件系统在何规模下失效」的量化结论，并在需要时落地替代方案。

## 决策

1. **默认仍是 JSONL**，保持现有产物布局与工具链（report / UI / diff）不变；
2. 新增 `TraceStore` 协议与 SQLite 实现，通过 profile 切换：

   ```yaml
   store:
     backend: sqlite      # jsonl（默认）| sqlite
     sqlite_name: traces.db
   ```

3. `LoopRunner` 只依赖协议，报告与 UI 不做改动；
4. 不引入 DuckDB：SQLite 是标准库、单文件、支持索引与 SQL 聚合，当前数据规模用不到列存。

## 依据（实测，Windows 11 / Python 3.13）

| 规模 | JSONL 全量读 | JSONL 单任务查 | SQLite 全量读 | SQLite 单任务查 | JSONL 写入 / SQLite 写入 |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 88 ms | 36 ms | 57 ms | 2.4 ms | 60 / 170 ms |
| 5,000 | 398 ms | 171 ms | 413 ms | 2.0 ms | 280 / 809 ms |
| 20,000 | 1.84 s | 675 ms | 1.92 s | 1.9 ms | 1.1 / 3.2 s |
| 50,000 | 5.15 s | 1.68 s | 4.59 s | 2.2 ms | 2.9 / 8.0 s |

结论：

- JSONL 的全量读与单任务查均为 O(N)，50k 时单任务查 1.68 s；
- SQLite 索引查询恒定在 2 ms 量级，但写入慢 2–3 倍、磁盘占用约 1.8 倍；
- 单 run 轨迹数 **≤ 1,000 条**时 JSONL 交互无感；**> 5,000 条**后报告页刷新进入可感知/秒级，应切换 SQLite。

## 影响

- 默认零变化；需要时改一行配置。
- SQLite 模式下 `iter_<n>/trajectory.jsonl` 不再生成，直接读文件的脚本需改用 `TraceStore`（或同时保留两种后端）。
- 报告页仍依赖 `metrics.json` / `report.md` / `attribution.jsonl`，这些不随存储后端变化。
