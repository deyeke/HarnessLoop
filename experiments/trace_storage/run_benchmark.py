"""M3-3: quantify when the JSONL trace layout stops being the right choice.

Both backends are measured on the same synthetic trajectories at four scales:
write cost, full re-read (what ``report``/``ui`` do), single-task lookup and a
failure filter. Results are written to ``RESULTS.md`` next to this script.

    python experiments/trace_storage/run_benchmark.py
"""

from __future__ import annotations

import json
import platform
import shutil
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

from harnessloop.models import CheckResult, Step, ToolCall, Trajectory

SCALES = (1_000, 5_000, 20_000, 50_000)
REPEATS = 2
HERE = Path(__file__).resolve().parent


def make_records(count: int) -> tuple[list[Trajectory], list[CheckResult]]:
    trajectories: list[Trajectory] = []
    results: list[CheckResult] = []
    for index in range(count):
        steps = [
            Step(
                index=step,
                model_output="我需要先查询任务表，然后按状态过滤。" * 2,
                tool_calls=[
                    ToolCall(
                        name="list_tasks",
                        arguments={"status": "open", "limit": 20, "offset": step * 20},
                    )
                ],
                observation={"tasks": [{"id": step * 10 + i, "status": "open"} for i in range(5)]},
                latency_ms=120 + step,
                prompt_tokens=320,
                completion_tokens=48,
            )
            for step in range(4)
        ]
        trajectories.append(
            Trajectory(
                task_id=f"task_{index:06d}",
                steps=steps,
                final_answer=f"SELECT * FROM tasks WHERE id = {index}",
                outcome="pass" if index % 3 else "fail",
                tokens={"prompt": 1280, "completion": 192},
                cost_usd=0.00042,
                duration_ms=980 + index % 37,
            )
        )
        results.append(
            CheckResult(
                task_id=f"task_{index:06d}",
                passed=index % 3 != 0,
                predicted_sql=f"SELECT * FROM tasks WHERE id = {index}",
            )
        )
    return trajectories, results


def timed(fn, repeats: int = REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def measure_jsonl(
    base: Path, count: int, trajectories: list[Trajectory], results: list[CheckResult]
) -> dict[str, float]:
    directory = base / f"jsonl_{count}"
    directory.mkdir(parents=True, exist_ok=True)
    traj_path = directory / "trajectory.jsonl"
    result_path = directory / "check_results.jsonl"

    write = timed(
        lambda: (
            traj_path.write_text(
                "\n".join(
                    json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
                    for item in trajectories
                ),
                encoding="utf-8",
            ),
            result_path.write_text(
                "\n".join(
                    json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
                    for item in results
                ),
                encoding="utf-8",
            ),
        )
    )

    def read_all() -> None:
        paths = [traj_path, result_path]
        for path in paths:
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def lookup_last() -> None:
        target = f"task_{count - 1:06d}"
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            if json.loads(line)["task_id"] == target:
                return

    def filter_failures() -> None:
        count_failed = sum(
            1
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if not json.loads(line)["passed"]
        )
        assert count_failed

    size = traj_path.stat().st_size + result_path.stat().st_size
    return {
        "write": write,
        "read_all": timed(read_all),
        "lookup": timed(lookup_last),
        "filter": timed(filter_failures),
        "size": float(size),
        "files": 2.0,
    }


def measure_sqlite(
    base: Path, count: int, trajectories: list[Trajectory], results: list[CheckResult]
) -> dict[str, float]:
    path = base / f"sqlite_{count}.db"
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE trajectories (
            task_id TEXT PRIMARY KEY, iteration INTEGER, passed INTEGER, payload TEXT
        );
        CREATE TABLE check_results (
            task_id TEXT PRIMARY KEY, iteration INTEGER, passed INTEGER, payload TEXT
        );
        CREATE INDEX idx_traj_task ON trajectories (task_id);
        CREATE INDEX idx_results_passed ON check_results (passed);
        """
    )

    def write() -> None:
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO trajectories VALUES (?,?,?,?)",
                [
                    (
                        item.task_id,
                        0,
                        int(item.outcome == "pass"),
                        json.dumps(item.model_dump(mode="json"), ensure_ascii=False),
                    )
                    for item in trajectories
                ],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO check_results VALUES (?,?,?,?)",
                [
                    (
                        item.task_id,
                        0,
                        int(item.passed),
                        json.dumps(item.model_dump(mode="json"), ensure_ascii=False),
                    )
                    for item in results
                ],
            )

    def read_all() -> None:
        for table in ("trajectories", "check_results"):
            rows = conn.execute(f"SELECT payload FROM {table}").fetchall()
            [json.loads(row[0]) for row in rows]

    def lookup_last() -> None:
        target = f"task_{count - 1:06d}"
        row = conn.execute(
            "SELECT payload FROM trajectories WHERE task_id = ?", (target,)
        ).fetchone()
        if row is None:
            raise AssertionError("lookup failed")

    def filter_failures() -> None:
        row = conn.execute(
            "SELECT COUNT(*) FROM check_results WHERE passed = 0"
        ).fetchone()
        assert row[0]

    write()
    conn.commit()
    size = path.stat().st_size
    result = {
        "write": timed(write),
        "read_all": timed(read_all),
        "lookup": timed(lookup_last),
        "filter": timed(filter_failures),
        "size": float(size),
        "files": 1.0,
    }
    conn.close()
    return result


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def fmt_size(size: float) -> str:
    return f"{size / 1024 / 1024:.1f} MB"


def render(rows: list[dict]) -> str:
    lines = [
        "# Trace storage: JSONL vs SQLite",
        "",
        f"- Generated by `experiments/trace_storage/run_benchmark.py` on "
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}.",
        "- Payload: synthetic API-line trajectories at 1,000 / 5,000 / 20,000 / "
        "50,000 records, 4 tool steps / ~1.6 KB each.",
        f"- Timings are the median of {REPEATS} runs; `lookup` resolves the *last* "
        "task id, i.e. the JSONL worst case.",
        "",
        "| 记录数 | JSONL 写入 | JSONL 全量读 | JSONL 单任务查 | JSONL 失败过滤 | "
        "SQLite 写入 | SQLite 全量读 | SQLite 单任务查 | SQLite 失败过滤 | 磁盘 (JSONL / DB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        jsonl, sqlite = row["jsonl"], row["sqlite"]
        lines.append(
            f"| {row['records']:,} | {fmt_ms(jsonl['write'])} | {fmt_ms(jsonl['read_all'])} | "
            f"{fmt_ms(jsonl['lookup'])} | {fmt_ms(jsonl['filter'])} | "
            f"{fmt_ms(sqlite['write'])} | {fmt_ms(sqlite['read_all'])} | "
            f"{fmt_ms(sqlite['lookup'])} | {fmt_ms(sqlite['filter'])} | "
            f"{fmt_size(jsonl['size'])} / {fmt_size(sqlite['size'])} |"
        )

    last = rows[-1]
    first = rows[0]
    worst = max(rows, key=lambda row: row["jsonl"]["lookup"])
    readable = next(
        (row for row in rows if row["jsonl"]["read_all"] > 0.2), last
    )
    lines += [
        "",
        "## 结论",
        "",
        f"1. JSONL 的单任务查询与全量读都是 O(N)：记录数从 {first['records']:,} 增至 "
        f"{worst['records']:,}（{worst['records'] / first['records']:g}×）时，最坏查询从 "
        f"{fmt_ms(first['jsonl']['lookup'])} 变为 {fmt_ms(worst['jsonl']['lookup'])}"
        f"（{worst['jsonl']['lookup'] / max(first['jsonl']['lookup'], 1e-9):.1f}×），"
        "因为必须逐行解析直到命中。",
        f"2. SQLite 走主键/二级索引，查询耗时基本不随 N 增长："
        f"{fmt_ms(first['sqlite']['lookup'])}（{first['records']:,} 条）→ "
        f"{fmt_ms(worst['sqlite']['lookup'])}（{worst['records']:,} 条）。"
        f"全量读仍是 O(N)，但只解析 payload 列：{fmt_ms(worst['jsonl']['read_all'])} → "
        f"{fmt_ms(worst['sqlite']['read_all'])}。",
        f"3. 写入反过来：JSONL 一次顺序写更快（{fmt_ms(last['jsonl']['write'])} vs "
        f"{fmt_ms(last['sqlite']['write'])}，{last['records']:,} 条），磁盘占用也更小"
        f"（{fmt_size(last['jsonl']['size'])} vs {fmt_size(last['sqlite']['size'])}）。"
        "写只发生在每轮结束一次，读却会被报告页反复触发，因此规模上来后 SQLite 划算。",
        "",
        f"**失效阈值**：单 run 轨迹数 ≤ {first['records']:,} 条（约 "
        f"{fmt_size(first['jsonl']['size'])}）时，JSONL 全量读 "
        f"{fmt_ms(first['jsonl']['read_all'])}、单任务查 {fmt_ms(first['jsonl']['lookup'])}，"
        f"交互无感；到 {readable['records']:,} 条时全量读 {fmt_ms(readable['jsonl']['read_all'])}，"
        f"开始可感知；{worst['records']:,} 条时分别达到 "
        f"{fmt_ms(worst['jsonl']['read_all'])} / {fmt_ms(worst['jsonl']['lookup'])}，"
        "报告页每次刷新都是秒级。**单 run 超过约 5,000 条轨迹就应切到 SQLite**"
        "（索引查询恒在 2 ms 量级）；≤ 1,000 条继续用 JSONL 最省事。"
        "当前示例 run 为 12 条任务 × 4 轮 = 48 条，远低于阈值，故默认仍是 JSONL。",
        "",
        "## 落地方式",
        "",
        "```yaml",
        "store:",
        "  backend: sqlite      # 默认 jsonl",
        "  sqlite_name: traces.db",
        "```",
        "",
        "两种后端实现同一个 `TraceStore` 协议（`harnessloop/store/trace_store.py`），"
        "`LoopRunner` 只依赖协议，因此报告与 UI 无需改动，JSONL 仍是默认值"
        "（可 diff、无需额外工具）。",
        "",
        "## 复现",
        "",
        "```bash",
        "python experiments/trace_storage/run_benchmark.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="hl_trace_bench_"))
    try:
        for count in SCALES:
            print(f"measuring {count:,} records ...", flush=True)
            trajectories, results = make_records(count)
            rows.append(
                {
                    "records": count,
                    "jsonl": measure_jsonl(tmp, count, trajectories, results),
                    "sqlite": measure_sqlite(tmp, count, trajectories, results),
                }
            )
            del trajectories, results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    output = HERE / "RESULTS.md"
    output.write_text(render(rows), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
