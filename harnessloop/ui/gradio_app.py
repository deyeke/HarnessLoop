"""Read-only Gradio report page (M2-7): browse runs without re-running them.

``harnessloop ui --runs-dir examples/sql_golden/runs`` opens a local page that
shows, per run: the pass@1 curve across iterations, the metrics / decision
table, per-iteration attribution distribution, harness diff and the generated
markdown report. It only reads the run artifacts — nothing can trigger a new
loop from here (v0.2 boundary: 报告页不做任何写操作).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# -- artifact readers --------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def list_runs(runs_dir: str | Path) -> list[str]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    runs = []
    for child in sorted(root.iterdir()):
        if (child / "latest.json").is_file():
            runs.append(child.name)
    return runs


def load_run(runs_dir: str | Path, run_id: str) -> dict[str, Any]:
    root = Path(runs_dir) / run_id
    iterations = sorted(
        (p for p in root.glob("iter_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    rows: list[list[Any]] = []
    curve: list[dict[str, Any]] = []
    reports: dict[str, str] = {}
    diffs: dict[str, str] = {}
    attributions: dict[str, list[dict[str, Any]]] = {}
    for folder in iterations:
        name = folder.name
        metrics = _read_json(folder / "metrics.json") or {}
        decision = _read_json(folder / "decision.json") or {}
        rows.append(
            [
                name,
                f"{metrics.get('passed', '?')}/{metrics.get('total', '?')}",
                round(metrics.get("pass_rate", 0.0) * 100, 1),
                metrics.get("prompt_tokens"),
                metrics.get("completion_tokens"),
                metrics.get("cost_usd"),
                "ACCEPT" if decision.get("accepted") else "REJECT",
                decision.get("reason", ""),
            ]
        )
        curve.append(
            {"iteration": name, "pass_rate": round(metrics.get("pass_rate", 0.0) * 100, 1)}
        )
        report_path = folder / "report.md"
        reports[name] = (
            report_path.read_text(encoding="utf-8") if report_path.is_file() else "（无报告）"
        )
        candidate = folder / "candidate_harness.json"
        # harness.json in an iter dir is the POST-decision state; the "previous"
        # side of the diff is therefore the previous iteration's harness.json.
        prev_index = int(name.split("_")[-1]) - 1
        prev_folder = root / f"iter_{prev_index}" if prev_index >= 0 else None
        previous = (
            _read_json(prev_folder / "harness.json") if prev_folder else {}
        ) or {}
        if candidate.is_file():
            import difflib

            current = _read_json(candidate) or {}
            diff = difflib.unified_diff(
                json.dumps(previous, ensure_ascii=False, indent=2).splitlines(),
                json.dumps(current, ensure_ascii=False, indent=2).splitlines(),
                fromfile=f"iter_{prev_index}/harness" if prev_folder else "initial",
                tofile=f"{name}/candidate",
                lineterm="",
            )
            diffs[name] = "\n".join(diff)
        else:
            diffs[name] = "（基线轮，无候选 harness）"
        attributions[name] = _read_jsonl(folder / "attribution.jsonl")

    latest = _read_json(root / "latest.json") or {}
    return {
        "run_id": run_id,
        "latest": latest,
        "rows": rows,
        "curve": curve,
        "reports": reports,
        "diffs": diffs,
        "attributions": attributions,
    }


def attribution_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "（本轮无归因记录）"
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get("label", "?"))
        counts[label] = counts.get(label, 0) + 1
    total = len(records)
    lines = [f"共 {total} 条归因："]
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {label}: {count} ({count / total:.0%})")
    return "\n".join(lines)


# -- UI ------------------------------------------------------------------------ #


def build_demo(runs_dir: str | Path) -> Any:
    try:
        import gradio as gr
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("报告页需要 gradio（pip install gradio）") from exc

    runs = list_runs(runs_dir)

    def _run_changed(run_id: str):
        view = load_run(runs_dir, run_id)
        df = pd.DataFrame(
            view["rows"],
            columns=[
                "iter", "pass", "pass_rate%", "prompt", "completion", "cost_usd",
                "decision", "reason",
            ],
        )
        curve_df = pd.DataFrame(view["curve"])
        iteration = view["latest"].get("harness", "").split("/")[0] or (
            view["rows"][-1][0] if view["rows"] else None
        )
        return (
            df,
            curve_df,
            gr.update(choices=list(view["reports"]), value=iteration),
            view["reports"].get(iteration, ""),
            view["diffs"].get(iteration, ""),
            attribution_summary(view["attributions"].get(iteration, [])),
        )

    def _iteration_changed(run_id: str, iteration: str):
        view = load_run(runs_dir, run_id)
        return (
            view["reports"].get(iteration, "（无报告）"),
            view["diffs"].get(iteration, ""),
            attribution_summary(view["attributions"].get(iteration, [])),
        )

    with gr.Blocks(title="HarnessLoop 迭代报告") as demo:
        gr.Markdown("# HarnessLoop 只读迭代报告")
        run_select = gr.Dropdown(
            choices=runs, value=runs[0] if runs else None, label="运行 (run)"
        )
        with gr.Row():
            with gr.Column(scale=3):
                curve_plot = gr.LinePlot(
                    pd.DataFrame(columns=["iteration", "pass_rate"]),
                    x="iteration",
                    y="pass_rate",
                    label="Pass@1 曲线 (%)",
                )
                metrics_table = gr.DataFrame(
                    headers=[
                        "iter", "pass", "pass_rate%", "prompt", "completion",
                        "cost_usd", "decision", "reason",
                    ],
                    label="迭代指标",
                )
            with gr.Column(scale=2):
                iteration_select = gr.Dropdown(label="迭代", choices=[])
                attribution_md = gr.Markdown(label="归因分布")
        diff_code = gr.Code(label="Harness Diff", language="markdown")
        report_md = gr.Markdown(label="报告")

        inputs = [run_select]
        outputs = [
            metrics_table,
            curve_plot,
            iteration_select,
            report_md,
            diff_code,
            attribution_md,
        ]
        run_select.change(_run_changed, inputs=inputs, outputs=outputs)
        if runs:
            demo.load(_run_changed, inputs=inputs, outputs=outputs)
        iteration_select.change(
            _iteration_changed,
            inputs=[run_select, iteration_select],
            outputs=[report_md, diff_code, attribution_md],
        )
    return demo


def launch(runs_dir: str | Path, *, port: int = 7860, share: bool = False) -> Any:
    demo = build_demo(runs_dir)
    demo.launch(server_port=port, share=share, prevent_thread_lock=True)
    return demo, str(getattr(demo, "local_url", "") or "")
