"""CLI thin shell over the library API."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from harnessloop.config import DEFAULT_HARNESS, load_config
from harnessloop.loop import LoopRunner
from harnessloop.optimize.diff import diff_harness
from harnessloop.store.workspace import load_harness

ENV_EXAMPLE = """# HarnessLoop LLM 连接配置
# 复制为 .env 后填入真实值；.env 不会被提交，也不会写入 runs/。

HARNESSLOOP_MODEL=gpt-4o-mini
HARNESSLOOP_PROVIDER=openai
HARNESSLOOP_API_BASE=https://api.openai.com/v1
HARNESSLOOP_API_KEY=sk-your-key-here
HARNESSLOOP_TEMPERATURE=0.0
"""


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.rounds is not None:
        config.loop.max_rounds = args.rounds
    if args.out:
        config.report.out_dir = args.out
    runner = LoopRunner.build(config, run_id=args.run_id)
    summary = asyncio.run(runner.run())
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.out:
        config.report.out_dir = args.out
    harness_path = args.harness or config.harness.path
    if not harness_path:
        print("error: --harness is required when config has no harness.path", file=sys.stderr)
        return 2
    harness = load_harness(harness_path)
    config.loop.max_rounds = 0
    runner = LoopRunner.build(
        config, run_id=args.run_id, initial_harness=harness
    )
    summary = asyncio.run(runner.run())
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"error: run directory not found: {run_dir}", file=sys.stderr)
        return 2
    for report in sorted(run_dir.glob("iter_*/report.md")):
        print(report)
    latest = run_dir / "latest.json"
    if latest.exists():
        print(f"latest: {latest.read_text(encoding='utf-8').strip()}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    previous = load_harness(args.previous)
    candidate = load_harness(args.candidate)
    change = diff_harness(previous, candidate)
    print(change.diff or "（无修改）")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from harnessloop.ui.gradio_app import build_demo

    demo = build_demo(args.runs_dir)
    demo.launch(server_port=args.port, share=args.share)
    return 0


def _cmd_plugins(args: argparse.Namespace) -> int:
    from harnessloop.attribution.labels import get_registry as get_label_registry
    from harnessloop.attribution.plugins import get_registry as get_rule_registry
    from harnessloop.optimize.plugins import (
        BUILTIN_EDIT_KINDS,
    )
    from harnessloop.optimize.plugins import (
        get_registry as get_edit_op_registry,
    )

    if args.config:
        from harnessloop.attribution.labels import load_label_plugins
        from harnessloop.attribution.plugins import load_rule_plugins
        from harnessloop.optimize.plugins import load_edit_op_plugins

        config = load_config(args.config)
        load_rule_plugins(config.plugins.attribution)
        load_label_plugins(config.plugins.labels)
        load_edit_op_plugins(config.plugins.edit_ops)

    rules = get_rule_registry().rules()
    print(f"attribution rules ({len(rules)}):")
    if not rules:
        print("  (none — built-in label mapping only)")
    for rule in rules:
        print(f"  {rule.name:<40} priority={rule.priority:<4} {rule.description}")
    labels = get_label_registry().labels()
    custom_labels = [label for label in labels if not label.builtin]
    print(f"attribution labels ({len(labels)}):")
    for label in labels:
        kind = "built-in" if label.builtin else "custom"
        print(
            f"  {label.name:<40} {kind:<9} "
            f"{label.facet.value}/{label.field_path} {label.description}"
        )
    if not custom_labels:
        print("  (no custom labels — plugins.labels is empty)")
    ops = get_edit_op_registry().names()
    print(f"edit ops ({len(BUILTIN_EDIT_KINDS) + len(ops)}):")
    for name in sorted(BUILTIN_EDIT_KINDS):
        print(f"  {name:<40} built-in")
    for name in ops:
        plugin = get_edit_op_registry().get(name)
        print(f"  {name:<40} {plugin.description if plugin else ''}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    directory = Path(args.dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    harness_path = directory / "harness.json"
    config_path = directory / "config.yaml"
    env_example_path = directory / ".env.example"
    if not harness_path.exists():
        harness_path.write_text(
            json.dumps(DEFAULT_HARNESS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not env_example_path.exists():
        env_example_path.write_text(ENV_EXAMPLE, encoding="utf-8")
    if not config_path.exists():
        config_path.write_text(
            """# HarnessLoop profile
# LLM 连接信息（model / api_base / api_key）请写在同目录的 .env 中，
# 参考 .env.example。

dataset:
  path: dataset.jsonl
  kind: sql

harness:
  path: harness.json

loop:
  max_rounds: 3
  concurrency: 8
  budget_usd_per_round: 1.50
  seed: 42

opt:
  max_hypotheses: 3
  mutator: rule
  gate:
    strategy: pareto
    min_gain_pp: 2.0

report:
  out_dir: runs
""",
            encoding="utf-8",
        )
    print(f"created {harness_path}")
    print(f"created {config_path}")
    print(f"created {env_example_path}")
    print(
        "next: copy .env.example to .env and fill in HARNESSLOOP_API_KEY, "
        "add dataset.jsonl (SQL tasks), then run `harnessloop run --config config.yaml`"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harnessloop",
        description="Harness-only self-improvement loop for agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the full closed loop")
    run_parser.add_argument("--config", required=True, help="profile YAML path")
    run_parser.add_argument("--rounds", type=int, default=None)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--out", default=None, help="override runs directory")
    run_parser.set_defaults(handler=_cmd_run)

    eval_parser = subparsers.add_parser("eval", help="evaluate one harness, no optimisation")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--harness", default=None, help="harness snapshot path")
    eval_parser.add_argument("--run-id", default=None)
    eval_parser.add_argument("--out", default=None)
    eval_parser.set_defaults(handler=_cmd_eval)

    report_parser = subparsers.add_parser("report", help="list iteration reports")
    report_parser.add_argument("--run", required=True, help="run directory")
    report_parser.set_defaults(handler=_cmd_report)

    diff_parser = subparsers.add_parser("diff", help="diff two harness snapshots")
    diff_parser.add_argument("previous")
    diff_parser.add_argument("candidate")
    diff_parser.set_defaults(handler=_cmd_diff)

    init_parser = subparsers.add_parser("init", help="write a starter profile")
    init_parser.add_argument("--dir", default="harnessloop_demo")
    init_parser.set_defaults(handler=_cmd_init)

    ui_parser = subparsers.add_parser(
        "ui", help="open the read-only Gradio report page"
    )
    ui_parser.add_argument("--runs-dir", required=True, help="parent directory of run folders")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true", help="create a public gradio link")
    ui_parser.set_defaults(handler=_cmd_ui)

    plugins_parser = subparsers.add_parser(
        "plugins",
        help="list attribution labels/rules and edit ops (optionally from a profile)",
    )
    plugins_parser.add_argument("--config", default=None, help="profile YAML to load plugins from")
    plugins_parser.set_defaults(handler=_cmd_plugins)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
