"""Configuration loading: YAML profile + ``.env`` + environment overrides.

Priority (low → high): code defaults < profile file < ``.env`` / env vars < CLI.

LLM 凭证统一走 ``.env``（或真实环境变量），不写入 YAML、不写入 ``runs/``：

    HARNESSLOOP_MODEL=gpt-4o-mini
    HARNESSLOOP_API_BASE=https://api.openai.com/v1
    HARNESSLOOP_API_KEY=sk-...
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    path: str
    kind: Literal["sql", "api"] = "sql"


class HarnessConfig(BaseModel):
    path: str | None = None


class LoopSettings(BaseModel):
    max_rounds: int = 3
    concurrency: int = 8
    budget_usd_per_round: float | None = None
    seed: int = 42


class GateSettings(BaseModel):
    strategy: Literal["threshold", "pareto"] = "pareto"
    min_gain_pp: float = 2.0
    max_equal_change_fields: int = 1


class GepaSettings(BaseModel):
    """Bounded reflective-mutation search per round (real ``gepa`` engine)."""

    max_metric_calls: int = 40
    reflection_minibatch_size: int = 3


class OpikSettings(BaseModel):
    """opik_optimizer backend.

    ``backend='auto'`` runs the native multi-turn adapter when
    ``opik_optimizer`` is installed and falls back to the meta-prompt mutator
    otherwise; ``native`` / ``metaprompt`` pin the choice.
    """

    backend: Literal["auto", "native", "metaprompt"] = "auto"
    model: str | None = None
    max_trials: int = 4
    candidates: int = 3
    sample_size: int = 8


class PluginSettings(BaseModel):
    """Third-party extension points. Specs are ``pkg.mod:attr`` or ``file.py[:attr]``."""

    attribution: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    edit_ops: list[str] = Field(default_factory=list)


class MCPSettings(BaseModel):
    command: list[str]
    cwd: str | None = None
    timeout_s: float = 30.0
    env: dict[str, str] = Field(default_factory=dict)


class ExecutorSettings(BaseModel):
    kind: Literal["litellm", "mcp"] = "litellm"
    mcp: MCPSettings | None = None

    @model_validator(mode="after")
    def _mcp_needs_command(self) -> ExecutorSettings:
        if self.kind == "mcp" and (self.mcp is None or not self.mcp.command):
            raise ValueError("executor.kind=mcp requires executor.mcp.command")
        return self


class StoreSettings(BaseModel):
    """Trace storage backend. ``sqlite`` keeps artifacts queryable at scale."""

    backend: Literal["jsonl", "sqlite"] = "jsonl"
    sqlite_name: str = "traces.db"


class OptSettings(BaseModel):
    hypothesis_model: str | None = None
    max_hypotheses: int = 3
    mutator: Literal["rule", "gepa", "opik"] = "rule"
    gepa: GepaSettings = Field(default_factory=GepaSettings)
    opik: OpikSettings = Field(default_factory=OpikSettings)
    gate: GateSettings = Field(default_factory=GateSettings)


class ReportSettings(BaseModel):
    out_dir: str = "runs"


class CompareSettings(BaseModel):
    order_sensitive: bool = False
    float_tolerance: float = 1e-6
    compare_column_names: bool = False
    column_order_sensitive: bool = False


class PricingSettings(BaseModel):
    """USD per 1M tokens. ``None`` falls back to the built-in price table."""

    input_per_1m: float | None = None
    output_per_1m: float | None = None


class LoopConfig(BaseModel):
    provider: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_base: str | None = None
    # 仅来自环境变量 / .env，永不写入 YAML、meta.json 或报告
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    dataset: DatasetConfig
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    loop: LoopSettings = Field(default_factory=LoopSettings)
    opt: OptSettings = Field(default_factory=OptSettings)
    executor: ExecutorSettings = Field(default_factory=ExecutorSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    compare: CompareSettings = Field(default_factory=CompareSettings)
    pricing: PricingSettings = Field(default_factory=PricingSettings)

    def llm_model(self, override: str | None = None) -> str:
        name = override or self.model
        if self.provider and "/" not in name:
            return f"{self.provider}/{name}"
        return name


class _EnvOverrides(BaseSettings):
    """Fields that can be overridden per shell session / ``.env``."""

    model_config = SettingsConfigDict(env_prefix="HARNESSLOOP_", extra="ignore")

    model: str | None = None
    provider: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    dataset: str | None = None
    harness: str | None = None
    out_dir: str | None = None
    max_rounds: int | None = None
    concurrency: int | None = None
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None


def _env_files(profile_path: Path) -> tuple[Path, ...]:
    """``.env`` in the profile directory wins over the current working directory."""
    candidates = (Path.cwd() / ".env", profile_path.parent / ".env")
    unique = {path.resolve() for path in candidates if path.is_file()}
    return tuple(sorted(unique, key=lambda path: path.parent != Path.cwd().resolve()))


def load_config(path: str | Path) -> LoopConfig:
    """Load a YAML profile and resolve relative paths against the profile."""
    profile_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {profile_path}")

    env_files = _env_files(profile_path)
    env = _EnvOverrides(_env_file=env_files or None)
    if env.model:
        raw["model"] = env.model
    if env.provider:
        raw["provider"] = env.provider
    if env.api_base:
        raw["api_base"] = env.api_base
    if env.temperature is not None:
        raw["temperature"] = env.temperature
    if env.dataset:
        raw.setdefault("dataset", {})["path"] = env.dataset
    if env.harness:
        raw.setdefault("harness", {})["path"] = env.harness
    if env.out_dir:
        raw.setdefault("report", {})["out_dir"] = env.out_dir
    if env.max_rounds is not None:
        raw.setdefault("loop", {})["max_rounds"] = env.max_rounds
    if env.concurrency is not None:
        raw.setdefault("loop", {})["concurrency"] = env.concurrency
    if env.input_price_per_1m is not None:
        raw.setdefault("pricing", {})["input_per_1m"] = env.input_price_per_1m
    if env.output_price_per_1m is not None:
        raw.setdefault("pricing", {})["output_per_1m"] = env.output_price_per_1m

    config = LoopConfig.model_validate(raw)
    config.api_key = env.api_key

    # Resolve paths relative to the profile, keeping absolute paths untouched.
    base = profile_path.parent
    dataset_path = Path(config.dataset.path).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = base / dataset_path
    config.dataset.path = str(dataset_path.resolve())

    if config.harness.path:
        harness_path = Path(config.harness.path).expanduser()
        if not harness_path.is_absolute():
            harness_path = base / harness_path
        config.harness.path = str(harness_path.resolve())

    out_dir = Path(config.report.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    config.report.out_dir = str(out_dir.resolve())

    config.plugins.attribution = [
        _resolve_plugin_spec(spec, base) for spec in config.plugins.attribution
    ]
    config.plugins.labels = [
        _resolve_plugin_spec(spec, base) for spec in config.plugins.labels
    ]
    config.plugins.edit_ops = [
        _resolve_plugin_spec(spec, base) for spec in config.plugins.edit_ops
    ]

    if config.executor.mcp and config.executor.mcp.cwd:
        cwd = Path(config.executor.mcp.cwd).expanduser()
        if not cwd.is_absolute():
            cwd = base / cwd
        config.executor.mcp.cwd = str(cwd.resolve())
    return config


def _resolve_plugin_spec(spec: str, base: Path) -> str:
    """Resolve a ``file.py[:attr]`` spec against the profile directory."""
    target, sep, attr = spec.partition(":")
    if not target.endswith(".py"):
        return spec
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = str(path.resolve())
    return f"{resolved}:{attr}" if sep else resolved


DEFAULT_HARNESS = {
    "system_prompt": (
        "你是 SQL 分析助手。根据数据库 schema 回答用户问题，只输出一条可执行的 SQL。"
    ),
    "tools": [
        {
            "name": "run_sql",
            "description": "在目标数据库上执行一条只读 SQL 并返回结果预览，用于自检查询是否正确。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "单条只读 SELECT 语句"}
                },
                "required": ["sql"],
            },
        },
        {
            "name": "list_tables",
            "description": "列出数据库中的全部表名。",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "describe_table",
            "description": "查看指定表的列定义。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "表名"}
                },
                "required": ["table"],
            },
        },
    ],
    "few_shot_examples": [],
    "output_format": "```sql\n<query>\n```",
}
