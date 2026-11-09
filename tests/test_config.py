from __future__ import annotations

from pathlib import Path

import yaml

from harnessloop.config import load_config


def test_load_config_resolves_relative_paths(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    profile = tmp_path / "config.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "dataset": {"path": "data/tasks.jsonl", "kind": "sql"},
                "harness": {"path": "harness.json"},
                "report": {"out_dir": "runs"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(profile)
    assert Path(config.dataset.path) == (tmp_path / "data/tasks.jsonl").resolve()
    assert Path(config.harness.path or "") == (tmp_path / "harness.json").resolve()
    assert Path(config.report.out_dir) == (tmp_path / "runs").resolve()
    assert config.llm_model() == "openai/gpt-4o-mini"
    assert config.llm_model("openai/gpt-4o") == "openai/gpt-4o"


def test_env_override(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    profile = tmp_path / "config.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "model": "gpt-4o-mini",
                "dataset": {"path": "tasks.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HARNESSLOOP_MODEL", "gpt-4o")
    config = load_config(profile)
    assert config.model == "gpt-4o"


ENV_KEYS = (
    "HARNESSLOOP_MODEL",
    "HARNESSLOOP_PROVIDER",
    "HARNESSLOOP_API_BASE",
    "HARNESSLOOP_API_KEY",
    "HARNESSLOOP_TEMPERATURE",
)


def _clear_env(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_dotenv_in_profile_dir(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    profile = tmp_path / "config.yaml"
    profile.write_text(
        yaml.safe_dump({"dataset": {"path": "tasks.jsonl"}}), encoding="utf-8"
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "HARNESSLOOP_MODEL=gpt-4o",
                "HARNESSLOOP_PROVIDER=openai",
                "HARNESSLOOP_API_BASE=https://gateway.example.com/v1",
                "HARNESSLOOP_API_KEY=sk-from-dotenv",
                "HARNESSLOOP_TEMPERATURE=0.3",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(profile)
    assert config.model == "gpt-4o"
    assert config.llm_model() == "openai/gpt-4o"
    assert config.api_base == "https://gateway.example.com/v1"
    assert config.api_key == "sk-from-dotenv"
    assert config.temperature == 0.3


def test_real_env_wins_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    profile = tmp_path / "config.yaml"
    profile.write_text(
        yaml.safe_dump({"dataset": {"path": "tasks.jsonl"}}), encoding="utf-8"
    )
    (tmp_path / ".env").write_text("HARNESSLOOP_MODEL=gpt-4o-mini", encoding="utf-8")
    monkeypatch.setenv("HARNESSLOOP_MODEL", "gpt-4o")
    config = load_config(profile)
    assert config.model == "gpt-4o"


def test_api_key_never_serialized(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    profile = tmp_path / "config.yaml"
    profile.write_text(
        yaml.safe_dump({"dataset": {"path": "tasks.jsonl"}}), encoding="utf-8"
    )
    (tmp_path / ".env").write_text("HARNESSLOOP_API_KEY=sk-secret", encoding="utf-8")
    config = load_config(profile)
    assert config.api_key == "sk-secret"
    dumped = config.model_dump()
    assert "api_key" not in dumped
    assert "sk-secret" not in repr(config)

