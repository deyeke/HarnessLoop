from __future__ import annotations

import json

from harnessloop.cli import main


def test_cli_init_creates_files(tmp_path, capsys) -> None:
    target = tmp_path / "demo"
    assert main(["init", "--dir", str(target)]) == 0
    assert (target / "harness.json").exists()
    assert (target / "config.yaml").exists()
    assert "created" in capsys.readouterr().out


def test_cli_diff(tmp_path, capsys) -> None:
    previous = {"system_prompt": "base", "tools": [], "few_shot_examples": []}
    candidate = {"system_prompt": "base\n\n使用 LEFT JOIN", "tools": [], "few_shot_examples": []}
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(previous), encoding="utf-8")
    second.write_text(json.dumps(candidate), encoding="utf-8")

    assert main(["diff", str(first), str(second)]) == 0
    assert "LEFT JOIN" in capsys.readouterr().out
