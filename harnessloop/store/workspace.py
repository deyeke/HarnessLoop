"""Filesystem run store.

v0.1 deliberately uses plain files: ``runs/run_<id>/iter_<n>/`` with JSONL
trajectories and immutable harness snapshots. No database, no queue.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from harnessloop.models import HarnessState


def _to_jsonable(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return {key: _to_jsonable(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [_to_jsonable(value) for value in item]
    return item


class RunDir:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_id = self.path.name

    @classmethod
    def create(cls, out_dir: str | Path, run_id: str | None = None) -> RunDir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        identifier = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        candidate = out / identifier
        suffix = 1
        while candidate.exists() and any(candidate.iterdir()):
            candidate = out / f"{identifier}_{suffix}"
            suffix += 1
        return cls(candidate)

    def iter_dir(self, iteration: int) -> Path:
        directory = self.path / f"iter_{iteration}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_text(self, path: str | Path, text: str, *, overwrite: bool = True) -> Path:
        target = path if isinstance(path, Path) else self.path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"immutable artifact already exists: {target}")
        target.write_text(text, encoding="utf-8")
        return target

    def write_json(self, path: str | Path, data: Any, *, overwrite: bool = True) -> Path:
        return self.write_text(
            path,
            json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2),
            overwrite=overwrite,
        )

    def write_jsonl(self, path: str | Path, rows: Iterable[Any]) -> Path:
        lines = [
            json.dumps(_to_jsonable(row), ensure_ascii=False) for row in rows
        ]
        return self.write_text(path, "\n".join(lines) + ("\n" if lines else ""))

    def save_harness(
        self,
        iteration: int,
        harness: HarnessState,
        *,
        name: str = "harness.json",
        overwrite: bool = False,
    ) -> Path:
        path = self.iter_dir(iteration) / name
        return self.write_json(path, harness, overwrite=overwrite)

    def set_latest(self, iteration: int, harness_path: str) -> None:
        self.write_json(
            "latest.json",
            {"iteration": iteration, "harness": harness_path},
        )

    def report_paths(self) -> list[str]:
        return [str(path) for path in sorted(self.path.glob("iter_*/report.md"))]


def load_harness(path: str | Path) -> HarnessState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessState.model_validate(payload)
