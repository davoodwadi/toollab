from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

from tool_lab.models.base import _to_serializable



class ResultWriter:
    def __init__(self, output_root: str | Path, experiment_name: str, provider: str, model_name: str) -> None:
        self.session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        experiment_slug = f"{_slugify(experiment_name)}-{_slugify(provider)}-{_slugify(model_name)}"
        self.session_dir = Path(output_root) / experiment_slug
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, payload: dict[str, Any]) -> None:
        path = self.session_dir / f"config.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def write_events(self, record: dict[str, Any]) -> None:
        """Flatten each run's trace into individual rows and append to events.jsonl."""
        lines = []
        run_meta = {k: record[k] for k in record.keys() if k!='trace'}
        for event in record.get("trace", []):
            flat = {**run_meta, **_to_serializable(event)}
            lines.append(json.dumps(flat))
        if lines:
            with (self.session_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")



def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return slug.lower() or "run"
