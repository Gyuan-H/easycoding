"""Persistence for per-run state, trace, and report artifacts."""

import json
from pathlib import Path
import tempfile


SCHEMA_VERSION = 1


class RunStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id):
        return self.root / run_id

    def start_run(self, task_state):
        directory = self.run_dir(task_state.run_id)
        directory.mkdir(parents=True, exist_ok=False)
        self.write_task_state(task_state)
        return directory

    def _atomic_json(self, destination, value):
        destination = Path(destination)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=destination.parent, suffix=".tmp"
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            temporary = Path(handle.name)
        temporary.replace(destination)

    def write_task_state(self, task_state):
        payload = {"schema_version": SCHEMA_VERSION, **task_state.to_dict()}
        self._atomic_json(self.run_dir(task_state.run_id) / "task_state.json", payload)

    def append_trace(self, run_id, event):
        path = self.run_dir(run_id) / "trace.jsonl"
        payload = {"schema_version": SCHEMA_VERSION, **dict(event)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def write_report(self, run_id, report):
        payload = {"schema_version": SCHEMA_VERSION, **dict(report)}
        self._atomic_json(self.run_dir(run_id) / "report.json", payload)
        return self.run_dir(run_id) / "report.json"

