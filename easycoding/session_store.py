"""Small JSON-backed session store."""

import json
from pathlib import Path
import tempfile


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        destination = self.path(session["id"])
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=self.root, suffix=".tmp"
        ) as handle:
            json.dump(session, handle, ensure_ascii=False, indent=2)
            temporary = Path(handle.name)
        temporary.replace(destination)
        return destination

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        candidates = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return candidates[-1].stem if candidates else None

