"""Small, inspectable working memory with file freshness checks."""

import hashlib
import re
from pathlib import Path

from .durable_memory import DurableMemoryStore, TOPICS


def _hash_file(root, relative):
    path = (Path(root) / relative).resolve()
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def default_state():
    return {
        "task_summary": "",
        "recent_files": [],
        "file_summaries": {},
        "episodic_notes": [],
    }


class LayeredMemory:
    def __init__(self, root, state=None):
        self.root = Path(root).resolve()
        self.durable = DurableMemoryStore(self.root)
        self.state = state if isinstance(state, dict) else default_state()
        for key, value in default_state().items():
            self.state.setdefault(key, value)

    def set_task_summary(self, text):
        self.state["task_summary"] = str(text)[:300]

    def remember_file(self, path):
        relative = self._relative(path)
        files = [item for item in self.state["recent_files"] if item != relative]
        files.append(relative)
        self.state["recent_files"] = files[-8:]

    def set_file_summary(self, path, summary):
        relative = self._relative(path)
        self.state["file_summaries"][relative] = {
            "summary": str(summary)[:500],
            "freshness": _hash_file(self.root, relative),
        }

    def invalidate_file_summary(self, path):
        self.state["file_summaries"].pop(self._relative(path), None)

    def append_note(self, text, tags=()):
        note = {"text": str(text)[:500], "tags": [str(tag) for tag in tags]}
        self.state["episodic_notes"].append(note)
        self.state["episodic_notes"] = self.state["episodic_notes"][-12:]

    def retrieve(self, query, limit=3, include_durable=True):
        tokens = set(re.findall(r"[\w./-]+", str(query).lower()))
        ranked = []
        for index, note in enumerate(self.state["episodic_notes"]):
            text = note.get("text", "")
            tags = " ".join(note.get("tags", []))
            note_tokens = set(re.findall(r"[\w./-]+", (text + " " + tags).lower()))
            score = len(tokens & note_tokens)
            if score or any(tag.lower() in str(query).lower() for tag in note.get("tags", [])):
                ranked.append((score, index, {**note, "kind": "working"}))
        if include_durable:
            for index, note in enumerate(self.durable.retrieve(query, limit=limit)):
                text = note.get("text", "")
                tags = " ".join(note.get("tags", []))
                note_tokens = set(re.findall(r"[\w./-]+", (text + " " + tags).lower()))
                score = len(tokens & note_tokens)
                ranked.append((max(1, score), len(ranked) + index, note))
        ranked.sort(reverse=True)
        return [note for _, _, note in ranked[:limit]]

    def promote_explicit(self, user_message):
        return self.durable.promote_explicit(user_message)

    def render(self, durable_details=False):
        lines = ["Working memory:", f"- task: {self.state['task_summary'] or '-'}"]
        files = self.state["recent_files"]
        lines.append("- recent_files: " + (", ".join(files) if files else "-"))
        for path, item in sorted(self.state["file_summaries"].items()):
            if item.get("freshness") and item["freshness"] == _hash_file(self.root, path):
                lines.append(f"- {path}: {item.get('summary', '')}")
        counts = self.durable.topic_counts()
        lines.append("Durable memory topics:")
        for topic, labels in TOPICS.items():
            lines.append(f"- {labels[0]} / {labels[1]}: {counts[topic]}")
        if durable_details:
            for note in self.durable.all_notes():
                lines.append(f"- [{note['topic']}] {note.get('text', '')}")
        return "\n".join(lines)

    def _relative(self, path):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve().relative_to(self.root).as_posix()
