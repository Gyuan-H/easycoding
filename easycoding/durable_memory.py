"""Conservative, workspace-scoped durable memory storage."""

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path


TOPICS = {
    "project-conventions": ("Project convention", "项目约定"),
    "key-decisions": ("Decision", "决策"),
    "dependency-facts": ("Dependency", "依赖"),
    "user-preferences": ("Preference", "偏好"),
}

_LABEL_TO_TOPIC = {
    label.lower(): topic
    for topic, labels in TOPICS.items()
    for label in labels
}
_LABEL_PATTERN = "|".join(
    re.escape(label) for labels in TOPICS.values() for label in labels
)
_INTENT_RE = re.compile(
    r"\b(?:remember|save|store|persist|capture|note)\b|记住|保存|长期记录|长期记忆|持久记忆",
    re.IGNORECASE,
)
_LABELED_FACT_RE = re.compile(
    rf"(?P<label>{_LABEL_PATTERN})\s*[:：]\s*(?P<text>[^\n\r]+)",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|passwd|client[_ -]?secret)"
    r"\s*[:=：]\s*\S+|\bsk-[A-Za-z0-9_-]{8,}",
    re.IGNORECASE,
)
_TRANSIENT_RE = re.compile(
    r"\b(?:traceback|exception|temporary failure|failed at|stack trace)\b|"
    r"(?:临时|偶发)(?:错误|失败)|错误堆栈|本次报错",
    re.IGNORECASE,
)
_SHELL_OUTPUT_RE = re.compile(
    r"(?:^|\s)(?:stdout|stderr|exit_code)\s*[:=]|shell command output|命令输出",
    re.IGNORECASE,
)
_TEMP_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\Users\\[^\\]+\\AppData\\Local\\Temp\\|/tmp/|\\Temp\\)",
    re.IGNORECASE,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _normalize(text):
    return " ".join(str(text).strip().lower().split())


def _tokens(text):
    value = str(text).lower()
    result = set(re.findall(r"[a-z0-9_./-]{2,}", value))
    for group in re.findall(r"[\u4e00-\u9fff]+", value):
        if len(group) == 1:
            result.add(group)
        else:
            result.update(group[index:index + 2] for index in range(len(group) - 1))
    return result


def _subject_key(topic, text):
    normalized = _normalize(text).replace("：", ":")
    for separator in (":", "="):
        if separator in normalized:
            subject = normalized.split(separator, 1)[0].strip(" -")
            if len(subject) >= 2:
                return f"{topic}:{subject[:120]}"
    return f"{topic}:{normalized[:120]}"


def _validate_fact(text):
    value = str(text).strip()
    if len(value) < 3:
        return "too_short"
    if len(value) > 500:
        return "too_long"
    if _SECRET_RE.search(value):
        return "sensitive_information"
    if _TRANSIENT_RE.search(value):
        return "transient_failure"
    if _SHELL_OUTPUT_RE.search(value):
        return "shell_output"
    if _TEMP_PATH_RE.search(value):
        return "temporary_path"
    return ""


class DurableMemoryStore:
    """JSON-backed facts that survive session resets within one workspace."""

    def __init__(self, root):
        self.root = Path(root).resolve() / ".easycoding" / "memory"
        self.index_path = self.root / "index.json"
        self.topics_root = self.root / "topics"

    def promote_explicit(self, user_message):
        result = {
            "intent_detected": bool(_INTENT_RE.search(str(user_message))),
            "promoted": [],
            "deduplicated": [],
            "superseded": [],
            "rejected": [],
        }
        if not result["intent_detected"]:
            return result
        facts = []
        for match in _LABELED_FACT_RE.finditer(str(user_message)):
            label = match.group("label").lower()
            facts.append((_LABEL_TO_TOPIC[label], match.group("text").strip()))
        if not facts:
            result["rejected"].append({"reason": "no_labeled_fact"})
            return result
        for topic, text in facts:
            reason = _validate_fact(text)
            if reason:
                result["rejected"].append({"topic": topic, "reason": reason})
                continue
            change = self.promote(topic, text)
            status = change.pop("status")
            if status == "duplicate":
                result["deduplicated"].append(change)
            else:
                result["promoted"].append(change)
                if status == "superseded":
                    result["superseded"].append({
                        "topic": topic,
                        "subject": change["subject"],
                        "previous_id": change["previous_id"],
                        "replacement_id": change["id"],
                    })
        return result

    def promote(self, topic, text):
        if topic not in TOPICS:
            raise ValueError(f"unknown durable-memory topic: {topic}")
        self._ensure_layout()
        payload = self._read_topic(topic)
        normalized = _normalize(text)
        subject = _subject_key(topic, text)
        for note in payload["notes"]:
            if _normalize(note.get("text", "")) == normalized:
                return {
                    "status": "duplicate", "topic": topic,
                    "id": note["id"], "subject": note.get("subject", subject),
                }
        now = _now()
        note_id = "mem_" + hashlib.sha256(
            f"{topic}\0{subject}\0{normalized}".encode("utf-8")
        ).hexdigest()[:16]
        note = {
            "id": note_id,
            "subject": subject,
            "text": str(text).strip(),
            "created_at": now,
            "updated_at": now,
        }
        previous = next(
            (item for item in payload["notes"] if item.get("subject") == subject), None
        )
        if previous:
            note["created_at"] = previous.get("created_at", now)
            payload["notes"] = [
                item for item in payload["notes"] if item.get("subject") != subject
            ]
        payload["notes"].append(note)
        _atomic_json(self._topic_path(topic), payload)
        self._write_index()
        change = {
            "status": "superseded" if previous else "promoted",
            "topic": topic, "id": note_id, "subject": subject, "text": note["text"],
        }
        if previous:
            change["previous_id"] = previous.get("id", "")
        return change

    def retrieve(self, query, limit=3):
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        ranked = []
        for topic in TOPICS:
            labels = " ".join(TOPICS[topic])
            for index, note in enumerate(self._read_topic(topic)["notes"]):
                score = len(query_tokens & _tokens(note.get("text", "") + " " + labels))
                if score:
                    ranked.append((score, note.get("updated_at", ""), index, topic, note))
        ranked.sort(reverse=True)
        return [
            {
                "id": note.get("id", ""),
                "text": note.get("text", ""),
                "tags": list(TOPICS[topic]),
                "kind": "durable",
                "topic": topic,
            }
            for _, _, _, topic, note in ranked[:max(0, int(limit))]
        ]

    def topic_counts(self):
        return {
            topic: len(self._read_topic(topic)["notes"])
            for topic in TOPICS
        }

    def all_notes(self):
        return [
            {**note, "topic": topic}
            for topic in TOPICS
            for note in self._read_topic(topic)["notes"]
        ]

    def _ensure_layout(self):
        self.topics_root.mkdir(parents=True, exist_ok=True)
        for topic in TOPICS:
            path = self._topic_path(topic)
            if not path.exists():
                _atomic_json(path, {"schema_version": 1, "topic": topic, "notes": []})
        if not self.index_path.exists():
            self._write_index()

    def _topic_path(self, topic):
        return self.topics_root / f"{topic}.json"

    def _read_topic(self, topic):
        path = self._topic_path(topic)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {"schema_version": 1, "topic": topic, "notes": []}
        if not isinstance(payload, dict) or not isinstance(payload.get("notes"), list):
            return {"schema_version": 1, "topic": topic, "notes": []}
        return payload

    def _write_index(self):
        _atomic_json(self.index_path, {
            "schema_version": 1,
            "updated_at": _now(),
            "topics": {
                topic: {
                    "count": len(self._read_topic(topic)["notes"]),
                    "labels": list(TOPICS[topic]),
                }
                for topic in TOPICS
            },
        })
