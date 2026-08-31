"""Runtime object that wires models, tools, context, state, and persistence."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

from . import checkpoint as checkpointlib
from .agent_loop import run_agent_loop
from .context_manager import ContextManager
from .memory import LayeredMemory
from .prompt_prefix import build_prompt_prefix
from .run_store import RunStore
from .security import redact
from .session_store import SessionStore
from .tool_executor import ToolExecutor
from .tools import BASE_TOOL_SPECS
from .workspace import WorkspaceContext


def _now():
    return datetime.now(timezone.utc).isoformat()


class EasyCoding:
    def __init__(
        self, model_client, workspace, session_store=None, session=None,
        approval_policy="ask", approval_callback=None, max_steps=6,
        max_new_tokens=512, allowed_tools=None, read_only=False,
        secret_env_names=(),
        durable_memory_enabled=True, resume_enabled=True,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root).resolve()
        self.approval_policy = approval_policy
        self.read_only = bool(read_only)
        self.max_steps = int(max_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.durable_memory_enabled = bool(durable_memory_enabled)
        self.resume_enabled = bool(resume_enabled)
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        state_root = self.root / ".easycoding"
        self.session_store = session_store or SessionStore(state_root / "sessions")
        self.run_store = RunStore(state_root / "runs")
        self.session = session or {
            "id": "session_" + uuid.uuid4().hex[:12],
            "created_at": _now(),
            "workspace_root": str(self.root),
            "history": [],
            "memory": {},
            "checkpoints": {"current_id": "", "items": {}},
        }
        self.memory = LayeredMemory(self.root, self.session.setdefault("memory", {}))
        self.tools = {
            name: spec for name, spec in BASE_TOOL_SPECS.items()
            if self.allowed_tools is None or name in self.allowed_tools
        }
        self.prefix_state = build_prompt_prefix(self.workspace, self.tools)
        self.tool_executor = ToolExecutor(
            self.root, BASE_TOOL_SPECS, approval_policy=approval_policy,
            allowed_tools=self.allowed_tools, read_only=read_only,
            approval_callback=approval_callback,
        )
        self.context_manager = ContextManager(self)
        self.secret_values = [os.environ.get(name, "") for name in secret_env_names]
        self.resume_state = checkpointlib.evaluate_resume_state(self)
        self.current_task_state = None
        self.last_durable_changes = self._empty_durable_changes()

    @staticmethod
    def _normalize_allowed_tools(allowed_tools):
        if allowed_tools is None:
            return None
        normalized = tuple(str(name).strip() for name in allowed_tools)
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence")
        unknown = sorted(set(normalized) - set(BASE_TOOL_SPECS))
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        return normalized

    @classmethod
    def from_session(cls, session_id, **kwargs):
        store = kwargs.get("session_store")
        if store is None:
            workspace = kwargs["workspace"]
            store = SessionStore(Path(workspace.repo_root) / ".easycoding" / "sessions")
            kwargs["session_store"] = store
        session = store.load(session_id)
        if Path(session.get("workspace_root", "")).resolve() != Path(kwargs["workspace"].repo_root).resolve():
            raise ValueError("session belongs to a different workspace")
        return cls(session=session, **kwargs)

    @property
    def session_path(self):
        return self.session_store.path(self.session["id"])

    def ask(self, user_message):
        self.last_durable_changes = self._empty_durable_changes()
        return run_agent_loop(self, str(user_message))

    @staticmethod
    def _empty_durable_changes():
        return {
            "intent_detected": False, "promoted": [], "deduplicated": [],
            "superseded": [], "rejected": [],
        }

    def promote_durable_memory(self, user_message, run_id):
        if not self.durable_memory_enabled:
            self.last_durable_changes = {
                **self._empty_durable_changes(), "disabled": True,
            }
            return self.last_durable_changes
        changes = self.memory.promote_explicit(user_message)
        self.last_durable_changes = changes
        for event_name, key in (
            ("durable_promoted", "promoted"),
            ("durable_deduplicated", "deduplicated"),
            ("durable_superseded", "superseded"),
            ("durable_rejected", "rejected"),
        ):
            for item in changes[key]:
                self.emit_trace(run_id, event_name, item)
        return changes

    def refresh_runtime_state(self):
        refreshed = WorkspaceContext.build(self.workspace.cwd)
        new_prefix = build_prompt_prefix(refreshed, self.tools)
        self.workspace = refreshed
        self.root = Path(refreshed.repo_root).resolve()
        if new_prefix.hash != self.prefix_state.hash:
            self.prefix_state = new_prefix
        self.resume_state = checkpointlib.evaluate_resume_state(self)

    def record(self, event):
        payload = dict(event)
        payload.setdefault("created_at", _now())
        safe = json.loads(redact(json.dumps(payload, ensure_ascii=False), self.secret_values))
        self.session.setdefault("history", []).append(safe)
        self.session["history"] = self.session["history"][-100:]
        self.session_store.save(self.session)

    def emit_trace(self, run_id, event, details=None):
        payload = {"event": event, "created_at": _now(), **dict(details or {})}
        safe = json.loads(redact(json.dumps(payload, ensure_ascii=False), self.secret_values))
        self.run_store.append_trace(run_id, safe)

    def finish_run(self, task_state, prompt_metadata):
        self.run_store.write_task_state(task_state)
        report = {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "attempts": task_state.attempts,
            "tool_steps": task_state.tool_steps,
            "checkpoint_id": task_state.checkpoint_id,
            "prompt_metadata": prompt_metadata,
            "model_metadata": getattr(self.model_client, "last_completion_metadata", {}),
            "resume_state": self.resume_state,
            "durable_memory_changes": self.last_durable_changes,
            "feature_flags": {
                "durable_memory_enabled": self.durable_memory_enabled,
                "resume_enabled": self.resume_enabled,
            },
        }
        safe = json.loads(redact(json.dumps(report, ensure_ascii=False), self.secret_values))
        self.run_store.write_report(task_state.run_id, safe)
        self.emit_trace(task_state.run_id, "run_finished", {"status": task_state.status, "stop_reason": task_state.stop_reason})
        self.session_store.save(self.session)
        return task_state.final_answer

    def create_checkpoint(self, task_state, trigger, blocker="", next_step=""):
        item = checkpointlib.create_checkpoint(self, task_state, trigger, blocker, next_step)
        self.resume_state = checkpointlib.evaluate_resume_state(self)
        self.session_store.save(self.session)
        self.emit_trace(task_state.run_id, "checkpoint_created", {"checkpoint_id": item["checkpoint_id"], "trigger": trigger})
        return item

    def checkpoint_text(self):
        return checkpointlib.render_checkpoint(self)

    def update_memory_after_tool(self, name, args, result):
        path = str(args.get("path", ""))
        accepted = result.status in {"success", "partial_success"}
        if path and accepted and name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(path)
        if path and name == "read_file" and result.status == "success":
            summary = " | ".join(result.text.splitlines()[:3])[:500]
            self.memory.set_file_summary(path, summary)
            self.memory.append_note(summary, tags=(path,))
        elif path and name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(path)

    def memory_text(self):
        return self.memory.render(durable_details=True)

    def reset(self):
        self.session["history"] = []
        self.session["memory"] = {}
        self.session["checkpoints"] = {"current_id": "", "items": {}}
        self.memory = LayeredMemory(self.root, self.session["memory"])
        self.resume_state = checkpointlib.evaluate_resume_state(self)
        self.session_store.save(self.session)

    @staticmethod
    def parse(raw):
        text = str(raw).strip()
        match = re.fullmatch(r"<(tool|final)>(.*?)</\1>", text, re.DOTALL)
        if match is None:
            return "retry", {
                "code": "invalid_protocol",
                "message": "model must return exactly one <tool> or <final> block with no extra text",
            }
        kind, body = match.groups()
        if any(marker in body for marker in ("<tool>", "</tool>", "<final>", "</final>")):
            return "retry", {
                "code": "invalid_protocol",
                "message": "model response must not contain nested or multiple protocol blocks",
            }
        if kind == "tool":
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return "retry", {
                    "code": "invalid_protocol",
                    "message": "tool payload is not valid JSON",
                }
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str) or not isinstance(payload.get("args"), dict):
                return "retry", {
                    "code": "invalid_protocol",
                    "message": "tool payload must contain string name and object args",
                }
            if not payload["name"].strip():
                return "retry", {
                    "code": "invalid_protocol",
                    "message": "tool name must not be empty",
                }
            return "tool", payload
        answer = body.strip()
        if not answer:
            return "retry", {
                "code": "invalid_protocol",
                "message": "final answer must not be empty",
            }
        return "final", answer
