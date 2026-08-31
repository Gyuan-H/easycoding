"""Bounded read-only child-agent execution."""

import json
from pathlib import Path
import time
import uuid

from .security import resolve_in_workspace
from .tool_types import ToolRunOutput
from .workspace import WorkspaceContext, clip


DELEGATE_ALLOWED_TOOLS = ("list_files", "read_file", "search")
MAX_DELEGATE_OUTPUT = 4000


def _read_trace(parent, run_id):
    path = parent.run_store.run_dir(run_id) / "trace.jsonl"
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _evidence_from_trace(trace):
    evidence = []
    seen = set()
    for item in trace:
        if item.get("event") != "tool_executed" or item.get("status") != "success":
            continue
        name = item.get("name")
        if name not in DELEGATE_ALLOWED_TOOLS:
            continue
        args = item.get("args", {})
        path = str(args.get("path", "."))
        key = (name, path, args.get("start"), args.get("pattern"))
        if key in seen:
            continue
        seen.add(key)
        entry = {"tool": name, "path": path}
        if name == "read_file":
            entry["line"] = int(args.get("start", 1))
        if name == "search":
            entry["pattern"] = str(args.get("pattern", ""))[:200]
        evidence.append(entry)
        if len(evidence) >= 20:
            break
    return evidence


def _provider_with_timeout(client, timeout_seconds):
    current = client
    for _ in range(3):
        if hasattr(current, "timeout"):
            original = current.timeout
            current.timeout = min(int(original), int(timeout_seconds))
            return current, original
        current = getattr(current, "client", None)
        if current is None:
            break
    return None, None


def execute_delegate(parent, args):
    """Run one child EasyCoding instance and return a JSON tool result."""
    from .runtime import EasyCoding

    if parent.delegation_depth >= 1:
        return ToolRunOutput(
            json.dumps({"status": "rejected", "reason": "nested_delegate"}),
            error_code="nested_delegate",
        )
    task = str(args["task"]).strip()
    max_steps = int(args.get("max_steps", 3))
    timeout_seconds = int(args.get("timeout_seconds", 60))
    raw_paths = [str(item) for item in args.get("paths", ["."])]
    scopes = []
    rendered_paths = []
    for raw_path in raw_paths:
        scope = resolve_in_workspace(parent.root, raw_path, must_exist=True)
        scopes.append(scope)
        rendered_paths.append(scope.relative_to(parent.root).as_posix() or ".")

    parent_run_id = parent.current_task_state.run_id
    delegation_id = "delegate_" + uuid.uuid4().hex[:12]
    started = time.perf_counter()
    child = EasyCoding(
        parent.model_client,
        WorkspaceContext.build(parent.workspace.cwd),
        approval_policy="never",
        max_steps=max_steps,
        max_new_tokens=min(parent.max_new_tokens, 768),
        allowed_tools=DELEGATE_ALLOWED_TOOLS,
        read_only=True,
        durable_memory_enabled=False,
        resume_enabled=False,
        path_scopes=scopes,
        agent_role="delegate",
        parent_run_id=parent_run_id,
        delegation_id=delegation_id,
        delegation_depth=parent.delegation_depth + 1,
    )
    child.secret_values = list(parent.secret_values)
    parent.emit_trace(parent_run_id, "delegate_started", {
        "delegation_id": delegation_id,
        "parent_run_id": parent_run_id,
        "child_session_id": child.session["id"],
        "task": task[:500],
        "paths": rendered_paths,
        "allowed_tools": list(DELEGATE_ALLOWED_TOOLS),
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
    })
    timeout_target, original_timeout = _provider_with_timeout(
        parent.model_client, timeout_seconds
    )
    error = ""
    answer = ""
    try:
        request = (
            "Delegated read-only investigation. Use only list_files, read_file, and search. "
            "Do not modify files or invoke another delegate. Return a concise evidence-based "
            f"summary. Allowed paths: {', '.join(rendered_paths)}.\n\nTask: {task}"
        )
        answer = child.ask(request)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        if timeout_target is not None:
            timeout_target.timeout = original_timeout

    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    state = child.current_task_state
    child_run_id = state.run_id if state else ""
    trace = _read_trace(parent, child_run_id) if child_run_id else []
    tool_events = [item for item in trace if item.get("event") == "tool_executed"]
    timed_out = duration_ms > timeout_seconds * 1000
    completed = bool(state and state.status == "completed" and not error and not timed_out)
    status = "success" if completed else "failed"
    if timed_out:
        error = f"delegate exceeded {timeout_seconds}s wall-clock budget"
    if not error and not completed:
        error = state.stop_reason if state else "delegate did not start"
    record = {
        "delegation_id": delegation_id,
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
        "child_session_id": child.session["id"],
        "status": status,
        "task": task[:500],
        "paths": rendered_paths,
        "allowed_tools": list(DELEGATE_ALLOWED_TOOLS),
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "attempts": state.attempts if state else 0,
        "tool_steps": state.tool_steps if state else 0,
        "stop_reason": state.stop_reason if state else "delegate_error",
        "tool_error_codes": [
            item.get("tool_error_code", "") for item in tool_events
            if item.get("tool_error_code")
        ],
        "summary": clip(answer, 2000) if completed else "",
        "error": error,
        "evidence": _evidence_from_trace(trace),
    }
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    rendered = clip(serialized, MAX_DELEGATE_OUTPUT)
    record["output_truncated"] = rendered != serialized
    parent.delegate_records.append(record)
    parent.emit_trace(
        parent_run_id,
        "delegate_completed" if completed else "delegate_failed",
        record,
    )
    return ToolRunOutput(
        rendered,
        error_code="" if completed else ("delegate_timeout" if timed_out else "delegate_failed"),
        metadata={"timed_out": timed_out, "output_truncated": rendered != serialized},
    )
