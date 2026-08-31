"""Checkpoint creation and conservative resume-state evaluation."""

import hashlib
from pathlib import Path
import uuid

from .security import DEFAULT_SHELL_ENV_ALLOWLIST


SCHEMA_VERSION = 1


def file_freshness(root, relative):
    try:
        return hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()
    except OSError:
        return ""


def runtime_identity(agent):
    return {
        "cwd": str(agent.root),
        "model": str(getattr(agent.model_client, "model", "unknown")),
        "model_client": agent.model_client.__class__.__name__,
        "approval_policy": agent.approval_policy,
        "read_only": agent.read_only,
        "max_steps": agent.max_steps,
        "max_new_tokens": agent.max_new_tokens,
        "allowed_tools": list(agent.allowed_tools) if agent.allowed_tools is not None else None,
        "shell_env_allowlist": list(DEFAULT_SHELL_ENV_ALLOWLIST),
        "workspace_fingerprint": agent.workspace.fingerprint(),
        "tool_signature": agent.prefix_state.tool_signature,
        "durable_memory_enabled": getattr(agent, "durable_memory_enabled", True),
        "resume_enabled": getattr(agent, "resume_enabled", True),
    }


def current_checkpoint(agent):
    state = agent.session.setdefault("checkpoints", {"current_id": "", "items": {}})
    checkpoint_id = state.get("current_id", "")
    return state.get("items", {}).get(checkpoint_id) if checkpoint_id else None


def create_checkpoint(agent, task_state, trigger, blocker="", next_step=""):
    state = agent.session.setdefault("checkpoints", {"current_id": "", "items": {}})
    previous = current_checkpoint(agent)
    checkpoint_id = "ckpt_" + uuid.uuid4().hex[:10]
    key_files = []
    for path in agent.memory.state.get("recent_files", [])[-4:]:
        key_files.append({"path": path, "freshness": file_freshness(agent.root, path)})
    item = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": previous.get("checkpoint_id", "") if previous else "",
        "trigger": trigger,
        "current_goal": task_state.user_request,
        "completed": [],
        "excluded": [],
        "current_blocker": blocker,
        "next_step": next_step,
        "key_files": key_files,
        "workspace_fingerprint": agent.workspace.fingerprint(),
        "runtime_identity": runtime_identity(agent),
        "summary": task_state.final_answer or task_state.stop_reason,
    }
    state["items"][checkpoint_id] = item
    state["current_id"] = checkpoint_id
    task_state.checkpoint_id = checkpoint_id
    return item


def evaluate_resume_state(agent):
    if not getattr(agent, "resume_enabled", True):
        result = {"status": "disabled", "stale_paths": [], "mismatch_fields": []}
        agent.session["resume_state"] = result
        return result
    checkpoint = current_checkpoint(agent)
    if not checkpoint:
        result = {"status": "no-checkpoint", "stale_paths": [], "mismatch_fields": []}
        agent.session["resume_state"] = result
        return result
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        result = {"status": "schema-mismatch", "stale_paths": [], "mismatch_fields": []}
        agent.session["resume_state"] = result
        return result
    stale = [
        item["path"] for item in checkpoint.get("key_files", [])
        if item.get("freshness") != file_freshness(agent.root, item.get("path", ""))
    ]
    current_identity = runtime_identity(agent)
    saved_identity = checkpoint.get("runtime_identity", {})
    mismatch = sorted(
        key for key in current_identity
        if key in saved_identity and current_identity.get(key) != saved_identity.get(key)
    )
    if mismatch or checkpoint.get("workspace_fingerprint") != agent.workspace.fingerprint():
        status = "workspace-mismatch"
    elif stale:
        status = "partial-stale"
    else:
        status = "ready"
    for path in stale:
        agent.memory.invalidate_file_summary(path)
    result = {
        "status": status,
        "stale_paths": stale,
        "mismatch_fields": mismatch,
        "stale_summary_invalidations": len(stale),
    }
    agent.session["resume_state"] = result
    return result


def render_checkpoint(agent):
    if not getattr(agent, "resume_enabled", True):
        return ""
    item = current_checkpoint(agent)
    if not item:
        return ""
    resume = agent.resume_state
    return "\n".join(
        [
            "Task checkpoint:",
            f"- resume_status: {resume.get('status', 'no-checkpoint')}",
            f"- current_goal: {item.get('current_goal', '-')}",
            f"- current_blocker: {item.get('current_blocker', '-') or '-'}",
            f"- next_step: {item.get('next_step', '-') or '-'}",
            f"- stale_paths: {', '.join(resume.get('stale_paths', [])) or '-'}",
        ]
    )
