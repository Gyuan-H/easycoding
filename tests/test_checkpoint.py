from easycoding import checkpoint
from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.task_state import TaskState
from easycoding.workspace import WorkspaceContext


def test_checkpoint_detects_stale_key_file(tmp_path):
    (tmp_path / "a.txt").write_text("old", encoding="utf-8")
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    agent.memory.remember_file("a.txt")
    state = TaskState.create("continue")
    checkpoint.create_checkpoint(agent, state, "test")
    assert checkpoint.evaluate_resume_state(agent)["status"] == "ready"
    (tmp_path / "a.txt").write_text("new", encoding="utf-8")
    assert checkpoint.evaluate_resume_state(agent)["status"] == "partial-stale"


def test_resume_state_without_checkpoint(tmp_path):
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    assert checkpoint.evaluate_resume_state(agent) == {
        "status": "no-checkpoint", "stale_paths": [], "mismatch_fields": []
    }


def test_checkpoint_schema_mismatch(tmp_path):
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    state = TaskState.create("continue")
    item = checkpoint.create_checkpoint(agent, state, "test")
    item["schema_version"] = 999
    assert checkpoint.evaluate_resume_state(agent)["status"] == "schema-mismatch"


def test_checkpoint_runtime_identity_mismatch(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path),
        approval_policy="auto",
    )
    state = TaskState.create("continue")
    checkpoint.create_checkpoint(agent, state, "test")
    agent.approval_policy = "never"
    resume = checkpoint.evaluate_resume_state(agent)
    assert resume["status"] == "workspace-mismatch"
    assert resume["mismatch_fields"] == ["approval_policy"]


def test_runtime_identity_captures_execution_contract(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path),
        approval_policy="auto", max_steps=4, max_new_tokens=321,
        allowed_tools=["read_file"], read_only=True,
    )
    identity = checkpoint.runtime_identity(agent)
    assert set(identity) >= {
        "cwd", "model", "model_client", "approval_policy", "read_only",
        "max_steps", "max_new_tokens", "allowed_tools", "shell_env_allowlist",
        "workspace_fingerprint", "tool_signature",
    }
    assert identity["allowed_tools"] == ["read_file"]
    assert identity["durable_memory_enabled"] is True
    assert identity["resume_enabled"] is True


def test_disabled_resume_context_does_not_render_checkpoint(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path),
        resume_enabled=False,
    )
    checkpoint.create_checkpoint(agent, TaskState.create("continue"), "test")
    assert checkpoint.evaluate_resume_state(agent)["status"] == "disabled"
    assert checkpoint.render_checkpoint(agent) == ""


def test_stale_resume_invalidates_cached_file_summary(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old", encoding="utf-8")
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    agent.memory.remember_file("a.txt")
    agent.memory.set_file_summary("a.txt", "old summary")
    checkpoint.create_checkpoint(agent, TaskState.create("continue"), "test")
    path.write_text("new", encoding="utf-8")
    resume = checkpoint.evaluate_resume_state(agent)
    assert resume["status"] == "partial-stale"
    assert resume["stale_summary_invalidations"] == 1
    assert "a.txt" not in agent.memory.state["file_summaries"]
