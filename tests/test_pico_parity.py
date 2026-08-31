import json

from easycoding import checkpoint
from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.task_state import TaskState
from easycoding.workspace import WorkspaceContext


def _trace(agent):
    path = agent.run_store.run_dir(agent.current_task_state.run_id) / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_tool_execution_creates_a_checkpoint(tmp_path):
    (tmp_path / "README.md").write_text("evidence", encoding="utf-8")
    agent = EasyCoding(
        ScriptedModelClient([
            '<tool>{"name":"read_file","args":{"path":"README.md"}}</tool>',
            "<final>done</final>",
        ]),
        WorkspaceContext.build(tmp_path), allowed_tools=["read_file"],
    )
    agent.ask("read")
    triggers = [
        item["trigger"] for item in _trace(agent)
        if item["event"] == "checkpoint_created"
    ]
    assert triggers == ["tool_executed", "run_finished"]


def test_context_reduction_creates_checkpoint_before_model_request(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>done</final>"]),
        WorkspaceContext.build(tmp_path),
    )
    agent.context_manager.total_budget = 80
    agent.ask("retain this current request")
    trace = _trace(agent)
    reduction = next(
        index for index, item in enumerate(trace)
        if item["event"] == "checkpoint_created"
        and item["trigger"] == "context_reduction"
    )
    request = next(
        index for index, item in enumerate(trace)
        if item["event"] == "model_requested"
    )
    assert reduction < request


def test_runtime_mismatch_is_visible_in_prompt_metadata_and_trace(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>re-anchored</final>"]),
        WorkspaceContext.build(tmp_path), approval_policy="auto",
    )
    checkpoint.create_checkpoint(agent, TaskState.create("previous"), "seed")
    agent.approval_policy = "never"
    assert agent.ask("continue") == "re-anchored"
    trace = _trace(agent)
    detected = next(item for item in trace if item["event"] == "resume_state_detected")
    assert detected["status"] == "workspace-mismatch"
    assert detected["mismatch_fields"] == ["approval_policy"]
    prompt_event = next(item for item in trace if item["event"] == "prompt_built")
    assert prompt_event["resume_status"] == "workspace-mismatch"

