import json

from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.workspace import WorkspaceContext


def test_tool_loop_writes_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("# EasyCoding Fixture\n", encoding="utf-8")
    model = ScriptedModelClient([
        '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
        "<final>The project is EasyCoding Fixture.</final>",
    ])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), approval_policy="never")
    answer = agent.ask("Read README.md and report the project name.")
    state = agent.current_task_state
    run_dir = tmp_path / ".easycoding" / "runs" / state.run_id
    assert answer == "The project is EasyCoding Fixture."
    assert state.status == "completed"
    assert state.attempts == 2
    assert state.tool_steps == 1
    assert "EasyCoding Fixture" in model.prompts[1]
    assert (run_dir / "task_state.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    assert (run_dir / "report.json").exists()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["stop_reason"] == "final_answer_returned"
    assert set(report) >= {
        "schema_version", "run_id", "task_id", "status", "stop_reason",
        "attempts", "tool_steps", "checkpoint_id", "prompt_metadata",
        "model_metadata", "resume_state",
    }
    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [item["event"] for item in trace]
    assert events[0] == "run_started"
    assert "prompt_built" in events
    assert "model_requested" in events
    assert "model_parsed" in events
    assert "tool_executed" in events
    assert "checkpoint_created" in events
    assert events[-1] == "run_finished"
    tool_event = next(item for item in trace if item["event"] == "tool_executed")
    assert set(tool_event) >= {
        "schema_version", "name", "args", "status", "text",
        "tool_error_code", "affected_paths", "workspace_changed",
    }


def test_step_and_retry_limits_are_distinct(tmp_path):
    (tmp_path / "step").mkdir()
    (tmp_path / "retry").mkdir()
    step_agent = EasyCoding(
        ScriptedModelClient([
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
        ]),
        WorkspaceContext.build(tmp_path / "step"), max_steps=1,
    )
    step_agent.ask("list")
    assert step_agent.current_task_state.stop_reason == "step_limit_reached"

    retry_agent = EasyCoding(
        ScriptedModelClient(["bad"] * 5), WorkspaceContext.build(tmp_path / "retry"),
        max_steps=1,
    )
    retry_agent.ask("retry")
    assert retry_agent.current_task_state.stop_reason == "retry_limit_reached"


def test_tool_budget_reserves_one_finalization_response(tmp_path):
    (tmp_path / "README.md").write_text("evidence", encoding="utf-8")
    model = ScriptedModelClient([
        '<tool>{"name":"read_file","args":{"path":"README.md"}}</tool>',
        '<final>Summarized from the existing evidence.</final>',
    ])
    agent = EasyCoding(
        model, WorkspaceContext.build(tmp_path), max_steps=1,
        allowed_tools=["read_file"],
    )
    assert agent.ask("read and summarize") == "Summarized from the existing evidence."
    assert agent.current_task_state.status == "completed"
    assert agent.current_task_state.tool_steps == 1
    assert agent.current_task_state.attempts == 2
    assert "tool budget is exhausted" in model.prompts[-1]


def test_allowed_tools_filter_prompt_but_keep_stable_rejection(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>done</final>"]),
        WorkspaceContext.build(tmp_path), allowed_tools=["read_file"],
    )
    assert set(agent.tools) == {"read_file"}
    assert "- read_file:" in agent.prefix_state.text
    assert "- write_file:" not in agent.prefix_state.text
    result = agent.tool_executor.execute("write_file", {"path": "x", "text": "x"})
    assert result.tool_error_code == "tool_not_allowed"


def test_unknown_allowed_tool_is_rejected_at_construction(tmp_path):
    try:
        EasyCoding(
            ScriptedModelClient(["<final>done</final>"]),
            WorkspaceContext.build(tmp_path), allowed_tools=["not_a_tool"],
        )
    except ValueError as exc:
        assert "unknown allowed tool" in str(exc)
    else:
        raise AssertionError("unknown tool should be rejected")


def test_strict_parser_retries_plain_text():
    kind, _ = EasyCoding.parse("plain answer")
    assert kind == "retry"
