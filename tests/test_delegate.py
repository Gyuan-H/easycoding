import json

from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.workspace import WorkspaceContext


def run_paths(agent):
    root = agent.root / ".easycoding" / "runs"
    return sorted(root.glob("*/report.json"))


def parent_report(agent):
    path = agent.run_store.run_dir(agent.current_task_state.run_id) / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def child_trace(agent):
    child_id = parent_report(agent)["delegation"]["children"][0]["child_run_id"]
    path = agent.run_store.run_dir(child_id) / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_delegate_reads_file_and_returns_evidence(tmp_path):
    (tmp_path / "README.md").write_text("# Evidence\n", encoding="utf-8")
    model = ScriptedModelClient([
        '<tool>{"name":"delegate","args":{"task":"read README","paths":["README.md"]}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"README.md"}}</tool>',
        '<final>README contains Evidence.</final>',
        '<final>Delegation complete.</final>',
    ])
    agent = EasyCoding(
        model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"],
    )
    assert agent.ask("delegate the README inspection") == "Delegation complete."
    report = parent_report(agent)
    assert report["delegation"]["count"] == 1
    child = report["delegation"]["children"][0]
    assert child["status"] == "success"
    assert child["evidence"] == [{"tool": "read_file", "path": "README.md", "line": 1}]
    assert "README contains Evidence" in model.prompts[-1]
    assert len(run_paths(agent)) == 2


def test_delegate_rejects_write_shell_and_nested_delegate(tmp_path):
    model = ScriptedModelClient([
        '<tool>{"name":"delegate","args":{"task":"attempt forbidden actions"}}</tool>',
        '<tool>{"name":"write_file","args":{"path":"bad.txt","text":"bad"}}</tool>',
        '<tool>{"name":"run_shell","args":{"command":"echo forbidden"}}</tool>',
        '<tool>{"name":"delegate","args":{"task":"nested"}}</tool>',
        '<final>Both forbidden actions were rejected.</final>',
        '<final>Parent recovered.</final>',
    ])
    agent = EasyCoding(
        model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"],
    )
    assert agent.ask("test child boundaries") == "Parent recovered."
    assert not (tmp_path / "bad.txt").exists()
    events = [item for item in child_trace(agent) if item["event"] == "tool_executed"]
    assert [item["tool_error_code"] for item in events] == [
        "tool_not_allowed", "tool_not_allowed", "tool_not_allowed"
    ]
    child = parent_report(agent)["delegation"]["children"][0]
    assert child["tool_error_codes"] == [
        "tool_not_allowed", "tool_not_allowed", "tool_not_allowed"
    ]


def test_delegate_enforces_path_scope(tmp_path):
    (tmp_path / "allowed").mkdir()
    (tmp_path / "allowed" / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    model = ScriptedModelClient([
        '<tool>{"name":"delegate","args":{"task":"inspect scope","paths":["allowed"]}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"outside.txt"}}</tool>',
        '<final>Outside path was rejected.</final>',
        '<final>done</final>',
    ])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"])
    agent.ask("scope test")
    event = next(item for item in child_trace(agent) if item["event"] == "tool_executed")
    assert event["tool_error_code"] == "path_not_allowed"


def test_delegate_has_independent_step_budget_and_finalization(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    model = ScriptedModelClient([
        '<tool>{"name":"delegate","args":{"task":"read once","max_steps":1}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"a.txt"}}</tool>',
        '<final>One read was enough.</final>',
        '<final>done</final>',
    ])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"])
    agent.ask("budget test")
    child = parent_report(agent)["delegation"]["children"][0]
    assert child["attempts"] == 2
    assert child["tool_steps"] == 1
    assert child["status"] == "success"


def test_delegate_failure_is_returned_to_parent_without_crashing(tmp_path):
    class FailChildModel:
        model = "fail-child"
        supports_prompt_cache = False

        def __init__(self):
            self.calls = 0
            self.last_completion_metadata = {}

        def complete(self, prompt, max_new_tokens=512, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '<tool>{"name":"delegate","args":{"task":"child fails"}}</tool>'
            if self.calls == 2:
                raise RuntimeError("injected child failure")
            return '<final>Parent handled the child failure.</final>'

    model = FailChildModel()
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"])
    assert agent.ask("failure test") == "Parent handled the child failure."
    report = parent_report(agent)
    assert report["delegation"]["failure_count"] == 1
    assert report["delegation"]["children"][0]["status"] == "failed"


def test_delegate_schema_rejects_invalid_budget_before_child_run(tmp_path):
    agent = EasyCoding(
        ScriptedModelClient(["<final>x</final>"]),
        WorkspaceContext.build(tmp_path), allowed_tools=["delegate"],
    )
    result = agent.tool_executor.execute("delegate", {"task": "x", "max_steps": 4})
    assert result.status == "rejected"
    assert result.tool_error_code == "argument_out_of_range"
    assert not agent.delegate_records


def test_child_report_and_parent_trace_preserve_relationship(tmp_path):
    model = ScriptedModelClient([
        '<tool>{"name":"delegate","args":{"task":"summarize"}}</tool>',
        '<final>child summary</final>',
        '<final>parent summary</final>',
    ])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"])
    agent.ask("relationship test")
    parent = parent_report(agent)
    child_record = parent["delegation"]["children"][0]
    child_path = agent.run_store.run_dir(child_record["child_run_id"]) / "report.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    assert child["run_relationship"]["agent_role"] == "delegate"
    assert child["run_relationship"]["parent_run_id"] == parent["run_id"]
    assert child["run_relationship"]["delegation_id"] == child_record["delegation_id"]
    trace_path = agent.run_store.run_dir(parent["run_id"]) / "trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert "delegate_started" in [item["event"] for item in trace]
    assert "delegate_completed" in [item["event"] for item in trace]


def test_delegate_summary_is_bounded_before_parent_injection(tmp_path):
    long_summary = "x" * 6000
    model = ScriptedModelClient([
        '<tool>{"name":"delegate","args":{"task":"produce bounded summary"}}</tool>',
        f"<final>{long_summary}</final>",
        '<final>parent done</final>',
    ])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), allowed_tools=["delegate"])
    agent.ask("output limit test")
    child = parent_report(agent)["delegation"]["children"][0]
    assert len(child["summary"]) < len(long_summary)
    assert "[truncated" in child["summary"]
    assert len(model.prompts[-1]) < 12000
