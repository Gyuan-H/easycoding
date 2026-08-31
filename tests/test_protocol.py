import json

import pytest

from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.workspace import WorkspaceContext


@pytest.mark.parametrize("raw", [
    "plain text",
    'prefix <final>answer</final>',
    '<final>answer</final> suffix',
    '<final>one</final><final>two</final>',
    '<tool>{"name":"list_files","args":{}}</tool><final>done</final>',
    '<tool>{bad json}</tool>',
    '<tool>[]</tool>',
    '<tool>{"name":"","args":{}}</tool>',
    '<tool>{"name":"list_files","args":[]}</tool>',
    '<final>   </final>',
])
def test_invalid_protocol_has_stable_error(raw):
    kind, payload = EasyCoding.parse(raw)
    assert kind == "retry"
    assert payload["code"] == "invalid_protocol"
    assert payload["message"]


def test_parser_accepts_exactly_one_complete_block():
    kind, payload = EasyCoding.parse(
        '  <tool>{"name":"list_files","args":{}}</tool>  '
    )
    assert (kind, payload) == (
        "tool", {"name": "list_files", "args": {}}
    )
    assert EasyCoding.parse("<final> done </final>") == ("final", "done")


def test_protocol_error_is_recorded_in_trace_and_metrics(tmp_path):
    model = ScriptedModelClient(["bad", "<final>recovered</final>"])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path), max_steps=1)
    assert agent.ask("test retry") == "recovered"

    trace_path = agent.run_store.run_dir(agent.current_task_state.run_id) / "trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    retry = next(item for item in trace if item["event"] == "model_retry")
    parsed = next(
        item for item in trace
        if item["event"] == "model_parsed" and item["kind"] == "retry"
    )
    assert retry["error_code"] == "invalid_protocol"
    assert parsed["error_code"] == "invalid_protocol"

