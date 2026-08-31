import pytest

from easycoding.tool_executor import ToolExecutor
from easycoding.tools import BASE_TOOL_SPECS


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")


@pytest.mark.parametrize(("name", "args", "code"), [
    ("read_file", {}, "missing_argument"),
    ("read_file", {"path": 3}, "invalid_argument_type"),
    ("read_file", {"path": "x", "start": True}, "invalid_argument_type"),
    ("read_file", {"path": "x", "extra": 1}, "unexpected_argument"),
    ("read_file", {"path": "x", "start": 0}, "argument_out_of_range"),
    ("read_file", {"path": "x", "start": 5, "end": 2}, "argument_out_of_range"),
    ("read_file", {"path": "x", "start": 1, "end": 1002}, "argument_out_of_range"),
    ("run_shell", {"command": "echo ok", "timeout": 121}, "argument_out_of_range"),
    ("search", {"pattern": ""}, "argument_out_of_range"),
])
def test_invalid_arguments_are_rejected_with_stable_codes(executor, name, args, code):
    result = executor.execute(name, args)
    assert result.status == "rejected"
    assert result.tool_error_code == code
    assert result.workspace_changed is False


def test_schema_defaults_are_applied_before_execution(executor, tmp_path):
    (tmp_path / "sample.txt").write_text("line one\nline two\n", encoding="utf-8")
    result = executor.execute("read_file", {"path": "sample.txt"})
    assert result.status == "success"
    assert "1: line one" in result.text


def test_read_end_default_remains_relative_to_start(executor, tmp_path):
    (tmp_path / "sample.txt").write_text(
        "\n".join(f"line {number}" for number in range(1, 502)), encoding="utf-8"
    )
    result = executor.execute("read_file", {"path": "sample.txt", "start": 500})
    assert result.status == "success"
    assert "500: line 500" in result.text
    assert "501: line 501" in result.text


def test_unknown_tool_keeps_stable_error_code(executor):
    result = executor.execute("does_not_exist", {})
    assert result.status == "rejected"
    assert result.tool_error_code == "unknown_tool"
