import json
import pytest

from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.security import shell_env
from easycoding.shell_security import assess_shell_command
from easycoding.tool_executor import ToolExecutor
from easycoding.tools import BASE_TOOL_SPECS
from easycoding.workspace import WorkspaceContext


@pytest.mark.parametrize(("command", "risk"), [
    ("echo ok", "read_only"),
    ("git status", "read_only"),
    ("python -m pytest -q", "mutating"),
    ("echo ok > result.txt", "mutating"),
    ("del result.txt", "high_risk"),
    ("git reset --hard", "high_risk"),
])
def test_shell_risk_classification(tmp_path, command, risk):
    assert assess_shell_command(command, tmp_path).risk_level == risk


@pytest.mark.parametrize(("command", "code"), [
    ("type ..\\outside.txt", "workspace_escape"),
    ("cd ..", "workspace_escape"),
    ("echo %USERPROFILE%", "workspace_escape"),
    ("echo %TEMP%", "workspace_escape"),
    ("type \\outside.txt", "workspace_escape"),
    ("shutdown /s", "unsafe_command"),
    ("echo first\necho second", "command_not_allowed"),
])
def test_blocked_commands_never_execute(tmp_path, command, code):
    executor = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")
    result = executor.execute("run_shell", {"command": command})
    assert result.status == "rejected"
    assert result.tool_error_code == code
    assert result.risk_level == "high_risk"


def test_high_risk_command_requires_callback_even_in_auto_mode(tmp_path):
    decisions = []

    def approve(name, args):
        decisions.append((name, args))
        return False

    executor = ToolExecutor(
        tmp_path, BASE_TOOL_SPECS, approval_policy="auto",
        approval_callback=approve,
    )
    result = executor.execute("run_shell", {"command": "del missing.txt"})
    assert result.tool_error_code == "approval_denied"
    assert result.approval_required is True
    assert result.approval_granted is False
    assert len(decisions) == 1


def test_high_risk_command_can_run_after_explicit_approval(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("x", encoding="utf-8")
    executor = ToolExecutor(
        tmp_path, BASE_TOOL_SPECS, approval_policy="auto",
        approval_callback=lambda name, args: True,
    )
    result = executor.execute("run_shell", {"command": "del victim.txt"})
    assert result.status == "success"
    assert result.risk_level == "high_risk"
    assert result.approval_granted is True
    assert not victim.exists()


def test_timeout_has_stable_code_and_metadata(tmp_path):
    command = 'python -c "import time; time.sleep(2)"'
    executor = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")
    result = executor.execute("run_shell", {"command": command, "timeout": 1})
    assert result.status == "error"
    assert result.tool_error_code == "command_timeout"
    assert result.timed_out is True


def test_large_output_is_truncated_and_reported(tmp_path):
    command = 'python -c "print(\'x\' * 5000)"'
    executor = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")
    result = executor.execute("run_shell", {"command": command})
    assert result.status == "partial_success"
    assert result.tool_error_code == "output_limit_exceeded"
    assert result.output_truncated is True
    assert "truncated" in result.text


def test_shell_security_metadata_is_written_to_trace(tmp_path):
    model = ScriptedModelClient([
        '<tool>{"name":"run_shell","args":{"command":"echo ok"}}</tool>',
        "<final>done</final>",
    ])
    agent = EasyCoding(
        model, WorkspaceContext.build(tmp_path), approval_policy="auto", max_steps=2
    )
    assert agent.ask("run echo") == "done"
    path = agent.run_store.run_dir(agent.current_task_state.run_id) / "trace.jsonl"
    trace = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    event = next(item for item in trace if item["event"] == "tool_executed")
    assert event["risk_level"] == "read_only"
    assert event["approval_required"] is True
    assert event["approval_granted"] is True
    assert event["exit_code"] == 0
    assert event["timed_out"] is False
    assert event["output_truncated"] is False


def test_shell_environment_uses_an_allowlist(tmp_path):
    result = shell_env(
        tmp_path,
        env={"PATH": "bin", "TEMP": "temp", "SECRET_TOKEN": "do-not-copy"},
    )
    assert result["PATH"] == "bin"
    assert result["TEMP"] == "temp"
    assert result["EASYCODING_WORKSPACE"] == str(tmp_path.resolve())
    assert "SECRET_TOKEN" not in result
