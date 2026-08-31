import os
from pathlib import Path

import pytest

from easycoding.security import resolve_in_workspace
from easycoding.tool_executor import ToolExecutor
from easycoding.tools import BASE_TOOL_SPECS, ToolSpec


def test_path_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes workspace"):
        resolve_in_workspace(tmp_path, "../outside.txt")


def test_list_files_hides_internal_agent_and_cache_directories(tmp_path):
    for name in (".easycoding", ".git", ".venv", "__pycache__", ".pytest_cache", "easycoding.egg-info"):
        (tmp_path / name).mkdir()
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")
    executor = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")
    result = executor.execute("list_files", {"path": "."})
    assert result.status == "success"
    assert "visible.txt" in result.text
    assert ".easycoding" not in result.text
    assert "__pycache__" not in result.text
    assert "egg-info" not in result.text


def test_patch_requires_unique_match(tmp_path):
    (tmp_path / "sample.txt").write_text("x x", encoding="utf-8")
    executor = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")
    result = executor.execute("patch_file", {"path": "sample.txt", "old_text": "x", "new_text": "y"})
    assert result.status == "error"
    assert result.tool_error_code == "invalid_arguments"


def test_allowed_tools_and_approval_are_enforced(tmp_path):
    whitelist = ToolExecutor(
        tmp_path, BASE_TOOL_SPECS, approval_policy="auto", allowed_tools=["read_file"]
    )
    assert whitelist.execute("write_file", {"path": "x", "text": "x"}).tool_error_code == "tool_not_allowed"

    approval = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="never")
    assert approval.execute("write_file", {"path": "x", "text": "x"}).tool_error_code == "approval_denied"


def test_third_identical_call_is_rejected(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    executor = ToolExecutor(tmp_path, BASE_TOOL_SPECS, approval_policy="auto")
    args = {"path": "a.txt"}
    assert executor.execute("read_file", args).status == "success"
    assert executor.execute("read_file", args).status == "success"
    result = executor.execute("read_file", args)
    assert result.status == "rejected"
    assert result.tool_error_code == "repeated_call"


def test_failure_after_workspace_change_is_partial_success(tmp_path):
    def change_then_fail(context, args):
        (context.root / "changed.txt").write_text("changed", encoding="utf-8")
        raise RuntimeError("failed after write")

    tools = {"change_then_fail": ToolSpec({}, True, "test tool", change_then_fail)}
    executor = ToolExecutor(tmp_path, tools, approval_policy="auto")
    result = executor.execute("change_then_fail", {})
    assert result.status == "partial_success"
    assert result.tool_error_code == "tool_error"
    assert result.workspace_changed is True
    assert result.affected_paths == ("changed.txt",)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation may require developer mode")
def test_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside-easycoding.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes workspace"):
        resolve_in_workspace(tmp_path, "link.txt", must_exist=True)
