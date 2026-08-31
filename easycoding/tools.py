"""Explicit tool registry and tool implementations."""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from .security import resolve_in_workspace
from .tool_types import ToolExecutionError, ToolRunOutput
from .workspace import clip


IGNORED_PATH_NAMES = {
    ".git", ".easycoding", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "node_modules",
}


def _ignored_path(path, root):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(
        part in IGNORED_PATH_NAMES or part.endswith(".egg-info") for part in parts
    )


@dataclass(frozen=True)
class ToolSpec:
    schema: dict
    risky: bool
    description: str
    run: object


def _list_files(context, args):
    path = resolve_in_workspace(context.root, args.get("path", "."), must_exist=True)
    if not path.is_dir():
        raise ValueError("list_files path must be a directory")
    items = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if _ignored_path(child, context.root):
            continue
        suffix = "/" if child.is_dir() else ""
        items.append(child.relative_to(context.root).as_posix() + suffix)
        if len(items) >= 200:
            break
    return "\n".join(items) or "(empty directory)"


def _read_file(context, args):
    path = resolve_in_workspace(context.root, args["path"], must_exist=True)
    if not path.is_file():
        raise ValueError("read_file path must be a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", start + 199))
    if start < 1 or end < start or end - start > 1000:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rendered = [f"{index}: {lines[index - 1]}" for index in range(start, min(end, len(lines)) + 1)]
    return "\n".join(rendered) or "(no lines in requested range)"


def _search(context, args):
    pattern = str(args.get("pattern", ""))
    if not pattern:
        raise ValueError("search pattern must not be empty")
    base = resolve_in_workspace(context.root, args.get("path", "."), must_exist=True)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    matches = []
    paths = [base] if base.is_file() else base.rglob("*")
    for path in paths:
        if not path.is_file() or _ignored_path(path, context.root):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{path.relative_to(context.root).as_posix()}:{number}:{line}")
                if len(matches) >= 100:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def _write_file(context, args):
    path = resolve_in_workspace(context.root, args["path"])
    if path.exists() and path.is_dir():
        raise ValueError("write_file path must not be a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = str(args.get("text", ""))
    path.write_text(text, encoding="utf-8")
    return f"wrote {len(text)} chars to {path.relative_to(context.root).as_posix()}"


def _patch_file(context, args):
    path = resolve_in_workspace(context.root, args["path"], must_exist=True)
    if not path.is_file():
        raise ValueError("patch_file path must be a file")
    old = str(args.get("old_text", ""))
    new = str(args.get("new_text", ""))
    if not old:
        raise ValueError("old_text must not be empty")
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"patched {path.relative_to(context.root).as_posix()}"


def _run_shell(context, args):
    command = str(args.get("command", "")).strip()
    timeout = int(args.get("timeout", 20))
    if not command:
        raise ValueError("shell command must not be empty")
    if not 1 <= timeout <= 120:
        raise ValueError("timeout must be between 1 and 120 seconds")
    try:
        result = subprocess.run(
            command, cwd=context.root, shell=True, capture_output=True, text=True,
            timeout=timeout, env=context.shell_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            "command_timeout", f"command timed out after {timeout}s",
            timed_out=True,
        ) from exc
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    raw = f"exit_code={result.returncode}\n{output}".strip()
    rendered = clip(raw)
    output_truncated = rendered != raw
    if result.returncode != 0:
        raise ToolExecutionError(
            "tool_error", rendered, exit_code=result.returncode,
            output_truncated=output_truncated,
        )
    return ToolRunOutput(
        rendered,
        error_code="output_limit_exceeded" if output_truncated else "",
        metadata={
            "exit_code": result.returncode,
            "timed_out": False,
            "output_truncated": output_truncated,
        },
    )


BASE_TOOL_SPECS = {
    "list_files": ToolSpec({
        "type": "object", "properties": {
            "path": {"type": "string", "default": ".", "minLength": 1},
        }, "additionalProperties": False,
    }, False, "List files inside the workspace.", _list_files),
    "read_file": ToolSpec({
        "type": "object", "properties": {
            "path": {"type": "string", "minLength": 1},
            "start": {"type": "integer", "default": 1, "minimum": 1},
            "end": {"type": "integer", "minimum": 1},
        }, "required": ["path"], "additionalProperties": False,
        "x-rules": [{
            "kind": "ordered_range", "lower": "start", "upper": "end",
            "maximumSpan": 1000,
        }],
    }, False, "Read a line range from a UTF-8 text file.", _read_file),
    "search": ToolSpec({
        "type": "object", "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "path": {"type": "string", "default": ".", "minLength": 1},
        }, "required": ["pattern"], "additionalProperties": False,
    }, False, "Regex search text files in the workspace.", _search),
    "write_file": ToolSpec({
        "type": "object", "properties": {
            "path": {"type": "string", "minLength": 1},
            "text": {"type": "string"},
        }, "required": ["path", "text"], "additionalProperties": False,
    }, True, "Write a UTF-8 text file.", _write_file),
    "patch_file": ToolSpec({
        "type": "object", "properties": {
            "path": {"type": "string", "minLength": 1},
            "old_text": {"type": "string", "minLength": 1},
            "new_text": {"type": "string"},
        }, "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }, True, "Replace text that occurs exactly once.", _patch_file),
    "run_shell": ToolSpec({
        "type": "object", "properties": {
            "command": {"type": "string", "minLength": 1},
            "timeout": {"type": "integer", "default": 20, "minimum": 1, "maximum": 120},
        }, "required": ["command"], "additionalProperties": False,
    }, True, "Run a shell command at the workspace root.", _run_shell),
}
