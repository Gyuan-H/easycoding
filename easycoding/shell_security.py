"""Conservative command-level guardrails for the shell tool.

This policy is intentionally a heuristic defense-in-depth layer. It does not
replace an operating-system sandbox.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class ShellAssessment:
    risk_level: str
    allowed: bool = True
    error_code: str = ""
    reason: str = ""
    requires_explicit_approval: bool = False


_CATASTROPHIC = (
    (r"(?i)\brm\s+-[^\r\n]*r[^\r\n]*f[^\r\n]*\s+/(?:\s|$)", "recursive deletion of filesystem root"),
    (r"(?i)\b(?:format|diskpart)\b", "disk-management command"),
    (r"(?i)\b(?:shutdown|reboot|restart-computer|stop-computer)\b", "machine power command"),
    (r"(?i)\bdel\b[^\r\n]*(?:[a-z]:\\|\\\\)[^\r\n]*/s", "recursive deletion of an absolute Windows path"),
    (r"(?i)\b(?:rd|rmdir)\b[^\r\n]*/s[^\r\n]*(?:[a-z]:\\|\\\\)", "recursive deletion of an absolute Windows path"),
    (r"(?i)\bremove-item\b[^\r\n]*(?:[a-z]:\\|\\\\)[^\r\n]*-recurse", "recursive deletion of an absolute Windows path"),
)

_HIGH_RISK = (
    r"(?i)\b(?:rm|del|erase|rmdir|remove-item)\b",
    r"(?i)\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f|checkout\s+--)",
    r"(?i)\b(?:drop\s+(?:database|table)|truncate\s+table)\b",
)

_MUTATING = (
    r"(?<![<>])>(?![>&])",
    r"(?i)(?:^|[&|;]\s*)\s*(?:mkdir|md|touch|copy|cp|move|mv|ren|rename|tee)\b",
    r"(?i)\bgit\s+(?:add|commit|merge|rebase|cherry-pick|tag|push)\b",
    r"(?i)\b(?:write_text|write_bytes|mkdir|unlink|rename|replace)\s*\(",
    r"(?i)\bopen\s*\([^\r\n]*[\"'][wax+][^\"']*[\"']",
)

_READ_ONLY_COMMANDS = {
    "echo", "dir", "ls", "pwd", "cd", "type", "cat", "more",
    "get-childitem", "get-content", "select-string", "rg", "grep",
    "find", "findstr", "where", "which", "whoami", "git",
}


def _inside(root, candidate):
    try:
        return os.path.commonpath([str(root), str(candidate)]) == str(root)
    except ValueError:
        return False


def _workspace_escape_reason(command, root):
    if re.search(
        r"(?i)(?:%userprofile%|%home%|%temp%|%tmp%|%appdata%|"
        r"\$env:(?:userprofile|home|temp|tmp|appdata)|\$(?:home|HOME|tmp|TMP)|~[\\/])",
        command,
    ):
        return "environment or home expansion may escape the declared workspace boundary"
    if re.search(r"(?:^|[\s\"'(=])\.\.[\\/]", command) or re.search(
        r"(?i)\b(?:cd|pushd|set-location)\s+[\"']?\.\.(?:[\"']?\s|$)", command
    ):
        return "parent-directory traversal is not allowed in shell commands"
    windows_paths = re.findall(
        r"(?i)(?:^|[\s\"'(=])([a-z]:[\\/][^\s\"';)]*)", command
    )
    for raw in windows_paths:
        candidate = Path(raw).resolve()
        if not _inside(root, candidate):
            return f"absolute path escapes workspace: {raw}"
    windows_rooted = re.findall(r"(?:^|[\s\"'(=])(\\\\[^\s\"';)]+|\\[^\s\"';)]+)", command)
    for raw in windows_rooted:
        candidate = Path(raw).resolve()
        if not _inside(root, candidate):
            return f"absolute path escapes workspace: {raw}"
    if os.name != "nt":
        unix_paths = re.findall(r"(?:^|[\s\"'(=])(/[^\s\"';)]*)", command)
        for raw in unix_paths:
            candidate = Path(raw).resolve()
            if not _inside(root, candidate):
                return f"absolute path escapes workspace: {raw}"
    return ""


def assess_shell_command(command, root):
    text = str(command).strip()
    workspace = Path(root).resolve()
    if "\x00" in text or "\r" in text or "\n" in text:
        return ShellAssessment(
            "high_risk", False, "command_not_allowed",
            "NUL and multiline shell commands are not allowed", True,
        )
    for pattern, reason in _CATASTROPHIC:
        if re.search(pattern, text):
            return ShellAssessment("high_risk", False, "unsafe_command", reason, True)
    escape_reason = _workspace_escape_reason(text, workspace)
    if escape_reason:
        return ShellAssessment(
            "high_risk", False, "workspace_escape", escape_reason, True
        )
    if any(re.search(pattern, text) for pattern in _HIGH_RISK):
        return ShellAssessment("high_risk", True, requires_explicit_approval=True)
    if any(re.search(pattern, text) for pattern in _MUTATING):
        return ShellAssessment("mutating")
    segments = [item.strip() for item in re.split(r"(?:&&|\|\||[|;])", text) if item.strip()]
    commands = []
    for segment in segments:
        match = re.match(r"(?i)(?:call\s+)?([\w.-]+)", segment)
        commands.append(match.group(1).lower() if match else "")
    if commands and all(item in _READ_ONLY_COMMANDS for item in commands):
        return ShellAssessment("read_only")
    return ShellAssessment("mutating")
