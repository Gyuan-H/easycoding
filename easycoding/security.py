"""Security helpers for paths, child process environments, and redaction."""

import os
from pathlib import Path
import re


DEFAULT_SHELL_ENV_ALLOWLIST = (
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
    "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV",
)

_CREDENTIAL_RE = re.compile(
    r"((?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|passwd|client[_ -]?secret)"
    r"\s*[:=：]\s*)([^\s\"'\\}]+)|\bsk-[A-Za-z0-9_-]{8,}",
    re.IGNORECASE,
)


def resolve_in_workspace(root, raw_path, *, must_exist=False):
    workspace = Path(root).resolve()
    raw = Path(str(raw_path))
    candidate = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()
    try:
        inside = os.path.commonpath([str(workspace), str(candidate)]) == str(workspace)
    except ValueError:
        inside = False
    if not inside:
        raise ValueError(f"path escapes workspace: {raw_path}")
    if must_exist and not candidate.exists():
        raise ValueError(f"path does not exist: {raw_path}")
    return candidate


def shell_env(root, allowlist=DEFAULT_SHELL_ENV_ALLOWLIST, env=None):
    source = os.environ if env is None else env
    result = {name: source[name] for name in allowlist if name in source}
    result["EASYCODING_WORKSPACE"] = str(Path(root).resolve())
    return result


def redact(value, secrets):
    text = str(value)
    for secret in sorted((str(item) for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return _CREDENTIAL_RE.sub(
        lambda match: (match.group(1) or "") + "[REDACTED]", text
    )
