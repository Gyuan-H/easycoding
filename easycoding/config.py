"""Small project .env loader and provider configuration helpers."""

import os
from pathlib import Path


def load_project_env(root, override=False):
    path = Path(root) / ".env"
    if not path.is_file():
        return {}
    loaded = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and (override or name not in os.environ):
            os.environ[name] = value
            loaded[name] = value
    return loaded


def env(name, legacy=(), default=""):
    for candidate in (name, *legacy):
        value = os.environ.get(candidate)
        if value:
            return value
    return default


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
