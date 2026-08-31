"""Redacted provider call recording and deterministic offline replay."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .providers import ProviderError, ProviderStatus


RECORDING_SCHEMA_VERSION = 1
SENSITIVE_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _now():
    return datetime.now(timezone.utc).isoformat()


def environment_secrets(env=None):
    source = os.environ if env is None else env
    return tuple(
        str(value) for name, value in source.items()
        if value and len(str(value)) >= 6
        and any(part in name.upper() for part in SENSITIVE_NAME_PARTS)
    )


def redact_value(value, secrets):
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(part in str(key).upper() for part in SENSITIVE_NAME_PARTS)
                else redact_value(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, secrets) for item in value]
    if not isinstance(value, str):
        return value
    safe = value
    for secret in sorted({str(item) for item in secrets if item}, key=len, reverse=True):
        safe = safe.replace(secret, "[REDACTED]")
    return safe


def _write_json_atomic(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=destination.parent, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(destination)


class RecordingModelClient:
    """Wrap a provider and persist each redacted call for later replay."""

    def __init__(self, client, path, secrets=()):
        self.client = client
        self.path = Path(path)
        self.provider = getattr(client, "provider", type(client).__name__)
        self.model = getattr(client, "model", "")
        self.supports_prompt_cache = bool(
            getattr(client, "supports_prompt_cache", False)
        )
        self.secrets = tuple(secrets) + environment_secrets()
        self.calls = []
        self.created_at = _now()
        self.last_completion_metadata = {}
        self._persist()

    def _persist(self):
        _write_json_atomic(self.path, {
            "schema_version": RECORDING_SCHEMA_VERSION,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "supports_prompt_cache": self.supports_prompt_cache,
            "calls": self.calls,
        })

    def complete(self, prompt, max_new_tokens=512, **kwargs):
        prompt = str(prompt)
        request = {
            "prompt": redact_value(prompt, self.secrets),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "max_new_tokens": int(max_new_tokens),
            "options": redact_value(dict(kwargs), self.secrets),
        }
        started = time.perf_counter()
        try:
            response = self.client.complete(prompt, max_new_tokens, **kwargs)
        except Exception as exc:
            duration = round((time.perf_counter() - started) * 1000, 3)
            self.calls.append({
                "request": request,
                "error": {
                    "code": getattr(exc, "code", "provider_error"),
                    "message": redact_value(str(exc), self.secrets),
                },
                "metadata": {"duration_ms": duration},
            })
            self._persist()
            raise
        metadata = dict(getattr(self.client, "last_completion_metadata", {}) or {})
        metadata.setdefault("duration_ms", round((time.perf_counter() - started) * 1000, 3))
        self.last_completion_metadata = redact_value(metadata, self.secrets)
        self.calls.append({
            "request": request,
            "response": redact_value(str(response), self.secrets),
            "metadata": self.last_completion_metadata,
        })
        self._persist()
        return str(response)

    def check(self):
        return self.client.check()


class ReplayModelClient:
    """Replay a versioned provider recording without network access."""

    def __init__(self, path, strict_prompts=False):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != RECORDING_SCHEMA_VERSION:
            raise ValueError("unsupported provider recording schema")
        calls = payload.get("calls")
        if not isinstance(calls, list) or not calls or not all(isinstance(item, dict) for item in calls):
            raise ValueError("provider recording must contain at least one call")
        self.provider = str(payload.get("provider", "replay"))
        self.model = str(payload.get("model", "replay"))
        self.supports_prompt_cache = bool(payload.get("supports_prompt_cache", False))
        self.calls = calls
        self.strict_prompts = bool(strict_prompts)
        self.index = 0
        self.call_history = []
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens=512, **kwargs):
        if self.index >= len(self.calls):
            raise ProviderError("replay_exhausted", "provider recording has no calls left")
        call = self.calls[self.index]
        self.index += 1
        request = call.get("request", {})
        if self.strict_prompts:
            actual = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
            if request.get("prompt_sha256") != actual:
                raise ProviderError("replay_prompt_mismatch", "prompt hash differs from recording")
        self.last_completion_metadata = dict(call.get("metadata", {}) or {})
        self.call_history.append(call)
        if isinstance(call.get("error"), dict):
            error = call["error"]
            raise ProviderError(error.get("code", "provider_error"), error.get("message", "recorded error"))
        if "response" not in call:
            raise ProviderError("invalid_replay", "recorded call has no response or error")
        return str(call["response"])

    def check(self):
        return ProviderStatus(
            provider="replay", endpoint=str(self.path), model=self.model,
            status="ready", reachable=True, model_available=True,
            reason=f"loaded {len(self.calls)} offline calls",
        )
