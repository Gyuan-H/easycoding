"""Model provider adapters."""

from dataclasses import asdict, dataclass
import json
import time
import urllib.error
import urllib.parse
import urllib.request


class ProviderError(RuntimeError):
    """Stable, user-safe error raised by an external model provider."""

    def __init__(self, code, message):
        self.code = str(code)
        super().__init__(f"provider {self.code}: {message}")


@dataclass(frozen=True)
class ProviderStatus:
    """Serializable result of a non-generating provider preflight check."""

    provider: str
    endpoint: str
    model: str
    status: str
    reachable: object = None
    model_available: object = None
    reason: str = ""
    suggestion: str = ""

    @property
    def ok(self):
        return self.status in {"ready", "configured"}

    def to_dict(self):
        return asdict(self)


def _redact(text, secrets):
    safe = str(text)
    for secret in sorted((str(item) for item in secrets if item), key=len, reverse=True):
        safe = safe.replace(secret, "[REDACTED]")
    return safe


def _validate_base_url(base_url):
    parsed = urllib.parse.urlparse(str(base_url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an absolute http:// or https:// URL")


def _request_json(request, timeout, provider, secrets=()):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = _redact(body[:500], secrets)
        raise ProviderError("http_error", f"{provider} HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        detail = _redact(str(exc), secrets)
        raise ProviderError("request_failed", f"{provider} request failed: {detail}") from exc
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError("invalid_response", f"{provider} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ProviderError("invalid_response", f"{provider} returned a non-object JSON response")
    return data


def _post_json(url, payload, headers, timeout, provider, secrets=()):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers)},
        method="POST",
    )
    return _request_json(request, timeout, provider, secrets)


def _get_json(url, headers, timeout, provider, secrets=()):
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    return _request_json(request, timeout, provider, secrets)


class ScriptedModelClient:
    """Deterministic model used for harness tests and offline demos."""

    supports_prompt_cache = False
    provider = "scripted"

    def __init__(self, outputs, model="scripted"):
        self.outputs = iter(outputs)
        self.model = model
        self.prompts = []
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens=512, **kwargs):
        self.prompts.append(str(prompt))
        try:
            result = next(self.outputs)
        except StopIteration as exc:
            raise RuntimeError("scripted model has no output left") from exc
        self.last_completion_metadata = {
            "model": self.model,
            "input_chars": len(str(prompt)),
            "output_chars": len(str(result)),
            "cache_hit": False,
        }
        return str(result)

    def check(self):
        return ProviderStatus(
            provider="scripted", endpoint="offline", model=self.model,
            status="ready", reachable=True, model_available=True,
            reason="deterministic offline provider is ready",
        )


class OpenAICompatibleModelClient:
    """Minimal adapter for an OpenAI-compatible ``/responses`` endpoint."""

    provider = "openai"

    def __init__(self, model, base_url, api_key, timeout=120, supports_prompt_cache=False):
        self.model = model
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout = int(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.supports_prompt_cache = bool(supports_prompt_cache)
        self.last_completion_metadata = {}

    def check(self):
        endpoint = self.base_url + "/responses"
        try:
            _validate_base_url(self.base_url)
        except ValueError as exc:
            return ProviderStatus(
                provider="openai", endpoint=endpoint, model=self.model,
                status="invalid_configuration", reason=str(exc),
                suggestion="set --base-url to an absolute HTTP(S) API base URL",
            )
        if not self.api_key:
            return ProviderStatus(
                provider="openai", endpoint=endpoint, model=self.model,
                status="missing_credentials", reason="OpenAI-compatible API key is missing",
                suggestion="set EASYCODING_OPENAI_API_KEY or pass --api-key",
            )
        return ProviderStatus(
            provider="openai", endpoint=endpoint, model=self.model,
            status="configured", reason="endpoint and credentials are configured; no billable request was sent",
            suggestion="run a normal prompt to verify remote model access",
        )

    def complete(
        self, prompt, max_new_tokens=512, prompt_cache_key=None,
        prompt_cache_retention=None, **kwargs,
    ):
        if not self.api_key:
            raise ProviderError("missing_credentials", "missing OpenAI-compatible API key")
        payload = {
            "model": self.model,
            "input": str(prompt),
            "max_output_tokens": int(max_new_tokens),
        }
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        started = time.perf_counter()
        data = _post_json(
            self.base_url + "/responses", payload,
            {
                "Authorization": f"Bearer {self.api_key}",
            },
            self.timeout, "OpenAI-compatible", (self.api_key,),
        )
        text = self._extract_text(data)
        if not text.strip():
            raise ProviderError("empty_response", "OpenAI-compatible model returned an empty response")
        usage = dict(data.get("usage", {}) or {})
        details = dict(usage.get("input_tokens_details", {}) or {})
        cached = int(details.get("cached_tokens", 0) or 0)
        self.last_completion_metadata = {
            "model": self.model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": cached,
            "cache_hit": cached > 0,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return text

    @staticmethod
    def _extract_text(data):
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        parts = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text", "")))
        return "\n".join(parts)


class OllamaModelClient:
    """Adapter for Ollama's non-streaming ``/api/generate`` endpoint."""

    supports_prompt_cache = False
    provider = "ollama"

    def __init__(self, model, base_url="http://localhost:11434", timeout=120):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.timeout = int(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.last_completion_metadata = {}

    def check(self):
        endpoint = self.base_url + "/api/tags"
        try:
            _validate_base_url(self.base_url)
        except ValueError as exc:
            return ProviderStatus(
                provider="ollama", endpoint=endpoint, model=self.model,
                status="invalid_configuration", reason=str(exc),
                suggestion="set --base-url to an absolute Ollama HTTP(S) base URL",
            )
        try:
            data = _get_json(endpoint, {}, self.timeout, "Ollama")
        except ProviderError as exc:
            return ProviderStatus(
                provider="ollama", endpoint=endpoint, model=self.model,
                status="unavailable", reachable=False, reason=str(exc),
                suggestion="install or start Ollama, then retry the check",
            )
        models = data.get("models", [])
        if not isinstance(models, list):
            return ProviderStatus(
                provider="ollama", endpoint=endpoint, model=self.model,
                status="unavailable", reachable=True,
                reason="Ollama /api/tags response does not contain a models list",
                suggestion="check the Ollama server version and endpoint",
            )
        installed = {
            str(item.get("name") or item.get("model") or "")
            for item in models if isinstance(item, dict)
        }
        candidates = {self.model}
        if ":" not in self.model:
            candidates.add(self.model + ":latest")
        if not candidates.intersection(installed):
            return ProviderStatus(
                provider="ollama", endpoint=endpoint, model=self.model,
                status="model_missing", reachable=True, model_available=False,
                reason=f"model is not installed; available models: {', '.join(sorted(installed)) or '(none)'}",
                suggestion=f"ollama pull {self.model}",
            )
        return ProviderStatus(
            provider="ollama", endpoint=endpoint, model=self.model,
            status="ready", reachable=True, model_available=True,
            reason="Ollama is reachable and the configured model is installed",
        )

    def complete(self, prompt, max_new_tokens=512, **kwargs):
        prompt = str(prompt)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": int(max_new_tokens)},
        }
        started = time.perf_counter()
        data = _post_json(
            self.base_url + "/api/generate", payload, {}, self.timeout, "Ollama"
        )
        text = data.get("response", "")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("empty_response", "Ollama returned an empty response")
        self.last_completion_metadata = {
            "model": self.model,
            "input_chars": len(prompt),
            "output_chars": len(text),
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "total_tokens": (
                int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
            ),
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "cache_hit": False,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return text
