import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from easycoding.providers import (
    OllamaModelClient,
    OpenAICompatibleModelClient,
    ProviderError,
    ScriptedModelClient,
)
from easycoding.runtime import EasyCoding
from easycoding.workspace import WorkspaceContext


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_openai_responses_request_and_metadata():
    response = {
        "output_text": "<final>done</final>",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 6},
        },
    }
    client = OpenAICompatibleModelClient(
        "test-model", "https://example.test/v1/", "secret-key",
        timeout=9, supports_prompt_cache=True,
    )
    with patch("easycoding.providers.urllib.request.urlopen", return_value=FakeResponse(response)) as urlopen:
        text = client.complete(
            "prompt", 123, prompt_cache_key="cache-key",
            prompt_cache_retention="in_memory",
        )

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://example.test/v1/responses"
    assert urlopen.call_args.kwargs["timeout"] == 9
    assert request.get_header("Authorization") == "Bearer secret-key"
    assert payload == {
        "model": "test-model",
        "input": "prompt",
        "max_output_tokens": 123,
        "prompt_cache_key": "cache-key",
        "prompt_cache_retention": "in_memory",
    }
    assert text == "<final>done</final>"
    assert client.last_completion_metadata["cached_tokens"] == 6
    assert client.last_completion_metadata["cache_hit"] is True


def test_ollama_generate_request_and_metadata():
    response = {
        "response": "<final>local</final>",
        "prompt_eval_count": 20,
        "eval_count": 5,
        "total_duration": 1000,
        "load_duration": 200,
    }
    client = OllamaModelClient("local-model", "http://localhost:11434/", timeout=7)
    with patch("easycoding.providers.urllib.request.urlopen", return_value=FakeResponse(response)) as urlopen:
        text = client.complete("hello", 77)

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:11434/api/generate"
    assert urlopen.call_args.kwargs["timeout"] == 7
    assert payload == {
        "model": "local-model",
        "prompt": "hello",
        "stream": False,
        "options": {"num_predict": 77},
    }
    assert text == "<final>local</final>"
    assert client.last_completion_metadata["input_tokens"] == 20
    assert client.last_completion_metadata["output_tokens"] == 5
    assert client.last_completion_metadata["total_tokens"] == 25
    assert client.last_completion_metadata["cache_hit"] is False


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (FakeResponse(b"not-json"), "invalid_response"),
        (FakeResponse({"response": ""}), "empty_response"),
    ],
)
def test_ollama_rejects_invalid_or_empty_responses(response, code):
    client = OllamaModelClient("local-model")
    with patch("easycoding.providers.urllib.request.urlopen", return_value=response):
        with pytest.raises(ProviderError) as captured:
            client.complete("hello")
    assert captured.value.code == code


def test_provider_network_error_has_stable_code():
    client = OllamaModelClient("local-model")
    with patch(
        "easycoding.providers.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(ProviderError) as captured:
            client.complete("hello")
    assert captured.value.code == "request_failed"
    assert "connection refused" in str(captured.value)


def test_openai_http_error_redacts_api_key():
    secret = "top-secret-key"
    error = urllib.error.HTTPError(
        "https://example.test/v1/responses", 401, "unauthorized", {},
        io.BytesIO(f"rejected credential {secret}".encode("utf-8")),
    )
    client = OpenAICompatibleModelClient("test-model", "https://example.test/v1", secret)
    with patch("easycoding.providers.urllib.request.urlopen", side_effect=error):
        with pytest.raises(ProviderError) as captured:
            client.complete("hello")
    assert captured.value.code == "http_error"
    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_api_key_does_not_reach_run_artifacts(tmp_path):
    secret = "artifact-secret-key"
    error = urllib.error.HTTPError(
        "https://example.test/v1/responses", 401, "unauthorized", {},
        io.BytesIO(f"rejected credential {secret}".encode("utf-8")),
    )
    client = OpenAICompatibleModelClient("test-model", "https://example.test/v1", secret)
    agent = EasyCoding(client, WorkspaceContext.build(tmp_path), max_steps=1)
    with patch("easycoding.providers.urllib.request.urlopen", side_effect=error):
        answer = agent.ask("hello")

    assert "[REDACTED]" in answer
    artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / ".easycoding").rglob("*")
        if path.is_file()
    )
    assert secret not in artifacts


def test_openai_requires_credentials_without_network_call():
    client = OpenAICompatibleModelClient("test-model", "https://example.test/v1", "")
    with patch("easycoding.providers.urllib.request.urlopen") as urlopen:
        with pytest.raises(ProviderError) as captured:
            client.complete("hello")
    assert captured.value.code == "missing_credentials"
    urlopen.assert_not_called()


def test_scripted_preflight_is_ready():
    status = ScriptedModelClient(["<final>x</final>"]).check()
    assert status.ok is True
    assert status.to_dict()["status"] == "ready"
    assert status.reachable is True


def test_openai_preflight_is_configuration_only():
    client = OpenAICompatibleModelClient("test-model", "https://example.test/v1", "secret")
    with patch("easycoding.providers.urllib.request.urlopen") as urlopen:
        status = client.check()
    assert status.ok is True
    assert status.status == "configured"
    assert status.reachable is None
    urlopen.assert_not_called()


def test_openai_preflight_reports_missing_credentials():
    status = OpenAICompatibleModelClient(
        "test-model", "https://example.test/v1", ""
    ).check()
    assert status.ok is False
    assert status.status == "missing_credentials"
    assert "EASYCODING_OPENAI_API_KEY" in status.suggestion


def test_provider_preflight_rejects_invalid_base_url():
    openai = OpenAICompatibleModelClient("test-model", "localhost:8000", "secret").check()
    ollama = OllamaModelClient("local-model", "localhost:11434").check()
    assert openai.status == "invalid_configuration"
    assert ollama.status == "invalid_configuration"


def test_ollama_preflight_finds_configured_model():
    client = OllamaModelClient("local-model", "http://localhost:11434")
    with patch(
        "easycoding.providers.urllib.request.urlopen",
        return_value=FakeResponse({"models": [{"name": "local-model:latest"}]}),
    ) as urlopen:
        status = client.check()
    request = urlopen.call_args.args[0]
    assert request.method == "GET"
    assert request.full_url == "http://localhost:11434/api/tags"
    assert status.ok is True
    assert status.status == "ready"
    assert status.model_available is True


def test_ollama_preflight_reports_missing_model():
    client = OllamaModelClient("wanted:3b")
    with patch(
        "easycoding.providers.urllib.request.urlopen",
        return_value=FakeResponse({"models": [{"model": "other:latest"}]}),
    ):
        status = client.check()
    assert status.ok is False
    assert status.status == "model_missing"
    assert status.reachable is True
    assert status.model_available is False
    assert status.suggestion == "ollama pull wanted:3b"


def test_ollama_preflight_reports_unavailable_server():
    client = OllamaModelClient("local-model")
    with patch(
        "easycoding.providers.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        status = client.check()
    assert status.ok is False
    assert status.status == "unavailable"
    assert status.reachable is False
    assert "connection refused" in status.reason
