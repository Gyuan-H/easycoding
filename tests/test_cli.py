import json
from unittest.mock import patch

import pytest

from easycoding.cli import _build_model, build_arg_parser, main
from easycoding.providers import OllamaModelClient, OpenAICompatibleModelClient


def test_cli_builds_ollama_provider():
    args = build_arg_parser().parse_args([
        "--provider", "ollama",
        "--model", "local-model",
        "--base-url", "http://ollama.test:11434",
        "--timeout", "33",
    ])
    client = _build_model(args)
    assert isinstance(client, OllamaModelClient)
    assert client.model == "local-model"
    assert client.base_url == "http://ollama.test:11434"
    assert client.timeout == 33


def test_ollama_uses_environment_configuration(monkeypatch):
    monkeypatch.setenv("EASYCODING_PROVIDER", "ollama")
    monkeypatch.setenv("EASYCODING_OLLAMA_MODEL", "env-model")
    monkeypatch.setenv("EASYCODING_OLLAMA_BASE_URL", "http://env.test:11434/")
    client = _build_model(build_arg_parser().parse_args([]))
    assert isinstance(client, OllamaModelClient)
    assert client.model == "env-model"
    assert client.base_url == "http://env.test:11434"


def test_openai_prompt_cache_requires_explicit_enable(monkeypatch):
    monkeypatch.setenv("EASYCODING_OPENAI_API_KEY", "secret")
    default_client = _build_model(build_arg_parser().parse_args(["--provider", "openai"]))
    enabled_client = _build_model(build_arg_parser().parse_args([
        "--provider", "openai", "--prompt-cache"
    ]))
    assert isinstance(default_client, OpenAICompatibleModelClient)
    assert default_client.supports_prompt_cache is False
    assert enabled_client.supports_prompt_cache is True


def test_openai_prompt_cache_environment_flag(monkeypatch):
    monkeypatch.setenv("EASYCODING_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("EASYCODING_OPENAI_PROMPT_CACHE", "true")
    client = _build_model(build_arg_parser().parse_args(["--provider", "openai"]))
    assert client.supports_prompt_cache is True


def test_invalid_environment_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("EASYCODING_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="unsupported provider"):
        _build_model(build_arg_parser().parse_args([]))


def test_cli_accepts_memory_and_resume_ablation_switches():
    args = build_arg_parser().parse_args([
        "--no-durable-memory", "--no-resume-context"
    ])
    assert args.durable_memory is False
    assert args.resume_context is False


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_check_provider_cli_reports_ready_ollama(tmp_path, capsys):
    with patch(
        "easycoding.providers.urllib.request.urlopen",
        return_value=FakeResponse({"models": [{"name": "local-model:latest"}]}),
    ):
        exit_code = main([
            "--provider", "ollama", "--model", "local-model",
            "--cwd", str(tmp_path), "--check-provider",
        ])
    output = capsys.readouterr()
    assert exit_code == 0
    assert "Status: ready" in output.out
    assert "Server reachable: yes" in output.out
    assert output.err == ""


def test_check_provider_cli_reports_missing_ollama_model(tmp_path, capsys):
    with patch(
        "easycoding.providers.urllib.request.urlopen",
        return_value=FakeResponse({"models": []}),
    ):
        exit_code = main([
            "--provider", "ollama", "--model", "missing:3b",
            "--cwd", str(tmp_path), "--check-provider",
        ])
    output = capsys.readouterr()
    assert exit_code == 1
    assert "Status: model_missing" in output.err
    assert "Suggestion: ollama pull missing:3b" in output.err


def test_check_provider_cli_reports_missing_openai_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("EASYCODING_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    exit_code = main([
        "--provider", "openai", "--cwd", str(tmp_path), "--check-provider",
    ])
    output = capsys.readouterr()
    assert exit_code == 1
    assert "Status: missing_credentials" in output.err
    assert "EASYCODING_OPENAI_API_KEY" in output.err
