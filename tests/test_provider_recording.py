import json

import pytest

from easycoding.provider_recording import RecordingModelClient, ReplayModelClient
from easycoding.providers import ProviderError, ScriptedModelClient


def test_recording_redacts_prompt_response_metadata_and_environment(tmp_path, monkeypatch):
    secret = "super-secret-token"
    monkeypatch.setenv("EASYCODING_TEST_SECRET_TOKEN", secret)
    client = ScriptedModelClient([f"<final>{secret}</final>"])
    path = tmp_path / "recording.json"
    recorder = RecordingModelClient(client, path)

    assert recorder.complete(f"prompt {secret}") == f"<final>{secret}</final>"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert secret not in text
    assert "[REDACTED]" in payload["calls"][0]["request"]["prompt"]
    assert "[REDACTED]" in payload["calls"][0]["response"]
    assert len(payload["calls"][0]["request"]["prompt_sha256"]) == 64


def test_replay_restores_outputs_metadata_and_exhaustion(tmp_path):
    path = tmp_path / "recording.json"
    RecordingModelClient(
        ScriptedModelClient(["<final>done</final>"]), path
    ).complete("prompt")
    replay = ReplayModelClient(path, strict_prompts=True)
    assert replay.complete("prompt") == "<final>done</final>"
    assert replay.last_completion_metadata["model"] == "scripted"
    with pytest.raises(ProviderError) as captured:
        replay.complete("prompt")
    assert captured.value.code == "replay_exhausted"


def test_replay_strict_prompt_mismatch_has_stable_code(tmp_path):
    path = tmp_path / "recording.json"
    RecordingModelClient(
        ScriptedModelClient(["<final>done</final>"]), path
    ).complete("original")
    with pytest.raises(ProviderError) as captured:
        ReplayModelClient(path, strict_prompts=True).complete("different")
    assert captured.value.code == "replay_prompt_mismatch"


def test_provider_errors_are_recorded_and_redacted(tmp_path):
    secret = "private-provider-key"

    class FailingClient:
        provider = "openai"
        model = "test"
        supports_prompt_cache = False
        last_completion_metadata = {}

        def complete(self, prompt, max_new_tokens=512, **kwargs):
            raise ProviderError("http_error", f"rejected {secret}")

    path = tmp_path / "error.json"
    recorder = RecordingModelClient(FailingClient(), path, secrets=(secret,))
    with pytest.raises(ProviderError):
        recorder.complete("prompt")
    text = path.read_text(encoding="utf-8")
    assert secret not in text
    assert json.loads(text)["calls"][0]["error"]["code"] == "http_error"


def test_sensitive_metadata_keys_are_always_redacted(tmp_path):
    client = ScriptedModelClient(["<final>done</final>"])
    client.last_completion_metadata = {"api_key": "short"}
    path = tmp_path / "metadata.json"
    recorder = RecordingModelClient(client, path)
    recorder.complete("prompt", api_key="also-short")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["calls"][0]["request"]["options"]["api_key"] == "[REDACTED]"
