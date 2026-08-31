import json
from pathlib import Path
from unittest.mock import patch

from easycoding.provider_benchmark import ProviderBenchmark, main
from easycoding.provider_recording import ReplayModelClient


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks" / "provider" / "tasks.json"


def test_offline_provider_benchmark_covers_recovery_and_usage():
    factory = lambda task: ReplayModelClient(ROOT / task["recording"])
    result = ProviderBenchmark(ROOT, factory, "replay").run(CATALOG)
    metrics = result["metrics"]
    assert result["passed"] == result["task_count"] == 6
    assert result["pass_rate"] == 1.0
    assert metrics["protocol_compliance_rate"] == 14 / 15
    assert metrics["tool_call_validity_rate"] == 7 / 8
    assert metrics["retry_recovery_rate"] == 1.0
    assert metrics["provider_error_rate"] == 0.0
    assert metrics["average_latency_ms"] > 0
    assert metrics["input_tokens"] > 0
    assert metrics["output_tokens"] > 0
    assert metrics["total_tokens"] > 0


def test_default_cli_replay_never_opens_network(capsys):
    with patch("easycoding.providers.urllib.request.urlopen") as urlopen:
        code = main([str(CATALOG), "--repo-root", str(ROOT)])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["mode"] == "replay"
    assert result["passed"] == 6
    urlopen.assert_not_called()


def test_live_openai_without_key_is_reported_as_skipped(capsys, monkeypatch):
    monkeypatch.delenv("EASYCODING_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("easycoding.providers.urllib.request.urlopen") as urlopen:
        code = main([
            str(CATALOG), "--repo-root", str(ROOT), "--live",
            "--provider", "openai", "--model", "test-model",
        ])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "skipped"
    assert "missing" in result["skip_reason"].lower()
    urlopen.assert_not_called()

