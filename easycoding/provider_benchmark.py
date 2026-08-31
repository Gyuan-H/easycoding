"""Replay-first benchmark for OpenAI-compatible and Ollama providers."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from .evaluator import BenchmarkEvaluator
from .config import load_project_env
from .metrics import analyze_trace
from .provider_recording import RecordingModelClient, ReplayModelClient
from .providers import OllamaModelClient, OpenAICompatibleModelClient
from .runtime import EasyCoding
from .workspace import WorkspaceContext


INVALID_TOOL_CODES = {
    "missing_argument", "invalid_argument_type", "argument_out_of_range",
    "unexpected_argument", "unknown_tool",
}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _read_trace(path):
    rows = []
    malformed = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def _call_metrics(calls):
    metadata = [dict(item.get("metadata", {}) or {}) for item in calls]
    durations = [float(item["duration_ms"]) for item in metadata if item.get("duration_ms") is not None]
    return {
        "latencies_ms": durations,
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in metadata),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in metadata),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in metadata),
    }


class ProviderBenchmark:
    def __init__(self, repo_root, client_factory, mode):
        self.repo_root = Path(repo_root).resolve()
        self.client_factory = client_factory
        self.mode = mode

    def run(self, benchmark_path):
        tasks = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("provider benchmark must contain a non-empty JSON array")
        rows = []
        for task in tasks:
            try:
                rows.append(self.run_task(task))
            except Exception as exc:
                rows.append({
                    "id": task.get("id", "unknown"), "passed": False,
                    "status": "harness_error", "provider_error": False,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                    "model_responses": 0, "protocol_valid_responses": 0,
                    "tool_calls": 0, "valid_tool_calls": 0, "retry_events": 0,
                    "recovered_after_retry": False, "latencies_ms": [],
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                })
        return self._aggregate(rows)

    def run_task(self, task):
        fixture = (self.repo_root / task["fixture_repo"]).resolve()
        with tempfile.TemporaryDirectory(prefix=f"easycoding-provider-{task['id']}-") as temporary:
            workspace_root = Path(temporary) / fixture.name
            shutil.copytree(fixture, workspace_root)
            client = self.client_factory(task)
            agent = EasyCoding(
                client, WorkspaceContext.build(workspace_root),
                approval_policy=task.get("approval_policy", "auto"),
                max_steps=int(task.get("step_budget", 6)),
                max_new_tokens=int(task.get("max_new_tokens", 512)),
                allowed_tools=task["allowed_tools"],
            )
            answer = agent.ask(task["prompt"])
            state = agent.current_task_state
            verifier = BenchmarkEvaluator._run_verifier(workspace_root, task["verifier"])
            artifact_exists = (workspace_root / task["expected_artifact"]).exists()
            trace, malformed = _read_trace(
                agent.run_store.run_dir(state.run_id) / "trace.jsonl"
            )
            analysis = analyze_trace(
                trace, state.attempts, state.tool_steps, malformed
            )
            parsed = [item for item in trace if item.get("event") == "model_parsed"]
            tools = [item for item in trace if item.get("event") == "tool_executed"]
            retries = analysis["retry_events"]
            calls = getattr(client, "call_history", None)
            if calls is None:
                calls = getattr(client, "calls", [])
            usage = _call_metrics(calls)
            completed = state.status == "completed"
            passed = completed and artifact_exists and verifier["passed"] and analysis["integrity_ok"]
            return {
                "id": task["id"], "passed": passed, "status": state.status,
                "stop_reason": state.stop_reason, "final_answer": answer,
                "provider_error": state.stop_reason == "model_error",
                "artifact_exists": artifact_exists,
                "verifier_passed": verifier["passed"],
                "trace_integrity_ok": analysis["integrity_ok"],
                "attempts": state.attempts, "tool_steps": state.tool_steps,
                "model_responses": len(parsed),
                "protocol_valid_responses": sum(item.get("kind") != "retry" for item in parsed),
                "tool_calls": len(tools),
                "valid_tool_calls": sum(
                    item.get("tool_error_code") not in INVALID_TOOL_CODES for item in tools
                ),
                "retry_events": retries,
                "recovered_after_retry": bool(retries and completed),
                **usage,
            }

    def _aggregate(self, rows):
        responses = sum(item["model_responses"] for item in rows)
        tool_calls = sum(item["tool_calls"] for item in rows)
        retried = sum(bool(item["retry_events"]) for item in rows)
        latencies = [value for item in rows for value in item["latencies_ms"]]
        task_count = len(rows)
        return {
            "mode": self.mode,
            "task_count": task_count,
            "passed": sum(bool(item["passed"]) for item in rows),
            "pass_rate": _rate(sum(bool(item["passed"]) for item in rows), task_count),
            "tasks": rows,
            "metrics": {
                "protocol_compliance_rate": _rate(
                    sum(item["protocol_valid_responses"] for item in rows), responses
                ),
                "tool_call_validity_rate": _rate(
                    sum(item["valid_tool_calls"] for item in rows), tool_calls
                ),
                "retry_recovery_rate": _rate(
                    sum(bool(item["recovered_after_retry"]) for item in rows), retried
                ),
                "provider_error_rate": _rate(
                    sum(bool(item["provider_error"]) for item in rows), task_count
                ),
                "average_latency_ms": _rate(sum(latencies), len(latencies)),
                "input_tokens": sum(item["input_tokens"] for item in rows),
                "output_tokens": sum(item["output_tokens"] for item in rows),
                "total_tokens": sum(item["total_tokens"] for item in rows),
                "stop_reasons": dict(Counter(
                    item.get("stop_reason", "") for item in rows if item.get("stop_reason")
                )),
            },
        }


def _live_client(args):
    if args.provider == "openai":
        key = args.api_key or os.environ.get("EASYCODING_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        return OpenAICompatibleModelClient(
            args.model or os.environ.get("EASYCODING_OPENAI_MODEL", "gpt-5"),
            args.base_url or os.environ.get("EASYCODING_OPENAI_API_BASE", "https://api.openai.com/v1"),
            key, timeout=args.timeout,
        )
    return OllamaModelClient(
        args.model or os.environ.get("EASYCODING_OLLAMA_MODEL", "qwen2.5-coder:7b"),
        args.base_url or os.environ.get("EASYCODING_OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout=args.timeout,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay provider recordings by default; use --live explicitly for network calls."
    )
    parser.add_argument("benchmark")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--provider", choices=("openai", "ollama"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--record-dir", default=".easycoding/provider-recordings")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    load_project_env(root)
    if not args.live:
        factory = lambda task: ReplayModelClient(root / task["recording"])
        result = ProviderBenchmark(root, factory, "replay").run(args.benchmark)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] == result["task_count"] else 1
    if not args.provider:
        parser.error("--provider is required with --live")
    probe = _live_client(args)
    status = probe.check()
    if not status.ok:
        print(json.dumps({
            "mode": "live", "status": "skipped", "skip_reason": status.reason,
            "suggestion": status.suggestion, "provider_status": status.to_dict(),
        }, ensure_ascii=False, indent=2))
        return 0
    record_dir = root / args.record_dir

    def factory(task):
        client = _live_client(args)
        secret = getattr(client, "api_key", "")
        return RecordingModelClient(
            client, record_dir / f"{task['id']}.json", secrets=(secret,)
        )

    result = ProviderBenchmark(root, factory, f"live:{args.provider}").run(args.benchmark)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] == result["task_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
