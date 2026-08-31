"""Reproducible fixture-based benchmark evaluator."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .metrics import aggregate_benchmark, analyze_trace
from .providers import ScriptedModelClient
from .runtime import EasyCoding
from .security import shell_env
from .workspace import WorkspaceContext


REQUIRED_FIELDS = {
    "id", "prompt", "fixture_repo", "allowed_tools", "step_budget",
    "expected_artifact", "verifier",
}

DEFAULT_EXPECTED = {
    "artifact_exists": True,
    "verifier_passed": True,
    "stop_reason": "final_answer_returned",
}


def validate_task(task):
    missing = REQUIRED_FIELDS - set(task)
    if missing:
        raise ValueError(f"benchmark task missing fields: {sorted(missing)}")
    if not isinstance(task["allowed_tools"], list) or not task["allowed_tools"]:
        raise ValueError("allowed_tools must be a non-empty list")
    if task.get("approval_policy", "auto") not in {"auto", "ask", "never"}:
        raise ValueError("approval_policy must be auto, ask, or never")
    if not isinstance(task.get("expected", {}), dict):
        raise ValueError("expected must be an object")


class BenchmarkEvaluator:
    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve()

    def run(self, benchmark_path):
        path = Path(benchmark_path)
        tasks = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise ValueError("benchmark must contain a JSON array")
        rows = []
        for index, task in enumerate(tasks):
            try:
                rows.append(self.run_task(task))
            except Exception as exc:
                rows.append(self._harness_error_row(task, index, exc))
        result = {
            "task_count": len(rows),
            "passed": sum(bool(row["passed"]) for row in rows),
            "pass_rate": sum(bool(row["passed"]) for row in rows) / len(rows) if rows else 0.0,
            "tasks": rows,
        }
        result["metrics"] = aggregate_benchmark(result)
        return result

    def run_task(self, task):
        validate_task(task)
        fixture = (self.repo_root / task["fixture_repo"]).resolve()
        if not fixture.is_dir():
            raise ValueError(f"fixture not found: {fixture}")
        with tempfile.TemporaryDirectory(prefix=f"easycoding-{task['id']}-") as temporary:
            workspace_root = Path(temporary) / fixture.name
            shutil.copytree(fixture, workspace_root)
            workspace = WorkspaceContext.build(workspace_root)
            model = ScriptedModelClient(task.get("scripted_outputs", ["<final>Done.</final>"]))
            agent = EasyCoding(
                model, workspace, approval_policy=task.get("approval_policy", "auto"),
                max_steps=int(task["step_budget"]), allowed_tools=task["allowed_tools"],
                read_only=bool(task.get("read_only", False)),
            )
            answer = agent.ask(task["prompt"])
            state = agent.current_task_state
            verifier = self._run_verifier(workspace_root, task["verifier"])
            artifact_exists = (workspace_root / task["expected_artifact"]).exists()
            within_budget = state.tool_steps <= int(task["step_budget"])
            trace, malformed_lines = self._read_trace(
                agent.run_store.run_dir(state.run_id) / "trace.jsonl"
            )
            trace_analysis = analyze_trace(
                trace, expected_attempts=state.attempts,
                expected_tool_steps=state.tool_steps,
                malformed_lines=malformed_lines,
            )
            tool_events = [item for item in trace if item.get("event") == "tool_executed"]
            actual = {
                "artifact_exists": artifact_exists,
                "verifier_passed": verifier["passed"],
                "stop_reason": state.stop_reason,
                "tool_statuses": [item.get("status", "") for item in tool_events],
                "tool_error_codes": [item.get("tool_error_code", "") for item in tool_events],
                "trace_events": [item.get("event", "") for item in trace],
                "shell_risk_levels": [
                    item.get("risk_level", "") for item in tool_events
                    if item.get("name") == "run_shell"
                ],
                "shell_timeout_count": sum(
                    bool(item.get("timed_out")) for item in tool_events
                    if item.get("name") == "run_shell"
                ),
                "shell_output_truncation_count": sum(
                    bool(item.get("output_truncated")) for item in tool_events
                    if item.get("name") == "run_shell"
                ),
            }
            expected = {**DEFAULT_EXPECTED, **task.get("expected", {})}
            contract_failures = self._compare_contract(expected, actual)
            passed = within_budget and trace_analysis["integrity_ok"] and not contract_failures
            failure = self._failure_category(
                expected, actual, within_budget, trace_analysis, contract_failures
            )
            return {
                "id": task["id"],
                "passed": passed,
                "failure_category": failure,
                "within_budget": within_budget,
                "artifact_exists": artifact_exists,
                "verifier_passed": verifier["passed"],
                "verifier_output": verifier["output"],
                "stop_reason": state.stop_reason,
                "attempts": state.attempts,
                "tool_steps": state.tool_steps,
                "final_answer": answer,
                "tool_statuses": actual["tool_statuses"],
                "tool_error_codes": actual["tool_error_codes"],
                "trace_events": actual["trace_events"],
                "trace_event_count": trace_analysis["event_count"],
                "trace_model_requests": trace_analysis["model_requests"],
                "trace_retry_events": trace_analysis["retry_events"],
                "protocol_error_codes": trace_analysis["protocol_error_codes"],
                "shell_risk_levels": actual["shell_risk_levels"],
                "shell_timeout_count": actual["shell_timeout_count"],
                "shell_output_truncation_count": actual["shell_output_truncation_count"],
                "trace_integrity_ok": trace_analysis["integrity_ok"],
                "trace_issues": trace_analysis["issues"],
                "contract_failures": contract_failures,
            }

    @staticmethod
    def _read_trace(path):
        rows = []
        malformed = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
        return rows, malformed

    @staticmethod
    def _failure_category(expected, actual, within_budget, trace_analysis, failures):
        if not failures and within_budget and trace_analysis["integrity_ok"]:
            return ""
        fields = {item.get("field") for item in failures}
        if not trace_analysis["integrity_ok"]:
            return "trace_invalid"
        if not within_budget or (
            actual["stop_reason"] == "step_limit_reached"
            and expected.get("stop_reason") != "step_limit_reached"
        ):
            return "budget_exceeded"
        if "artifact_exists" in fields:
            return "missing_artifact" if expected.get("artifact_exists") else "unexpected_artifact"
        if "verifier_passed" in fields:
            return "verifier_failed"
        if "stop_reason" in fields:
            return "failure_stop_reason"
        return "contract_mismatch"

    @staticmethod
    def _harness_error_row(task, index, exc):
        task_id = task.get("id", f"task_{index}") if isinstance(task, dict) else f"task_{index}"
        return {
            "id": task_id,
            "passed": False,
            "failure_category": "harness_error",
            "harness_error": f"{type(exc).__name__}: {exc}"[:1000],
            "within_budget": False,
            "artifact_exists": False,
            "verifier_passed": False,
            "stop_reason": "harness_error",
            "attempts": 0,
            "tool_steps": 0,
            "tool_statuses": [],
            "tool_error_codes": [],
            "trace_events": [],
            "trace_event_count": 0,
            "trace_model_requests": 0,
            "trace_retry_events": 0,
            "protocol_error_codes": {},
            "shell_risk_levels": [],
            "shell_timeout_count": 0,
            "shell_output_truncation_count": 0,
            "trace_integrity_ok": False,
            "trace_issues": ["harness_error"],
            "contract_failures": [],
        }

    @staticmethod
    def _compare_contract(expected, actual):
        failures = []
        for field in ("artifact_exists", "verifier_passed", "stop_reason", "tool_statuses", "tool_error_codes"):
            if field in expected and expected[field] != actual[field]:
                failures.append({"field": field, "expected": expected[field], "actual": actual[field]})
        for event in expected.get("required_trace_events", []):
            if event not in actual["trace_events"]:
                failures.append({"field": "required_trace_events", "expected": event, "actual": "missing"})
        return failures

    @staticmethod
    def _run_verifier(root, verifier):
        if verifier.get("type") != "python":
            raise ValueError("only trusted python verifier scripts are supported")
        script = (Path(root) / verifier["script"]).resolve()
        if script.parent != Path(root).resolve() and Path(root).resolve() not in script.parents:
            raise ValueError("verifier escapes fixture")
        result = subprocess.run(
            [sys.executable, str(script)], cwd=root, capture_output=True,
            text=True, timeout=20, check=False, env=shell_env(root),
        )
        return {"passed": result.returncode == 0, "output": (result.stdout + result.stderr)[:2000]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    print(json.dumps(BenchmarkEvaluator(args.repo_root).run(args.benchmark), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
