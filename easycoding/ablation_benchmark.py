"""Controlled durable-memory and checkpoint-context ablation benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

from . import checkpoint as checkpointlib
from .metrics import analyze_trace
from .runtime import EasyCoding
from .task_state import TaskState
from .workspace import WorkspaceContext


CONFIGS = (
    {"id": "full", "durable_memory_enabled": True, "resume_enabled": True},
    {"id": "no_memory", "durable_memory_enabled": False, "resume_enabled": True},
    {"id": "no_resume", "durable_memory_enabled": True, "resume_enabled": False},
    {"id": "neither", "durable_memory_enabled": False, "resume_enabled": False},
)
REQUIRED_TASK_FIELDS = {"id", "fixture_repo", "request", "required_evidence"}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


class EvidenceModelClient:
    """Deterministic judge whose answer depends only on retained prompt evidence."""

    model = "evidence-aware-fixture"
    supports_prompt_cache = False

    def __init__(self, required_evidence):
        self.required_evidence = [str(item) for item in required_evidence]
        self.prompts = []
        self.last_completion_metadata = {
            "provider": "fixture", "model": self.model, "mode": "ablation",
        }

    def complete(self, prompt, max_new_tokens=512, **kwargs):
        self.prompts.append(str(prompt))
        retained = all(item in str(prompt) for item in self.required_evidence)
        return (
            "<final>Evidence available.</final>" if retained
            else "<final>Missing required evidence.</final>"
        )


def validate_benchmark(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("ablation benchmark must be an object containing tasks")
    for task in payload["tasks"]:
        if not isinstance(task, dict):
            raise ValueError("ablation task must be an object")
        missing = REQUIRED_TASK_FIELDS - set(task)
        if missing:
            raise ValueError(f"ablation task missing fields: {sorted(missing)}")
        if not isinstance(task["required_evidence"], list):
            raise ValueError("required_evidence must be a list")
        if not isinstance(task.get("durable_facts", []), list):
            raise ValueError("durable_facts must be a list")


class AblationBenchmark:
    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve()

    def run(self, benchmark_path):
        path = Path(benchmark_path).resolve()
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        validate_benchmark(payload)
        configurations = []
        for config in CONFIGS:
            rows = [self._run_task(task, config) for task in payload["tasks"]]
            configurations.append({
                **config,
                "tasks": rows,
                "metrics": self._aggregate(rows),
            })
        full_metrics = configurations[0]["metrics"]
        for item in configurations:
            metrics = item["metrics"]
            metrics["delta_vs_full"] = {
                "pass_rate": metrics["pass_rate"] - full_metrics["pass_rate"],
                "evidence_retention_rate": (
                    metrics["evidence_retention_rate"]
                    - full_metrics["evidence_retention_rate"]
                ),
                "avg_attempts": metrics["avg_attempts"] - full_metrics["avg_attempts"],
                "avg_tool_steps": metrics["avg_tool_steps"] - full_metrics["avg_tool_steps"],
                "avg_prompt_chars": (
                    metrics["avg_prompt_chars"] - full_metrics["avg_prompt_chars"]
                ),
            }
        observed = {
            item["id"]: item["metrics"]["pass_rate"] for item in configurations
        }
        expected = payload.get("expected_pass_rates", {})
        failures = [
            {"configuration": name, "expected": float(rate), "actual": observed.get(name)}
            for name, rate in expected.items()
            if observed.get(name) != float(rate)
        ]
        return {
            "schema_version": 1,
            "task_count": len(payload["tasks"]),
            "configuration_count": len(configurations),
            "provenance": {
                "benchmark": str(path),
                "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
                "evaluator": "easycoding.ablation_benchmark",
                "model": EvidenceModelClient.model,
                "mode": "deterministic-controlled-ablation",
                "python": platform.python_version(),
            },
            "expected_pass_rates": expected,
            "observed_pass_rates": observed,
            "contract_failures": failures,
            "configurations": configurations,
        }

    def _run_task(self, task, config):
        fixture = (self.repo_root / task["fixture_repo"]).resolve()
        if not fixture.is_dir() or self.repo_root not in fixture.parents:
            raise ValueError(f"fixture not found inside repo root: {fixture}")
        with tempfile.TemporaryDirectory(prefix=f"easycoding-ablation-{task['id']}-") as temporary:
            workspace_root = Path(temporary) / fixture.name
            shutil.copytree(fixture, workspace_root)
            model = EvidenceModelClient(task["required_evidence"])
            agent = EasyCoding(
                model, WorkspaceContext.build(workspace_root),
                approval_policy="never", max_steps=1,
                allowed_tools=task.get("allowed_tools", ["read_file"]),
                durable_memory_enabled=config["durable_memory_enabled"],
                resume_enabled=config["resume_enabled"],
            )
            for fact in task.get("durable_facts", []):
                agent.memory.durable.promote(fact["topic"], fact["text"])
            checkpoint = task.get("checkpoint")
            if checkpoint:
                state = TaskState.create(checkpoint.get("current_goal", task["request"]))
                checkpointlib.create_checkpoint(
                    agent, state, "ablation_seed",
                    blocker=checkpoint.get("current_blocker", ""),
                    next_step=checkpoint.get("next_step", ""),
                )
                agent.resume_state = checkpointlib.evaluate_resume_state(agent)
            answer = agent.ask(task["request"])
            state = agent.current_task_state
            prompt = model.prompts[-1]
            missing = [
                item for item in task["required_evidence"] if item not in prompt
            ]
            report_path = agent.run_store.run_dir(state.run_id) / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            trace, malformed = self._read_trace(
                agent.run_store.run_dir(state.run_id) / "trace.jsonl"
            )
            analysis = analyze_trace(
                trace, expected_attempts=state.attempts,
                expected_tool_steps=state.tool_steps, malformed_lines=malformed,
            )
            passed = (
                not missing and answer == "Evidence available."
                and state.status == "completed" and analysis["integrity_ok"]
            )
            metadata = report["prompt_metadata"]
            return {
                "id": task["id"],
                "category": task.get("category", "unspecified"),
                "passed": passed,
                "failure_category": "" if passed else (
                    "missing_evidence" if missing else "runtime_failure"
                ),
                "required_evidence_count": len(task["required_evidence"]),
                "retained_evidence_count": len(task["required_evidence"]) - len(missing),
                "missing_evidence": missing,
                "attempts": state.attempts,
                "tool_steps": state.tool_steps,
                "prompt_chars": metadata["prompt_chars"],
                "durable_memory_hits": metadata["durable_memory_hits"],
                "resume_context_hits": metadata["resume_context_hits"],
                "trace_integrity_ok": analysis["integrity_ok"],
                "trace_issues": analysis["issues"],
                "feature_flags": report["feature_flags"],
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
    def _aggregate(rows):
        count = len(rows)
        evidence_total = sum(row["required_evidence_count"] for row in rows)
        return {
            "task_count": count,
            "passed": sum(row["passed"] for row in rows),
            "pass_rate": _rate(sum(row["passed"] for row in rows), count),
            "evidence_retention_rate": _rate(
                sum(row["retained_evidence_count"] for row in rows), evidence_total
            ),
            "avg_attempts": _rate(sum(row["attempts"] for row in rows), count),
            "avg_tool_steps": _rate(sum(row["tool_steps"] for row in rows), count),
            "total_attempts": sum(row["attempts"] for row in rows),
            "total_tool_steps": sum(row["tool_steps"] for row in rows),
            "avg_prompt_chars": _rate(sum(row["prompt_chars"] for row in rows), count),
            "durable_memory_hits": sum(row["durable_memory_hits"] for row in rows),
            "resume_context_hits": sum(row["resume_context_hits"] for row in rows),
            "trace_integrity_rate": _rate(
                sum(row["trace_integrity_ok"] for row in rows), count
            ),
            "failure_categories": {
                name: sum(row["failure_category"] == name for row in rows)
                for name in sorted({row["failure_category"] for row in rows if row["failure_category"]})
            },
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run controlled durable-memory/resume-context ablations."
    )
    parser.add_argument("benchmark")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    result = AblationBenchmark(args.repo_root).run(args.benchmark)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["contract_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
