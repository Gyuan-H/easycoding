"""Deterministic benchmark for long-context assembly using real project files."""

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from . import checkpoint as checkpointlib
from .context_manager import ContextManager, REDUCTION_ORDER
from .metrics import analyze_trace
from .providers import ScriptedModelClient
from .runtime import EasyCoding
from .security import resolve_in_workspace
from .task_state import TaskState
from .workspace import WorkspaceContext


REQUIRED_FIELDS = {"id", "profile", "fixture_repo", "request", "total_budget"}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("long-context task must be an object")
    missing = REQUIRED_FIELDS - set(task)
    if missing:
        raise ValueError(f"long-context task missing fields: {sorted(missing)}")
    if int(task["total_budget"]) <= 0:
        raise ValueError("total_budget must be greater than zero")
    for field in (
        "prefix_files", "history_files", "memory_note_files", "workspace_files",
        "file_summary_files", "checkpoint_files", "required_evidence",
        "required_reduced_sections", "allowed_tools", "scripted_outputs",
    ):
        if field in task and not isinstance(task[field], list):
            raise ValueError(f"{field} must be a list")


class LongContextBenchmark:
    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve()

    def run(self, benchmark_path):
        tasks = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise ValueError("long-context benchmark must contain a JSON array")
        rows = []
        for index, task in enumerate(tasks):
            try:
                validate_task(task)
                first = self._run_once(task)
                second = self._run_once(task)
                first["deterministic"] = first["signature"] == second["signature"]
                first["passed"] = first["passed"] and first["deterministic"]
                if not first["deterministic"]:
                    first["contract_failures"].append("nondeterministic_metadata")
                first.pop("signature", None)
                rows.append(first)
            except Exception as exc:
                rows.append(self._harness_error(task, index, exc))
        result = {"task_count": len(rows), "tasks": rows}
        result["passed"] = sum(bool(row.get("passed")) for row in rows)
        result["context_pass_rate"] = _rate(result["passed"], len(rows))
        result["metrics"] = self.aggregate(rows)
        return result

    def _run_once(self, task):
        fixture = resolve_in_workspace(
            self.repo_root, task["fixture_repo"], must_exist=True
        )
        if not fixture.is_dir():
            raise ValueError(f"fixture is not a directory: {fixture}")
        with tempfile.TemporaryDirectory(prefix=f"easycoding-context-{task['id']}-") as temporary:
            workspace_root = Path(temporary) / fixture.name
            shutil.copytree(fixture, workspace_root)
            self._copy_workspace_files(workspace_root, task.get("workspace_files", []))
            self._copy_prefix_documents(workspace_root, task.get("prefix_files", []))
            workspace = WorkspaceContext.build(workspace_root)
            model = ScriptedModelClient(task.get(
                "scripted_outputs", ["<final>Context benchmark completed.</final>"]
            ))
            agent = EasyCoding(
                model, workspace, approval_policy="never",
                max_steps=int(task.get("max_steps", 2)),
                allowed_tools=task.get("allowed_tools"),
            )
            agent.context_manager = ContextManager(
                agent,
                total_budget=int(task["total_budget"]),
                section_budgets=task.get("section_budgets"),
            )
            repeated_reads = self._seed_history(
                agent, task.get("history_files", []), int(task.get("history_repeat", 1))
            )
            self._seed_memory(agent, workspace_root, task)
            request = self._build_request(task)
            if task.get("checkpoint"):
                self._seed_checkpoint(agent, request, task.get("checkpoint_files", []))

            answer = agent.ask(request)
            state = agent.current_task_state
            prompt = model.prompts[-1]
            report_path = agent.run_store.run_dir(state.run_id) / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            metadata = report["prompt_metadata"]
            trace, malformed = self._read_trace(
                agent.run_store.run_dir(state.run_id) / "trace.jsonl"
            )
            trace_analysis = analyze_trace(
                trace, expected_attempts=state.attempts,
                expected_tool_steps=state.tool_steps, malformed_lines=malformed,
            )

            floors = metadata["effective_section_floors"]
            floor_violations = [
                name for name, floor in floors.items()
                if metadata["rendered_chars"][name] < floor
            ]
            reduction_order = metadata["reduction_order"]
            order_violation = not self._is_ordered_subsequence(
                reduction_order, REDUCTION_ORDER
            )
            required_sections = set(task.get("required_reduced_sections", []))
            missing_reductions = sorted(required_sections - set(reduction_order))
            expected_reduction = task.get("expect_reduction")
            reduction_expectation_met = (
                expected_reduction is None
                or bool(reduction_order) is bool(expected_reduction)
            )
            required_evidence = [str(item) for item in task.get("required_evidence", [])]
            missing_evidence = [item for item in required_evidence if item not in prompt]
            request_retained = request in prompt
            relevant_minimum = int(task.get("min_relevant_memory", 0))
            relevant_ok = metadata["relevant_memory_count"] >= relevant_minimum
            expected_repeated = task.get("expected_repeated_reads")
            repeated_reads_ok = (
                expected_repeated is None or repeated_reads == int(expected_repeated)
            )
            contract_failures = []
            if not request_retained:
                contract_failures.append("current_request_not_retained")
            if floor_violations:
                contract_failures.append("section_floor_violation")
            if order_violation:
                contract_failures.append("reduction_order_violation")
            if missing_reductions or not reduction_expectation_met:
                contract_failures.append("reduction_expectation_failed")
            if missing_evidence:
                contract_failures.append("required_evidence_missing")
            if not relevant_ok:
                contract_failures.append("relevant_memory_count_too_low")
            if not repeated_reads_ok:
                contract_failures.append("repeated_read_count_mismatch")
            if not trace_analysis["integrity_ok"]:
                contract_failures.append("trace_invalid")
            passed = state.status == "completed" and not contract_failures
            signature = {
                "answer": answer,
                "status": state.status,
                "stop_reason": state.stop_reason,
                "attempts": state.attempts,
                "tool_steps": state.tool_steps,
                "metadata": {
                    key: metadata[key] for key in (
                        "original_chars", "rendered_chars", "section_budgets",
                        "section_floors", "effective_section_floors",
                        "section_reduction_rates", "current_request_chars",
                        "current_request_retention_rate", "prompt_chars",
                        "soft_budget", "budget_overflow_chars", "budget_reductions",
                        "reduction_order", "total_reduction_rate",
                        "history_event_count", "relevant_memory_count",
                    )
                },
                "missing_evidence": missing_evidence,
                "trace_counts": trace_analysis["event_counts"],
            }
            return {
                "id": task["id"],
                "profile": task["profile"],
                "passed": passed,
                "contract_failures": contract_failures,
                "prompt_chars": metadata["prompt_chars"],
                "soft_budget": metadata["soft_budget"],
                "budget_overflow_chars": metadata["budget_overflow_chars"],
                "current_request_chars": metadata["current_request_chars"],
                "current_request_retention_rate": 1.0 if request_retained else 0.0,
                "section_reduction_rates": metadata["section_reduction_rates"],
                "total_reduction_rate": metadata["total_reduction_rate"],
                "reduction_order": reduction_order,
                "floor_violations": floor_violations,
                "reduction_order_violation": order_violation,
                "required_evidence_count": len(required_evidence),
                "retained_evidence_count": len(required_evidence) - len(missing_evidence),
                "missing_evidence": missing_evidence,
                "history_event_count": metadata["history_event_count"],
                "relevant_memory_count": metadata["relevant_memory_count"],
                "repeated_read_count": repeated_reads,
                "follow_up_tool_steps": state.tool_steps,
                "attempts": state.attempts,
                "trace_integrity_ok": trace_analysis["integrity_ok"],
                "trace_event_count": trace_analysis["event_count"],
                "signature": signature,
            }

    def _source_text(self, relative):
        path = resolve_in_workspace(self.repo_root, relative, must_exist=True)
        if not path.is_file():
            raise ValueError(f"context source must be a file: {relative}")
        return path.read_text(encoding="utf-8", errors="replace")

    def _copy_workspace_files(self, workspace_root, files):
        for relative in files:
            source = resolve_in_workspace(self.repo_root, relative, must_exist=True)
            destination = resolve_in_workspace(workspace_root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def _copy_prefix_documents(self, workspace_root, files):
        document_names = ("AGENTS.md", "package.json", "pyproject.toml")
        if len(files) > len(document_names):
            raise ValueError("prefix_files supports at most three project documents")
        for relative, document_name in zip(files, document_names):
            content = self._source_text(relative)
            rendered = content[:1000] + f"\n[END SOURCE:{relative}]\n"
            (Path(workspace_root) / document_name).write_text(rendered, encoding="utf-8")

    def _seed_history(self, agent, files, repeat):
        paths = []
        for _ in range(max(0, repeat)):
            for relative in files:
                paths.append(str(relative))
                agent.session["history"].append({
                    "role": "tool",
                    "tool_name": "read_file",
                    "tool_args": {"path": str(relative)},
                    "content": self._source_text(relative),
                })
        counts = Counter(paths)
        return sum(max(0, count - 1) for count in counts.values())

    def _seed_memory(self, agent, workspace_root, task):
        tag = str(task.get("memory_query_tag", "context"))
        for relative in task.get("memory_note_files", []):
            text = self._source_text(relative)
            note = text[:400] + f"\n[MEMORY:{relative}]"
            agent.memory.append_note(note, tags=(tag, relative))
        summary_files = set(task.get("file_summary_files", []))
        for relative in summary_files:
            workspace_path = resolve_in_workspace(workspace_root, relative, must_exist=True)
            if not workspace_path.is_file():
                raise ValueError(f"summary source is not a workspace file: {relative}")
            text = workspace_path.read_text(encoding="utf-8", errors="replace")
            agent.memory.remember_file(relative)
            agent.memory.set_file_summary(
                relative, text[:400] + f" [FILE SUMMARY:{relative}]"
            )

    def _seed_checkpoint(self, agent, request, files):
        for relative in files:
            agent.memory.remember_file(relative)
        state = TaskState.create(request)
        checkpointlib.create_checkpoint(
            agent, state, "long_context_seed", next_step="continue benchmark request"
        )
        agent.resume_state = checkpointlib.evaluate_resume_state(agent)

    def _build_request(self, task):
        request = str(task["request"])
        if task.get("request_file"):
            request += (
                f"\n\nAttached real project source ({task['request_file']}):\n"
                + self._source_text(task["request_file"])
            )
        return request

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
    def _is_ordered_subsequence(observed, expected):
        positions = {name: index for index, name in enumerate(expected)}
        try:
            indexes = [positions[name] for name in observed]
        except KeyError:
            return False
        return len(indexes) == len(set(indexes)) and indexes == sorted(indexes)

    @staticmethod
    def _harness_error(task, index, exc):
        task_id = task.get("id", f"task_{index}") if isinstance(task, dict) else f"task_{index}"
        profile = task.get("profile", "unknown") if isinstance(task, dict) else "unknown"
        return {
            "id": task_id,
            "profile": profile,
            "passed": False,
            "deterministic": False,
            "contract_failures": ["harness_error"],
            "harness_error": f"{type(exc).__name__}: {exc}"[:1000],
            "prompt_chars": 0,
            "soft_budget": 0,
            "budget_overflow_chars": 0,
            "current_request_chars": 0,
            "current_request_retention_rate": 0.0,
            "section_reduction_rates": {},
            "total_reduction_rate": 0.0,
            "reduction_order": [],
            "floor_violations": [],
            "reduction_order_violation": False,
            "required_evidence_count": 0,
            "retained_evidence_count": 0,
            "missing_evidence": [],
            "history_event_count": 0,
            "relevant_memory_count": 0,
            "repeated_read_count": 0,
            "follow_up_tool_steps": 0,
            "attempts": 0,
            "trace_integrity_ok": False,
            "trace_event_count": 0,
        }

    @staticmethod
    def aggregate(rows):
        count = len(rows)
        evidence_total = sum(int(row.get("required_evidence_count", 0)) for row in rows)
        return {
            "task_count": count,
            "profile_count": len({row.get("profile") for row in rows}),
            "context_pass_rate": _rate(sum(bool(row.get("passed")) for row in rows), count),
            "request_retention_rate": _rate(
                sum(float(row.get("current_request_retention_rate", 0.0)) for row in rows), count
            ),
            "evidence_retention_rate": _rate(
                sum(int(row.get("retained_evidence_count", 0)) for row in rows), evidence_total
            ),
            "avg_prompt_chars": _rate(sum(int(row.get("prompt_chars", 0)) for row in rows), count),
            "avg_reduction_rate": _rate(
                sum(float(row.get("total_reduction_rate", 0.0)) for row in rows), count
            ),
            "floor_violation_count": sum(len(row.get("floor_violations", [])) for row in rows),
            "reduction_order_violation_count": sum(
                bool(row.get("reduction_order_violation")) for row in rows
            ),
            "determinism_failure_count": sum(not bool(row.get("deterministic")) for row in rows),
            "trace_integrity_rate": _rate(
                sum(bool(row.get("trace_integrity_ok")) for row in rows), count
            ),
            "total_repeated_reads": sum(int(row.get("repeated_read_count", 0)) for row in rows),
            "avg_follow_up_tool_steps": _rate(
                sum(int(row.get("follow_up_tool_steps", 0)) for row in rows), count
            ),
            "contract_failures": dict(Counter(
                failure for row in rows for failure in row.get("contract_failures", [])
            )),
        }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    result = LongContextBenchmark(args.repo_root).run(args.benchmark)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["context_pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
