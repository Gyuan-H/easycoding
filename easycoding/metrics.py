"""Aggregations and integrity checks for benchmark and run artifacts."""

from collections import Counter
import json
from pathlib import Path


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def analyze_trace(trace, expected_attempts=None, expected_tool_steps=None, malformed_lines=0):
    """Return event counts and structural issues for one run trace."""
    rows = [item for item in trace if isinstance(item, dict)]
    events = [str(item.get("event", "")) for item in rows]
    counts = Counter(event for event in events if event)
    issues = []
    if malformed_lines:
        issues.append(f"malformed_json_lines:{malformed_lines}")
    if len(rows) != len(trace):
        issues.append("non_object_event")
    if not rows:
        issues.append("empty_trace")
    else:
        if events[0] != "run_started":
            issues.append("first_event_not_run_started")
        if events[-1] != "run_finished":
            issues.append("last_event_not_run_finished")
    if counts["run_started"] != 1:
        issues.append(f"run_started_count:{counts['run_started']}")
    if counts["run_finished"] != 1:
        issues.append(f"run_finished_count:{counts['run_finished']}")
    for index, item in enumerate(rows):
        if "schema_version" not in item:
            issues.append(f"missing_schema_version:{index}")
        elif item.get("schema_version") != 1:
            issues.append(f"unsupported_schema_version:{index}:{item.get('schema_version')}")
        if not item.get("created_at"):
            issues.append(f"missing_created_at:{index}")
        if not item.get("event"):
            issues.append(f"missing_event:{index}")
        if item.get("event") == "tool_executed":
            required = {
                "name", "args", "status", "text", "tool_error_code",
                "affected_paths", "workspace_changed",
            }
            if item.get("name") == "run_shell":
                required.update({
                    "risk_level", "approval_required", "approval_granted",
                    "exit_code", "timed_out", "output_truncated",
                })
            missing = sorted(required - set(item))
            if missing:
                issues.append(f"tool_event_missing_fields:{index}:{','.join(missing)}")
        previous = rows[index - 1] if index else {}
        if item.get("event") == "model_requested":
            prior_index = index - 1
            while prior_index >= 0 and rows[prior_index].get("event") in {
                "checkpoint_created", "resume_state_detected",
            }:
                prior_index -= 1
            if prior_index < 0 or rows[prior_index].get("event") != "prompt_built":
                issues.append(f"model_requested_without_prompt:{index}")
        if item.get("event") == "model_parsed" and previous.get("event") != "model_requested":
            issues.append(f"model_parsed_without_request:{index}")
        if item.get("event") == "tool_executed" and not (
            previous.get("event") == "model_parsed" and previous.get("kind") == "tool"
        ):
            issues.append(f"tool_executed_without_tool_parse:{index}")
        if item.get("event") == "model_retry" and not (
            previous.get("event") == "model_parsed" and previous.get("kind") == "retry"
        ):
            issues.append(f"model_retry_without_retry_parse:{index}")
    model_requests = counts["model_requested"]
    tool_executions = counts["tool_executed"]
    retries = counts["model_retry"]
    protocol_error_codes = Counter(
        str(item.get("error_code")) for item in rows
        if item.get("event") == "model_retry" and item.get("error_code")
    )
    if expected_attempts is not None and model_requests != int(expected_attempts):
        issues.append(f"attempt_trace_mismatch:{expected_attempts}:{model_requests}")
    if expected_tool_steps is not None and tool_executions != int(expected_tool_steps):
        issues.append(f"tool_step_trace_mismatch:{expected_tool_steps}:{tool_executions}")
    return {
        "event_count": len(rows),
        "event_counts": dict(counts),
        "model_requests": model_requests,
        "tool_executions": tool_executions,
        "retry_events": retries,
        "protocol_error_codes": dict(protocol_error_codes),
        "integrity_ok": not issues,
        "issues": issues,
    }


def aggregate_benchmark(result):
    rows = result.get("tasks", [])
    tool_statuses = Counter()
    tool_error_codes = Counter()
    contract_failures = Counter()
    trace_events = Counter()
    protocol_error_codes = Counter()
    shell_risk_levels = Counter()
    for row in rows:
        tool_statuses.update(item for item in row.get("tool_statuses", []) if item)
        tool_error_codes.update(item for item in row.get("tool_error_codes", []) if item)
        contract_failures.update(
            item.get("field", "unknown") for item in row.get("contract_failures", [])
        )
        trace_events.update(item for item in row.get("trace_events", []) if item)
        protocol_error_codes.update(row.get("protocol_error_codes", {}))
        shell_risk_levels.update(item for item in row.get("shell_risk_levels", []) if item)
    task_count = len(rows)
    tool_count = sum(tool_statuses.values())
    trace_ok = sum(bool(row.get("trace_integrity_ok")) for row in rows)
    return {
        "task_count": task_count,
        "passed": sum(bool(row.get("passed")) for row in rows),
        "pass_rate": _rate(sum(bool(row.get("passed")) for row in rows), task_count),
        "within_budget_rate": _rate(sum(bool(row.get("within_budget")) for row in rows), task_count),
        "artifact_success_rate": _rate(sum(bool(row.get("artifact_exists")) for row in rows), task_count),
        "verifier_pass_rate": _rate(sum(bool(row.get("verifier_passed")) for row in rows), task_count),
        "avg_attempts": _rate(sum(int(row.get("attempts", 0)) for row in rows), task_count),
        "avg_tool_steps": _rate(sum(int(row.get("tool_steps", 0)) for row in rows), task_count),
        "total_attempts": sum(int(row.get("attempts", 0)) for row in rows),
        "total_tool_steps": sum(int(row.get("tool_steps", 0)) for row in rows),
        "failure_categories": dict(Counter(
            row.get("failure_category", "") for row in rows if row.get("failure_category")
        )),
        "stop_reasons": dict(Counter(row.get("stop_reason", "") for row in rows if row.get("stop_reason"))),
        "tool_statuses": dict(tool_statuses),
        "tool_error_codes": dict(tool_error_codes),
        "tool_success_rate": _rate(tool_statuses["success"], tool_count),
        "tool_rejection_rate": _rate(tool_statuses["rejected"], tool_count),
        "tool_error_rate": _rate(tool_statuses["error"] + tool_statuses["partial_success"], tool_count),
        "contract_failures": dict(contract_failures),
        "trace_integrity_rate": _rate(trace_ok, task_count),
        "avg_trace_events": _rate(sum(int(row.get("trace_event_count", 0)) for row in rows), task_count),
        "trace_event_counts": dict(trace_events),
        "model_requests": sum(int(row.get("trace_model_requests", 0)) for row in rows),
        "retry_events": sum(int(row.get("trace_retry_events", 0)) for row in rows),
        "protocol_error_codes": dict(protocol_error_codes),
        "shell_risk_levels": dict(shell_risk_levels),
        "shell_timeout_count": sum(int(row.get("shell_timeout_count", 0)) for row in rows),
        "shell_output_truncation_count": sum(
            int(row.get("shell_output_truncation_count", 0)) for row in rows
        ),
    }


def _read_trace(path):
    rows = []
    malformed = 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], 0
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def aggregate_run_reports(runs_root):
    reports = []
    malformed_reports = 0
    trace_analyses = []
    paths = list(Path(runs_root).glob("*/report.json"))
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("report must be an object")
        except (OSError, ValueError, json.JSONDecodeError):
            malformed_reports += 1
            continue
        reports.append(report)
        trace, malformed_lines = _read_trace(path.with_name("trace.jsonl"))
        trace_analyses.append(analyze_trace(
            trace,
            expected_attempts=report.get("attempts"),
            expected_tool_steps=report.get("tool_steps"),
            malformed_lines=malformed_lines,
        ))
    run_count = len(reports)
    trace_events = Counter()
    protocol_error_codes = Counter()
    for analysis in trace_analyses:
        trace_events.update(analysis["event_counts"])
        protocol_error_codes.update(analysis["protocol_error_codes"])
    return {
        "discovered_report_count": len(paths),
        "run_count": run_count,
        "malformed_report_count": malformed_reports,
        "avg_attempts": _rate(sum(int(item.get("attempts", 0)) for item in reports), run_count),
        "avg_tool_steps": _rate(sum(int(item.get("tool_steps", 0)) for item in reports), run_count),
        "status_counts": dict(Counter(item.get("status", "") for item in reports if item.get("status"))),
        "stop_reasons": dict(Counter(item.get("stop_reason", "") for item in reports if item.get("stop_reason"))),
        "resume_states": dict(Counter(
            item.get("resume_state", {}).get("status", "")
            for item in reports if isinstance(item.get("resume_state"), dict)
        )),
        "trace_integrity_rate": _rate(
            sum(bool(item["integrity_ok"]) for item in trace_analyses), run_count
        ),
        "trace_event_counts": dict(trace_events),
        "protocol_error_codes": dict(protocol_error_codes),
        "trace_issues": dict(Counter(
            issue for analysis in trace_analyses for issue in analysis["issues"]
        )),
    }
