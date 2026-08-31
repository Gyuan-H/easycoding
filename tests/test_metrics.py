import json

from easycoding.metrics import aggregate_run_reports, analyze_trace


def event(event_name, **details):
    return {
        "schema_version": 1,
        "event": event_name,
        "created_at": "2026-01-01T00:00:00+00:00",
        **details,
    }


def test_trace_integrity_and_runtime_counts():
    trace = [
        event("run_started"),
        event("prompt_built"),
        event("model_requested"),
        event("model_parsed", kind="tool"),
        event(
            "tool_executed", name="read_file", args={"path": "README.md"},
            status="success", text="ok", tool_error_code="",
            affected_paths=[], workspace_changed=False,
        ),
        event("checkpoint_created"),
        event("run_finished"),
    ]
    result = analyze_trace(trace, expected_attempts=1, expected_tool_steps=1)
    assert result["integrity_ok"] is True
    assert result["event_count"] == 7
    assert result["model_requests"] == 1
    assert result["tool_executions"] == 1
    assert result["retry_events"] == 0
    assert result["protocol_error_codes"] == {}
    assert result["issues"] == []


def test_trace_metrics_count_protocol_error_codes():
    trace = [
        event("run_started"),
        event("prompt_built"),
        event("model_requested"),
        event("model_parsed", kind="retry", error_code="invalid_protocol"),
        event(
            "model_retry", reason="extra text", error_code="invalid_protocol"
        ),
        event("run_finished"),
    ]
    result = analyze_trace(trace, expected_attempts=1, expected_tool_steps=0)
    assert result["integrity_ok"] is True
    assert result["protocol_error_codes"] == {"invalid_protocol": 1}


def test_trace_integrity_reports_corruption_and_counter_mismatch():
    trace = [{"event": "model_requested"}]
    result = analyze_trace(
        trace, expected_attempts=2, expected_tool_steps=1, malformed_lines=1
    )
    assert result["integrity_ok"] is False
    assert "malformed_json_lines:1" in result["issues"]
    assert "first_event_not_run_started" in result["issues"]
    assert "last_event_not_run_finished" in result["issues"]
    assert "missing_schema_version:0" in result["issues"]
    assert "missing_created_at:0" in result["issues"]
    assert "attempt_trace_mismatch:2:1" in result["issues"]
    assert "tool_step_trace_mismatch:1:0" in result["issues"]
    assert "model_requested_without_prompt:0" in result["issues"]


def test_aggregate_run_reports_tracks_malformed_and_missing_trace(tmp_path):
    good = tmp_path / "run_good"
    missing_trace = tmp_path / "run_missing_trace"
    malformed = tmp_path / "run_malformed"
    good.mkdir()
    missing_trace.mkdir()
    malformed.mkdir()
    (good / "report.json").write_text(json.dumps({
        "status": "completed",
        "stop_reason": "final_answer_returned",
        "attempts": 1,
        "tool_steps": 0,
        "resume_state": {"status": "ready"},
    }), encoding="utf-8")
    (good / "trace.jsonl").write_text("\n".join(json.dumps(item) for item in [
        event("run_started"),
        event("prompt_built"),
        event("model_requested"),
        event("model_parsed", kind="final"),
        event("checkpoint_created"),
        event("run_finished"),
    ]), encoding="utf-8")
    (missing_trace / "report.json").write_text(json.dumps({
        "status": "stopped",
        "stop_reason": "retry_limit_reached",
        "attempts": 2,
        "tool_steps": 0,
        "resume_state": {"status": "partial-stale"},
    }), encoding="utf-8")
    (malformed / "report.json").write_text("not-json", encoding="utf-8")

    result = aggregate_run_reports(tmp_path)

    assert result["discovered_report_count"] == 3
    assert result["run_count"] == 2
    assert result["malformed_report_count"] == 1
    assert result["avg_attempts"] == 1.5
    assert result["avg_tool_steps"] == 0.0
    assert result["status_counts"] == {"completed": 1, "stopped": 1}
    assert result["stop_reasons"] == {
        "final_answer_returned": 1,
        "retry_limit_reached": 1,
    }
    assert result["resume_states"] == {"ready": 1, "partial-stale": 1}
    assert result["trace_integrity_rate"] == 0.5
    assert result["trace_issues"]["empty_trace"] == 1
