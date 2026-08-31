import json
from pathlib import Path

import pytest

from easycoding.context_benchmark import LongContextBenchmark, validate_task


def load_catalog():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "long_context" / "tasks.json"
    return root, path, json.loads(path.read_text(encoding="utf-8"))


def test_long_context_catalog_has_twelve_distinct_real_profiles():
    root, _, tasks = load_catalog()
    assert len(tasks) == 12
    assert len({task["id"] for task in tasks}) == 12
    assert len({task["profile"] for task in tasks}) == 12
    source_fields = (
        "request_file", "prefix_files", "history_files", "memory_note_files",
        "workspace_files", "file_summary_files", "checkpoint_files",
    )
    referenced = set()
    for task in tasks:
        validate_task(task)
        for field in source_fields:
            values = task.get(field, [])
            if isinstance(values, str):
                values = [values]
            referenced.update(values)
    assert len(referenced) >= 9
    for relative in referenced:
        source = root / relative
        assert source.is_file(), relative
        assert source.stat().st_size >= 500, relative


def test_long_context_benchmark_passes_all_contracts():
    root, path, _ = load_catalog()
    result = LongContextBenchmark(root).run(path)
    metrics = result["metrics"]

    assert result["task_count"] == 12
    assert result["passed"] == 12
    assert result["context_pass_rate"] == 1.0
    assert metrics["profile_count"] == 12
    assert metrics["request_retention_rate"] == 1.0
    assert metrics["evidence_retention_rate"] == 1.0
    assert metrics["floor_violation_count"] == 0
    assert metrics["reduction_order_violation_count"] == 0
    assert metrics["determinism_failure_count"] == 0
    assert metrics["trace_integrity_rate"] == 1.0
    assert metrics["contract_failures"] == {}
    assert metrics["avg_prompt_chars"] > 3000
    assert metrics["avg_reduction_rate"] > 0.1
    assert metrics["total_repeated_reads"] >= 50
    assert metrics["avg_follow_up_tool_steps"] == 1 / 12

    rows = {row["id"]: row for row in result["tasks"]}
    assert rows["lc02_long_current_request"]["budget_overflow_chars"] > 0
    assert rows["lc05_repeated_file_reads"]["repeated_read_count"] == 11
    assert rows["lc09_relevant_memory_saturation"]["relevant_memory_count"] == 3
    assert rows["lc12_comprehensive_pressure"]["reduction_order"] == [
        "relevant_memory", "history", "memory", "prefix"
    ]


def test_long_context_task_schema_rejects_invalid_budget():
    with pytest.raises(ValueError, match="total_budget"):
        validate_task({
            "id": "bad", "profile": "bad", "fixture_repo": ".",
            "request": "bad", "total_budget": 0,
        })
