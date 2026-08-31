import json
from pathlib import Path

from easycoding.evaluator import BenchmarkEvaluator
from easycoding.metrics import aggregate_benchmark


def test_fixture_benchmark_passes():
    root = Path(__file__).resolve().parents[1]
    result = BenchmarkEvaluator(root).run(root / "benchmarks" / "tasks.json")
    assert result["task_count"] == 10
    assert result["pass_rate"] == 1.0
    metrics = aggregate_benchmark(result)
    assert result["metrics"] == metrics
    assert metrics["avg_attempts"] == 2.6
    assert metrics["avg_tool_steps"] == 1.2
    assert metrics["total_attempts"] == 26
    assert metrics["total_tool_steps"] == 12
    assert metrics["within_budget_rate"] == 1.0
    assert metrics["artifact_success_rate"] == 1.0
    assert metrics["verifier_pass_rate"] == 1.0
    assert metrics["stop_reasons"] == {
        "final_answer_returned": 8,
        "step_limit_reached": 1,
        "retry_limit_reached": 1,
    }
    assert metrics["tool_statuses"] == {
        "success": 7,
        "rejected": 3,
        "error": 1,
        "partial_success": 1,
    }
    assert metrics["contract_failures"] == {}
    assert metrics["failure_categories"] == {}
    assert metrics["trace_integrity_rate"] == 1.0
    assert metrics["model_requests"] == 26
    assert metrics["retry_events"] == 5
    assert metrics["tool_success_rate"] == 7 / 12
    assert metrics["tool_rejection_rate"] == 3 / 12
    assert metrics["tool_error_rate"] == 2 / 12


def test_evaluator_classifies_failure_paths(tmp_path):
    root = Path(__file__).resolve().parents[1]
    common = {
        "prompt": "exercise a failure category",
        "fixture_repo": "benchmarks/fixtures/readme_typo",
        "allowed_tools": ["read_file", "list_files"],
        "step_budget": 1,
        "expected_artifact": "README.md",
        "verifier": {"type": "python", "script": "verify_unchanged.py"},
    }
    tasks = [
        {
            **common,
            "id": "missing_artifact",
            "expected_artifact": "missing.txt",
            "scripted_outputs": ["<final>done</final>"],
        },
        {
            **common,
            "id": "verifier_failed",
            "verifier": {"type": "python", "script": "verify.py"},
            "scripted_outputs": ["<final>done</final>"],
        },
        {
            **common,
            "id": "unexpected_artifact",
            "expected": {"artifact_exists": False},
            "scripted_outputs": ["<final>done</final>"],
        },
        {
            **common,
            "id": "budget_exceeded",
            "scripted_outputs": [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            ],
        },
        {
            **common,
            "id": "failure_stop_reason",
            "scripted_outputs": ["bad"] * 5,
        },
        {
            **common,
            "id": "contract_mismatch",
            "expected": {"tool_statuses": ["success"]},
            "scripted_outputs": ["<final>done</final>"],
        },
        {
            **common,
            "id": "harness_error",
            "fixture_repo": "benchmarks/fixtures/does_not_exist",
            "scripted_outputs": ["<final>done</final>"],
        },
    ]
    benchmark = tmp_path / "failures.json"
    benchmark.write_text(json.dumps(tasks), encoding="utf-8")

    result = BenchmarkEvaluator(root).run(benchmark)

    assert result["passed"] == 0
    assert result["pass_rate"] == 0.0
    assert [row["failure_category"] for row in result["tasks"]] == [
        "missing_artifact",
        "verifier_failed",
        "unexpected_artifact",
        "budget_exceeded",
        "failure_stop_reason",
        "contract_mismatch",
        "harness_error",
    ]
    assert result["metrics"]["failure_categories"] == {
        "missing_artifact": 1,
        "verifier_failed": 1,
        "unexpected_artifact": 1,
        "budget_exceeded": 1,
        "failure_stop_reason": 1,
        "contract_mismatch": 1,
        "harness_error": 1,
    }


def test_evaluator_rejects_a_corrupt_trace(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    task = {
        "id": "trace_invalid",
        "prompt": "finish normally",
        "fixture_repo": "benchmarks/fixtures/readme_typo",
        "allowed_tools": ["read_file"],
        "step_budget": 1,
        "expected_artifact": "README.md",
        "verifier": {"type": "python", "script": "verify_unchanged.py"},
        "scripted_outputs": ["<final>done</final>"],
    }
    monkeypatch.setattr(
        BenchmarkEvaluator, "_read_trace", staticmethod(lambda path: ([], 1))
    )
    row = BenchmarkEvaluator(root).run_task(task)
    assert row["passed"] is False
    assert row["failure_category"] == "trace_invalid"
    assert row["trace_integrity_ok"] is False
    assert "malformed_json_lines:1" in row["trace_issues"]
