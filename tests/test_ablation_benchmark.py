from easycoding import checkpoint
from easycoding.ablation_benchmark import AblationBenchmark
from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.task_state import TaskState
from easycoding.workspace import WorkspaceContext


def test_feature_flags_control_prompt_injection(tmp_path):
    enabled_model = ScriptedModelClient(["<final>done</final>"])
    enabled = EasyCoding(enabled_model, WorkspaceContext.build(tmp_path))
    enabled.memory.durable.promote("project-conventions", "Python version: 3.12")
    checkpoint.create_checkpoint(
        enabled, TaskState.create("continue"), "test", next_step="resume marker R-1"
    )
    enabled.resume_state = checkpoint.evaluate_resume_state(enabled)
    enabled.ask("Python version resume marker")
    assert "Python version: 3.12" in enabled_model.prompts[0]
    assert "resume marker R-1" in enabled_model.prompts[0]

    disabled_model = ScriptedModelClient(["<final>done</final>"])
    disabled = EasyCoding(
        disabled_model, WorkspaceContext.build(tmp_path),
        durable_memory_enabled=False, resume_enabled=False,
    )
    disabled.ask("Python version resume marker")
    assert "Python version: 3.12" not in disabled_model.prompts[0]
    assert "resume marker R-1" not in disabled_model.prompts[0]
    report = disabled.run_store.run_dir(disabled.current_task_state.run_id) / "report.json"
    assert report.exists()


def test_ablation_matrix_has_expected_causal_rates():
    root = WorkspaceContext.build(".").repo_root
    result = AblationBenchmark(root).run("benchmarks/ablation/tasks.json")
    assert result["contract_failures"] == []
    assert result["observed_pass_rates"] == {
        "full": 1.0,
        "no_memory": 0.5,
        "no_resume": 0.5,
        "neither": 0.25,
    }
    rows = {item["id"]: item for item in result["configurations"]}
    assert rows["full"]["metrics"]["durable_memory_hits"] == 2
    assert rows["full"]["metrics"]["resume_context_hits"] == 2
    assert rows["neither"]["metrics"]["durable_memory_hits"] == 0
    assert rows["neither"]["metrics"]["resume_context_hits"] == 0
    assert rows["full"]["metrics"]["trace_integrity_rate"] == 1.0
    assert len(result["provenance"]["benchmark_sha256"]) == 64
