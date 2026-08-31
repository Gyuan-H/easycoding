from easycoding.providers import ScriptedModelClient
from easycoding.prompt_prefix import PromptPrefix
from easycoding.runtime import EasyCoding
from easycoding.workspace import WorkspaceContext


def test_file_summary_expires_after_change(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old", encoding="utf-8")
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    agent.memory.set_file_summary("a.txt", "old summary")
    assert "old summary" in agent.memory.render()
    path.write_text("new", encoding="utf-8")
    assert "old summary" not in agent.memory.render()


def test_current_request_is_not_clipped(tmp_path):
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    request = "z" * 13000
    prompt, metadata = agent.context_manager.build(request)
    assert request in prompt
    assert metadata["current_request_chars"] == 13000


def test_budget_reduction_order_and_section_floors(tmp_path):
    agent = EasyCoding(ScriptedModelClient(["<final>x</final>"]), WorkspaceContext.build(tmp_path))
    agent.prefix_state = PromptPrefix("p" * 3600, "hash", "workspace", "tools")
    agent.memory.render = lambda: "m" * 2000
    agent.memory.retrieve = lambda query, limit=3: [{"text": "r" * 700}] * 3
    agent.session["history"] = [
        {"role": "tool", "content": "h" * 1200, "tool_name": "search"}
        for _ in range(12)
    ]
    agent.context_manager.total_budget = 7000
    request = "q" * 3000

    prompt, metadata = agent.context_manager.build(request)

    assert request in prompt
    assert [item["section"] for item in metadata["budget_reductions"]] == [
        "relevant_memory", "history", "memory", "prefix"
    ]
    floors = {"prefix": 900, "memory": 400, "relevant_memory": 300, "history": 1300}
    assert all(metadata["rendered_chars"][name] >= floor for name, floor in floors.items())
    assert metadata["section_budgets"] == {
        "prefix": 3600, "memory": 1600, "relevant_memory": 1200, "history": 5200
    }
    assert metadata["reduction_order"] == [
        "relevant_memory", "history", "memory", "prefix"
    ]
    assert metadata["current_request_retention_rate"] == 1.0
    assert metadata["budget_overflow_chars"] >= 0
    assert 0.0 <= metadata["total_reduction_rate"] <= 1.0
    assert metadata["history_event_count"] == 12
    assert metadata["relevant_memory_count"] == 3
