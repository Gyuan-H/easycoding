import json

from easycoding.providers import ScriptedModelClient
from easycoding.runtime import EasyCoding
from easycoding.workspace import WorkspaceContext


def make_agent(root, output="<final>done</final>"):
    return EasyCoding(ScriptedModelClient([output]), WorkspaceContext.build(root))


def report_for(agent):
    path = agent.root / ".easycoding" / "runs" / agent.current_task_state.run_id / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def trace_for(agent):
    path = agent.root / ".easycoding" / "runs" / agent.current_task_state.run_id / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_explicit_english_fact_survives_a_new_session(tmp_path):
    first = make_agent(tmp_path)
    first.ask("Remember this. Project convention: Python version: 3.12")

    second_model = ScriptedModelClient(["<final>3.12</final>"])
    second = EasyCoding(second_model, WorkspaceContext.build(tmp_path))
    assert second.ask("Which Python version does this project use?") == "3.12"
    assert "Python version: 3.12" in second_model.prompts[0]
    assert report_for(second)["prompt_metadata"]["durable_memory_hits"] == 1
    assert "durable_memory_hits" in [item["event"] for item in trace_for(second)]


def test_chinese_label_is_supported_and_visible_in_memory_command(tmp_path):
    agent = make_agent(tmp_path)
    agent.ask("请记住。项目约定：测试框架：pytest")
    assert "测试框架：pytest" in agent.memory_text()
    assert "Project convention / 项目约定: 1" in agent.memory_text()


def test_duplicate_fact_is_not_written_twice(tmp_path):
    first = make_agent(tmp_path)
    first.ask("Remember. Dependency: HTTP client: urllib")
    second = make_agent(tmp_path)
    second.ask("Remember. Dependency: HTTP client: urllib")
    notes = second.memory.durable.all_notes()
    assert len(notes) == 1
    changes = report_for(second)["durable_memory_changes"]
    assert len(changes["deduplicated"]) == 1
    assert not changes["promoted"]


def test_same_subject_supersedes_old_fact(tmp_path):
    make_agent(tmp_path).ask("Remember. Decision: Python version: 3.11")
    second = make_agent(tmp_path)
    second.ask("Remember. Decision: Python version: 3.12")
    notes = second.memory.durable.all_notes()
    assert [item["text"] for item in notes] == ["Python version: 3.12"]
    changes = report_for(second)["durable_memory_changes"]
    assert len(changes["promoted"]) == 1
    assert len(changes["superseded"]) == 1
    assert "durable_superseded" in [item["event"] for item in trace_for(second)]


def test_secret_is_rejected_without_leaking_into_artifacts(tmp_path):
    secret = "sk-1234567890abcdef"
    agent = make_agent(tmp_path)
    agent.ask(f"Remember. Preference: api_key: {secret}")
    assert not agent.memory.durable.all_notes()
    report_text = json.dumps(report_for(agent), ensure_ascii=False)
    trace_text = json.dumps(trace_for(agent), ensure_ascii=False)
    assert secret not in report_text
    assert secret not in trace_text
    assert report_for(agent)["durable_memory_changes"]["rejected"] == [
        {"topic": "user-preferences", "reason": "sensitive_information"}
    ]


def test_no_explicit_intent_does_not_create_durable_store(tmp_path):
    agent = make_agent(tmp_path)
    agent.ask("Project convention: Python version: 3.12")
    assert not (tmp_path / ".easycoding" / "memory").exists()
    assert not report_for(agent)["durable_memory_changes"]["intent_detected"]


def test_explicit_intent_requires_a_supported_label(tmp_path):
    agent = make_agent(tmp_path)
    agent.ask("Remember that we discussed testing today.")
    assert not agent.memory.durable.all_notes()
    assert report_for(agent)["durable_memory_changes"]["rejected"] == [
        {"reason": "no_labeled_fact"}
    ]


def test_only_relevant_durable_memory_enters_prompt(tmp_path):
    make_agent(tmp_path).ask(
        "Remember these facts.\n"
        "Project convention: Python version: 3.12\n"
        "Dependency: Database engine: SQLite\n"
        "Preference: Documentation language: Chinese"
    )
    model = ScriptedModelClient(["<final>done</final>"])
    agent = EasyCoding(model, WorkspaceContext.build(tmp_path))
    agent.ask("Which database engine is used?")
    prompt = model.prompts[0]
    assert "Database engine: SQLite" in prompt
    assert "Python version: 3.12" not in prompt
    assert "Documentation language: Chinese" not in prompt
    assert report_for(agent)["prompt_metadata"]["durable_memory_hits"] == 1


def test_reset_clears_working_but_not_durable_memory(tmp_path):
    agent = make_agent(tmp_path)
    agent.ask("Remember. Preference: Output language: Chinese")
    agent.memory.append_note("temporary working note")
    agent.reset()
    assert not agent.memory.state["episodic_notes"]
    assert [item["text"] for item in agent.memory.durable.all_notes()] == [
        "Output language: Chinese"
    ]
