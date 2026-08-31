from pathlib import Path

from easycoding.workspace import WorkspaceContext


def test_workspace_non_git_and_doc_fingerprint(tmp_path):
    (tmp_path / "README.md").write_text("alpha", encoding="utf-8")
    first = WorkspaceContext.build(tmp_path)
    assert first.repo_root == str(tmp_path.resolve())
    assert "README.md" in first.project_docs
    (tmp_path / "README.md").write_text("beta", encoding="utf-8")
    second = WorkspaceContext.build(tmp_path)
    assert first.fingerprint() != second.fingerprint()

