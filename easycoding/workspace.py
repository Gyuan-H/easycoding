"""Build a small, stable description of the active repository."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess


DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
MAX_TOOL_OUTPUT = 4000


def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _git(cwd, *args, limit=2000):
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return clip(completed.stdout.strip(), limit) if completed.returncode == 0 else ""


@dataclass
class WorkspaceContext:
    cwd: str
    repo_root: str
    branch: str = ""
    default_branch: str = ""
    status: str = ""
    recent_commits: str = ""
    project_docs: dict = field(default_factory=dict)

    @classmethod
    def build(cls, cwd):
        current = Path(cwd).expanduser().resolve()
        if not current.exists() or not current.is_dir():
            raise ValueError(f"workspace directory does not exist: {current}")
        root_text = _git(current, "rev-parse", "--show-toplevel")
        root = Path(root_text).resolve() if root_text else current
        docs = {}
        for base in dict.fromkeys((root, current)):
            for name in DOC_NAMES:
                path = base / name
                if not path.is_file():
                    continue
                try:
                    key = path.relative_to(root).as_posix()
                except ValueError:
                    key = path.name
                if key not in docs:
                    docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)
        default_ref = _git(root, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        return cls(
            cwd=str(current),
            repo_root=str(root),
            branch=_git(root, "branch", "--show-current"),
            default_branch=default_ref.removeprefix("origin/"),
            status=_git(root, "status", "--short", limit=1500),
            recent_commits=_git(root, "log", "--oneline", "-5", limit=1500),
            project_docs=docs,
        )

    def fingerprint(self):
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self):
        return {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": self.recent_commits,
            "project_docs": self.project_docs,
        }

    def text(self):
        lines = [
            "Workspace:",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch or '-'}",
            f"- default_branch: {self.default_branch or '-'}",
            f"- status: {self.status or 'clean or unavailable'}",
            f"- recent_commits: {self.recent_commits or '-'}",
        ]
        for path, content in sorted(self.project_docs.items()):
            lines.extend((f"\nProject document: {path}", content))
        return "\n".join(lines)

