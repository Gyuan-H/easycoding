"""Serializable state for one call to ``EasyCoding.ask``."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    run_id: str
    task_id: str
    user_request: str
    status: str = "running"
    attempts: int = 0
    tool_steps: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    checkpoint_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create(cls, user_request):
        timestamp = now()
        return cls(
            run_id="run_" + uuid.uuid4().hex[:12],
            task_id="task_" + uuid.uuid4().hex[:12],
            user_request=str(user_request),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def record_attempt(self):
        self.attempts += 1
        self.updated_at = now()

    def record_tool(self, name):
        self.tool_steps += 1
        self.last_tool = str(name)
        self.updated_at = now()

    def finish(self, status, stop_reason, final_answer=""):
        self.status = status
        self.stop_reason = stop_reason
        self.final_answer = str(final_answer)
        self.updated_at = now()

    def to_dict(self):
        return asdict(self)

