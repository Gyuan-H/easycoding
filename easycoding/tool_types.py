"""Shared result and error types used by tool implementations."""

from dataclasses import dataclass, field


class ToolExecutionError(RuntimeError):
    def __init__(self, code, message, **metadata):
        super().__init__(message)
        self.code = str(code)
        self.metadata = dict(metadata)


@dataclass(frozen=True)
class ToolRunOutput:
    text: str
    error_code: str = ""
    metadata: dict = field(default_factory=dict)

