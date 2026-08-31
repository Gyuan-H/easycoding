"""EasyCoding: a teaching-sized local coding agent harness."""

__version__ = "0.1.0"

from .provider_recording import RecordingModelClient, ReplayModelClient
from .providers import ScriptedModelClient
from .runtime import EasyCoding
from .workspace import WorkspaceContext

__all__ = [
    "EasyCoding", "RecordingModelClient", "ReplayModelClient",
    "ScriptedModelClient", "WorkspaceContext", "__version__",
]
