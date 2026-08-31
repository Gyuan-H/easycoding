"""Stable prompt prefix construction."""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class PromptPrefix:
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str


def tool_signature(tools):
    payload = {
        name: {
            "schema": spec.schema,
            "risky": spec.risky,
            "description": spec.description,
        }
        for name, spec in sorted(tools.items())
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt_prefix(workspace, tools):
    rendered_tools = []
    for name, spec in sorted(tools.items()):
        rendered_tools.append(
            f"- {name}: {spec.description} schema={json.dumps(spec.schema, sort_keys=True)} "
            f"risky={str(spec.risky).lower()}"
        )
    text = "\n".join(
        [
            "You are EasyCoding, a local coding agent working inside one repository.",
            "Rules:",
            "- Inspect evidence before making claims.",
            "- Use only registered tools and stay inside the workspace.",
            "- Return exactly one tool call or one final answer per turn.",
            '- Tool format: <tool>{"name":"tool_name","args":{...}}</tool>',
            "- Final format: <final>answer</final>",
            "Tools:",
            *rendered_tools,
            "",
            workspace.text(),
        ]
    )
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=tool_signature(tools),
    )

