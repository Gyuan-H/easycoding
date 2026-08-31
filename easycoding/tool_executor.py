"""Guarded execution gateway for all model-requested tools."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .security import resolve_in_workspace, shell_env
from .shell_security import ShellAssessment, assess_shell_command
from .tool_types import ToolExecutionError, ToolRunOutput
from .workspace import clip


class ToolArgumentError(ValueError):
    """A stable, model-correctable tool argument validation failure."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_tool_arguments(schema, args):
    """Validate and normalize arguments against the supported object schema subset."""
    if not schema:
        return dict(args)
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for name in required:
        if name not in args:
            raise ToolArgumentError("missing_argument", f"missing required argument: {name}")
    if schema.get("additionalProperties", True) is False:
        unexpected = sorted(set(args) - set(properties))
        if unexpected:
            raise ToolArgumentError(
                "unexpected_argument", f"unexpected argument: {unexpected[0]}"
            )
    normalized = dict(args)
    for name, rule in properties.items():
        if name not in normalized:
            if "default" in rule:
                normalized[name] = rule["default"]
            continue
        value = normalized[name]
        expected = rule.get("type")
        valid = {
            "string": isinstance(value, str),
            "integer": type(value) is int,
            "number": type(value) in {int, float},
            "boolean": type(value) is bool,
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
        }.get(expected, True)
        if not valid:
            raise ToolArgumentError(
                "invalid_argument_type", f"argument {name} must be {expected}"
            )
        if expected in {"integer", "number"}:
            if "minimum" in rule and value < rule["minimum"]:
                raise ToolArgumentError(
                    "argument_out_of_range",
                    f"argument {name} must be at least {rule['minimum']}",
                )
            if "maximum" in rule and value > rule["maximum"]:
                raise ToolArgumentError(
                    "argument_out_of_range",
                    f"argument {name} must be at most {rule['maximum']}",
                )
        if expected == "string" and len(value) < rule.get("minLength", 0):
            raise ToolArgumentError(
                "argument_out_of_range", f"argument {name} must not be empty"
            )
        if expected == "string" and "maxLength" in rule and len(value) > rule["maxLength"]:
            raise ToolArgumentError(
                "argument_out_of_range",
                f"argument {name} must be at most {rule['maxLength']} characters",
            )
        if expected == "array":
            if len(value) < rule.get("minItems", 0):
                raise ToolArgumentError(
                    "argument_out_of_range", f"argument {name} has too few items"
                )
            if "maxItems" in rule and len(value) > rule["maxItems"]:
                raise ToolArgumentError(
                    "argument_out_of_range", f"argument {name} has too many items"
                )
            item_rule = rule.get("items", {})
            item_type = item_rule.get("type")
            for index, item in enumerate(value):
                if item_type == "string" and not isinstance(item, str):
                    raise ToolArgumentError(
                        "invalid_argument_type",
                        f"argument {name}[{index}] must be string",
                    )
                if item_type == "string" and len(item) < item_rule.get("minLength", 0):
                    raise ToolArgumentError(
                        "argument_out_of_range",
                        f"argument {name}[{index}] must not be empty",
                    )
    for rule in schema.get("x-rules", []):
        if rule.get("kind") != "ordered_range":
            continue
        lower_name = rule["lower"]
        upper_name = rule["upper"]
        if upper_name not in normalized:
            continue
        lower = normalized[lower_name]
        upper = normalized[upper_name]
        if upper < lower or upper - lower > rule.get("maximumSpan", upper - lower):
            raise ToolArgumentError(
                "argument_out_of_range",
                f"arguments {lower_name} and {upper_name} form an invalid range",
            )
    return normalized


@dataclass
class ToolResult:
    status: str
    text: str
    tool_error_code: str = ""
    affected_paths: tuple = ()
    workspace_changed: bool = False
    risk_level: str = ""
    approval_required: bool = False
    approval_granted: bool = False
    exit_code: object = None
    timed_out: bool = False
    output_truncated: bool = False

    def to_dict(self):
        payload = asdict(self)
        payload["affected_paths"] = list(self.affected_paths)
        return payload


class ToolContext:
    def __init__(self, root, delegate_callback=None):
        self.root = Path(root).resolve()
        self.delegate_callback = delegate_callback

    def shell_env(self):
        return shell_env(self.root)

    def delegate(self, args):
        if self.delegate_callback is None:
            raise ToolExecutionError("delegate_unavailable", "delegate runtime is unavailable")
        return self.delegate_callback(dict(args))


def _snapshot(root):
    root = Path(root)
    result = {}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or ".easycoding" in path.parts:
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        result[path.relative_to(root).as_posix()] = digest
    return result


class ToolExecutor:
    def __init__(
        self, root, tools, approval_policy="ask", allowed_tools=None,
        read_only=False, approval_callback=None, delegate_callback=None,
        path_scopes=None,
    ):
        self.context = ToolContext(root, delegate_callback=delegate_callback)
        self.tools = dict(tools)
        self.approval_policy = approval_policy
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.read_only = bool(read_only)
        self.approval_callback = approval_callback
        self.path_scopes = tuple(
            Path(path).resolve() for path in (path_scopes or ())
        )
        self.recent_calls = []

    def execute(self, name, args):
        if name not in self.tools:
            return ToolResult("rejected", f"unknown tool: {name}", "unknown_tool")
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return ToolResult("rejected", f"tool not allowed: {name}", "tool_not_allowed")
        spec = self.tools[name]
        try:
            args = validate_tool_arguments(spec.schema, args)
        except ToolArgumentError as exc:
            return ToolResult("rejected", f"error: {exc}", exc.code)
        if name in {"list_files", "read_file", "search"} and self.path_scopes:
            try:
                candidate = resolve_in_workspace(
                    self.context.root, args.get("path", ".")
                )
            except ValueError as exc:
                return ToolResult("rejected", f"error: {exc}", "workspace_escape")
            if not any(
                candidate == scope or scope in candidate.parents
                for scope in self.path_scopes
            ):
                return ToolResult(
                    "rejected", f"path outside delegated scope: {args.get('path', '.')}",
                    "path_not_allowed",
                )
        assessment = ShellAssessment("tool")
        if name == "run_shell":
            assessment = assess_shell_command(args["command"], self.context.root)
            if not assessment.allowed:
                return ToolResult(
                    "rejected", f"error: {assessment.reason}", assessment.error_code,
                    risk_level=assessment.risk_level,
                    approval_required=assessment.requires_explicit_approval,
                )
        signature = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if len(self.recent_calls) >= 2 and all(item == signature for item in self.recent_calls[-2:]):
            return ToolResult("rejected", f"repeated identical tool call: {name}", "repeated_call")
        self.recent_calls.append(signature)
        self.recent_calls = self.recent_calls[-3:]
        if spec.risky and self.read_only:
            return ToolResult(
                "rejected", f"read-only mode rejects {name}", "read_only",
                risk_level=assessment.risk_level,
            )
        approval_required = bool(spec.risky)
        approved = not approval_required or self._approved(
            name, args, force_explicit=assessment.requires_explicit_approval
        )
        if not approved:
            return ToolResult(
                "rejected", f"approval denied for {name}", "approval_denied",
                risk_level=assessment.risk_level,
                approval_required=approval_required,
                approval_granted=False,
            )
        before = _snapshot(self.context.root)
        metadata = {}
        try:
            output = spec.run(self.context, dict(args))
            if isinstance(output, ToolRunOutput):
                text = output.text
                error_code = output.error_code
                metadata = dict(output.metadata)
            else:
                text = clip(output)
                error_code = ""
        except ToolExecutionError as exc:
            text = f"error: {exc}"
            error_code = exc.code
            metadata = dict(exc.metadata)
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
            text = f"error: {exc}"
            error_code = "invalid_arguments" if isinstance(exc, (KeyError, TypeError, ValueError)) else "tool_error"
        after = _snapshot(self.context.root)
        affected = tuple(sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path)))
        changed = bool(affected)
        if (error_code and changed) or error_code == "output_limit_exceeded":
            status = "partial_success"
        elif error_code:
            status = "error"
        else:
            status = "success"
        return ToolResult(
            status, text, error_code, affected, changed,
            risk_level=assessment.risk_level,
            approval_required=approval_required,
            approval_granted=approved,
            exit_code=metadata.get("exit_code"),
            timed_out=bool(metadata.get("timed_out", False)),
            output_truncated=bool(metadata.get("output_truncated", False)),
        )

    def _approved(self, name, args, force_explicit=False):
        if self.approval_policy == "auto" and not force_explicit:
            return True
        if self.approval_policy == "never":
            return False
        return bool(self.approval_callback and self.approval_callback(name, args))
