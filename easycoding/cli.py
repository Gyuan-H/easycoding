"""Command-line entry point and runtime assembly."""

import argparse
import os
import sys

from .config import env, env_bool, load_project_env
from .providers import (
    OllamaModelClient, OpenAICompatibleModelClient, ProviderError,
    ScriptedModelClient,
)
from .runtime import EasyCoding
from .session_store import SessionStore
from .workspace import WorkspaceContext


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="EasyCoding - a teaching-sized local coding agent harness.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot task.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--provider", choices=("scripted", "openai", "ollama"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=120, help="Provider request timeout in seconds.")
    parser.add_argument(
        "--prompt-cache", action=argparse.BooleanOptionalAction, default=None,
        help="Explicitly enable compatible OpenAI prompt-cache parameters.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask")
    parser.add_argument("--resume", default=None, help="Session id or 'latest'.")
    parser.add_argument(
        "--durable-memory", action=argparse.BooleanOptionalAction, default=True,
        help="Inject and promote workspace-scoped durable memory.",
    )
    parser.add_argument(
        "--resume-context", action=argparse.BooleanOptionalAction, default=True,
        help="Evaluate and inject the current checkpoint into prompts.",
    )
    parser.add_argument(
        "--check-provider", action="store_true",
        help="Check provider configuration without running an agent task.",
    )
    return parser


def _approval_callback(name, args):
    if not sys.stdin.isatty():
        return False
    answer = input(f"Approve risky tool {name} with {args}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _build_model(args):
    provider = (args.provider or env("EASYCODING_PROVIDER", default="scripted")).strip().lower()
    if provider == "scripted":
        return ScriptedModelClient([
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "<final>EasyCoding inspected the workspace with list_files.</final>",
        ])
    if provider == "ollama":
        model = args.model or env("EASYCODING_OLLAMA_MODEL", default="qwen2.5-coder:7b")
        base_url = args.base_url or env("EASYCODING_OLLAMA_BASE_URL", default="http://localhost:11434")
        return OllamaModelClient(model, base_url, timeout=args.timeout)
    if provider == "openai":
        model = args.model or env("EASYCODING_OPENAI_MODEL", ("OPENAI_MODEL",), "gpt-5")
        base_url = args.base_url or env("EASYCODING_OPENAI_API_BASE", ("OPENAI_API_BASE",), "https://api.openai.com/v1")
        api_key = args.api_key or env("EASYCODING_OPENAI_API_KEY", ("OPENAI_API_KEY",))
        supports_cache = args.prompt_cache
        if supports_cache is None:
            supports_cache = env_bool("EASYCODING_OPENAI_PROMPT_CACHE", default=False)
        return OpenAICompatibleModelClient(
            model, base_url, api_key, timeout=args.timeout,
            supports_prompt_cache=supports_cache,
        )
    raise ValueError(f"unsupported provider: {provider}")


def build_agent(args):
    workspace = WorkspaceContext.build(args.cwd)
    load_project_env(workspace.repo_root)
    store = SessionStore(os.path.join(workspace.repo_root, ".easycoding", "sessions"))
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    kwargs = dict(
        model_client=_build_model(args),
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        approval_callback=_approval_callback,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=("EASYCODING_OPENAI_API_KEY", "OPENAI_API_KEY"),
        durable_memory_enabled=args.durable_memory,
        resume_enabled=args.resume_context,
    )
    return EasyCoding.from_session(session_id=session_id, **kwargs) if session_id else EasyCoding(**kwargs)


def _render_provider_status(status):
    def value(item):
        if item is True:
            return "yes"
        if item is False:
            return "no"
        return "not checked"

    lines = [
        f"Provider: {status.provider}",
        f"Endpoint: {status.endpoint}",
        f"Model: {status.model}",
        f"Server reachable: {value(status.reachable)}",
        f"Model available: {value(status.model_available)}",
        f"Status: {status.status}",
    ]
    if status.reason:
        lines.append(f"Reason: {status.reason}")
    if status.suggestion:
        lines.append(f"Suggestion: {status.suggestion}")
    return "\n".join(lines)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        if args.check_provider:
            workspace = WorkspaceContext.build(args.cwd)
            load_project_env(workspace.repo_root)
            status = _build_model(args).check()
            stream = sys.stdout if status.ok else sys.stderr
            print(_render_provider_status(status), file=stream)
            return 0 if status.ok else 1
        agent = build_agent(args)
    except (ProviderError, ValueError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.prompt:
        try:
            print(agent.ask(" ".join(args.prompt)))
            return 0
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    print(f"easycoding | workspace={agent.workspace.cwd} | session={agent.session['id']}")
    while True:
        try:
            user_input = input("easycoding> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print("/help /memory /session /reset /exit")
        elif user_input == "/memory":
            print(agent.memory_text())
        elif user_input == "/session":
            print(agent.session_path)
        elif user_input == "/reset":
            agent.reset()
            print("session reset")
        else:
            print(agent.ask(user_input))
