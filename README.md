# EasyCoding

EasyCoding is a teaching-sized local Coding Agent Harness reconstructed from the Pico architecture notes.

It demonstrates:

- a deterministic model/tool loop;
- repository-aware prompt construction;
- guarded file and shell tools;
- strict model-response protocol and executable tool argument schemas;
- shell risk classification, explicit approval escalation, and resource reporting;
- redacted provider recording, offline replay, and optional live evaluation;
- prompt-level tool allowlists, bounded finalization, and recovery checkpoints;
- section-based context budgets;
- working memory, file freshness, and explicit durable memory;
- sessions, run traces, reports, checkpoints, and conservative resume checks;
- fixture-based benchmark evaluation.

## Quick start

```bash
python -m easycoding --help
python -m easycoding --provider scripted --cwd . "inspect the workspace"
python -m pytest -q
python -m easycoding.evaluator benchmarks/tasks.json --repo-root .
```

The scripted provider is deterministic and requires no API key. For a real OpenAI-compatible `/responses` endpoint:

```bash
set EASYCODING_OPENAI_API_KEY=...
python -m easycoding --provider openai --model gpt-5 --cwd . "inspect README.md"
```

Prompt-cache parameters are opt-in because not every OpenAI-compatible server supports them. Enable them with `--prompt-cache` or `EASYCODING_OPENAI_PROMPT_CACHE=true`.

For a local Ollama server, first make sure the requested model is already available, then run:

```bash
python -m easycoding --provider ollama --model YOUR_MODEL --base-url http://localhost:11434 --cwd . "inspect README.md"
```

Ollama configuration can also be supplied through `EASYCODING_PROVIDER=ollama`, `EASYCODING_OLLAMA_MODEL`, and `EASYCODING_OLLAMA_BASE_URL`. Use `--timeout` to control the provider request timeout for both HTTP providers.

Check provider readiness without starting an agent task:

```bash
python -m easycoding --provider ollama --model YOUR_MODEL --check-provider
python -m easycoding --provider openai --model YOUR_MODEL --check-provider
```

The Ollama check calls `/api/tags` and verifies that the configured model is installed. The OpenAI-compatible check validates the endpoint and credentials without sending a model request, so it cannot incur generation usage. A failed check exits with code 1 and prints a concrete suggestion, such as `ollama pull YOUR_MODEL`.

Windows CMD uses `^` for multiline commands; PowerShell uses a backtick. A single-line command works in both shells and avoids mixing their continuation syntax.

## Security boundary

EasyCoding validates paths, resolves symlinks, constrains tool registration, filters child-process environment variables, applies approval policies, and records workspace changes. Shell commands are classified as `read_only`, `mutating`, or `high_risk`; obvious workspace escapes and catastrophic commands are rejected, and high-risk commands require an explicit callback decision even with `--approval auto`. This is a command-level heuristic, not an OS-level sandbox. Only run it against repositories and commands you trust.

## Deterministic regression baseline

`benchmarks/tasks.json` contains ten scripted scenarios covering the normal read/write path, tool allowlists, approval denial, path escape rejection, repeated calls, step and retry limits, and partial success after a workspace change. Each task declares its expected stop reason and tool-result sequence; the evaluator compares those contracts against `trace.jsonl` instead of treating every non-happy-path run as a failure.

Run the complete baseline with:

```bash
python -m pytest -q
python -m easycoding.evaluator benchmarks/tasks.json --repo-root .
```

The current baseline is 122 passing tests and one platform-specific symlink test skipped on Windows, including durable-memory and controlled-ablation coverage. Provider HTTP tests are fully mocked and do not require network access, API keys, or a running Ollama server.

The evaluator output includes a top-level `metrics` object with pass rate, budget/artifact/verifier rates, average and total attempts/tool steps, fine-grained failure categories, tool success/rejection/error rates, stop reasons, protocol error codes, Trace event counts, retry counts, and Trace integrity rate. Trace integrity checks cover schema fields, start/end markers, event ordering, malformed JSON lines, and consistency with report attempt/tool counters.

Model responses must be exactly one complete `<tool>...</tool>` or `<final>...</final>` block with no surrounding text or additional blocks. Tool calls are validated before approval and execution. Stable validation codes include `missing_argument`, `invalid_argument_type`, `argument_out_of_range`, `unexpected_argument`, `invalid_protocol`, and `unknown_tool`.

`allowed_tools` is part of the execution contract: disallowed tools are omitted from the stable prompt prefix and are still rejected with `tool_not_allowed` if a model requests one. Internal state and cache directories such as `.easycoding`, `.git`, `.venv`, `__pycache__`, and `*.egg-info` are hidden from listing and search tools.

After the normal tool budget is exhausted, EasyCoding reserves one model-only finalization request. It asks for a non-empty `<final>` answer and will not execute another tool. Checkpoints are created after every tool result, when context is reduced, when resume freshness or runtime identity is invalid, and when a run completes or stops. Runtime identity now covers the model client, budgets, allowed tools, Shell environment contract, workspace fingerprint, and tool signature.

Run the dedicated Shell guardrail benchmark with:

```bash
python -m pytest tests/test_shell_security.py -q
python -m easycoding.evaluator benchmarks/shell_security/tasks.json --repo-root .
```

Shell Trace events include risk level, approval requirement and decision, exit code, timeout state, and output truncation state. Stable Shell codes include `unsafe_command`, `command_not_allowed`, `command_timeout`, `output_limit_exceeded`, and `workspace_escape`.

## Provider recording and benchmark

The Provider benchmark runs from deterministic recordings by default and never opens the network:

```bash
python -m easycoding.provider_benchmark benchmarks/provider/tasks.json --repo-root .
```

Its six scenarios cover file reading, search, writing, patching, invalid tool-argument recovery, and invalid protocol recovery. Metrics include protocol compliance, tool-call validity, retry recovery, provider errors, average latency, and input/output/total token usage. The checked-in recordings are deterministic seed fixtures for offline regression; they are not measurements of a named live model.

Network and potentially billable requests require the explicit `--live` flag:

```bash
python -m easycoding.provider_benchmark benchmarks/provider/tasks.json --repo-root . --live --provider ollama --model YOUR_MODEL
python -m easycoding.provider_benchmark benchmarks/provider/tasks.json --repo-root . --live --provider openai --model YOUR_MODEL
```

Unavailable providers or missing credentials produce a structured `skipped` result. Successful live calls write redacted recordings under `.easycoding/provider-recordings/`, one file per scenario. Recordings retain prompts and responses after configured secret replacement, so review them before sharing or committing because repository content itself is not automatically treated as a secret.

## Real long-context benchmark

`benchmarks/long_context/tasks.json` defines twelve distinct context profiles built from the repository's real Python, test, and documentation files. The profiles cover a long current request, long and repeated read history, large tool outputs, multiple files, large project documents, saturated working and relevant memory, checkpoint injection, mixed Chinese/Unicode/code, and combined over-budget pressure.

Run it with:

```bash
python -m easycoding.context_benchmark benchmarks/long_context/tasks.json --repo-root .
```

Each profile runs twice to check deterministic context metadata. The result reports request and evidence retention, per-section reduction rates, soft-budget overflow, floor/order violations, repeated reads, Trace integrity, average prompt size, and aggregate context pass rate. The benchmark uses real repository text rather than repeated-character filler.

## Durable memory

Durable facts are workspace-scoped and survive both new sessions and `/reset`. EasyCoding writes them only when the user explicitly asks to remember or save a labeled fact. Supported labels are `Project convention` / `项目约定`, `Decision` / `决策`, `Dependency` / `依赖`, and `Preference` / `偏好`.

```text
Remember this. Project convention: Python version: 3.12
请记住。偏好：输出语言：中文
```

Facts are stored under `.easycoding/memory/`, deduplicated, and replaced when a new fact has the same labeled subject. API keys, access tokens, passwords, transient failures, raw Shell output, temporary paths, oversized values, and unlabeled requests are rejected. At most three relevant facts enter a prompt; unrelated facts remain outside the model context. Run reports contain `durable_memory_changes`, while prompt metadata and Trace record `durable_memory_hits`.

```bash
python -m pytest tests/test_durable_memory.py -q
```

## Memory and resume ablation

The two context features can be disabled independently without disabling session or run artifacts:

```bash
python -m easycoding --no-durable-memory --cwd . "inspect the workspace"
python -m easycoding --no-resume-context --cwd . "continue the task"
```

The controlled ablation benchmark runs every task under `full`, `no_memory`, `no_resume`, and `neither`. A deterministic evidence-aware fixture model succeeds only when all task evidence is genuinely present in the assembled prompt. Results include pass and evidence-retention rates, attempts, tool steps, prompt characters, durable-memory hits, resume-context hits, Trace integrity, deltas from the full configuration, and benchmark/model/Python provenance.

```bash
python -m easycoding.ablation_benchmark benchmarks/ablation/tasks.json --repo-root .
```
