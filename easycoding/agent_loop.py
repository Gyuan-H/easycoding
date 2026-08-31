"""The model/tool/state control loop."""

from .task_state import TaskState, now


def run_agent_loop(agent, user_message):
    task_state = TaskState.create(user_message)
    agent.current_task_state = task_state
    agent.run_store.start_run(task_state)
    agent.memory.set_task_summary(user_message)
    agent.record({"role": "user", "content": user_message})
    agent.emit_trace(task_state.run_id, "run_started", {
        "user_request": str(user_message)[:300],
        "agent_role": agent.agent_role,
        "parent_run_id": agent.parent_run_id,
        "delegation_id": agent.delegation_id,
        "delegation_depth": agent.delegation_depth,
    })
    max_attempts = agent.max_steps * 2 + 3
    last_prompt_metadata = {}

    while task_state.tool_steps < agent.max_steps and task_state.attempts < max_attempts:
        task_state.record_attempt()
        agent.run_store.write_task_state(task_state)
        agent.refresh_runtime_state()
        prompt, prompt_metadata = agent.context_manager.build(user_message)
        last_prompt_metadata = prompt_metadata
        if prompt_metadata.get("durable_memory_hits"):
            agent.emit_trace(task_state.run_id, "durable_memory_hits", {
                "count": prompt_metadata["durable_memory_hits"],
                "topics": prompt_metadata.get("durable_memory_topics", []),
            })
        if prompt_metadata.get("resume_context_hits"):
            agent.emit_trace(task_state.run_id, "resume_context_hits", {
                "count": prompt_metadata["resume_context_hits"],
                "status": prompt_metadata.get("resume_status", "no-checkpoint"),
            })
        agent.emit_trace(task_state.run_id, "prompt_built", prompt_metadata)
        resume_status = prompt_metadata.get("resume_status")
        if resume_status in {"partial-stale", "workspace-mismatch", "schema-mismatch"}:
            agent.emit_trace(
                task_state.run_id, "resume_state_detected",
                {
                    "status": resume_status,
                    "stale_paths": prompt_metadata.get("stale_paths", []),
                    "mismatch_fields": prompt_metadata.get(
                        "runtime_identity_mismatch_fields", []
                    ),
                },
            )
            agent.create_checkpoint(
                task_state,
                "freshness_mismatch" if resume_status == "partial-stale" else resume_status,
                blocker=resume_status,
                next_step="re-anchor on the current workspace before continuing",
            )
        if prompt_metadata.get("budget_reductions"):
            agent.create_checkpoint(
                task_state, "context_reduction",
                next_step="continue with the reduced context and retained current request",
            )
        agent.emit_trace(
            task_state.run_id, "model_requested",
            {"attempt": task_state.attempts, "purpose": "action"},
        )
        try:
            cache_kwargs = {}
            if getattr(agent.model_client, "supports_prompt_cache", False):
                cache_kwargs = {
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                    "prompt_cache_retention": "in_memory",
                }
            raw = agent.model_client.complete(prompt, agent.max_new_tokens, **cache_kwargs)
        except RuntimeError as exc:
            task_state.finish("failed", "model_error", str(exc))
            agent.create_checkpoint(task_state, "model_error", blocker="model_error", next_step="retry the model request")
            return agent.finish_run(task_state, last_prompt_metadata)

        kind, payload = agent.parse(raw)
        parse_details = {"kind": kind}
        if kind == "retry" and isinstance(payload, dict):
            parse_details["error_code"] = payload.get("code", "invalid_protocol")
        parse_details["purpose"] = "action"
        agent.emit_trace(task_state.run_id, "model_parsed", parse_details)
        if kind == "tool":
            name = payload["name"]
            args = payload["args"]
            task_state.record_tool(name)
            result = agent.tool_executor.execute(name, args)
            agent.update_memory_after_tool(name, args, result)
            agent.record(
                {
                    "role": "tool",
                    "tool_name": name,
                    "tool_args": args,
                    "content": result.text,
                    "tool_result": result.to_dict(),
                }
            )
            agent.emit_trace(
                task_state.run_id,
                "tool_executed",
                {"name": name, "args": args, **result.to_dict()},
            )
            agent.run_store.write_task_state(task_state)
            if result.workspace_changed:
                agent.refresh_runtime_state()
            trigger = "tool_partial_success" if result.status == "partial_success" else "tool_executed"
            blocker = result.tool_error_code if result.status != "success" else ""
            next_step = (
                "inspect workspace changes before retry"
                if result.status == "partial_success"
                else f"decide the next action after {name}"
            )
            agent.create_checkpoint(
                task_state, trigger, blocker=blocker, next_step=next_step,
            )
            continue
        if kind == "final":
            task_state.finish("completed", "final_answer_returned", payload)
            agent.record({"role": "assistant", "content": payload})
            agent.promote_durable_memory(user_message, task_state.run_id)
            agent.create_checkpoint(task_state, "run_finished", next_step="task completed")
            return agent.finish_run(task_state, last_prompt_metadata)

        retry_code = "invalid_protocol"
        retry_message = str(payload)
        if isinstance(payload, dict):
            retry_code = str(payload.get("code", retry_code))
            retry_message = str(payload.get("message", "invalid model response"))
        agent.record({"role": "system", "content": retry_message})
        agent.emit_trace(
            task_state.run_id, "model_retry",
            {"reason": retry_message, "error_code": retry_code},
        )

    if task_state.tool_steps >= agent.max_steps:
        task_state.record_attempt()
        agent.run_store.write_task_state(task_state)
        agent.refresh_runtime_state()
        prompt, prompt_metadata = agent.context_manager.build(user_message)
        prompt += (
            "\n\nRuntime notice: the tool budget is exhausted. Do not call another tool. "
            "Use the evidence already present in history and return exactly one non-empty "
            "<final>...</final> answer."
        )
        prompt_metadata = {**prompt_metadata, "finalization": True}
        last_prompt_metadata = prompt_metadata
        if prompt_metadata.get("durable_memory_hits"):
            agent.emit_trace(task_state.run_id, "durable_memory_hits", {
                "count": prompt_metadata["durable_memory_hits"],
                "topics": prompt_metadata.get("durable_memory_topics", []),
            })
        if prompt_metadata.get("resume_context_hits"):
            agent.emit_trace(task_state.run_id, "resume_context_hits", {
                "count": prompt_metadata["resume_context_hits"],
                "status": prompt_metadata.get("resume_status", "no-checkpoint"),
            })
        agent.emit_trace(
            task_state.run_id, "prompt_built",
            {**prompt_metadata, "purpose": "finalization"},
        )
        agent.emit_trace(
            task_state.run_id, "model_requested",
            {"attempt": task_state.attempts, "purpose": "finalization"},
        )
        try:
            cache_kwargs = {}
            if getattr(agent.model_client, "supports_prompt_cache", False):
                cache_kwargs = {
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                    "prompt_cache_retention": "in_memory",
                }
            raw = agent.model_client.complete(prompt, agent.max_new_tokens, **cache_kwargs)
        except RuntimeError as exc:
            task_state.finish("failed", "model_error", str(exc))
            agent.create_checkpoint(
                task_state, "model_error", blocker="model_error",
                next_step="retry the finalization model request",
            )
            return agent.finish_run(task_state, last_prompt_metadata)
        kind, payload = agent.parse(raw)
        parse_details = {"kind": kind, "purpose": "finalization"}
        if kind == "retry" and isinstance(payload, dict):
            parse_details["error_code"] = payload.get("code", "invalid_protocol")
        agent.emit_trace(task_state.run_id, "model_parsed", parse_details)
        if kind == "final":
            task_state.finish("completed", "final_answer_returned", payload)
            agent.record({"role": "assistant", "content": payload})
            agent.promote_durable_memory(user_message, task_state.run_id)
            agent.create_checkpoint(task_state, "run_finished", next_step="task completed")
            return agent.finish_run(task_state, last_prompt_metadata)
        if kind == "retry":
            retry_code = str(payload.get("code", "invalid_protocol")) if isinstance(payload, dict) else "invalid_protocol"
            retry_message = str(payload.get("message", payload)) if isinstance(payload, dict) else str(payload)
            agent.record({"role": "system", "content": retry_message})
            agent.emit_trace(
                task_state.run_id, "model_retry",
                {"reason": retry_message, "error_code": retry_code, "purpose": "finalization"},
            )
        else:
            agent.record({
                "role": "system",
                "content": "tool calls are not allowed during finalization",
            })
            agent.emit_trace(
                task_state.run_id, "finalization_rejected",
                {"reason": "tool_call_after_budget", "tool_name": payload.get("name", "")},
            )
        task_state.finish("stopped", "step_limit_reached", "Stopped after reaching the tool-step limit.")
    else:
        task_state.finish("stopped", "retry_limit_reached", "Stopped after reaching the retry limit.")
    agent.create_checkpoint(
        task_state, task_state.stop_reason, blocker=task_state.stop_reason,
        next_step="review the trace and continue from the checkpoint",
    )
    return agent.finish_run(task_state, last_prompt_metadata)
