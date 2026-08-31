"""Section-based prompt assembly and character-budget reduction."""

from .workspace import clip


DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 3600,
    "memory": 1600,
    "relevant_memory": 1200,
    "history": 5200,
}
REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")


def _tail_clip(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    if limit <= 30:
        return text[-limit:]
    removed = len(text) - limit
    marker = ""
    for _ in range(3):
        marker = f"...[{removed} chars removed]\n"
        removed = len(text) - max(0, limit - len(marker))
    tail_budget = max(0, limit - len(marker))
    return marker + text[-tail_budget:] if tail_budget else marker[:limit]


class ContextManager:
    def __init__(self, agent, total_budget=DEFAULT_TOTAL_BUDGET, section_budgets=None):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update(section_budgets)

    def build(self, user_message):
        if self.agent.durable_memory_enabled:
            relevant = self.agent.memory.retrieve(user_message, limit=3)
        else:
            relevant = self.agent.memory.retrieve(
                user_message, limit=3, include_durable=False
            )
        relevant_text = "Relevant memory:\n" + "\n".join(
            f"- {item.get('text', '')}" for item in relevant
        ) if relevant else "Relevant memory:\n-"
        checkpoint = self.agent.checkpoint_text()
        originals = {
            "prefix": self.agent.prefix_state.text + ("\n\n" + checkpoint if checkpoint else ""),
            "memory": self.agent.memory.render(),
            "relevant_memory": relevant_text,
            "history": self._render_history(),
        }
        rendered = {
            name: _tail_clip(text, self.section_budgets[name])
            for name, text in originals.items()
        }
        initially_rendered = dict(rendered)
        reductions = []
        request = str(user_message)
        over = sum(len(value) for value in rendered.values()) + len(request) - self.total_budget
        floors = {name: max(20, budget // 4) for name, budget in self.section_budgets.items()}
        effective_floors = {
            name: min(floors[name], len(initially_rendered[name])) for name in rendered
        }
        for name in REDUCTION_ORDER:
            if over <= 0:
                break
            available = max(0, len(rendered[name]) - floors[name])
            remove = min(over, available)
            if remove:
                before = len(rendered[name])
                rendered[name] = _tail_clip(rendered[name], before - remove)
                actual = before - len(rendered[name])
                over -= actual
                reductions.append({"section": name, "removed_chars": actual})
        prompt = "\n\n".join(
            [
                rendered["prefix"],
                rendered["memory"],
                rendered["relevant_memory"],
                rendered["history"],
                "Current request:\n" + request,
            ]
        )
        metadata = {
            "original_chars": {name: len(text) for name, text in originals.items()},
            "rendered_chars": {name: len(text) for name, text in rendered.items()},
            "section_budgets": dict(self.section_budgets),
            "section_floors": floors,
            "effective_section_floors": effective_floors,
            "section_reduction_rates": {
                name: (
                    1.0 - (len(rendered[name]) / len(originals[name]))
                    if originals[name] else 0.0
                )
                for name in rendered
            },
            "current_request_chars": len(request),
            "current_request_retention_rate": 1.0,
            "prompt_chars": len(prompt),
            "soft_budget": self.total_budget,
            "budget_overflow_chars": max(0, len(prompt) - self.total_budget),
            "budget_reductions": reductions,
            "reduction_order": [item["section"] for item in reductions],
            "total_reduction_rate": (
                1.0 - (sum(len(value) for value in rendered.values()) /
                       sum(len(value) for value in originals.values()))
                if sum(len(value) for value in originals.values()) else 0.0
            ),
            "history_event_count": len(self.agent.session.get("history", [])),
            "relevant_memory_count": len(relevant),
            "durable_memory_hits": sum(
                item.get("kind") == "durable" for item in relevant
            ),
            "durable_memory_topics": sorted({
                item.get("topic", "") for item in relevant
                if item.get("kind") == "durable" and item.get("topic")
            }),
            "resume_context_hits": int(bool(checkpoint)),
            "checkpoint_context_included": bool(checkpoint),
            "prefix_hash": self.agent.prefix_state.hash,
            "workspace_fingerprint": self.agent.prefix_state.workspace_fingerprint,
            "prompt_cache_key": self.agent.prefix_state.hash,
            "resume_status": self.agent.resume_state.get("status", "no-checkpoint"),
            "stale_paths": list(self.agent.resume_state.get("stale_paths", [])),
            "runtime_identity_mismatch_fields": list(
                self.agent.resume_state.get("mismatch_fields", [])
            ),
            "stale_summary_invalidations": int(
                self.agent.resume_state.get("stale_summary_invalidations", 0)
            ),
        }
        return prompt, metadata

    def _render_history(self):
        history = list(self.agent.session.get("history", []))
        recent_start = max(0, len(history) - 6)
        lines = ["History:"]
        seen_old_reads = set()
        for index, event in enumerate(history):
            recent = index >= recent_start
            role = str(event.get("role", "event"))
            content = str(event.get("content", ""))
            if not recent and event.get("tool_name") == "read_file":
                path = str(event.get("tool_args", {}).get("path", ""))
                if path in seen_old_reads:
                    continue
                seen_old_reads.add(path)
            limit = 900 if recent else 80
            lines.append(f"- {role}: {clip(content.replace(chr(10), ' '), limit)}")
        return "\n".join(lines)
